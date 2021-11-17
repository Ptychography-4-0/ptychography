import math

import numpy as np
import numba
import numba.cuda
import scipy.constants as const
import scipy.sparse
import scipy.ndimage

from libertem.corrections import coordinates
from libertem.common.numba import rmatmul


# Calculation of the relativistic electron wavelength in meters
def wavelength(U):
    '''
    Calculate the electron wavelength

    Parameters
    ----------

    U : float
        Acceleration voltage in kV

    Returns
    -------

    wavelength : float
        Wavelength in m

    Examples
    --------

    >>> wavelength(300)
    1.9687489006848795e-12
    '''
    e = const.elementary_charge  # Elementary charge  !!! 1.602176634×10−19
    h = const.Planck  # Planck constant    !!! 6.62607004 × 10-34
    c = const.speed_of_light  # Speed of light
    m_0 = const.electron_mass  # Electron rest mass

    T = e*U*1000
    lambda_e = h*c/(math.sqrt(T**2+2*T*m_0*(c**2)))
    return lambda_e


@numba.njit
def offset(s1, s2):
    o1, ss1 = s1
    o2, ss2 = s2
    return o2 - o1


@numba.njit
def shift_by(sl, shift):
    origin, shape = sl
    return (
        origin + shift,
        shape
    )


@numba.njit
def shift_to(s1, origin):
    o1, ss1 = s1
    return (
        origin,
        ss1
    )


@numba.njit
def intersection(s1, s2):
    o1, ss1 = s1
    o2, ss2 = s2
    # Adapted from libertem.common.slice
    new_origin = np.maximum(o1, o2)
    new_shape = np.minimum(
        (o1 + ss1) - new_origin,
        (o2 + ss2) - new_origin,
    )
    new_shape = np.maximum(0, new_shape)
    return (new_origin, new_shape)


@numba.njit
def get_shifted(arr_shape: tuple, tile_origin: tuple, tile_shape: tuple, shift: tuple):
    '''
    Calculate the slices to cut out a shifted part of a 2D source
    array and place it into a target array, including tiling support.

    This works with negative and positive integer shifts.
    '''
    # TODO this could be adapted for full sig, nav, n-D etc support
    # and included as a method in Slice?
    full_slice = (np.array((0, 0)), arr_shape)
    tileslice = (tile_origin, tile_shape)
    shifted = shift_by(tileslice, shift)
    isect = intersection(full_slice, shifted)
    if np.prod(isect[1]) == 0:
        return (
            np.array([(0, 0), (0, 0)]),
            np.array([0, 0])
        )
    # We measure by how much we have clipped the zero point
    # This is zero if we didn't shift into the negative region beyond the original array
    clip = offset(shifted, isect)
    # Now we move the intersection to (0, 0) plus the amount we clipped
    # so that the overlap region is moved by the correct amount, in total
    targetslice = shift_by(shift_to(isect, np.array((0, 0))), clip)
    start = targetslice[0]
    length = targetslice[1]
    target_tup = np.stack((start, start+length), axis=1)
    offsets = isect[0] - targetslice[0]
    return (target_tup, offsets)


def to_slices(target_tup, offsets):
    target_slice = tuple(slice(s[0], s[1]) for s in target_tup)
    source_slice = tuple(slice(s[0] + o, s[1] + o) for (s, o) in zip(target_tup, offsets))
    return (target_slice, source_slice)


def bounding_box(array):
    # Based on https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    # But return values that work as start:stop slices
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    if np.any(rows):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return np.array(((y_min, y_max+1), (x_min, x_max+1)))
    else:
        return np.array([(0, 0), (0, 0)])


def diffraction_to_detector(
        lamb, diffraction_shape, pixel_size_real, pixel_size_detector,
        cy, cx, flip_y=False, scan_rotation=0.):
    '''
    Generate a function that transforms pixel coordinates from diffraction of
    real space to detector coordinates.

    When performing a forward calculation where a wave front is passed through an object
    function and projected in the far field, the projection is a Fourier transform.
    Each pixel in the fourier-transformed slice of real space corresponds to a diffraction
    angle in radian. This angle is a function of real space dimensions and wavelength.

    For ptychography, this forward-projected beam is put in relation with detector data.
    The projection and detector, however, are usually not calibrated in such a way that each
    pixel corresponds exactly to one diffracted pixel/beam from the object and illumination
    function. This function applies the correct scale, rotation and handedness to match
    the forward-projected beam with the detector data, given the necessary input parameters.

    cy, cx, flip_y and scan_rotation are chosen to correspond to the parameters in
    :meth:libertem.api.Context.create_com_analysis`.

    This function is designed to create a :code:`affine_transformation` parameter
    for :meth:`image_transformation_matrix`.

    Parameters
    ----------

    lamb : float
        Wavelength in m
    diffraction_shape : Tuple[int, int]
        Shape of the diffracted area
    pixel_size_real : float or tuple or ndarray
        Pixel size in m for the diffracted shape. Can be a Tuple[float, float] or array with
        two values for y and x
    pixel_size_detector : float or tuple or ndarray
        Pixel size in radian of the detector. Can be a Tuple[float, float] or array with two values
        for y and x. For free propagation into the far-field, this is the detector pixel size in
        m divided by the camera length for small angles.
    cy : float
        Y position of the central beam on the detector in pixel.
    cx : float
        X position of the central beam on the detector in pixel.
    flip_y : bool
        Flip the y axis of the detector coordinates
    scan_rotation : float
        Scan rotation in degrees

    Returns
    -------
    transform : callable(coords : numpy.ndarray) -> numpy.ndarray
        A function that accepts pixel coordinates in diffracted space as an
        array of shape (n, 2) with (y, x). Upper left
        corner is (0, 0). It returns pixel coordinates on the detector as floats of shape (n, 2).
    '''
    # Make sure broadcasting works as expected
    diffraction_shape = np.array(diffraction_shape)
    pixel_size_real = np.array(pixel_size_real)
    pixel_size_detector = np.array(pixel_size_detector)
    # Size of one pixel in radian of the diffracted shape.
    # Twice the diffraction_shape and half the pixel_size_real, i.e. the same physical
    # area at a finer pixel resolution, can capture higher diffraction orders and
    # therefore the FFT extends twice as far.
    # That means the pixel_size_diffracted should stay the same.
    # Longer wavelength means higher diffraction angles to get the same relative
    # path difference.
    pixel_size_diffracted = 1/pixel_size_real/diffraction_shape*lamb

    transformation = coordinates.identity()

    if flip_y:
        transformation = coordinates.flip_y() @ transformation

    transformation = coordinates.rotate_deg(scan_rotation) @ transformation

    transformation *= pixel_size_diffracted / pixel_size_detector

    def transform(coords: np.ndarray):
        # Shift the coordinates relative to the center of the
        # diffraction pattern
        relative_to_center = (coords - diffraction_shape / 2)

        return (relative_to_center @ transformation) + (cy, cx)

    return transform


def fftshift_coords(reconstruct_shape):
    '''
    Generate a function that performs an FFT shift of coordinates.

    On the detector, the central beam is near the center, while for native FFT
    the center is at the (0, 0) position of the result array. Instead of
    fft-shifting the result of a projection calculation to match the detector directly,
    one can instead calculate a transformation that picks from an unshifted FFT result
    in such a way that it matches a diffraction pattern in its native layout. This allows
    to combine the FFT shift with additional transformation steps. See also
    :meth:`image_transformation_matrix`.

    Parameters
    ----------

    reconstruct_shape : Tuple[int, int]
        Taget shape

    Returns
    -------
    Callable(coords:  numpy.ndarray)
        Function that accepts target coordinates with shape (n, 2), kind int
        and calculates the source coordinates with shape (n, 2), kind int
        so that taking from the source coordinates and placing into the target
        coordinates performs an FFT shift.
    '''
    reconstruct_shape = np.array(reconstruct_shape)

    def fftshift(coords):
        coords = np.array(coords)
        return (coords + (reconstruct_shape + 1) // 2) % reconstruct_shape

    return fftshift


def ifftshift_coords(reconstruct_shape):
    '''
    Generate a function that performs an inverse FFT shift of coordinates.

    On the detector, the central beam is near the center, while for native FFT
    the center is at the (0, 0) position of the result array. Instead of
    fft-shifting the result of a projection calculation to match the detector,
    one can instead calculate a transformation that picks from the detector data
    in such a way that an inverse FFT shift is performed. This allows to combine the
    inverse FFT shift with additional transformations. See also
    :meth:`image_transformation_matrix`.

    Parameters
    ----------

    reconstruct_shape : Tuple[int, int]
        Taget shape

    Returns
    -------
    Callable(coords:  numpy.ndarray)
        Function that accepts target coordinates with shape (n, 2), kind int
        and calculates the source coordinates with shape (n, 2), kind int
        so that taking from the source coordinates and placing into the target
        coordinates performs an inverse FFT shift.
    '''
    reconstruct_shape = np.array(reconstruct_shape)

    def ifftshift(coords):
        coords = np.array(coords)
        return (coords + reconstruct_shape // 2) % reconstruct_shape

    return ifftshift


@numba.njit(fastmath=True)
def _binning_elements(
        multi_target, multi_y_steps, multi_x_steps, multi_upleft,
        multi_y_vectors, multi_x_vectors):
    n_entries = int(np.sum(multi_y_steps*multi_x_steps))
    source_array = np.empty((n_entries, 2), dtype=np.float32)
    target_array = np.empty((n_entries, 2), dtype=np.int32)
    index = 0
    for i in range(len(multi_target)):
        for y in range(multi_y_steps[i]):
            for x in range(multi_x_steps[i]):
                source_coord = (
                    multi_upleft[i]
                    + y / multi_y_steps[i] * multi_y_vectors[i]
                    + x / multi_x_steps[i] * multi_x_vectors[i]
                )
                source_array[index] = source_coord
                target_array[index] = multi_target[i]
                index += 1
    return source_array, target_array


@numba.njit(fastmath=True)
def _weights(targets, target_shape):
    counts = np.zeros(target_shape, dtype=np.int32)
    result = np.empty(len(targets), dtype=np.float32)
    for t in targets:
        counts[t[0], t[1]] += 1
    for i, t in enumerate(targets):
        result[i] = 1/counts[t[0], t[1]]
    return result


def image_transformation_matrix(
        source_shape, target_shape, affine_transformation, pre_transform=None,
        post_transform=None):
    '''
    Construct a sparse matrix that transforms a flattened source image stack to
    a flattened target image stack.

    A sparse matrix prodct can be a highly efficient method to apply a set of
    transformations to an image in one pass. This function constructs a sparse
    matrix that picks values from a source image to fill each pixel of the
    target image by first applying :code:`pre_transform()` to the target image
    indices to map them into a 2D vector space, then projecting the pixel
    outline in this vector space using :code:`affine_transformation()` to
    calculate the source pixel coordinates, and then using
    :code:`post_transform()` to map the source coordinates to indices in the
    source image. If the projected pixel is of size 1.5 or smaller in the source
    coordinates, the closest integer is chosen. If it is larger, the average of
    pixels within the projected pixel outline is chosen. This corresponds to
    scaling with order=0 in :meth:`scipy.ndimage.zoom`.

    :code:`pre_transform() and :code:`post_transform()` can also be used for
    shifting the center of :code:`affine_transformation()`.

    Parameters
    ----------
    source_shape, target_shape : Tuple[int, int]
        Shape of source and target image for bounds checking and index raveling
    affine_transformation : callable(coords -> coords)
        Transformation that maps intermediate coordinates, i.e. the result of
        applying :code:`pre_transform()`, to float source coordinates.
        It should be continuous, strictly monotone and approximately affine for the size
        of one pixel. :meth:`diffraction_to_detector` can be used to generate a suitable
        coordinate transformation function.
    pre_transform : callable(coords) -> coords
        Map target image indices to coordinates, typically euclidean. :code:`pre_transform()`
        should not change the scale of the coordinates. It is designed to be something like
        :meth:`ifftshift_coords` to un-scramble target coordinates so that coordinates that
        are close in the source image are also close in the
        un-scrambled intermediate coordinates generated by this function.
        This is identity by default.
    post_transform : callable(coords) -> coords(int)
        Map source image coordinates, typically euclidean, to source image indices.
        :code:`post_transform()` should not change the scale of the coordinates. By default it
        is :code:`np.round(...).astype(int)`.

    Returns
    -------
    scipy.sparse.csc_matrix
        Shape np.prod(source_shape), np.prod(target_shape)
    '''
    source_shape = tuple(source_shape)
    target_shape = tuple(target_shape)
    if pre_transform is None:
        def pre_transform(x):
            return x
    if post_transform is None:
        def post_transform(x):
            return np.round(x).astype(int)
    # Array with all coordinates in the target image
    target_coords = np.stack(
        np.mgrid[:target_shape[0], :target_shape[1]],
        axis=2
        ).reshape((np.prod(target_shape, dtype=np.int64), 2))
    # Obtain coordinates in a proper euclidean space
    intermediate_coords = pre_transform(target_coords)
    # Pixel corners
    upright = intermediate_coords + (0, 1)
    downleft = intermediate_coords + (1, 0)

    # Transform to euclidean source coordinates
    source_upleft = affine_transformation(intermediate_coords)
    source_upright = affine_transformation(upright)
    source_downleft = affine_transformation(downleft)
    # Calculate edge vectors and their lengths
    source_y_vectors = source_downleft - source_upleft
    source_y_vector_lengths = np.linalg.norm(source_y_vectors, axis=-1)
    source_x_vectors = source_upright - source_upleft
    source_x_vector_lengths = np.linalg.norm(source_x_vectors, axis=-1)

    # Determine which source pixels are so small that they only cover one pixel
    single_pixel = (source_y_vector_lengths < 1.5) & (source_x_vector_lengths < 1.5)

    # Extract and calculate source indices for single pixels
    single_target = target_coords[single_pixel]
    single_centers = source_upleft[single_pixel]
    single_source = post_transform(single_centers)

    # Crop source indices to the source image
    single_within_limits = np.all((single_source >= 0) & (single_source < source_shape), axis=-1)
    all_target = single_target[single_within_limits]
    all_source = single_source[single_within_limits]
    # They all have weight 1, being a single pixel
    all_data = np.ones(len(all_source), dtype=np.float32)

    # Prepare for multi-pixel / binning entries
    multi_y_vectors = source_y_vectors[~single_pixel]
    multi_y_vector_lengths = source_y_vector_lengths[~single_pixel]
    multi_x_vectors = source_x_vectors[~single_pixel]
    multi_x_vector_lengths = source_x_vector_lengths[~single_pixel]

    # Size of the bin for each target pixel
    multi_y_steps = np.round(multi_y_vector_lengths).astype(int)
    multi_x_steps = np.round(multi_x_vector_lengths).astype(int)

    if not np.all(single_pixel):
        # Numba workhorse to calculate all source coordinates
        # for the bins
        multi_intermediate_source, multi_target = _binning_elements(
            multi_target=target_coords[~single_pixel],
            multi_y_steps=multi_y_steps,
            multi_x_steps=multi_x_steps,
            multi_upleft=source_upleft[~single_pixel],
            multi_y_vectors=multi_y_vectors,
            multi_x_vectors=multi_x_vectors,
        )

        # Transform to source indices
        multi_source = post_transform(multi_intermediate_source)

        # Crop to source image
        multi_within_limits = np.all((multi_source >= 0) & (multi_source < source_shape), axis=-1)
        multi_source = multi_source[multi_within_limits]
        multi_target = multi_target[multi_within_limits]

        # Count how many entries per bin and use the inverse.
        # They have to be counted individually since they might be cropped
        multi_data = _weights(multi_target, target_shape)

        # Extend the arrays with multi-pixel portion
        all_target = np.concatenate((all_target, multi_target))
        all_source = np.concatenate((all_source, multi_source))
        all_data = np.concatenate((all_data, multi_data))

    # Convert to flat indices for matrix product
    flat_target = np.ravel_multi_index(all_target.T, target_shape)
    flat_source = np.ravel_multi_index(all_source.T, source_shape)

    # Construct the matrix
    result = scipy.sparse.csc_matrix(
        (all_data, (flat_source, flat_target)),
        dtype=np.float32,
        shape=(np.prod(source_shape), np.prod(target_shape))
    )
    return result


def apply_matrix(sources, matrix, target_shape):
    '''
    Apply a transformation matrix generated by :meth:`image_transformation_matrix` to
    a stack of images.

    Parameters
    ----------
    sources : array-like
        Array of shape (n, sy, sx) where (sy, sx) is the :code:`source_shape` parameter
        of :meth:`image_transformation_matrix`.
    matrix : array-like
        Matrix generated by :meth:`image_transformation_matrix` or equivalent
    target_shape : Tuple[int, int]
        :code:`source_shape` parameter of :meth:`image_transformation_matrix`.
        The result will be reshaped to :code:`(n, ) + target_shape`
    '''
    flat_sources = sources.reshape((-1, np.prod(sources.shape[-2:], dtype=int)))
    if isinstance(matrix, (scipy.sparse.csc_matrix, scipy.sparse.csr_matrix)):
        flat_result = rmatmul(flat_sources, matrix)
    else:
        flat_result = flat_sources @ matrix
    return flat_result.reshape(sources.shape[:-2] + tuple(target_shape))


def shifted_probes(probe, bins, xp=np, scipy=scipy):
    '''
    Calculated subpixel-shifted versions of the probe

    Parameters
    ----------
    probe : numpy.ndarray
    bins : int or Tuple[int, int]
        Number of antialiasing steps in y and x axis. Can be int as well

    Returns
    -------
    probes : numpy.ndarray
        4D, shape bins + probe.shape or (bins, bins) + probe.shape if bins is an int
    '''
    if isinstance(bins, int):
        bins = (bins, bins)
    assert isinstance(bins, (list, tuple))
    assert len(bins) == 2
    probes = xp.zeros(bins + probe.shape, dtype=probe.dtype)
    for y in range(bins[0]):
        for x in range(bins[1]):
            dy = y / bins[0]
            dx = x / bins[1]
            real = scipy.ndimage.shift(
                probe.real,
                shift=(dy, dx),
            )
            probes[y, x] = real
            if xp.iscomplexobj(probe):
                imag = scipy.ndimage.shift(
                    probe.imag,
                    shift=(dy, dx),
                )
                probes[y, x] += 1j*imag
    return probes


def aggregate_shifted_probes(probes, xp=np, scipy=scipy):
    '''
    Aggregate subpixel-shifted versions of the probe, undoing the shift

    Parameters
    ----------
    probes : numpy.ndarray
        4D, shape (y_bins, x_bins) + probe.shape

    Returns
    -------
    probe : numpy.ndarray
        2D, shape probe.shape
    '''
    probe = xp.zeros(probes.shape[2:], dtype=probes.dtype)
    for y in range(probes.shape[0]):
        for x in range(probes.shape[1]):
            dy = y / probes.shape[0]
            dx = x / probes.shape[1]
            real = scipy.ndimage.shift(
                probes[y, x].real,
                shift=(-dy, -dx),
            )
            probe += real
            if xp.iscomplexobj(probe):
                imag = scipy.ndimage.shift(
                    probes[y, x].imag,
                    shift=(-dy, -dx),
                )
                probe += 1j*imag
    return probe


@numba.njit(fastmath=True, nogil=True)
def rolled_object_probe_product_cpu(obj, probe, shifts, result_out, ifftshift=False):
    '''
    Multiply object and shifted illumination

    This function combines several steps that are relevant for ptychographic reconstruction:

    * Multiply an object function with a shifted illumination
    * Roll the object function indices around the edges
    * Optionally, perform an ifftshift to prepare the data for subsequent FFT

    These steps are combined in a single loop since each requires significant memory
    transfer if they are performed step-by-step. For performance reasons it doesn't perform a
    free subpixel shift, but picks the best option from a set of pre-calculated shifted versions.

    See :meth:`shifted_probes` for a function to calculate the shifted versions.

    Parameters
    ----------
    obj : numpy.ndarray
        2D array with the object
    probe : numpy.ndarray
        4D array with subpixel shifts of the probe, last two dimensions same size or
        smaller than obj.
    shifts : numpy.ndarray
        Array with shift vectors, shape (n, 2), kind float
    result_out : numpy.ndarray
        Array where the result is placed. Shape (n, ) + probe.shape
    ifftshift : bool
        place the product ifft-shifted into :code:`result_out`

    Returns
    -------
    subpixel_indices : np.ndarray
        The first two indices for :code:`probe`
    '''
    obj_y, obj_x = obj.shape
    assert len(shifts) == result_out.shape[0]
    assert probe.shape[2:] == result_out.shape[1:]
    assert len(probe.shape) == 4
    y_subpixels, x_subpixels = probe.shape[:2]
    int_shifts = shifts.astype(np.int32)
    subpixel_indices = (
        shifts * np.array((y_subpixels, x_subpixels))
    ).astype(np.int32) % np.array((y_subpixels, x_subpixels))
    for i in range(len(result_out)):
        for y in range(probe.shape[-2]):
            for x in range(probe.shape[-1]):
                source_y = (y + int_shifts[i, 0]) % obj_y
                source_x = (x + int_shifts[i, 1]) % obj_x
                y_subpixel = subpixel_indices[i, 0]
                x_subpixel = subpixel_indices[i, 1]
                if ifftshift:  # From source to target
                    target_y = (y + (probe.shape[-2] + 1) // 2) % probe.shape[-2]
                    target_x = (x + (probe.shape[-1] + 1) // 2) % probe.shape[-1]
                else:
                    target_y, target_x = y, x
                update = obj[source_y, source_x] * probe[y_subpixel, x_subpixel, y, x]
                result_out[i, target_y, target_x] = update
    return subpixel_indices


@numba.njit(fastmath=True, nogil=True)
def rolled_object_aggregation_cpu(obj_out, updates, shifts, fftshift=False):
    '''
    Aggregate shifted updates to an object function

    This function accumulates updates that are shifted relative to the object function
    using addition and rolls the indices within the object if necesssary.
    Optionally, it can fftshift the updates while integrating. Doing this in one loop allows to
    reduce the number of calls for each shift and reduces overall memory transfer.

    Parameters
    ----------

    obj_out : numpy.ndarray
        2D array with the object, modified in-place by this function
    updates : numpy.ndarray
        Array with updates, shape (n, ...)
    shifts : numpy.ndarray
        Array with shift vectors, shape (n, 2), kind int
    fftshift : bool
        Read the updates fft-shifted from :code:`updates`
    '''
    obj_y, obj_x = obj_out.shape
    assert len(shifts) == updates.shape[0]
    for i in range(updates.shape[0]):
        for y in range(updates.shape[1]):
            for x in range(updates.shape[2]):
                target_y = (y + shifts[i, 0]) % obj_y
                target_x = (x + shifts[i, 1]) % obj_x
                if fftshift:  # From target to source
                    source_y = (y + (updates.shape[1] + 1) // 2) % updates.shape[1]
                    source_x = (x + (updates.shape[2] + 1) // 2) % updates.shape[2]
                else:
                    source_y, source_x = y, x
                obj_out[target_y, target_x] += updates[i, source_y, source_x]


@numba.cuda.jit
def _rolled_object_probe_product_cuda(obj, probe, shifts, result_out, ifftshift):
    obj_y, obj_x = obj.shape
    y_subpixels, x_subpixels = probe.shape[:2]

    i, y, x = numba.cuda.grid(3)

    source_y = (y + int(shifts[i, 0])) % obj_y
    source_x = (x + int(shifts[i, 1])) % obj_x

    y_subpixel = int(shifts[i, 0] * y_subpixels) % y_subpixels
    x_subpixel = int(shifts[i, 1] * x_subpixels) % x_subpixels

    if i < result_out.shape[0] and y < result_out.shape[1] and x < result_out.shape[2]:
        if ifftshift:
            target_y = (y + (probe.shape[-2] + 1) // 2) % probe.shape[-2]
            target_x = (x + (probe.shape[-1] + 1) // 2) % probe.shape[-1]
        else:
            target_y, target_x = y, x
        update = obj[source_y, source_x] * probe[y_subpixel, x_subpixel, y, x]
        result_out[i, target_y, target_x] = update


def rolled_object_probe_product_cuda(obj, probe, shifts, result_out, ifftshift=False):
    '''
    Numba CUDA version of :meth:`rolled_object_probe_product_cpu`
    '''
    import cupy
    count = result_out.shape[0]
    threadsperblock = 32
    blockspergrid = (count + (threadsperblock - 1)) // threadsperblock
    assert len(shifts) == result_out.shape[0]
    assert probe.shape[2:] == result_out.shape[1:]
    assert len(probe.shape) == 4
    # Calculate subpixel indices here since calculating
    # on the GPU is impractical
    y_subpixels, x_subpixels = probe.shape[:2]
    subpixel_indices = (
        shifts * cupy.array((y_subpixels, x_subpixels))
    ).astype(np.int32) % cupy.array((y_subpixels, x_subpixels))
    _rolled_object_probe_product_cuda[
        (blockspergrid, result_out.shape[1], result_out.shape[2]), (32, 1, 1)
    ](obj, probe, shifts, result_out, ifftshift)
    return subpixel_indices


@numba.cuda.jit(device=True)
def add_complex_complex(a, coords, b):
    numba.cuda.atomic.add(
            a.imag, coords, b.imag
        )
    numba.cuda.atomic.add(
        a.real, coords, b.real
    )


@numba.cuda.jit(device=True)
def add_real_real(a, coords, b):
    numba.cuda.atomic.add(
        a, coords, b
    )


@numba.cuda.jit(device=True)
def add_complex_real(a, coords, b):
    numba.cuda.atomic.add(
        a.real, coords, b
    )


def _make_rolled_object_aggregation_cuda(add):
    @numba.cuda.jit
    def _rolled_object_aggregation_cuda(obj_out, updates, shifts, fftshift):
        obj_y, obj_x = obj_out.shape
        i, y, x = numba.cuda.grid(3)
        if i < updates.shape[0] and y < updates.shape[1] and x < updates.shape[2]:
            target_y = (y + shifts[i, 0]) % obj_y
            target_x = (x + shifts[i, 1]) % obj_x
            if fftshift:  # From target to source
                source_y = (y + (updates.shape[1] + 1) // 2) % updates.shape[1]
                source_x = (x + (updates.shape[2] + 1) // 2) % updates.shape[2]
            else:
                source_y, source_x = y, x
            add(obj_out, (target_y, target_x), updates[i, source_y, source_x])
    return _rolled_object_aggregation_cuda


_roac_complex_complex = _make_rolled_object_aggregation_cuda(add_complex_complex)
_roac_complex_real = _make_rolled_object_aggregation_cuda(add_complex_real)
_roac_real_real = _make_rolled_object_aggregation_cuda(add_real_real)


def rolled_object_aggregation_cuda(obj_out, updates, shifts, fftshift=False):
    '''
    Numba CUDA version of :meth:`rolled_object_aggregation_cpu`
    '''
    count = updates.shape[0]
    threadsperblock = 32
    blockspergrid = (count + (threadsperblock - 1)) // threadsperblock

    if obj_out.dtype.kind == 'c':
        if updates.dtype.kind == 'c':
            f = _roac_complex_complex
        else:
            f = _roac_complex_real
    else:
        f = _roac_real_real
    f[
        (blockspergrid, updates.shape[1], updates.shape[2]), (32, 1, 1)
    ](obj_out, updates, shifts, fftshift)


@numba.njit(fastmath=True, nogil=True)
def subpixel_probe_aggregation_cpu(probe_out, updates, subpixel_indices, fftshift=False):
    '''
    Aggregate subpixel_shifted updates to a probe function

    This function accumulates subpixel-shifted probe updates in separte bins
    selected by code:`subpixel_indices`.
    The subpixel aggregates can then be consolidated into a single probe update
    by undoing the subpixel shifts. Since subpixel shifting is expensive, it is advantageous to
    aggregate first.
    Optionally, it can fftshift the updates while integrating. Doing this in one loop allows to
    reduce the number of calls for each shift and reduces overall memory transfer.

    Parameters
    ----------

    probe_out : numpy.ndarray
        2D array with the object, modified in-place by this function
    updates : numpy.ndarray
        Array with updates, shape (n, ...)
    subpixel_indices : numpy.ndarray
        Array with subpixel indices, shape (n, 2), kind int
    fftshift : bool
        Read the updates fft-shifted from :code:`updates`
    '''
    assert len(subpixel_indices) == updates.shape[0]
    assert updates.shape[1:] == probe_out.shape[2:]
    assert np.min(subpixel_indices) >= 0
    # See https://github.com/numba/numba/issues/1269
    # assert np.all(np.amax(subpixel_indices, axis=0) < np.array(probe_out.shape[:2]))
    for i in range(updates.shape[0]):
        subpixel_y, subpixel_x = subpixel_indices[i]
        # Do it here because of https://github.com/numba/numba/issues/1269
        assert subpixel_y < probe_out.shape[0]
        assert subpixel_x < probe_out.shape[1]
        for y in range(updates.shape[1]):
            for x in range(updates.shape[2]):
                if fftshift:  # From target to source
                    source_y = (y + (updates.shape[1] + 1) // 2) % updates.shape[1]
                    source_x = (x + (updates.shape[2] + 1) // 2) % updates.shape[2]
                else:
                    source_y, source_x = y, x
                probe_out[subpixel_y, subpixel_x, y, x] += updates[i, source_y, source_x]


def _make_subpixel_probe_aggregation_cuda(add):
    @numba.cuda.jit
    def _subpixel_probe_aggregation_cuda(probe_out, updates, subpixel_indices, fftshift=False):
        pass
        i, y, x = numba.cuda.grid(3)
        if i < updates.shape[0] and y < updates.shape[1] and x < updates.shape[2]:
            subpixel_y, subpixel_x = subpixel_indices[i]
            # Do it here because of https://github.com/numba/numba/issues/1269
            assert subpixel_y < probe_out.shape[0]
            assert subpixel_x < probe_out.shape[1]
            if fftshift:  # From target to source
                source_y = (y + (updates.shape[1] + 1) // 2) % updates.shape[1]
                source_x = (x + (updates.shape[2] + 1) // 2) % updates.shape[2]
            else:
                source_y, source_x = y, x
            add(probe_out, (subpixel_y, subpixel_x, y, x), updates[i, source_y, source_x])

    return _subpixel_probe_aggregation_cuda


_spa_complex = _make_subpixel_probe_aggregation_cuda(add_complex_complex)
_spa_real = _make_subpixel_probe_aggregation_cuda(add_real_real)


def subpixel_probe_aggregation_cuda(probe_out, updates, subpixel_indices, fftshift=False):
    '''
    Numba CUDA version of :meth:`subpixel_probe_aggregation_cpu`
    '''
    assert len(subpixel_indices) == updates.shape[0]
    assert updates.shape[1:] == probe_out.shape[2:]
    assert np.min(subpixel_indices) >= 0
    assert probe_out.dtype.kind == updates.dtype.kind
    # See https://github.com/numba/numba/issues/1269
    # assert np.all(np.amax(subpixel_indices, axis=0) < np.array(probe_out.shape[:2]))
    count = updates.shape[0]
    threadsperblock = 32
    blockspergrid = (count + (threadsperblock - 1)) // threadsperblock

    if probe_out.dtype.kind == 'c':
        f = _spa_complex
    else:
        f = _spa_real

    f[
        (blockspergrid, updates.shape[1], updates.shape[2]), (32, 1, 1)
    ](probe_out, updates, subpixel_indices, fftshift)


@numba.njit(fastmath=True, nogil=True)
def rolled_object_probe_select_cpu(obj, probe, shifts, obj_out, probe_out, ifftshift=False):
    '''
    Select object and shifted illumination

    This function combines several steps that are relevant for ptychographic reconstruction:

    * Overlay an object function with a shifted illumination
    * Roll the object function indices around the edges
    * Optionally, perform an ifftshift to prepare the data for subsequent FFT

    It leaves the multiplication from :meth:`rolled_object_probe_product_cpu` to the user
    since some algorithms require both sides of the product to calculate the update function.

    These steps are combined in a single loop since each requires significant memory
    transfer if they are performed step-by-step. For performance reasons it doesn't perform a
    free subpixel shift, but picks the best option from a set of pre-calculated shifted versions.

    See :meth:`shifted_probes` for a function to calculate the shifted versions.

    Parameters
    ----------
    obj : numpy.ndarray
        2D array with the object
    probe : numpy.ndarray
        4D array with subpixel shifts of the probe, last two dimensions same size or
        smaller than obj.
    shifts : numpy.ndarray
        Array with shift vectors, shape (n, 2), kind float
    obj_out : numpy.ndarray
        Array where the selection from obj is placed. Shape (n, ) + probe.shape
    probe_out : numpy.ndarray
        Array where the selection from probe is placed. Shape (n, ) + probe.shape
    ifftshift : bool
        place the product ifft-shifted into :code:`obj_out` and :code:`probe_out`

    Returns
    -------
    subpixel_indices : np.ndarray
        The first two indices for :code:`probe`
    '''
    obj_y, obj_x = obj.shape
    assert len(shifts) == obj_out.shape[0]
    assert len(shifts) == probe_out.shape[0]
    assert probe.shape[2:] == obj_out.shape[1:]
    assert probe.shape[2:] == probe_out.shape[1:]
    assert len(probe.shape) == 4
    y_subpixels, x_subpixels = probe.shape[:2]
    int_shifts = shifts.astype(np.int32)
    subpixel_indices = (
        shifts * np.array((y_subpixels, x_subpixels))
    ).astype(np.int32) % np.array((y_subpixels, x_subpixels))
    for i in range(len(obj_out)):
        for y in range(probe.shape[-2]):
            for x in range(probe.shape[-1]):
                source_y = (y + int_shifts[i, 0]) % obj_y
                source_x = (x + int_shifts[i, 1]) % obj_x
                y_subpixel = subpixel_indices[i, 0]
                x_subpixel = subpixel_indices[i, 1]
                if ifftshift:  # From source to target
                    target_y = (y + (probe.shape[-2] + 1) // 2) % probe.shape[-2]
                    target_x = (x + (probe.shape[-1] + 1) // 2) % probe.shape[-1]
                else:
                    target_y, target_x = y, x
                obj_out[i, target_y, target_x] = obj[source_y, source_x]
                probe_out[i, target_y, target_x] = probe[y_subpixel, x_subpixel, y, x]
    return subpixel_indices


@numba.cuda.jit
def _rolled_object_probe_select_cuda(obj, probe, shifts, obj_out, probe_out, ifftshift):
    obj_y, obj_x = obj.shape
    y_subpixels, x_subpixels = probe.shape[:2]

    i, y, x = numba.cuda.grid(3)

    source_y = (y + int(shifts[i, 0])) % obj_y
    source_x = (x + int(shifts[i, 1])) % obj_x

    y_subpixel = int(shifts[i, 0] * y_subpixels) % y_subpixels
    x_subpixel = int(shifts[i, 1] * x_subpixels) % x_subpixels

    if i < obj_out.shape[0] and y < obj_out.shape[1] and x < obj_out.shape[2]:
        if ifftshift:
            target_y = (y + (probe.shape[-2] + 1) // 2) % probe.shape[-2]
            target_x = (x + (probe.shape[-1] + 1) // 2) % probe.shape[-1]
        else:
            target_y, target_x = y, x
        obj_out[i, target_y, target_x] = obj[source_y, source_x]
        probe_out[i, target_y, target_x] = probe[y_subpixel, x_subpixel, y, x]


def rolled_object_probe_select_cuda(obj, probe, shifts, obj_out, probe_out, ifftshift=False):
    '''
    Numba CUDA version of :meth:`rolled_object_probe_select_cpu`
    '''
    import cupy
    count = obj_out.shape[0]
    threadsperblock = 32
    blockspergrid = (count + (threadsperblock - 1)) // threadsperblock
    assert len(shifts) == obj_out.shape[0]
    assert len(shifts) == probe_out.shape[0]
    assert probe.shape[2:] == obj_out.shape[1:]
    assert probe.shape[2:] == probe_out.shape[1:]
    assert len(probe.shape) == 4
    # Calculate subpixel indices here since calculating
    # on the GPU is impractical
    y_subpixels, x_subpixels = probe.shape[:2]
    subpixel_indices = (
        shifts * cupy.array((y_subpixels, x_subpixels))
    ).astype(np.int32) % cupy.array((y_subpixels, x_subpixels))
    _rolled_object_probe_select_cuda[
        (blockspergrid, obj_out.shape[1], obj_out.shape[2]), (32, 1, 1)
    ](obj, probe, shifts, obj_out, probe_out, ifftshift)
    return subpixel_indices


@numba.njit(fastmath=True, nogil=True)
def trunc_divide_cpu(numerator, denominator, out, threshold=1e-6, fill=0):
    assert numerator.shape == denominator.shape
    assert numerator.shape == out.shape
    assert len(numerator.shape) == 3
    for i in range(numerator.shape[0]):
        for y in range(numerator.shape[1]):
            for x in range(numerator.shape[2]):
                if np.abs(denominator[i, y, x]) <= threshold:
                    out[i, y, x] = fill
                else:
                    out[i, y, x] = numerator[i, y, x] / denominator[i, y, x]


def trunc_divide_cuda(numerator, denominator, out, threshold=1e-6, fill=0):
    assert numerator.shape == denominator.shape
    assert numerator.shape == out.shape
    assert len(numerator.shape) == 3
    import cupy
    cupy.divide(
        numerator,
        denominator,
        out=out,
    )
    out[cupy.abs(denominator) <= threshold] = fill
