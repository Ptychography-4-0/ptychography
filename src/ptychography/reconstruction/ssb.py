import numpy as np
import sparse
import numba

from libertem.udf import UDF
from libertem.common.container import MaskContainer
from libertem.masks import circular
from libertem.corrections.coordinates import identity

from .common import wavelength, get_shifted


def empty_mask(mask_shape, dtype):
    return sparse.zeros(mask_shape, dtype=dtype)


def mask_pair_subpix(cy, cx, sy, sx, filter_center, semiconv_pix, cutoff, mask_shape):
    filter_positive = circular(
        centerX=cx+sx, centerY=cy+sy,
        imageSizeX=mask_shape[1], imageSizeY=mask_shape[0],
        radius=semiconv_pix,
        antialiased=True
    )

    filter_negative = circular(
        centerX=cx-sx, centerY=cy-sy,
        imageSizeX=mask_shape[1], imageSizeY=mask_shape[0],
        radius=semiconv_pix,
        antialiased=True
    )
    mask_positive = filter_center * filter_positive * (filter_negative == 0)
    mask_negative = filter_center * filter_negative * (filter_positive == 0)
    return mask_positive, mask_negative


@numba.njit
def mask_tile_pair(center_tile, tile_origin, tile_shape, filter_center, sy, sx):

    sy, sx, = np.int(np.round(sy)), np.int(np.round(sx))
    positive_tile = np.zeros_like(center_tile)
    negative_tile = np.zeros_like(center_tile)
    # We get from negative coordinates,
    # that means it looks like shifted to positive
    target_tup_p, offsets_p = get_shifted(
        arr_shape=np.array(filter_center.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((-sy, -sx))
    )
    # We get from positive coordinates,
    # that means it looks like shifted to negative
    target_tup_n, offsets_n = get_shifted(
        arr_shape=np.array(filter_center.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((sy, sx))
    )

    sta_y, sto_y, sta_x, sto_x = target_tup_p.flatten()
    off_y, off_x = offsets_p
    positive_tile[sta_y:sto_y, sta_x:sto_x] = filter_center[
        sta_y+off_y:sto_y+off_y,
        sta_x+off_x:sto_x+off_x
    ]
    sta_y, sto_y, sta_x, sto_x = target_tup_n.flatten()
    off_y, off_x = offsets_n
    negative_tile[sta_y:sto_y, sta_x:sto_x] = filter_center[
        sta_y+off_y:sto_y+off_y,
        sta_x+off_x:sto_x+off_x
    ]

    mask_positive = center_tile * positive_tile * (negative_tile == 0)
    mask_negative = center_tile * negative_tile * (positive_tile == 0)

    return (mask_positive, target_tup_p, offsets_p, mask_negative, target_tup_n, offsets_n)


def mask_pair_shift(cy, cx, sy, sx, filter_center, semiconv_pix, cutoff, mask_shape):
    (mask_positive, target_tup_p, offsets_p,
    mask_negative, target_tup_n, offsets_n) = mask_tile_pair(
        center_tile=np.array(filter_center),
        tile_origin=np.array((0, 0)),
        tile_shape=np.array(mask_shape),

        filter_center=np.array(filter_center),
        sy=sy,
        sx=sx
    )
    return mask_positive, mask_negative


def generate_mask(cy, cx, sy, sx, filter_center, semiconv_pix,
                  cutoff, mask_shape, dtype, method='subpix'):
    # 1st diffraction order and primary beam don't overlap
    if sx**2 + sy**2 > 4*np.sum(semiconv_pix**2):
        return empty_mask(mask_shape, dtype=dtype)

    if np.allclose((sy, sx), (0, 0)):
        # The zero order component (0, 0) is special, comes out zero with above code
        m_0 = filter_center / filter_center.sum()
        return sparse.COO(m_0.astype(dtype))

    params = dict(
        cy=cy, cx=cx, sy=sy, sx=sx,
        filter_center=filter_center,
        semiconv_pix=semiconv_pix,
        cutoff=cutoff,
        mask_shape=mask_shape,
    )

    if method == 'subpix':
        mask_positive, mask_negative = mask_pair_subpix(**params)
    elif method == 'shift':
        mask_positive, mask_negative = mask_pair_shift(**params)
    else:
        raise ValueError(f"Unsupported method {method}. Allowed are 'subpix' and 'shift'")

    non_zero_positive = mask_positive.sum()
    non_zero_negative = mask_negative.sum()

    if non_zero_positive >= cutoff and non_zero_negative >= cutoff:
        m = (
            mask_positive / non_zero_positive
            - mask_negative / non_zero_negative
        ) / 2
        return sparse.COO(m.astype(dtype))
    else:
        # Exclude small, missing or unbalanced trotters
        return empty_mask(mask_shape, dtype=dtype)


def generate_masks(reconstruct_shape, mask_shape, dtype, lamb, dpix, semiconv,
        semiconv_pix, transformation=None, center=None, cutoff=1, cutoff_freq=np.float32('inf'),
        method='subpix'):

    reconstruct_shape = np.array(reconstruct_shape)

    dpix = np.array(dpix)

    d_Kf = np.sin(semiconv)/lamb/semiconv_pix
    d_Qp = 1/dpix/reconstruct_shape

    if center is None:
        center = np.array(mask_shape) / 2

    if transformation is None:
        transformation = identity()

    cy, cx = center

    filter_center = circular(
        centerX=cx, centerY=cy,
        imageSizeX=mask_shape[1], imageSizeY=mask_shape[0],
        radius=semiconv_pix,
        antialiased=True
    )

    half_reconstruct = (reconstruct_shape[0]//2 + 1, reconstruct_shape[1])
    masks = []

    for row in range(half_reconstruct[0]):
        for column in range(half_reconstruct[1]):
            # Do an fftshift of q and p
            qp = np.array((row, column))
            flip = qp > (reconstruct_shape / 2)
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - reconstruct_shape[flip]

            if np.sum(real_qp**2) > cutoff_freq**2:
                masks.append(empty_mask(mask_shape, dtype=dtype))
                continue

            # Shift of diffraction order relative to zero order
            # without rotation in physical coordinates
            real_sy_phys, real_sx_phys = real_qp * d_Qp
            # We apply the transformation backwards to go
            # from physical orientation to detector orientation,
            # while the forward direction in center of mass analysis
            # goes from detector coordinates to physical coordinates
            # Afterwards, we transform from physical detector coordinates
            # to pixel coordinates
            sy, sx = ((real_sy_phys, real_sx_phys) @ transformation) / d_Kf

            masks.append(generate_mask(
                cy=cy, cx=cx, sy=sy, sx=sx,
                filter_center=filter_center,
                semiconv_pix=semiconv_pix,
                cutoff=cutoff,
                mask_shape=mask_shape,
                dtype=dtype,
                method=method,
            ))

    # Since we go through masks in order, this gives a mask stack with
    # flattened (q, p) dimension to work with dot product and mask container
    masks = sparse.stack(masks)
    return masks


class SSB_UDF(UDF):

    def __init__(self, U, dpix, semiconv, semiconv_pix,
                 dtype=np.float32, center=None, mask_container=None,
                 transformation=None, cutoff=1, method='subpix'):
        '''
        Parameters

        U: float
            The acceleration voltage U in kV

        center: (float, float)

        dpix: float or Iterable(y, x)
            STEM pixel size in m

        semiconv: float
            STEM semiconvergence angle in radians

        dtype: np.dtype
            dtype to perform the calculation in

        semiconv_pix: float
            Diameter of the primary beam in the diffraction pattern in pixels

        transformation: numpy.ndarray() of shape (2, 2) or None
            Transformation matrix to apply to shift vectors. This allows to adjust for scan rotation
            and mismatch of detector coordinate system handedness, such as flipped y axis for MIB.

        mask_container: MaskContainer
            Hack to pass in a precomputed mask stack when using with single thread live data
            or with an inline executor.
            The proper fix is https://github.com/LiberTEM/LiberTEM/issues/335

        cutoff : int
            Minimum number of pixels in a trotter

        method : 'subpix' or 'shift'
            Method to use for generating the mask stack
        '''
        super().__init__(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                         dtype=dtype, center=center, mask_container=mask_container,
                         transformation=transformation, cutoff=cutoff, method=method)

    def get_result_buffers(self):
        dtype = np.result_type(np.complex64, self.params.dtype)
        return {
            'pixels': self.buffer(
                kind="single", dtype=dtype, extra_shape=self.reconstruct_shape,
                where='device'
            ),
        }

    @property
    def reconstruct_shape(self):
        return tuple(self.meta.dataset_shape.nav)

    def get_task_data(self):
        # shorthand, cupy or numpy
        xp = self.xp

        if self.meta.device_class == 'cpu':
            backend = 'numpy'
        elif self.meta.device_class == 'cuda':
            backend = 'cupy'
        else:
            raise ValueError("Unknown device class")

        # Hack to pass a fixed external container
        # In particular useful for single-process live processing
        # or inline executor
        if self.params.mask_container is None:
            masks = generate_masks(
                reconstruct_shape=self.reconstruct_shape,
                mask_shape=tuple(self.meta.dataset_shape.sig),
                dtype=self.params.dtype,
                lamb=wavelength(self.params.U),
                dpix=self.params.dpix,
                semiconv=self.params.semiconv,
                semiconv_pix=self.params.semiconv_pix,
                center=self.params.center,
                transformation=self.params.transformation,
                cutoff=self.params.cutoff,
                method=self.params.method,
            )
            container = MaskContainer(
                mask_factories=lambda: masks, dtype=masks.dtype,
                use_sparse='scipy.sparse.csc', count=masks.shape[0], backend=backend
            )
        else:
            container = self.params.mask_container
            target_size = (self.reconstruct_shape[0] // 2 + 1)*self.reconstruct_shape[1]
            container_shape = container.computed_masks.shape
            expected_shape = (target_size, ) + tuple(self.meta.dataset_shape.sig)
            if container_shape != expected_shape:
                raise ValueError(
                    f"External mask container doesn't have the expected shape. "
                    f"Got {container_shape}, expected {expected_shape}. "
                    "Mask count (self.meta.dataset_shape.nav[0] // 2 + 1) "
                    "* self.meta.dataset_shape.nav[1], "
                    "Mask shape self.meta.dataset_shape.sig. "
                    "The methods generate_masks_*() help to generate a suitable mask stack."
                )
        ds_nav = tuple(self.meta.dataset_shape.nav)
        y_positions, x_positions = np.mgrid[0:ds_nav[0], 0:ds_nav[1]]

        # Precalculated values for Fourier transform
        row_steps = -2j*np.pi*np.linspace(0, 1, self.reconstruct_shape[0], endpoint=False)
        col_steps = -2j*np.pi*np.linspace(0, 1, self.reconstruct_shape[1], endpoint=False)

        if self.meta.roi is None:
            y_map = y_positions.flatten()
            x_map = x_positions.flatten()
        else:
            y_map = y_positions[self.meta.roi]
            x_map = x_positions[self.meta.roi]

        steps_dtype = np.result_type(np.complex64, self.params.dtype)

        cy, cx = self.params.center
        fy, fx = tuple(self.meta.dataset_shape.sig)

        return {
            "masks": container,
            # Frame positions in the dataset masked by ROI
            # to easily access position in dataset when
            # processing with ROI applied
            "y_map": xp.array(y_map),
            "x_map": xp.array(x_map),
            "row_steps": xp.array(row_steps.astype(steps_dtype)),
            "col_steps": xp.array(col_steps.astype(steps_dtype)),
            "backend": backend
        }

    def merge(self, dest, src):
        dest['pixels'][:] = dest['pixels'] + src['pixels']

    def process_tile(self, tile):
        # shorthand, cupy or numpy
        xp = self.xp

        tile_start = self.meta.slice.origin[0]
        tile_depth = tile.shape[0]

        buffer_frame = xp.zeros_like(self.results.pixels)
        half_y = buffer_frame.shape[0] // 2 + 1

        y_indices = self.task_data.y_map[tile_start:tile_start+tile_depth]
        x_indices = self.task_data.x_map[tile_start:tile_start+tile_depth]

        factors_dtype = np.result_type(np.complex64, self.params.dtype)

        fourier_factors_row = xp.exp(
            y_indices[:, np.newaxis, np.newaxis]
            * self.task_data.row_steps[np.newaxis, :half_y, np.newaxis]
        ).astype(factors_dtype)
        fourier_factors_col = xp.exp(
            x_indices[:, np.newaxis, np.newaxis]
            * self.task_data.col_steps[np.newaxis, np.newaxis, :]
        ).astype(factors_dtype)

        masks = self.task_data.masks.get(
            self.meta.slice, transpose=True, backend=self.task_data.backend
        )
        tile_flat = tile.reshape(tile.shape[0], -1)

        dot_result = tile_flat @ masks
        dot_result = dot_result.reshape((tile_depth, half_y, buffer_frame.shape[1]))

        buffer_frame[:half_y] = (dot_result*fourier_factors_row*fourier_factors_col).sum(axis=0)
        # patch accounts for even and odd sizes
        # FIXME make sure this is correct using an example that transmits also
        # the high spatial frequencies
        patch = buffer_frame.shape[0] % 2
        # We skip the first row since it would be outside the FOV
        extracted = buffer_frame[1:buffer_frame.shape[0] // 2 + patch]
        # The coordinates of the bottom half are inverted and
        # the zero column is rolled around to the front
        # The real part is inverted
        buffer_frame[half_y:] = -xp.conj(
            xp.roll(xp.flip(xp.flip(extracted, axis=0), axis=1), shift=1, axis=1)
        )
        self.results.pixels[:] += buffer_frame

    def get_backends(self):
        return ('numpy', 'cupy')

    def get_tiling_preferences(self):
        dtype = np.result_type(np.complex64, self.params.dtype)
        result_size = np.prod(self.reconstruct_shape) * dtype.itemsize
        if self.meta.device_class == 'cuda':
            total_size = 1000e6
            good_depth = max(1, total_size / result_size)
            return {
                "depth": good_depth,
                "total_size": total_size,
            }
        else:
            good_depth = max(1, 1e6 / result_size)
            return {
                "depth": int(good_depth),
                "total_size": 1e6,
            }


def get_results(udf_result):
    '''
    Derive real space wave front from Fourier space

    To be included in UDF after https://github.com/LiberTEM/LiberTEM/issues/628
    '''
    # Since we derive the wave front with a linear function from intensities,
    # but the wave front is inherently an amplitude,
    # we take the square root of the calculated amplitude and
    # combine it with the calculated phase.
    rec = np.fft.ifft2(udf_result["pixels"].data)
    amp = np.abs(rec)
    phase = np.angle(rec)
    return np.sqrt(amp) * np.exp(1j*phase)
