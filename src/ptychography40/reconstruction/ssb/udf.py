import functools

import numpy as np
import numba
import scipy.sparse
import sparseconverter

from libertem.udf import UDF
from libertem.common.container import MaskContainer

from ptychography40.reconstruction.ssb.trotters import generate_masks


@numba.njit(fastmath=True, cache=True, parallel=True, nogil=True)
def rmatmul_csc_fourier(n_threads, left_dense, right_data, right_indices, right_indptr,
                        coordinates, row_exp, col_exp, res_inout):
    '''
    Fold :meth:`SSB_UDF.merge_dot_result` into a sparse dot product from
    :meth:`libertem.common.numba._rmatmul_csc`

    The result can be directly merged into the result buffer instead of
    instantiating the intermediate dot result, which can be large. That way,
    this method can process entire memory-mapped partitions efficiently.
    Furthermore, it allows early skipping of trotters that are empty in this tile.

    It uses multithreading to process different parts of a tile in parallel.
    '''
    left_rows = left_dense.shape[0]
    p_size = col_exp.shape[1]
    q_size = row_exp.shape[1]
    # We subdivide in blocks per thread so that each thread
    # writes exclusively to its own part of an intermediate result buffer.
    # Using prange and automated '+=' merge leads to wrong results when using threading.
    blocksize = max(int(np.ceil(left_rows / n_threads)), 1)
    resbuf = np.zeros((n_threads, q_size, p_size), dtype=res_inout.dtype)
    # The blocks are processed in parallel
    for block in numba.prange(n_threads):
        start = block * blocksize
        stop = min((block + 1) * blocksize, left_rows)
        for left_row in range(start, stop):
            # Pixel coordinates in nav dimension
            y, x = coordinates[left_row]
            for q in range(q_size):
                for p in range(p_size):
                    # right_column is the mask index
                    right_column = q * p_size + p
                    # Descent into CSC data structure
                    offset = right_indptr[right_column]
                    items = right_indptr[right_column+1] - offset
                    if items > 0:
                        # We accumulate for the whole mask into acc
                        # before applying the phase factor
                        acc = 0
                        # Iterate over non-zero entries in this mask
                        for i in range(items):
                            index = i + offset
                            right_row = right_indices[index]
                            right_value = right_data[index]
                            acc += left_dense[left_row, right_row] * right_value
                        # Phase factor for this scan point and mask
                        factor = row_exp[y, q] * col_exp[x, p]
                        # Applying the factor, accumulate in per-thread buffer
                        resbuf[block, q, p] += acc * factor
    # Finally, accumulate per-thread buffer into result
    res_inout += np.sum(resbuf, axis=0)


class SSB_Base(UDF):
    def get_result_buffers(self):
        ''
        dtype = np.result_type(np.complex64, self.params.dtype)
        component_dtype = np.result_type(np.float32, self.params.dtype)
        return {
            'fourier': self.buffer(
                kind="single", dtype=dtype, extra_shape=self.reconstruct_shape,
                where='device'
            ),
            'complex': self.buffer(
                kind="single", dtype=dtype, extra_shape=self.reconstruct_shape,
                use='result_only',
            ),
            'amplitude': self.buffer(
                kind="single", dtype=component_dtype, extra_shape=self.reconstruct_shape,
                use='result_only',
            ),
            'phase': self.buffer(
                kind="single", dtype=component_dtype, extra_shape=self.reconstruct_shape,
                use='result_only',
            ),
        }

    def get_task_data(self):
        ''
        # shorthand, cupy or numpy
        xp = self.xp

        # Precalculated LUT for Fourier transform
        # The y axis is trimmed in half since the full trotter stack is symmetric,
        # i.e. the missing half can be reconstructed from the other results
        row_steps = -2j*np.pi*np.linspace(0, 1, self.reconstruct_shape[0], endpoint=False)
        col_steps = -2j*np.pi*np.linspace(0, 1, self.reconstruct_shape[1], endpoint=False)

        half_y = self.reconstruct_shape[0] // 2 + 1
        full_x = self.reconstruct_shape[1]

        # This creates a 2D array of row x spatial frequency
        row_exp = np.exp(
            row_steps[:, np.newaxis]
            * np.arange(half_y)[np.newaxis, :]
        )
        # This creates a 2D array of col x spatial frequency
        col_exp = np.exp(
            col_steps[:, np.newaxis]
            * np.arange(full_x)[np.newaxis, :]
        )

        steps_dtype = np.result_type(np.complex64, self.params.dtype)

        return {
            "row_exp": xp.array(row_exp.astype(steps_dtype)),
            "col_exp": xp.array(col_exp.astype(steps_dtype)),
        }

    @property
    def reconstruct_shape(self):
        return tuple(self.meta.dataset_shape.nav)

    def merge(self, dest, src):
        ''
        dest.fourier[:] += src.fourier

    def merge_dot_result(self, dot_result):
        # shorthand, cupy or numpy
        xp = self.xp

        tile_depth = dot_result.shape[0]

        # We calculate only half of the Fourier transform due to the
        # inherent symmetry of the mask stack. In this case we
        # cut the y axis in half. The "+ 1" accounts for odd sizes
        # The mask stack is already trimmed in y direction to only contain
        # one of the trotter pairs
        half_y = self.results.fourier.shape[0] // 2 + 1

        # Get the real x and y indices within the dataset navigation dimension
        # for the current tile
        y_indices = self.meta.coordinates[:, 0]
        x_indices = self.meta.coordinates[:, 1]

        # This loads the correct entries for the current tile from the pre-calculated
        # 1-D DFT matrices using the x and y indices of the frames in the current tile
        # fourier_factors_row is already trimmed for half_y, but the explicit index
        # is kept for clarity
        fourier_factors_row = self.task_data.row_exp[y_indices, :half_y, np.newaxis]
        fourier_factors_col = self.task_data.col_exp[x_indices, np.newaxis, :]

        # The masks are in order [row, col], but flattened. Here we undo the flattening
        dot_result = dot_result.reshape((tile_depth, half_y, self.results.fourier.shape[1]))

        # Calculate the part of the Fourier transform for this tile.
        # Reconstructed shape corresponds to depth of mask stack, see above.
        # Shape of operands:
        # dot_result: (tile_depth, reconstructed_shape_y // 2 + 1, reconstructed_shape_x)
        # fourier_factors_row: (tile_depth, reconstructed_shape_y // 2 + 1, 1)
        # fourier_factors_col: (tile_depth, 1, reconstructed_shape_x)
        # The einsum is equivalent to
        # (dot_result*fourier_factors_row*fourier_factors_col).sum(axis=0)
        # The product of the Fourier factors for row and column implicitly builds part
        # of the full 2D DFT matrix through NumPy broadcasting
        # The sum of axis 0 (tile depth) already performs the accumulation for the tile
        # stack before patching the missing half for the full result.
        # Einsum is about 3x faster in this scenario, likely because of not building a large
        # intermediate array before summation
        self.results.fourier[:half_y] += xp.einsum(
            'i...,i...,i...',
            dot_result,
            fourier_factors_row,
            fourier_factors_col
        )

    def postprocess(self):
        ''
        # shorthand, cupy or numpy
        xp = self.xp
        half_y = self.results.fourier.shape[0] // 2 + 1
        # patch accounts for even and odd sizes
        # FIXME make sure this is correct using an example that transmits also
        # the high spatial frequencies
        patch = self.results.fourier.shape[0] % 2
        # We skip the first row since it would be outside the FOV
        extracted = self.results.fourier[1:self.results.fourier.shape[0] // 2 + patch]
        # The coordinates of the bottom half are inverted and
        # the zero column is rolled around to the front
        # The real part is inverted
        self.results.fourier[half_y:] = -xp.conj(
            xp.roll(xp.flip(xp.flip(extracted, axis=0), axis=1), shift=1, axis=1)
        )

    def get_results(self):
        '''
        Since we derive the wave front with a linear function from intensities,
        but the wave front is inherently an amplitude,
        we take the square root of the calculated amplitude and
        combine it with the calculated phase.
        '''
        inverse = np.fft.ifft2(self.results.fourier)
        amp = np.sqrt(np.abs(inverse))
        phase = np.angle(inverse)
        return {
            'fourier': self.results.fourier,
            'complex': amp * np.exp(1j*phase),
            'amplitude': amp,
            'phase': phase,
        }


class SSB_UDF(SSB_Base):
    '''
    UDF to perform ptychography using the single side band (SSB) method :cite:`Pennycook2015`.

    Parameters
    ----------

    lamb: float
        The illumination wavelength in m. The function :meth:`ptychography40.common.wavelength`
        allows to calculate the electron wavelength as a function of acceleration voltage.
    dpix: float or Iterable(y, x)
        STEM pixel size in m
    semiconv: float
        STEM semiconvergence angle in radians
    dtype: np.dtype
        dtype to perform the calculation in
    semiconv_pix: float
        Diameter of the primary beam in the diffraction pattern in pixels
    cy, cx : float, optional
        Position of the optical axis on the detector in px, center of illumination.
        Default: Center of the detector
    transformation: numpy.ndarray() of shape (2, 2) or None
        Transformation matrix to apply to shift vectors. This allows to adjust for scan rotation
        and mismatch of detector coordinate system handedness, such as flipped y axis for MIB.
    cutoff : int
        Minimum number of pixels in a trotter
    method : 'subpix' or 'shift'
        Method to use for generating the mask stack
    mask_container: MaskContainer
        Allows to pass in a precomputed mask stack when using with single thread live data
        or with an inline executor as a work-around. The number of masks is sanity-checked
        to match the other parameters.
        The proper fix is https://github.com/LiberTEM/LiberTEM/issues/335
    '''
    def __init__(self, lamb, dpix, semiconv, semiconv_pix,
                 dtype=np.float32, cy=None, cx=None, transformation=None,
                 cutoff=1, method='subpix', mask_container=None,):
        super().__init__(lamb=lamb, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                         dtype=dtype, cy=cy, cx=cx, mask_container=mask_container,
                         transformation=transformation, cutoff=cutoff, method=method)

    EFFICIENT_THREADS = 4

    def get_task_data(self):
        ''
        result = super().get_task_data()

        if self.meta.array_backend in sparseconverter.CPU_BACKENDS:
            backend = 'numpy'
        elif self.meta.array_backend in sparseconverter.CUDA_BACKENDS:
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
                lamb=self.params.lamb,
                dpix=self.params.dpix,
                semiconv=self.params.semiconv,
                semiconv_pix=self.params.semiconv_pix,
                cy=self.params.cy,
                cx=self.params.cx,
                transformation=self.params.transformation,
                cutoff=self.params.cutoff,
                method=self.params.method,
            )
            container = MaskContainer(
                mask_factories=lambda: masks, dtype=masks.dtype,
                use_sparse='scipy.sparse.csr', count=masks.shape[0], backend=backend
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

        result['masks'] = container
        result['backend'] = backend
        return result

    def process_tile(self, tile):
        ''
        # We flatten the signal dimension of the tile in preparation of the dot product
        tile_flat = tile.reshape((tile.shape[0], -1))
        # We calculate only half of the Fourier transform due to the
        # inherent symmetry of the mask stack. In this case we
        # cut the y axis in half. The "+ 1" accounts for odd sizes
        # The mask stack is already trimmed in y direction to only contain
        # one of the trotter pairs
        half_y = self.results.fourier.shape[0] // 2 + 1
        tpw = self.meta.threads_per_worker
        if self.meta.array_backend == self.BACKEND_CUPY:
            # Load the preconditioned trotter stack from the mask container:
            # * Coutout for current tile
            # * Flattened signal dimension
            # * Clean CSR matrix
            # * On correct device
            masks = self.task_data.masks.get(
                self.meta.slice, transpose=False, backend=self.task_data.backend,
                sparse_backend='scipy.sparse.csr'
            )
            # Skip an empty tile since the contribution is 0
            if masks.nnz == 0:
                return
            # This performs the trotter integration
            # As of now, cupy doesn't seem to support __rmatmul__ with sparse matrices
            dot_result = masks.dot(tile_flat.T).T
            self.merge_dot_result(dot_result)
        elif ((tpw is None) or (tpw >= self.EFFICIENT_THREADS)
                and self.meta.array_backend == self.BACKEND_NUMPY):
            # Load the preconditioned trotter stack from the mask container:
            # * Coutout for current tile
            # * Flattened signal dimension
            # * Transposed for right hand side of dot product
            #   (rmatmul_csc_fourier() takes care of putting things where they belong
            #    in the result)
            # * Clean CSC matrix
            # * On correct device (CPU)
            masks = self.task_data.masks.get(
                self.meta.slice, transpose=True, backend=self.task_data.backend,
                sparse_backend='scipy.sparse.csc'
            )
            # Skip an empty tile since the contribution is 0
            if masks.nnz == 0:
                return

            # This combines the trotter integration with merge_dot_result
            # into a Numba loop that eliminates a potentially large intermediate result
            # and allows efficient multithreading on a large tile
            rmatmul_csc_fourier(
                n_threads=self.meta.threads_per_worker,
                left_dense=tile_flat,
                right_data=masks.data,
                right_indices=masks.indices,
                right_indptr=masks.indptr,
                coordinates=self.meta.coordinates,
                row_exp=self.task_data.row_exp,
                col_exp=self.task_data.col_exp,
                res_inout=self.results.fourier[:half_y]
            )
        elif self.meta.array_backend == self.BACKEND_NUMPY:
            masks = self.task_data.masks.get(
                self.meta.slice, transpose=False, backend=self.task_data.backend,
                sparse_backend='scipy.sparse.csr'
            )
            # Skip an empty tile since the contribution is 0
            if masks.nnz == 0:
                return
            # This performs the trotter integration
            # Transposed copy of the input data for fast scipy.sparse __matmul__()
            dot_result = masks.dot(tile_flat.T.copy()).T
            self.merge_dot_result(dot_result)
        elif self.meta.array_backend in (self.BACKEND_SCIPY_CSR, self.BACKEND_CUPY_SCIPY_CSR):
            masks = self.task_data.masks.get(
                self.meta.slice, transpose=True, backend=self.task_data.backend,
                sparse_backend='scipy.sparse.csr'
            )
            # Skip an empty tile since the contribution is 0
            if masks.nnz == 0:
                return
            dot_result = tile_flat.dot(masks)
            dot_result = sparseconverter.for_backend(
                dot_result,
                sparseconverter.get_backend(self.results.fourier)
            )
            self.merge_dot_result(dot_result)
        else:
            raise RuntimeError(f"Didn't find implementation for backend {self.meta.array_backend}")

    def get_backends(self):
        ''
        return (
            self.BACKEND_CUPY,
            self.BACKEND_CUPY_SCIPY_CSR,
            self.BACKEND_SCIPY_CSR,
            self.BACKEND_NUMPY
        )

    def get_tiling_preferences(self):
        ''
        dtype = np.result_type(np.complex64, self.params.dtype)
        result_size = np.prod(self.reconstruct_shape) * dtype.itemsize
        if self.meta.device_class == 'cuda':
            free, total = self.xp.cuda.runtime.memGetInfo()
            total_size = min(100e6, free // 4)
            good_depth = max(1, total_size / result_size * 4)
            return {
                "depth": good_depth,
                "total_size": total_size,
            }
        else:
            tpw = self.meta.threads_per_worker
            if (tpw is None) or (tpw >= self.EFFICIENT_THREADS):
                # The parallel Numba loop can process entire partitions efficiently
                # since it accumulates in the result without large intermediate
                # data
                return {
                    "depth": self.TILE_DEPTH_MAX,
                    "total_size": self.TILE_SIZE_MAX,
                }
            else:
                # We limit the depth of a tile so that the intermediate
                # results from processing a tile fit into the CPU cache.
                good_depth = max(1, 1e6 / result_size)
                return {
                    "depth": int(good_depth),
                    # We reduce the size of a tile since a transposed copy of
                    # each tile will be taken to speed up the sparse matrix product
                    "total_size": 0.5e6,
                }


def crop_bin_params(rec_params, mask_params, binning_factor: int):
    '''
    Calculate parameters and vectors for binned SSB

    Parameters
    ----------

    rec_params, mask_params : dict
        Parameters for unbinned reconstruction and mask generation
    binning_factor : int

    Returns
    -------
    (new_rec_params, new_mask_params, y_binner, x_binner)
        :code:`y_binner` and :code:`x_binner` are NumPy arrays that perform binning
        via :code:`y_binner @ frame @ x_binner`.

    '''
    center = int(np.ceil(rec_params["semiconv_pix"] / binning_factor))
    size = 2 * center

    def crop_bin_vector(length, origin):
        bins = np.zeros((length, size), dtype=np.float32)
        for i in range(size):
            start = origin + i*binning_factor
            stop = start + binning_factor
            bins[start:stop, i] = 1/binning_factor
        return bins

    new_rec_params = rec_params.copy()
    new_rec_params['cy'] = center
    new_rec_params['cx'] = center
    new_rec_params['semiconv_pix'] = rec_params['semiconv_pix'] / binning_factor
    new_rec_params['cutoff'] = int(np.ceil(rec_params['cutoff'] / binning_factor**2))
    new_mask_params = mask_params.copy()
    new_mask_params['mask_shape'] = (size, size)
    new_mask_params['method'] = 'subpix'

    y_binner = crop_bin_vector(
        length=mask_params['mask_shape'][0],
        origin=int(rec_params['cy']) - binning_factor * center,
    ).T

    x_binner = crop_bin_vector(
        length=mask_params['mask_shape'][1],
        origin=int(rec_params['cx']) - binning_factor * center,
    )

    return (new_rec_params, new_mask_params, y_binner, x_binner)


def _get_binner(constructor, y_binner, x_binner):
    '''
    Generate a function that returns the right subset of the binners
    for a given sig slice.

    Parameters
    ----------

    constructor : function
        This function is applied to the subset in order to allow construction
        of different array types, such as CuPy arrays or sparse matrices.
    y_binner, x_binner:
        Binning matrices for the full frame, as returned by :func:`crop_bin_params`.

    Returns
    -------

    function
        A cached function that accepts the signal slice and returns suitable
        binning info (y_binner, x_binner, noop) so that noop is :code:`True`
        if any of the binners is all zero and that :code:`y_binner @ tile @ x_binner`
        calculates the contribution of a tile to the binned result.

    '''
    @functools.lru_cache(maxsize=512)
    def get(sig_slice):
        y_origin, x_origin = sig_slice.origin
        y_shape, x_shape = sig_slice.shape
        y_res = y_binner[:, y_origin:y_origin+y_shape]
        x_res = x_binner[x_origin:x_origin+x_shape]
        return (
            constructor(y_res),
            constructor(x_res),
            np.allclose(y_res, 0) or np.allclose(x_res, 0)
        )

    return get


class BinnedSSB_UDF(SSB_Base):
    '''
    Variant of the :class:`SSB_UDF` that crops and bins the data
    before applying the trotters. See also :cite: `Yang2015a`.

    Different from :class:`SSB_UDF`, this UDF accepts the trotters directly as
    :class:`scipy.sparse.csr_matrix` since no mask container is used.

    It benefits greatly from the improvements in LiberTEM 0.9 that allow
    efficient sharing of large parameters.

    :func:`crop_bin_params` can calculate the parameters for the trotters and the
    :code:`y_binner, x_binner` parameters.

    Parameters
    ----------
    y_binner, x_binner : np.ndarray
        As generated by :func:`crop_bin_params`
    csr_trotters : scipy.sparse.csr_matrix
        Result of :func:`ptychography40.reconstruction.ssb.trotters.generate_masks` with parameters
        modified by :func:`crop_bin_params` and converted to CSR.
    dtype
        dtype for the calculation

    Examples
    --------

    >>> from libertem.corrections.coordinates import flip_y, rotate_deg
    >>> rec_params = {
    ...     "dtype": np.float32,
    ...     "lamb": 2e-12,
    ...     "dpix": 12.7e-12,
    ...     "semiconv": 22.1346e-3,
    ...     "semiconv_pix": 31,
    ...     "transformation": rotate_deg(88) @ flip_y(),
    ...     "cx": 123,
    ...     "cy": 126,
    ...     "cutoff": 16,
    ... }
    >>> mask_params = {
    ...     'reconstruct_shape': (128, 128),
    ...     'mask_shape': (256, 256),
    ...     'method': 'shift',
    ... }
    >>> binned_rec_params, binned_mask_params, y_binner, x_binner = crop_bin_params(
    ...     rec_params=rec_params,
    ...     mask_params=mask_params,
    ...     binning_factor=5,
    ... )
    >>> binned_trotters = generate_masks(**binned_rec_params, **binned_mask_params)
    >>> flat_shape = (binned_trotters.shape[0], np.prod(binned_trotters.shape[1:]))
    >>> csr_trotters = binned_trotters.reshape(flat_shape).tocsr()
    >>> binned_ssb_udf = BinnedSSB_UDF(
    ...     y_binner=y_binner,
    ...     x_binner=x_binner,
    ...     csr_trotters=csr_trotters,
    ... )

    '''
    def __init__(self, y_binner, x_binner, csr_trotters: scipy.sparse.csr_matrix, dtype=np.float32):
        # make sure the cropped and binned region has a size divisible by two
        binned_size = np.sqrt(csr_trotters.shape[1])
        assert np.allclose(binned_size % 2, 0)
        super().__init__(
            y_binner=y_binner,
            x_binner=x_binner,
            csr_trotters=csr_trotters,
            dtype=dtype
        )

    def get_task_data(self):
        result = super().get_task_data()
        result['binner'] = _get_binner(self.xp.array, self.params.y_binner, self.params.x_binner)
        if self.meta.array_backend in sparseconverter.CPU_BACKENDS:
            result['trotters'] = self.params.csr_trotters
        elif self.meta.array_backend in sparseconverter.CUDA_BACKENDS:
            import cupy.sparse
            result['trotters'] = cupy.sparse.csr_matrix(self.params.csr_trotters)
        else:
            raise RuntimeError("Should not happen")
        return result

    def process_partition(self, tile):
        '''
        Processing a partition reduces amplification
        from processing many signal slices per frame
        on higher-resolution detectors.
        '''
        try:
            sig_slice = self.meta.sig_slice
        except AttributeError:
            sig_slice = self.meta.slice.sig
        y_binner, x_binner, noop = self.task_data.binner(sig_slice)
        if not noop:
            # A bit faster in that order, for some reason
            binned = y_binner @ (tile @ x_binner)
            binned_flat = binned.reshape((binned.shape[0], binned.shape[1]*binned.shape[2]))
            masks = self.task_data.trotters
            # Alternative: Use rmatmul_csc_fourier
            # Since the matrix is much smaller, this doesn't bring
            # so much benefit here.
            # half_y = self.results.fourier.shape[0] // 2 + 1
            # csr left hand mask is same as csc right hand mask
            # rmatmul_csc_fourier(
            #     n_threads=self.meta.threads_per_worker,
            #     left_dense=binned_flat,
            #     right_data=masks.data,
            #     right_indices=masks.indices,
            #     right_indptr=masks.indptr,
            #     coordinates=self.meta.coordinates,
            #     row_exp=self.task_data.row_exp,
            #     col_exp=self.task_data.col_exp,
            #     res_inout=self.results.fourier[:half_y]
            # )
            dot_result = masks.dot(binned_flat.T).T
            self.merge_dot_result(dot_result)

    def get_backends(self):
        ''
        return (self.BACKEND_CUPY, self.BACKEND_SPARSE_GCXS, self.BACKEND_NUMPY)
