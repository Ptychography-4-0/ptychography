import numpy as np

from libertem.udf import UDF
from libertem.common.container import MaskContainer

from ptychography.reconstruction.ssb.trotters import generate_masks


class SSB_UDF(UDF):
    '''
    UDF to perform ptychography using the single side band (SSB) method :cite:`Pennycook2015`.
    '''
    def __init__(self, lamb, dpix, semiconv, semiconv_pix,
                 dtype=np.float32, cy=None, cx=None, transformation=None,
                 cutoff=1, method='subpix', mask_container=None,):
        '''
        Parameters
        ----------

        lamb: float
            The illumination wavelength in m. The function :meth:`ptychography.common.wavelength`
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
        super().__init__(lamb=lamb, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                         dtype=dtype, cy=cy, cx=cx, mask_container=mask_container,
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
            "masks": container,
            "row_exp": xp.array(row_exp.astype(steps_dtype)),
            "col_exp": xp.array(col_exp.astype(steps_dtype)),
            "backend": backend
        }

    def merge(self, dest, src):
        dest['pixels'][:] = dest['pixels'] + src['pixels']

    def merge_dot_result(self, dot_result):
        # shorthand, cupy or numpy
        xp = self.xp

        tile_depth = dot_result.shape[0]

        # We calculate only half of the Fourier transform due to the
        # inherent symmetry of the mask stack. In this case we
        # cut the y axis in half. The "+ 1" accounts for odd sizes
        # The mask stack is already trimmed in y direction to only contain
        # one of the trotter pairs
        half_y = self.results.pixels.shape[0] // 2 + 1

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
        dot_result = dot_result.reshape((tile_depth, half_y, self.results.pixels.shape[1]))

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
        self.results.pixels[:half_y] += xp.einsum(
            'i...,i...,i...',
            dot_result,
            fourier_factors_row,
            fourier_factors_col
        )

    def postprocess(self):
        # shorthand, cupy or numpy
        xp = self.xp
        half_y = self.results.pixels.shape[0] // 2 + 1
        # patch accounts for even and odd sizes
        # FIXME make sure this is correct using an example that transmits also
        # the high spatial frequencies
        patch = self.results.pixels.shape[0] % 2
        # We skip the first row since it would be outside the FOV
        extracted = self.results.pixels[1:self.results.pixels.shape[0] // 2 + patch]
        # The coordinates of the bottom half are inverted and
        # the zero column is rolled around to the front
        # The real part is inverted
        self.results.pixels[half_y:] = -xp.conj(
            xp.roll(xp.flip(xp.flip(extracted, axis=0), axis=1), shift=1, axis=1)
        )

    def process_tile(self, tile):
        # We flatten the signal dimension of the tile in preparation of the dot product
        tile_flat = tile.reshape(tile.shape[0], -1)
        if self.task_data.backend == 'cupy':
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
        else:
            # Load the preconditioned trotter stack from the mask container:
            # * Coutout for current tile
            # * Flattened signal dimension
            # * Not transposed for left hand side of dot product
            #   (the tile and the result will be transposed instead)
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
            # Since we restrict the size of the input matrix in get_tiling_preferences()
            # and the mask side of the product is much, much larger than the
            # tile, taking a transposed copy of the tile with optimized memory layout
            # for the fast scipy.sparse.csr __matmul__ is worth the effort here.
            # See also https://github.com/scipy/scipy/issues/13211
            # The optimal way to perform this dot product may also depend
            # on the shape of the data and the size of the zero order disk
            dot_result = (masks @ tile_flat.T.copy()).T

        self.merge_dot_result(dot_result)

    def get_backends(self):
        return ('numpy', 'cupy')

    def get_tiling_preferences(self):
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
            # We limit the depth of a tile so that the intermediate
            # results from processing a tile fit into the CPU cache.
            good_depth = max(1, 1e6 / result_size)
            return {
                "depth": int(good_depth),
                # We reduce the size of a tile since a transposed copy of
                # each tile will be taken to speed up the sparse matrix product
                "total_size": 0.5e6,
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
