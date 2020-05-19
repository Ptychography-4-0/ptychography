import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sparse

from libertem.udf import UDF
from libertem import api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.common.container import MaskContainer
from libertem.common import Slice, Shape
from libertem.masks import circular
from libertem.corrections.coordinates import identity

from .common import wavelength, get_shifted, to_slices


def mask_tile_pair(tileslice, filter_center, sy, sx):
    sy, sx, = np.int(np.round(sy)), np.int(np.round(sx))
    tileslice = tileslice.discard_nav()
    center_tile = tileslice.get(arr=filter_center)
    positive_tile = np.zeros_like(center_tile)
    negative_tile = np.zeros_like(center_tile)
    # We get from negative coordinates,
    # that means it looks like shifted to positive
    (target_slice_p, source_slice_p) = to_slices(*get_shifted(
        arr_shape=filter_center.shape,
        tile_origin=tileslice.origin,
        tile_shape=tuple(tileslice.shape),
        shift=(-sy, -sx)
    ))
    # We get from positive coordinates,
    # that means it looks like shifted to negative
    (target_slice_n, source_slice_n) = to_slices(*get_shifted(
        arr_shape=filter_center.shape,
        tile_origin=tileslice.origin,
        tile_shape=tuple(tileslice.shape),
        shift=(sy, sx)
    ))
    positive_tile[target_slice_p] = filter_center[source_slice_p]
    negative_tile[target_slice_n] = filter_center[source_slice_n]

    positive_mask = center_tile * positive_tile * (negative_tile == 0)
    negative_mask = center_tile * negative_tile * (positive_tile == 0)

    return (positive_mask, negative_mask)


def generate_masks(reconstruct_shape, mask_shape, dtype, wavelength, dpix, semiconv,
        semiconv_pix, transformation=None, center=None, cutoff=1):
    reconstruct_shape = np.array(reconstruct_shape)

    d_Kf = np.sin(semiconv)/wavelength/semiconv_pix
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
    full_slice = Slice(origin=(0, 0), shape=Shape(mask_shape, sig_dims=2))

    half_reconstruct = (reconstruct_shape[0]//2 + 1, reconstruct_shape[1])
    masks = []

    for row in range(half_reconstruct[0]):
        for column in range(half_reconstruct[1]):
            # Do an fftshift of q and p
            qp = np.array((row, column))
            flip = qp > (reconstruct_shape / 2)
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - reconstruct_shape[flip]

            # Shift of diffraction order relative to zero order
            # without rotation
            real_sy, real_sx = real_qp * d_Qp / d_Kf

            # We apply the transformation backwards to go
            # from physical coordinates to detector coordinates,
            # while the forward direction in center of mass analysis
            # goes from detector coordinates to physical coordinates
            sy, sx = (real_sy, real_sx) @ transformation

            mask_positive, mask_negative = mask_tile_pair(
                tileslice=full_slice,
                filter_center=filter_center,
                sy=sy,
                sx=sx
            )

            non_zero_positive = mask_positive.sum()
            non_zero_negative = mask_negative.sum()

            if non_zero_positive >= cutoff and non_zero_negative >= cutoff:
                m = (
                    mask_positive / non_zero_positive
                    - mask_negative / non_zero_negative
                ) / 2
                masks.append(sparse.COO(m.astype(dtype)))
            else:
                # Exclude small, missing or unbalanced trotters
                masks.append(sparse.zeros(mask_shape, dtype=dtype))

    # The zero order component (0, 0) is special, comes out zero with above code
    m_0 = filter_center / filter_center.sum()
    masks[0] = sparse.COO(m_0.astype(dtype))

    # Since we go through masks in order, this gives a mask stack with
    # flattened (q, p) dimension to work with dot product and mask container
    masks = sparse.stack(masks)
    return masks


class SSB_UDF(UDF):

    def __init__(self, U, dpix, semiconv, semiconv_pix,
                 dtype=np.float32, center=None, filter_center=None,
                 angle=0.0, cutoff=0):
        '''
        Parameters

        U: float
            The acceleration voltage U in kV

        center: (float, float)

        dpix: float
            STEM pixel size in m

        semiconv: float
            STEM semiconvergence angle in radians

        dtype: np.dtype
            dtype to perform the calculation in

        semiconv_pix: float
            Diameter of the primary beam in the diffraction pattern in pixels

        angle: float
            Angle of rotation of the detector vs scan in degrees

        mask_container: MaskContainer
            Hack to pass in a precomputed mask stack when using with single thread live data
            or with an inline executor.
            The proper fix is https://github.com/LiberTEM/LiberTEM/issues/335
        '''
        super().__init__(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                         dtype=dtype, center=center,
                         filter_center=filter_center, angle=angle, cutoff=cutoff)

    def get_result_buffers(self):
        dtype = np.result_type(np.complex64, self.params.dtype)
        return {
            'pixels': self.buffer(
                kind="single", dtype=dtype, extra_shape=self.reconstruct_shape,
                where='device'
             )
        }

    @property
    def reconstruct_shape(self):
        return tuple(self.meta.dataset_shape.nav)

    def get_task_data(self):
        # shorthand, cupy or numpy
        xp = self.xp
        # Hack to pass a fixed external container
        # In particular useful for single-process live processing
        # or inline executor
        if self.params.mask_container is None:
            masks = generate_masks(
                reconstruct_shape=self.reconstruct_shape,
                mask_shape=tuple(self.meta.dataset_shape.sig),
                dtype=self.params.dtype,
                wavelength=wavelength(self.params.U),
                dpix=self.params.dpix,
                semiconv=self.params.semiconv,
                semiconv_pix=self.params.semiconv_pix,
                center=self.params.center,
                angle=self.params.angle,
                cutoff=self.params.cutoff
            )
            container = MaskContainer(
                mask_factories=lambda: masks, dtype=masks.dtype,
                use_sparse='scipy.sparse.csr', count=masks.shape[0], backend=self.meta.backend
            )
        else:
            container = self.params.mask_container
            assert container.computed_masks.shape[0] == (self.reconstruct_shape[0] // 2 + 1)*self.reconstruct_shape[1]
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

        return {
            "masks": container,
            # Frame positions in the dataset masked by ROI
            # to easily access position in dataset when
            # processing with ROI applied
            "y_map": xp.array(y_map),
            "x_map": xp.array(x_map),
            "row_steps": xp.array(row_steps.astype(steps_dtype)),
            "col_steps": xp.array(col_steps.astype(steps_dtype)),
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

        masks = self.task_data.masks.get(self.meta.slice, transpose=True, backend=self.meta.backend)
        tile_flat = tile.reshape(tile.shape[0], -1)

        # Dot product from old matrix interface
        dot_result = tile_flat * masks

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
        if self.meta.backend == 'cupy':
            total_size = 10e6
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
