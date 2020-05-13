import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit
import scipy.constants as const
import sparse

from libertem.udf import UDF
from libertem import api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.common.container import MaskContainer


def main():
    path = r'C:\Users\lesnic\Nextcloud\Dieter\cGaN_sim_300kV\DPs\CBED_MSAP.raw'
    # Shape = (Number of probe scan pixels per scan row, number of scan rows,
    #          ...)
    shape = (50, 50, 189, 189)
    ctx = lt.Context(executor=InlineJobExecutor())
    data_s = ctx.load(
        "raw", path=path, dtype="float32",
        scan_size=shape[:2], detector_size=shape[-2:]
    )
    udf = SSB_UDF(U=300, dpix=0.5654/50*1e-9, semiconv=25e-3,
                  semiconv_pix=78.6649, dtype=np.float64)
    result = ctx.run_udf(udf=udf, dataset=data_s)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(np.abs(result['pixels']), norm=LogNorm())
    axes[1].imshow(np.angle(np.fft.ifft2(result['pixels'])))

    input("press return to continue")


# Calculation of the relativistic electron wavelength in meters
def wavelength(U):
    e = const.elementary_charge  # Elementary charge  !!! 1.602176634×10−19
    h = const.Planck  # Planck constant    !!! 6.62607004 × 10-34
    c = const.speed_of_light  # Speed of light
    m_0 = const.electron_mass  # Electron rest mass

    T = e*U*1000
    lambda_e = h*c/(math.sqrt(T**2+2*T*m_0*(c**2)))
    return lambda_e


@njit(fastmath=True)
def dot_product_transposed(Ax, Aj, Ap, n_cols, n_rows, Xx, dtype):

    tile_size = Xx.shape[0]
    Yy = np.zeros((n_rows, tile_size), dtype=dtype)
    pixel_stack = np.zeros(tile_size, dtype=dtype)

    for i in range(n_cols):
        offset = Ap[i]
        j = Ap[i+1] - offset
        if j > 0:
            pixel_stack[:] = Xx[:, i]
            for jj in range(j):

                current_row = Aj[jj + offset]
                Yy[current_row] += Ax[jj + offset] * pixel_stack

    return Yy


# FIXME calculate as sparse without instantiating the full
# dense stack, which is as large as the dataset
def generate_masks(reconstruct_shape, mask_shape, dtype, wavelength, dpix, semiconv,
        semiconv_pix, center=None):
    reconstruct_shape = np.array(reconstruct_shape)

    d_Kf = np.sin(semiconv)/wavelength/semiconv_pix
    d_Qp = 1/dpix/reconstruct_shape

    if center is None:
        center = np.array(mask_shape) / 2

    cy, cx = center

    y, x = np.ogrid[0:mask_shape[0], 0:mask_shape[1]]
    filter_center = (y - cy)**2 + (x - cx)**2 < semiconv_pix**2

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
            sx, sy = real_qp * d_Qp / d_Kf

            # 1st diffraction order and primary beam don't overlap
            if sx**2 + sy**2 > 4*semiconv_pix**2:
                masks.append(sparse.zeros(mask_shape, dtype=dtype))
                continue

            filter_positive = (
                (y - cy - sy)**2 + (x - cx - sx)**2 < semiconv_pix**2
            )
            filter_negative = (
                (y - cy + sy)**2 + (x - cx + sx)**2 < semiconv_pix**2
            )
            mask_positive = np.all(
                (filter_center, filter_positive, np.invert(filter_negative)),
                axis=0
            )
            mask_negative = np.all(
                (filter_center, filter_negative, np.invert(filter_positive)),
                axis=0
            )

            non_zero_positive = np.count_nonzero(mask_positive)
            non_zero_negative = np.count_nonzero(mask_negative)

            if non_zero_positive > 0 and non_zero_negative > 0:
                m = (
                    mask_positive / non_zero_positive
                    - mask_negative / non_zero_negative
                ) / 2
                masks.append(sparse.COO(m.astype(dtype)))
            else:
                # Assert that there are no unbalanced trotters
                assert non_zero_positive == 0
                assert non_zero_negative == 0
                masks.append(sparse.zeros(mask_shape, dtype=dtype))

    # The zero order component (0, 0) is special, comes out zero with above code
    m_0 = filter_center / np.count_nonzero(filter_center)
    masks[0] = sparse.COO(m_0.astype(dtype))

    # Since we go through masks in order, this gives a mask stack with
    # flattened (q, p) dimension to work with dot product and mask container
    masks = sparse.stack(masks)
    return masks


class SSB_UDF(UDF):

    def __init__(self, U, dpix, semiconv, semiconv_pix,
            dtype=np.float32, center=None, reconstruct_shape=None, mask_container=None):
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

        reconstruct_shape: (int, int)
            Shape to reconstruct into. Default dataset.shape.nav

        mask_container: MaskContainer
            Hack to pass in a precomputed mask stack when using with single thread live data
            or with an inline executor.
            The proper fix is https://github.com/LiberTEM/LiberTEM/issues/335
        '''
        super().__init__(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                         dtype=dtype, center=center, reconstruct_shape=reconstruct_shape,
                         mask_container=mask_container)

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
        if self.params.reconstruct_shape is None:
            shape = tuple(self.meta.dataset_shape.nav)
        else:
            shape = self.params.reconstruct_shape
        return shape

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
                center=self.params.center
            )
            container = MaskContainer(
                mask_factories=lambda: masks, dtype=masks.dtype,
                use_sparse='scipy.sparse.csr', count=masks.shape[0], backend=self.meta.backend
            )
        else:
            container = self.params.mask_container
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

        return {
            "masks": container,
            # Frame positions in the dataset masked by ROI
            # to easily access position in dataset when
            # processing with ROI applied
            "y_map": xp.array(y_map),
            "x_map": xp.array(x_map),
            "row_steps": xp.array(row_steps),
            "col_steps": xp.array(col_steps),
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

        fourier_factors_row = xp.exp(
            y_indices[:, np.newaxis, np.newaxis]
            * self.task_data.row_steps[np.newaxis, :half_y, np.newaxis]
        )
        fourier_factors_col = xp.exp(
            x_indices[:, np.newaxis, np.newaxis]
            * self.task_data.col_steps[np.newaxis, np.newaxis, :]
        )

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
        buffer_frame[half_y:] = -xp.conj(xp.roll(xp.flip(xp.flip(extracted, axis=0), axis=1), shift=1, axis=1))

        self.results.pixels[:] += buffer_frame

    def get_backends(self):
        return ('numpy', 'cupy')

    def get_tiling_preferences(self):
        result_size = np.prod(self.reconstruct_shape) * 8
        if self.meta.backend =='cupy':
            good_depth= min(1, 100e6 / result_size)
            return {
                "depth": good_depth,
                "total_size": 100e6,
            }
        else:
            good_depth= min(1, 1e6 / result_size)
            return {
                "depth": int(good_depth),
                "total_size": UDF.TILE_SIZE_MAX,
            }


if __name__ == '__main__':
    main()
