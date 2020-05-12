import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from libertem.udf import UDF
from libertem import api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.common.container import MaskContainer

from numba import njit
import scipy.constants as const


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
def Fourier_transform(tile_slice_i, tile_size, Nblock):

    FT = np.zeros((tile_size, Nblock[0]*Nblock[1]), dtype=np.complex128)
    current_tile_index = range(tile_slice_i, tile_slice_i + tile_size)
    tmp_x = np.zeros(Nblock[1], dtype=np.complex128)

    for i in current_tile_index:
        m = i // Nblock[0]
        n = i % Nblock[1]
        y_index = i - tile_slice_i

        for l in range(Nblock[1]):
            nl = n*l/Nblock[1]
            tmp_x[l] = np.exp(-2j*np.pi*nl)

        for k in range(Nblock[0]):
            mk = m*k/Nblock[0]
            tmp_y = np.exp(-2j*np.pi*mk)
            offset = k * Nblock[1]
            for l in range(Nblock[1]):
                FT[y_index, offset + l] = tmp_y * tmp_x[l]
    return FT


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


@njit
def Result_function(intermediate_result_1, FT, Nblock, dtype, n_rows, tile_size):

    current_tile_index = range(tile_size)
    N_rows = Nblock[0]*Nblock[1]

    A = np.zeros((N_rows,), dtype=dtype)

    left_boundary = (n_rows - Nblock[1])//Nblock[1]
    IR2_i = np.zeros((left_boundary*Nblock[1],), dtype=dtype)
    IR2_reshaped = np.zeros((left_boundary, Nblock[1],), dtype=dtype)
    result = np.zeros(N_rows, dtype=np.complex128)

    for i in current_tile_index:

        IR1_i = intermediate_result_1[:, i]

        IR2_i[0:left_boundary*Nblock[1]] = IR1_i[Nblock[1]:n_rows]
        IR2_reshaped = IR2_i.reshape(left_boundary, Nblock[1])

        IR2_reshaped = IR2_reshaped*(-1)
        IR2_reshaped = IR2_reshaped[::-1, :]
        part_IR2 = IR2_reshaped[:, 1:Nblock[1]]
        IR2_reshaped[:, 1:Nblock[1]] = part_IR2[:, ::-1]
        IR2_i = IR2_reshaped.flatten()

        A[0:n_rows] = IR1_i
        A[N_rows - left_boundary*Nblock[1]:N_rows] = IR2_i
        result += A*FT[i]

    return result.reshape(Nblock[0], Nblock[1])


def generate_masks(shape, dtype, U, dpix, semiconv, semiconv_pix, cy=None, cx=None):
    masks_dtype = dtype

    Nblock = np.array(shape[0:2])
    Nscatter = np.array(shape[2:4])

    # Calculation of the relativistic electron wavelength in meters
    lambda_e = wavelength(U)

    d_Kf = np.sin(semiconv)/lambda_e/semiconv_pix
    d_Qp = 1/dpix/Nblock

    if cx is None:
        cx = shape[-1] / 2
    if cy is None:
        cy = shape[-2] / 2

    y, x = np.ogrid[0:Nscatter[0], 0:Nscatter[1]]
    filter_center = (y - cy)**2 + (x - cx)**2 < semiconv_pix**2

    # size = self.meta.dataset_shape
    size2 = (Nblock[0]//2, Nblock[1], Nscatter[0], Nscatter[1])
    masks = np.zeros(size2, dtype=masks_dtype)

    for row in range(size2[0]):
        for column in range(Nblock[1]):
            qp = np.array((row, column))
            flip = qp > Nblock / 2,
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - Nblock[flip]
            sx, sy = real_qp * d_Qp / d_Kf

            # 1st diffraction order and primary beam don't overlap
            if sx**2 + sy**2 > 4*semiconv_pix**2:
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
                masks[row, column] = (
                    mask_positive / non_zero_positive -
                    mask_negative / non_zero_negative
                ) / 2
            else:
                # Assert that there are no unbalanced trotters
                assert non_zero_positive == 0
                assert non_zero_negative == 0
    masks = masks.reshape((Nblock[0]//2)*Nblock[1], Nscatter[0], Nscatter[1])
    return masks, filter_center


class SSB_UDF(UDF):

    def __init__(self, U, dpix, semiconv, semiconv_pix, dtype=np.float32, cy=None, cx=None):
        '''
        Parameters

        U: float
            The acceleration voltage U in kV

        cy: float

        cx: float

        dpix: float
            STEM pixel size in m

        semiconv: float
            STEM semiconvergence angle in radians

        semiconv_pix: float
            Diameter of the primary beam in the diffraction pattern in pixels
        '''
        super().__init__(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                         dtype=dtype, cy=cy, cx=cx)

    def get_result_buffers(self):

        return {
            'pixels': self.buffer(
                kind="single", dtype="complex128", extra_shape=(self.meta.dataset_shape[0:2])
             )
        }

    def merge(self, dest, src):

        dest['pixels'][:] = dest['pixels'] + src['pixels']

    def process_tile(self, tile):

        dtype = self.params.dtype
        tile_slice_i = self.meta.slice.origin[0]
        tile_shape = self.meta.slice.shape
        tile_slice_start = self.meta.slice.origin

        Nblock = np.array(self.meta.dataset_shape[0:2])

        masks = self.task_data.masks.get(self.meta.slice, transpose=False)
        masks_dtype = self.params.dtype

        filter_center = self.task_data['filter_center']
        filter_center_tile = filter_center[
            tile_slice_start[1]:tile_slice_start[1]+tile_shape[1],
            tile_slice_start[2]:tile_slice_start[2]+tile_shape[2]]
        non0_filter_center_tile = np.count_nonzero(filter_center_tile)
        point_0_0 = 0
        point_0_0 = (
            np.sum(tile.astype(masks_dtype)[..., filter_center_tile])/non0_filter_center_tile
        )

        tile_flat = tile.reshape(tile.shape[0], -1)

        tile_size = tile_flat.shape[0]
        n_rows = masks.shape[0]
        n_cols = masks.shape[1]
        Aj = masks.indices
        Ap = masks.indptr
        Ax = masks.data

        result = np.zeros(Nblock[0]*Nblock[1], dtype=np.complex128)
        intermediate_result = np.zeros((n_rows, tile_size), dtype=dtype)
        intermediate_result = dot_product_transposed(Ax, Aj, Ap, n_cols, n_rows, tile_flat, dtype)
        FT = Fourier_transform(tile_slice_i, tile_flat.shape[0], Nblock)
        result = Result_function(intermediate_result, FT, Nblock, dtype, n_rows, tile_size)

        self.results.pixels[:] = self.results.pixels + result
        self.results.pixels[0, 0] = self.results.pixels[0, 0] + point_0_0

    def get_task_data(self):
        masks, filter_center = generate_masks(
            shape=self.meta.dataset_shape,
            dtype=self.params.dtype,
            U=self.params.U,
            dpix=self.params.dpix,
            semiconv=self.params.semiconv,
            semiconv_pix=self.params.semiconv_pix,
            cx=self.params.cx,
            cy=self.params.cy,
        )

        return {
            "masks": MaskContainer(mask_factories=lambda: masks, dtype=masks.dtype,
                     use_sparse='scipy.sparse.csc', count=masks.shape[0]),
            "filter_center": filter_center
        }


if __name__ == '__main__':
    main()
