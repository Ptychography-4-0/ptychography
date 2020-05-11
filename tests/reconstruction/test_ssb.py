import numpy as np
import math
import pytest
from libertem import api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.memory import MemoryDataSet
from ptychography.reconstruction.ssb import (SSB_UDF, wavelength,
dot_product_transposed, Fourier_transform)
from scipy.sparse import csc_matrix


# @pytest.mark.parametrize("dtype,atol", [(np.float32, 30.05), (np.float64, 30.0)])  # 0.015  1e-8
def test_ssb():

    ctx = lt.Context(executor=InlineJobExecutor())
    dtype = np.float32
    shape = (5, 7, 71, 27)
    #  ? shape = np.random.uniform(1, 300, (4,1,))

    # The acceleration voltage U in keV
    U = 300
    # STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    dpix = 0.5654/50*1e-9
    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649

    input_data = np.random.uniform(0, 10000, shape)
    udf = SSB_UDF(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                  dtype=dtype)

    dataset = MemoryDataSet(
        data=input_data, tileshape=(20, 71, 27), num_partitions=2, sig_dims=2,
    )

    result = ctx.run_udf(udf=udf, dataset=dataset)

    result_f = reference_ssb(input_data, U=U, dpix=dpix, semiconv=semiconv,
                             semiconv_pix=semiconv_pix)

    atol = np.max(np.abs(result_f))*0.02

    # print(np.max(np.abs(np.abs(result['pixels']) - np.abs(result_f))))

    assert np.allclose(np.abs(result['pixels']), np.abs(result_f), atol=atol)


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-8), (np.float64, 1e-8)])
def test_masks(dtype, atol):

    shape = (50, 50, 189, 189)
    # The acceleration voltage U in keV
    U = 300
    # STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    dpix = 0.5654/50*1e-9
    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649

    masks1 = generate_masks(shape, dtype, U, dpix, semiconv, semiconv_pix)
    masks2 = generate_masks_udf(shape, dtype, U, dpix, semiconv, semiconv_pix)
    assert np.allclose(masks1, masks2, atol=atol)


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-8), (np.float64, 1e-8)])
def test_dot_product(dtype, atol):
    # dtype = np.float32
    A = np.random.uniform(0, 2, size=(10, 189*189))
    Xx = np.random.uniform(0, 1000, size=(1, 189*189))
    Acsr = csc_matrix(A)
    n_rows = Acsr.shape[0]
    n_cols = Acsr.shape[1]
    Aj = Acsr.indices
    Ax = Acsr.data
    Ap = Acsr.indptr

    Result_ssb = dot_product_transposed(Ax, Aj, Ap, n_cols, n_rows, Xx, dtype)
    Result_expected = Acsr.dot(Xx.T)
    assert np.allclose(Result_ssb, Result_expected, atol=atol)


def test_fourier_transform():

    shape = (50, 50, 189, 189)
    Nblock = np.array(shape[0:2])
    tile_slice_i = 2
    tile_size = 1

    FT_result_ssb = Fourier_transform(tile_slice_i, tile_size, Nblock)

    k = range(tile_slice_i, tile_slice_i + tile_size)
    j = np.ogrid[0:Nblock[0]*Nblock[1]]

    FT_expected = np.zeros((tile_size, Nblock[0]*Nblock[1]), dtype=np.complex128)
    for i in k:
        m_n = divmod(i, Nblock[0])
        k_l = divmod(j, Nblock[1])
        M = 1/Nblock[0]
        N = 1/Nblock[1]
        Fourier_exponent = np.exp(-2j*np.pi*(m_n[0]*k_l[0]*M + m_n[1]*k_l[1]*N))
        FT_expected[i-tile_slice_i] = Fourier_exponent

    assert np.allclose(FT_result_ssb, FT_expected)


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-14), (np.float64, 1e-14)])
def test_wavelength(dtype, atol):

    # Handbook of Physics in Medicine and Biology p.40-3
    # Table 40.1 'Electron wavelengths at some acceleration voltages used in TEM'

    U_voltages = np.array([100, 200, 300, 400, 1000], dtype=dtype)
    wavelength_expected = np.array([3.70e-12, 2.51e-12, 1.97e-12, 1.64e-12, 0.87e-12], dtype=dtype)

    shape = U_voltages.shape[0]
    wavelength_result_ssb = np.zeros((shape), dtype=dtype)

    for i in range(0, shape):
        wavelength_result_ssb[i] = wavelength(U_voltages[i])

    assert np.allclose(wavelength_result_ssb,  wavelength_expected, atol=atol)


def generate_masks_udf(shape, dtype, U, dpix, semiconv, semiconv_pix):

    masks_dtype = dtype

    Nblock = np.array(shape[0:2])
    Nscatter = np.array(shape[2:4])

    # Calculation of the relativistic electron wavelength in meters
    lambda_e = wavelength(U)

    d_Kf = np.sin(semiconv)/lambda_e/semiconv_pix
    d_Qp = 1/dpix/Nblock
    cy, cx = np.array(shape)[-2:]//2

    y, x = np.ogrid[0:Nscatter[0], 0:Nscatter[1]]
    filter_center = (y - cy)**2 + (x - cx)**2 < semiconv_pix**2

    # size = self.meta.dataset_shape
    size2 = (Nblock[0]//2, Nblock[1], Nscatter[0], Nscatter[1])
    masks = np.zeros(size2, dtype=masks_dtype)

    row = 0
    for column in range(Nblock[1]):

        qp = np.array((row, column))
        flip = qp > Nblock / 2
        real_qp = qp.copy()
        real_qp[flip] = qp[flip] - Nblock[flip]
        sx, sy = real_qp * d_Qp / d_Kf

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

        nonzero_masks = 2*np.count_nonzero(mask_positive)
        if nonzero_masks > 0 and column > 0:
            masks[row, column] = np.subtract(mask_positive.astype(masks_dtype),
                                        mask_negative.astype(masks_dtype))/nonzero_masks
        else:
            boundary_2 = column + 1
            boundary_1 = Nblock[1] - boundary_2 + 1

    nonzero_rows = list(range(1, boundary_1))
    nonzero_columns = list(range(boundary_1))+list(range(boundary_2, Nblock[1]))

    for row in nonzero_rows:
        for column in nonzero_columns:
            qp = np.array((row, column))
            flip = qp > Nblock / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - Nblock[flip]
            sx, sy = real_qp * d_Qp / d_Kf

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

            nonzero_masks = 2*np.count_nonzero(mask_positive)
            if nonzero_masks > 0:
                masks[row, column] = (
                    np.subtract(mask_positive.astype(masks_dtype),
                                mask_negative.astype(masks_dtype))/nonzero_masks
                )

    masks = masks.reshape(Nblock[0]*Nblock[1]//2, Nscatter[0]*Nscatter[1])
    return masks


def generate_masks(shape, dtype, U, dpix, semiconv, semiconv_pix):

    Nblock = np.array(shape[0:2])
    Nscatter = np.array(shape[2:4])

    # electron wavelength in m
    lamb = wavelength(U)
    # spatial freq. step size in scattering space
    d_Kf = np.sin(semiconv)/lamb/semiconv_pix
    # spatial freq. step size according to probe raster
    d_Qp = 1/dpix/Nblock

    cy, cx = np.array(shape)[-2:] // 2

    y, x = np.ogrid[0:Nscatter[0], 0:Nscatter[1]]
    filter_center = (y - cy)**2 + (x - cx)**2 < semiconv_pix**2

    masks = np.zeros(shape, dtype=dtype)

    for q in range(Nblock[0]):
        for p in range(Nblock[0]):

            qp = np.array((q, p))
            flip = qp > Nblock / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - Nblock[flip]
            sx, sy = real_qp * d_Qp / d_Kf

            filter_positive = (y - cy - sy)**2 + (x - cx - sx)**2 < semiconv_pix**2
            filter_negative = (y - cy + sy)**2 + (x - cx + sx)**2 < semiconv_pix**2

            mask_positive = np.all((filter_center, filter_positive,
                                    np.invert(filter_negative)), axis=0)
            mask_negative = np.all((filter_center, filter_negative,
                                    np.invert(filter_positive)), axis=0)

            divide = 0

            if np.count_nonzero(mask_positive) > 0:
                masks[q, p] += mask_positive
                divide += 1

            if np.count_nonzero(mask_negative) > 0:
                masks[q, p] -= mask_negative
                divide += 1

            if divide > 0:
                masks[q, p] /= divide
                masks[q, p] /= np.count_nonzero(mask_negative)

    size2 = (Nblock[0]//2, Nblock[1], Nscatter[0], Nscatter[1])
    masks1 = np.zeros(size2, dtype=dtype)

    masks1 = masks[0:Nblock[0]//2, 0:Nblock[1], :, :]
    masks1 = masks1.reshape(Nblock[0]*Nblock[1]//2, Nscatter[0]*Nscatter[1])

    return masks1


def reference_ssb(data, U, dpix, semiconv, semiconv_pix):

    # 'U' - The acceleration voltage U in keV
    # 'dpix' - STEM pixel size in m
    # 'semiconv' -  STEM semiconvergence angle in radians
    # 'semiconv_pix' - Diameter of the primary beam in the diffraction pattern in pixels

    reordered = np.moveaxis(data, (0, 1), (2, 3))
    ffts = np.fft.fft2(reordered)
    rearranged_ffts = np.moveaxis(ffts, (2, 3), (0, 1))

    Nblock = np.array(data.shape[0:2])
    Nscatter = np.array(data.shape[2:4])

    # electron wavelength in m
    lamb = wavelength(U)
    # spatial freq. step size in scattering space
    d_Kf = np.sin(semiconv)/lamb/semiconv_pix
    # spatial freq. step size according to probe raster
    d_Qp = 1/dpix/Nblock

    result_f = np.zeros(data.shape[:2], dtype=rearranged_ffts.dtype)
    cy, cx = np.array(data.shape)[-2:] // 2

    y, x = np.ogrid[0:Nscatter[0], 0:Nscatter[1]]
    filter_center = (y - cy)**2 + (x - cx)**2 < semiconv_pix**2

    for q in range(Nblock[0]):
        for p in range(Nblock[0]):
            qp = np.array((q, p))
            flip = qp > Nblock / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - Nblock[flip]
            sx, sy = real_qp * d_Qp / d_Kf

            filter_positive = (y - cy - sy)**2 + (x - cx - sx)**2 < semiconv_pix**2
            filter_negative = (y - cy + sy)**2 + (x - cx + sx)**2 < semiconv_pix**2

            mask_positive = np.all((filter_center, filter_positive,
                                    np.invert(filter_negative)), axis=0)
            mask_negative = np.all((filter_center, filter_negative,
                                    np.invert(filter_positive)), axis=0)

            f = rearranged_ffts[q, p]

            divide = 0

            if np.count_nonzero(mask_positive) > 0:
                result_f[q, p] += np.average(f[mask_positive])
                divide += 1
            if np.count_nonzero(mask_negative) > 0:
                result_f[q, p] -= np.average(f[mask_negative])
                divide += 1

            if divide > 0:
                result_f[q, p] /= divide

    result_f[0, 0] = np.average(rearranged_ffts[0, 0, filter_center])

    return result_f
