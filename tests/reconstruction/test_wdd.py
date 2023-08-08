import numpy as np
import pytest
import typing
from libertem.corrections.coordinates import identity, rotate_deg
from libertem.io.dataset.memory import MemoryDataSet
from libertem import api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.common import Shape
from ptychography40.reconstruction.wdd.params_recon import (
    f2d_matrix_replacement)
from ptychography40.reconstruction.wdd.dim_reduct import compress, decompress
from ptychography40.reconstruction.wdd.wdd_udf import (
    WDDUDF, wdd_per_frame_combined)
from ptychography40.reconstruction.wdd.params_recon import wdd_params_recon
from ptychography40.reconstruction.wdd.wiener_filter import probe_initial
from ptychography40.reconstruction.common import wavelength
if typing.TYPE_CHECKING:
    import numpy.typing as nt


def test_fourier_matrix():
    """
    Unit test to evaluate the implementation of
    1D Fourier matrices and fast Fourier transform
    """
    nav_shape = (64, 64)
    complex_dtype = np.complex128
    mat_row, mat_col = f2d_matrix_replacement(nav_shape, complex_dtype)

    # 1D Fourier matrix
    F1D = np.fft.fft(np.eye(nav_shape[0]), norm='ortho')

    assert np.allclose(mat_row, F1D)
    assert np.allclose(mat_col, F1D)


def test_2d_fourier_matrix():
    """
    Unit test to evaluate the implementation of
    2D Fourier matrices and 2D fast Fourier transform
    In the paper the 2D Fourier matrices should be
    implemented since we want to process per frame
    data
    """
    nav_shape = (64, 64)
    complex_dtype = np.complex128
    mat_row, mat_col = f2d_matrix_replacement(nav_shape, complex_dtype)

    # Generate random matrix
    A = (np.random.randn(
        nav_shape[0], nav_shape[1]
        )
        + 1j*np.random.randn(
        nav_shape[0], nav_shape[1]
        )
        )

    assert np.allclose(np.fft.fft2(A, norm='ortho'), mat_row@A@mat_col.T)


def test_2d_fourier_matrix_flatten():
    """
    Unit test to evaluate the implementation of
    2D Fourier matrix in terms of Kronecker product
    and the 2D fast Fourier transform
    """
    nav_shape = (64, 64)
    complex_dtype = np.complex128
    mat_row, mat_col = f2d_matrix_replacement(nav_shape, complex_dtype)

    # Generate random matrix
    A = (np.random.randn(
        nav_shape[0], nav_shape[1]
        )
        + 1j*np.random.randn(
        nav_shape[0], nav_shape[1])
        )
    A_flat = A.reshape(-1)

    F2D = np.kron(mat_row, mat_col)

    assert np.allclose(
        np.fft.fft2(A, norm='ortho'),
        (F2D@A_flat).reshape(nav_shape)
        )


def test_sum_outer_product():
    """
    Unit test to evaluate the implementation of
    2D Fourier transform with partial reconstruction.
    In live reconstruction we update the  reconstruction
    partially when a new frame arrives at the detector.
    """

    nav_shape = (16, 16)
    det_shape = (16, 16)
    complex_dtype = np.complex128
    mat_row, mat_col = f2d_matrix_replacement(nav_shape, complex_dtype)

    # Generate random matrix
    A = (np.random.randn(
        nav_shape[0], nav_shape[1],
        det_shape[0], det_shape[1]
        )
        + 1j*np.random.randn(
        nav_shape[0], nav_shape[1],
        det_shape[0], det_shape[1]
        )
        )
    A_flat = A.reshape(np.prod(nav_shape), np.prod(det_shape))

    fft2_A = np.fft.fft2(A, axes=(0, 1), norm='ortho')
    F2D = np.kron(mat_row, mat_col)
    # outer product
    result = np.zeros_like(A_flat)
    for idx in range(F2D.shape[1]):
        result += np.outer(F2D[:, idx], A_flat.T[:, idx])
    assert np.allclose(F2D@A_flat, result)
    assert np.allclose(fft2_A,
                       result.reshape(nav_shape+det_shape)
                       )


def wdd_reference(ds_shape: Shape,
                  idp: np.ndarray,
                  params: dict,
                  complex_dtype: "nt.DTypeLike" = np.complex64):
    """
    Implementation of reference for Wigner Distribution Deconvolution

    Parameters
    ----------
    ds_shape
        Dimension of 4D datasets
    idp
        4D diffraction data
    params
        Dictionary related to physical coordinates
    complex_dtype
        Pre defined complex dtype
    """
    if np.dtype(complex_dtype) == np.complex64:
        float_dtype = np.float32
    elif np.dtype(complex_dtype) == np.complex128:
        float_dtype = np.float64
    else:
        raise RuntimeError(f"unknown complex dtype: {complex_dtype}")

    # Get dimension from 4D data
    scan_dim = np.array(ds_shape.nav)
    # First step in WDD, to apply Fourier transform in the scanning position
    # by changin the scan position
    # Apply Fourier transform to scan position in order to get
    # spatial frequency
    idp_fft = np.fft.fft2(idp, axes=(0, 1))

    # Physical parameters
    cy = params['com'][0]
    cx = params['com'][1]

    d_Kf = np.sin(params['semiconv'])/params['lamb']/params['semiconv_pix']
    d_Qp = 1/params['dpix']/np.array(ds_shape.nav)

    # Generate grid for probe
    y, x = np.ogrid[0:ds_shape.sig[0], 0:ds_shape.sig[1]]
    probe_center = probe_initial(
                                (cy, cx),
                                y, x,
                                0., 0.,
                                params['semiconv_pix'],
                                float_dtype,
                                norm=True
                               )
    transformation = params['transformation']
    # Allocate object cut
    cut = np.zeros((scan_dim[0], scan_dim[1]), dtype=complex_dtype)
    for q in range(scan_dim[0]):
        for p in range(scan_dim[1]):

            qp = np.array((q, p))
            flip = qp > scan_dim / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - scan_dim[flip]

            # Shift of diffraction order relative to zero order
            # without rotation in physical coordinates
            real_sy_phys, real_sx_phys = real_qp * d_Qp
            # We apply the transformation backwards to go
            # from physical orientation to detector orientation,
            # while the forward direction in center of mass analysis
            # goes from detector coordinates to physical coordinates
            # Afterwards, we transform from physical detector coordinates
            # to pixel coordinates
            sy, sx = (np.array((real_sy_phys, real_sx_phys))
                      @ transformation) / d_Kf

            # Shift probe
            probe_shift = probe_initial((cy, cx),
                                        y, x,
                                        sy, sx,
                                        params['semiconv_pix'],
                                        float_dtype,
                                        norm=True,
                                        )
            # Product with its conjugate
            product_conjugate = np.conj(probe_center)*probe_shift

            idp_rQ = np.fft.ifft2(idp_fft[q, p])
            # Get the inverse Wigner function
            wigner_func_inv = np.fft.ifft2((product_conjugate))

            # Numerator Wiener filter
            num = np.conj(wigner_func_inv)
            # Denominator
            den = np.abs(wigner_func_inv)**2 + params['epsilon']
            # Wiener filter
            wiener_filt = idp_rQ*num/den
            cut[q, p] = np.sum(wiener_filt)

    real_cut = np.fft.ifft2(cut)
    real_cut = real_cut/np.max(np.abs(real_cut))
    return real_cut.conj().astype(complex_dtype)


@pytest.mark.parametrize(
     'complex_dtype, rtol, atol', [
         (np.complex64, 1e-6, 1e-6),
         (np.complex128, 1e-8, 1e-8)]
)
@pytest.mark.with_numba
def test_wdd_no_rot(complex_dtype, rtol, atol):
    lt_ctx = lt.Context(InlineJobExecutor(debug=True, inline_threads=6))
    scaling = 4
    shape = (16, 16, 189 // scaling, 197 // scaling)
    dpix = 0.5654/50*1e-9
    # The acceleration voltage U in keV
    U = 300
    lamb = wavelength(U)

    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling

    # Center of Mass
    cy = 93 // scaling
    cx = 97 // scaling
    com = (cy, cx)

    # Input Data
    input_data = (
        100000*np.random.uniform(0, 1, np.prod(shape))
    )
    input_data = input_data.astype(np.float64).reshape(shape)

    ds = MemoryDataSet(
        data=input_data, tileshape=(20, shape[2],
                                    shape[3]), num_partitions=2, sig_dims=2,
    )
    # Parameter reconstruction
    params = {'com': com,
              'dpix': dpix,
              'lamb': lamb,
              'semiconv_pix': semiconv_pix,
              'semiconv': semiconv,
              'transformation': identity(),
              'epsilon': 100
              }

    recon_parameters = wdd_params_recon(ds_shape=ds.shape,
                                        params=params,
                                        complex_dtype=complex_dtype,
                                        )
    # Compress-Decompress
    com_decom_input_data = np.zeros_like(input_data)
    for row in range(input_data.shape[0]):
        for col in range(input_data.shape[1]):
            com_decom_input_data[row, col] = decompress(
                compress(input_data[row, col],
                         recon_parameters.coeff),
                recon_parameters.coeff)

    # Create context
    live_wdd = lt_ctx.run_udf(dataset=ds,
                              roi=None,
                              udf=WDDUDF(recon_parameters,
                                         complex_dtype),
                              )

    live_wdd_recon = live_wdd['reconstructed']
    # Reference
    result_ref = wdd_reference(ds.shape,
                               com_decom_input_data,
                               params,
                               complex_dtype
                               )
    assert np.allclose(np.angle(result_ref),
                       np.angle(live_wdd_recon.data),
                       rtol, atol)


@pytest.mark.parametrize(
     'complex_dtype, rtol, atol', [
        (np.complex64, 1e-6, 1e-6),
        (np.complex128, 1e-8, 1e-8)]
)
@pytest.mark.with_numba
def test_wdd_udf(complex_dtype, rtol, atol):
    lt_ctx = lt.Context(InlineJobExecutor(debug=True, inline_threads=6))
    scaling = 4
    shape = (16, 16, 189 // scaling, 197 // scaling)
    dpix = 0.5654/50*1e-9
    # The acceleration voltage U in keV
    U = 300
    lamb = wavelength(U)

    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling

    # Center of Mass
    cy = 93 // scaling
    cx = 97 // scaling
    com = (cy, cx)

    # Input Data
    input_data = (
        100000*np.random.uniform(0, 1, np.prod(shape))
    )
    input_data = input_data.astype(np.float64).reshape(shape)

    ds = MemoryDataSet(
        data=input_data, tileshape=(20, shape[2], shape[3]),
        num_partitions=2, sig_dims=2,
    )
    # Parameter reconstruction
    params = {'com': com,
              'dpix': dpix,
              'lamb': lamb,
              'semiconv_pix': semiconv_pix,
              'semiconv': semiconv,
              'transformation': identity(),
              'epsilon': 100
              }

    recon_parameters = wdd_params_recon(ds_shape=ds.shape,
                                        params=params,
                                        complex_dtype=complex_dtype,
                                        )

    # Create context
    live_wdd = lt_ctx.run_udf(dataset=ds,
                              roi=None,
                              udf=WDDUDF(recon_parameters,
                                         complex_dtype),
                              )

    live_wdd_recon = live_wdd['reconstructed']
    # Reference
    result_ref = wdd_per_frame_combined(
        input_data,
        recon_parameters.coeff,
        recon_parameters.wiener_filter_compressed,
        recon_parameters.wiener_roi,
        recon_parameters.row_exp,
        recon_parameters.col_exp,
        complex_dtype,)
    assert np.allclose(np.angle(result_ref),
                       np.angle(live_wdd_recon.data), rtol, atol)


def test_wdd_rotate():
    complex_dtype = np.complex128
    lt_ctx = lt.Context(InlineJobExecutor(debug=True, inline_threads=6))
    scaling = 4
    det = 45
    shape = (29, 30, det, det)

    # The acceleration voltage U in keV
    U = 300
    lamb = wavelength(U)
    # STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    dpix = 0.5654/50*1e-9
    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling
    # Center of Mass
    cy = det // 2
    cx = det // 2
    com = (cy, cx)

    # Input data
    input_data = (
        np.random.uniform(0, 1, np.prod(shape))
        * np.linspace(1.0, 1000.0, num=np.prod(shape))
    )
    input_data = input_data.astype(np.float64).reshape(shape)

    # Rotate 90 degrees clockwise
    data_90deg = np.zeros_like(input_data)
    for y in range(det):
        for x in range(det):
            data_90deg[:, :, x, det-1-y] = input_data[:, :, y, x]

    # Parameters WDD
    params = {'com': com,
              'dpix': dpix,
              'lamb': lamb,
              'semiconv_pix': semiconv_pix,
              'semiconv': semiconv,
              'transformation': rotate_deg(-90.),
              'epsilon': 100
              }

    ds = MemoryDataSet(
        data=data_90deg, tileshape=(20, shape[2], shape[3]),
        num_partitions=2, sig_dims=2,
    )

    recon_parameters = wdd_params_recon(ds_shape=ds.shape,
                                        params=params,
                                        complex_dtype=complex_dtype,
                                        )
    # Create context
    live_wdd = lt_ctx.run_udf(
                            dataset=ds,
                            roi=None,
                            udf=WDDUDF(recon_parameters,
                                       complex_dtype),
                            )

    live_wdd_recon = live_wdd['reconstructed']

    # Compress decompress data for reference wd
    com_decom_input_data = np.zeros_like(input_data)
    for row in range(input_data.shape[0]):
        for col in range(input_data.shape[1]):
            com_decom_input_data[row, col] = decompress(
                compress(input_data[row, col],
                         recon_parameters.coeff),
                recon_parameters.coeff)
    # Reference
    params_ref = {'com': com,
                  'dpix': dpix,
                  'lamb': lamb,
                  'semiconv_pix': semiconv_pix,
                  'semiconv': semiconv,
                  'transformation': identity(),
                  'epsilon': 100
                  }
    result_ref = wdd_reference(ds.shape,
                               com_decom_input_data,
                               params_ref,
                               complex_dtype
                               )

    assert np.allclose(np.angle(result_ref), np.angle(live_wdd_recon.data))


def test_match_dtype():
    with pytest.raises(RuntimeError) as excinfo:
        def wdd_dtype(complex_dtype):

            scaling = 4
            shape = (29, 30, 189 // scaling, 197 // scaling)

            cy = 93 // scaling
            cx = 97 // scaling

            # Parameter reconstruction
            params = {'com': (cy, cx),
                      'dpix': 0.5654/50*1e-9,
                      'lamb': wavelength(300),
                      'semiconv_pix': 78.6649 / scaling,
                      'semiconv': 25e-3,
                      'transformation': identity(),
                      'epsilon': 100}

            recon_parameters = wdd_params_recon(ds_shape=shape,
                                                params=params,
                                                complex_dtype=complex_dtype,
                                                )
            return recon_parameters
        complex_dtype = np.complex256
        wdd_dtype(complex_dtype)

    assert f"unknown complex dtype: {complex_dtype}" in str(excinfo.value)
