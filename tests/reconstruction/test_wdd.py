from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_allclose
import pytest

from libertem.corrections.coordinates import identity, rotate_deg
from libertem.io.dataset.memory import MemoryDataSet
from libertem import api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.common import Shape

from ptychography40.reconstruction.wdd.params_recon import f2d_matrix_replacement
from ptychography40.reconstruction.wdd.dim_reduct import compress, decompress, get_sampled_basis
from ptychography40.reconstruction.wdd.wdd_udf import (
    WDDUDF, PatchWDDUDF, GridSpec, wdd_per_frame_combined)
from ptychography40.reconstruction.wdd.params_recon import wdd_params_recon
from ptychography40.reconstruction.wdd.wiener_filter import probe_initial, pre_computed_Wiener
from ptychography40.reconstruction.common import wavelength

if TYPE_CHECKING:
    import numpy.typing as nt


@pytest.mark.parametrize(
    'size, step', [
        (0, 23),  # Size not positive finite
        (-1, 23),  # Size not positive finite
        (23, 0),  # Step not positive finite
        (23, -1),  # Step not positive finite
        (4, 5),  # Step larger than size
        (4, "hello"),  # Type mismatch
        ("hello", 23),  # Type mismatch
    ]
)
def test_gridspec_errors(size, step):
    with pytest.raises(ValueError):
        GridSpec(size=size, step=step)


def test_gridspec_simple():
    g = GridSpec(size=3, step=3)
    assert g.min_patch_index == 0
    # patches 0 and 1
    assert g.max_patch_index(5) == 1
    # patches 0 and 1, filling the area completely
    assert g.max_patch_index(6) == 1
    # Starting patch 2
    assert g.max_patch_index(7) == 2
    # patch index, index within patch
    assert g.get_patch_indices(0) == [(0, 0)]
    assert g.get_patch_indices(2) == [(0, 2)]
    assert g.get_patch_indices(3) == [(1, 0)]
    assert g.get_patch_indices(-1) == [(-1, 2)]


def test_gridspec_overlap():
    g = GridSpec(size=4, step=3)
    assert g.min_patch_index == -1
    # patches 0 and 1
    assert g.max_patch_index(5) == 1
    # patches 0 and 1, filling the area completely
    assert g.max_patch_index(6) == 1
    # Starting patch 2
    assert g.max_patch_index(7) == 2
    # patch index, index within patch
    assert g.get_patch_indices(0) == [(0, 0), (-1, 3)]
    # Only one
    assert g.get_patch_indices(2) == [(0, 2)]
    # Overlap region
    assert g.get_patch_indices(3) == [(1, 0), (0, 3)]
    # only one
    assert g.get_patch_indices(-1) == [(-1, 2)]
    # Overlap region
    assert g.get_patch_indices(-3) == [(-1, 0), (-2, 3)]


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
    if transformation is None:
        transformation = identity()
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
              'transformation': None,
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
    # patchspec: (patch_shape, patch_trim, patch_overlap, raises)
    'complex_dtype, rtol, atol, patchspec', [
        (np.complex64, 1e-6, 1e-6, None),
        (np.complex128, 1e-8, 1e-8, None),
        # Not equivalent, but reasonably similar to real result
        # Since the exmple is small, we have to work with
        # small trim and overlap which increases the difference.
        (np.complex64, 1e-3, 1e-3, ((14, 14), 3, 3, False)),
        (np.complex64, 1e-3, 1e-3, ((19, 11), (3, 2), (2, 3), False)),
        (np.complex64, 1e-3, 1e-3, ((20, 21), (2, 3), (2, 3), False)),
        (np.complex64, 1e-3, 1e-3, ((20, 21), 8, 8, True)),
        (np.complex64, 1e-3, 1e-3, ((36, 38), None, None, False)),
    ]
)
@pytest.mark.with_numba
def test_wdd_udf(complex_dtype, rtol, atol, patchspec):
    lt_ctx = lt.Context(InlineJobExecutor(debug=True, inline_threads=6))
    scaling = 4
    shape = (37, 39, 189 // scaling, 197 // scaling)
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

    # Make sure we have a real signal for comparing
    # since random noise tends to average to near 0,
    # which magnifies errors
    obj = 1 - 0.2j * np.random.random(shape[2:])
    illum = np.random.random(shape[2:])

    input_data = np.zeros(shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            rolled = np.roll(illum, (-y, -x))
            wave = obj * rolled
            projected = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave)))
            intensity = np.abs(projected)**2
            input_data[y, x] = intensity

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
              'transformation': None,
              'epsilon': 100
              }
    if patchspec is None:
        patch_shape = None
        patch_trim = None
        patch_overlap = None
        raises = False
    else:
        patch_shape, patch_trim, patch_overlap, raises = patchspec

    recon_parameters = wdd_params_recon(
        ds_shape=ds.shape,
        params=params,
        complex_dtype=complex_dtype,
        patch_shape=patch_shape
    )

    recon_parameters_ref = wdd_params_recon(
        ds_shape=ds.shape,
        params=params,
        complex_dtype=complex_dtype,
    )

    if patchspec is None:
        udf = WDDUDF(
            recon_parameters=recon_parameters,
            complex_dtype=complex_dtype
        )
    else:
        if raises:
            with pytest.raises(ValueError):
                udf = PatchWDDUDF(
                    recon_parameters=recon_parameters,
                    complex_dtype=complex_dtype,
                    patch_trim=patch_trim,
                    patch_overlap=patch_overlap,
                )
            return
        else:
            udf = PatchWDDUDF(
                recon_parameters=recon_parameters,
                complex_dtype=complex_dtype,
                patch_trim=patch_trim,
                patch_overlap=patch_overlap,
            )
        for i in (0, 1):
            assert udf.patch_step[i] == (
                recon_parameters.patch_shape[i]
                - 2*udf.params.patch_trim[i]
                - udf.params.patch_overlap[i]
            )
        if isinstance(patch_overlap, int):
            _overlap = (patch_overlap, patch_overlap)
        else:
            _overlap = patch_overlap
        if isinstance(patch_trim, int):
            _trim = (patch_trim, patch_trim)
        else:
            _trim = patch_trim
        if _overlap is not None:
            assert udf.params.patch_overlap == _overlap
        if _trim is not None:
            assert udf.params.patch_trim == _trim

    # Create context
    live_wdd = lt_ctx.run_udf(
        dataset=ds,
        udf=udf,
        roi=None,
        plots=[['amplitude', 'phase']],
    )

    live_wdd_recon = live_wdd['reconstructed']
    # Reference
    result_ref = wdd_per_frame_combined(
        idp=input_data,
        coeff=recon_parameters_ref.coeff,
        wiener_filter_compressed=recon_parameters_ref.wiener_filter_compressed,
        scan_idx=recon_parameters_ref.wiener_roi,
        row_exp=recon_parameters_ref.row_exp,
        col_exp=recon_parameters_ref.col_exp,
        complex_dtype=complex_dtype,
    )
    if patchspec is None:
        res = live_wdd_recon.data
        ref = result_ref
    else:
        # Cut border where tiled and normal differ the most
        res = live_wdd_recon.data[3:-3, 3:-3]
        ref = result_ref[3:-3, 3:-3]
    assert_allclose(
        np.angle(res),
        np.angle(ref),
        rtol=rtol, atol=atol
    )
    # Check that the numerical tolerance is small enough to catch
    # a gross error in the result
    with pytest.raises(AssertionError):
        assert_allclose(
            np.angle(result_ref),
            # Transposed
            np.angle(live_wdd_recon.data).T,
            rtol=rtol, atol=atol
        )


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


def wiener_dtype(complex_dtype):
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
              'epsilon': 100}

    coeff = get_sampled_basis(order=16,
                              sig_shape=shape,
                              com=params['com'],
                              scale=5,
                              semiconv_pix=params['semiconv_pix'],
                              float_dtype=np.float32)

    wiener_filter_compressed, wiener_roi = pre_computed_Wiener(
                                nav_shape=shape[:2],
                                sig_shape=shape[2:],
                                order=16,
                                params=params,
                                coeff=coeff,
                                epsilon=params['epsilon'],
                                complex_dtype=complex_dtype,
                            )
    return wiener_filter_compressed, wiener_roi


def test_dtype_wiener():
    with pytest.raises(RuntimeError) as excinfo:
        complex_dtype = np.float32
        wiener_dtype(complex_dtype)
    assert f"unknown complex dtype: {complex_dtype}" in str(excinfo.value)


def test_dtype_wdd_recon():
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
        complex_dtype = np.float32
        wdd_dtype(complex_dtype)

    assert f"unknown complex dtype: {complex_dtype}" in str(excinfo.value)
