from typing import Tuple, Optional
from ptychography40.reconstruction.wdd.dim_reduct import compress
from libertem.corrections.coordinates import identity
import typing
import numpy as np
import numba

if typing.TYPE_CHECKING:
    import numpy.typing as nt


def probe_initial(
    com: Tuple,
    y: np.ndarray,
    x: np.ndarray,
    sy: float,
    sx: float,
    semiconv_pix: float,
    float_dtype: 'nt.DTypeLike',
    norm: Optional[bool] = False,
):
    """
    A function to generate circular aperture,
    since we are working on the Fourier space the probe is circular aperture

    Parameters
    ----------
    com
        Center of mass (cy,cx)
    y
        Grid on the y dimensional axes
    x
        Grid on the x dimensional axes
    sy
        Shift in physical coordinate y-axis
    sx
        Shift in physical coordinate x-axis

    semiconv_pix
        Radius on the pixel

    Return
    ------
    probe_center
        Circular aperture at the center
    """

    pupil = (y - com[0] + sy)**2 + (x - com[1] + sx)**2 < semiconv_pix**2
    probe = pupil.astype(float_dtype)
    norm_value = np.linalg.norm(probe)
    if norm and norm_value > 1e-8:
        probe = (probe / norm_value).astype(float_dtype)
    return probe


compress_jit = numba.njit(compress)
probe_init_jit = numba.njit(probe_initial)


def pre_computed_Wiener(
    nav_shape: Tuple,
    sig_shape: Tuple,
    order: int,
    params: dict,
    coeff: Tuple[np.ndarray, np.ndarray],
    epsilon: float,
    complex_dtype: "nt.DTypeLike" = np.complex64,
    transformation: Optional[np.ndarray] = None,
):

    """
    A wrapper function to choose if we use numba or
    not for calculating Wiener filter
    ds_shape:Shape,
    order:int,
    params:dict,
    coeff:Tuple[np.ndarray,np.ndarray],
    epsilon:float,
    complex_dtype: "nt.DTypeLike" = np.complex64,
    transformation: Optional[np.ndarray] = None,
    ):

    A wrapper function to choose if we use numba or not for
    calculating Wiener filter

    Parameters
    ----------
    nav_shape
        Nav dimension of reconstruction patch
    sig_shape
        Sig Dimension of 4D datasets
    order
        Maximum degree of polynomials
    coeff
        Matrix contains sampled Hermite-Gauss
        functions for dimensionality reduction
    params
        Dictionary related to physical coordinate
    scale
        Scaling Hermite-Gauss polynomials for both x and y direction
    epsilon
        Small factor to avoid zero division

    """

    if np.dtype(complex_dtype) == np.complex64:
        float_dtype = np.float32
    elif np.dtype(complex_dtype) == np.complex128:
        float_dtype = np.float64
    else:
        raise RuntimeError(f"unknown complex dtype: {complex_dtype}")

    if transformation is None:
        transformation = identity()

    # COM coordinate
    cy = params['com'][0]
    cx = params['com'][1]

    d_Kf = np.sin(params['semiconv'])/params['lamb']/params['semiconv_pix']
    d_Qp = 1/params['dpix']/np.array(nav_shape)

    # Generate grid for probe
    y, x = np.ogrid[0:sig_shape[0], 0:sig_shape[1]]
    wiener_filter_compressed = wiener_jit(tuple(nav_shape), order,
                                          y, x, cy, cx, d_Kf, d_Qp,
                                          params['semiconv_pix'],
                                          coeff, epsilon, complex_dtype,
                                          float_dtype, transformation)

    # Finding non-zero intersection
    scan_idx = roi_wiener_filter(wiener_filter_compressed)

    return wiener_filter_compressed, scan_idx


@numba.njit(parallel=True, cache=True)
def wiener_jit(
    scan_dim: Tuple,
    order: int,
    y: np.ndarray,
    x: np.ndarray,
    cy: float,
    cx: float,
    d_Kf: float,
    d_Qp: float,
    semiconv_pix: float,
    coeff: Tuple[np.ndarray, np.ndarray],
    epsilon: float,
    complex_dtype: 'nt.DTypeLike',
    float_dtype: 'nt.DTypeLike',
    transformation: np.ndarray,
):
    """
    A function to calculate compressed Wiener filter the result in
    compressed space, shape (q, p, order, order)
    It should be noted here we only implement Wiener filter with respect
    to initial probe, to estimate the probe
    after estimating the object, the shift direction should be reversed
    In this case we use numba for faster implementation

    Parameters
    ----------

    ds_shape
        Dimension of 4D datasets

    order
        Maximum degree of polynomials
    y
        Grid on the y dimensional axes
    x
        Grid on the x dimensional axes
    cy
        Center of mass coordinate on y axes
    cx
        Center of mass coordinate on x axes
    d_Kf
        Ratio between semiconv angle and pixel radius
    d_Qp
        Ratio between shape of the reconstructed area and step size

    semiconv_pix
        Radius of diffraction pattern in pixel

    coeff
        Matrix from sampled Hermite-Gauss polynomials
        for both x and y direction

    epsilon
        Small factor to avoid zero division
    """
    # Allocation for result
    wiener_filter_compressed = np.zeros(scan_dim + (order, order),
                                        dtype=complex_dtype)

    # Generate center probe (circular aperture)
    probe_center = probe_init_jit((cy, cx),
                                  y, x,
                                  0., 0.,
                                  semiconv_pix,
                                  float_dtype,
                                  norm=True
                                  )

    scan_dim = np.array(scan_dim)
    for q in numba.prange(scan_dim[0]):
        for p in range(scan_dim[1]):

            # Get physical coordinate
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
            probe_shift = probe_init_jit((cy, cx),
                                         y, x,
                                         sy, sx,
                                         semiconv_pix,
                                         float_dtype,
                                         norm=True,
                                         )
            # Autocorrelation
            product_conjugate = probe_center*np.conj(probe_shift)

            # Get the inverse Wigner function
            # Below is wrong it should be compressed space,
            # since we change ifft2 to compressed basis
            # wigner_func_inv =  np.fft.ifft2((product_conjugate)).
            # astype(np.complex64)
            wigner_func_inv = compress_jit(product_conjugate, coeff)

            # Wiener filter
            # Numerator
            num = np.conj(wigner_func_inv)
            # Denominator
            den = np.abs(wigner_func_inv)**2 + epsilon
            wiener_filter_compressed[q, p] = num/den

    return wiener_filter_compressed


def roi_wiener_filter(
    wiener_filter_compressed: np.ndarray
):

    """
    A function to calculate which scanning points has zero Wiener filter

    Parameters
    ----------
    wiener_filter_compressed
        Four-dimensional wiener filter

    Returns
    -------
        The non-zero scanning position
    """
    check_zero = np.sum(np.abs(wiener_filter_compressed), axis=(2, 3))

    return np.argwhere(~np.isclose(np.abs(check_zero), 0))
