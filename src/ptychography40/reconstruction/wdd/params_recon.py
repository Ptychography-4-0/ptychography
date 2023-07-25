from typing import NamedTuple, Tuple, TYPE_CHECKING
import numpy as np
from libertem.common import Shape
from ptychography40.reconstruction.wdd.dim_reduct import get_sampled_basis
from ptychography40.reconstruction.wdd.wiener_filter import pre_computed_Wiener
if TYPE_CHECKING:
    import numpy.typing as nt


def f2d_matrix_replacement(
    nav_shape: Shape,
    complex_dtype: 'nt.DTypeLike'
):

    """
    A Function to generate sampled Fourier basis, since we process the data
    per frame, it is better to use sampled Fourier basis not fft

    Parameters
    ----------
    nav_shape
        Dimension of scan points
    complex_dtype
        Pre defined data type of the elements

    Returns
    -------
    row_exp
        Sampled Fourier basis in terms of row dimension
    col_exp
        Sampled Fourier basis in terms of column dimension

    """
    reconstruct_shape = tuple(nav_shape)
    row_steps = -2j*np.pi*np.linspace(0, 1, reconstruct_shape[0],
                                      endpoint=False)
    col_steps = -2j*np.pi*np.linspace(0, 1, reconstruct_shape[1],
                                      endpoint=False)

    full_y = reconstruct_shape[0]
    full_x = reconstruct_shape[1]

    # This creates a 2D array of row x spatial frequency
    row_exp = np.exp(
        row_steps[:, np.newaxis]
        * np.arange(full_y)[np.newaxis, :]
    )
    # This creates a 2D array of col x spatial frequency
    col_exp = np.exp(
        col_steps[:, np.newaxis]
        * np.arange(full_x)[np.newaxis, :]
    )
    # Fourier matrix has normalization
    return ((1/np.sqrt(reconstruct_shape[0]))*row_exp.astype(complex_dtype),
            ((1 / np.sqrt(reconstruct_shape[1]))*col_exp).astype(complex_dtype)
            )


class WDDReconstructionPlan(NamedTuple):
    """
    Parameters that is needed for reconstruction

    Attributes
    -------
    roi
        Non zero index for overlapping region
    wiener_filter_compressed
        Wiener filter after dimensionality reduction
    row_exp
        Fourier matrix applied on the row space
    col_exp
        Fourier matrix applied on the column space
    coeff
        Matrix contains sampled Hermite-Gauss functions
        for dimensionality reduction
    """
    coeff: Tuple[np.ndarray, np.ndarray]
    wiener_filter_compressed: np.ndarray
    row_exp: np.ndarray
    col_exp: np.ndarray
    wiener_roi: np.ndarray


def wdd_params_recon(ds_shape: Shape,
                     params: dict,
                     order: int = 16,
                     scale: int = 5,
                     complex_dtype: "nt.DTypeLike" = np.complex64,
                     ):

    """
    params
        Dictionary related to physical coordinates
    order
        Maximum degree of Hermite-Gauss function
        default is 16
    scale
        The scaling radius for Hermite-Gaussian polynomials
    complex_dtype
        Pre-defined complex_dtype

    """

    if np.dtype(complex_dtype) == np.complex64:
        float_dtype = np.float32
    elif np.dtype(complex_dtype) == np.complex128:
        float_dtype = np.float64
    else:
        raise RuntimeError(f"unknown complex dtype: {complex_dtype}")

    # Generate Hermite-Gauss function
    coeff = get_sampled_basis(order=order,
                              sig_shape=ds_shape.sig,
                              com=params['com'],
                              scale=scale,
                              semiconv_pix=params['semiconv_pix'],
                              float_dtype=float_dtype
                              )

    # Fourier calculation
    row_exp, col_exp = f2d_matrix_replacement(nav_shape=ds_shape.nav,
                                              complex_dtype=complex_dtype)

    # Calculate Wiener filter
    wiener_filter_compressed, wiener_roi = pre_computed_Wiener(
        ds_shape,
        order=order,
        params=params,
        coeff=coeff,
        epsilon=params['epsilon'],
        complex_dtype=complex_dtype,
        transformation=params['transformation'],
    )

    return WDDReconstructionPlan(coeff, wiener_filter_compressed,
                                 row_exp, col_exp, wiener_roi)
