from typing import Tuple
import numpy as np
import typing
from libertem.common import Shape
if typing.TYPE_CHECKING:
    import numpy.typing as nt


def H0(coords: np.ndarray):
    '''
    A function to generate Gaussian envelope
    Parameters
    ----------

    coords
        Sampling grids

    Returns
    -------
    Sampled Gaussian function
    '''
    return np.exp(-0.5 * coords**2)


def vectors(
    coords: np.ndarray,
    L_lim: int,
    float_dtype: "nt.DTypeLike"
):

    '''
    A function to generate vector Hermite-Gauss polynomials

    Parameters
    ----------

    L_lim
        Maximum degree of polynomials

    coords
        Sampling grids
    float_dtype
        Using given float_dtype to avoid mismatch later
    Returns
    -------
    result
        Sampled Hermite-Gauss polynomials with ambient dimension L_lim
    '''

    result = np.zeros((len(coords), L_lim), dtype=float_dtype)

    Hnp1 = np.zeros_like(coords)
    Hn = H0(coords)
    Hnm1 = np.zeros_like(coords)

    result[:, 0] = Hn / np.sqrt(np.sum(Hn**2))

    for nu in range(0, L_lim-1):
        nuhalf = nu / 2
        Hnp1 = coords * Hn - nuhalf * Hnm1
        result[:, nu+1] = Hnp1 / np.sqrt(np.sum(Hnp1**2))
        Hnm1 = Hn
        Hn = Hnp1

    return result


def get_sampled_basis(
    order: np.ndarray,
    sig_shape: Shape,
    com: Tuple,
    semiconv_pix: float,
    scale: float,
    float_dtype: "nt.DTypeLike",
):
    '''
    A function to generate vector Hermite-Gauss polynomials
    for both coordinates x and y, or column and row spaces

    Parameters
    ----------

    order
        Maximum degree of polynomials

    ds_shape
        Dimension of four-dimensional datasets
    com
        Center of mass (cy, cx)
    semiconv_pix
        Radius of circular aperture on pixel
    Returns
    -------
    vy
        Matrix from sampled Hermite-Gauss polynomials with ambient dimension
        is equal to dimension of y direction input data times maximum order
    vx
        Matrix from sampled Hermite-Gauss polynomials with ambient dimension
        is equal to dimension of y direction input data times maximum order
    '''
    cy, cx = com
    y = np.arange(sig_shape[0])
    x = np.arange(sig_shape[1])

    xi_y = (y-cy) / (semiconv_pix / scale)
    xi_x = (x-cx) / (semiconv_pix / scale)

    v_y = vectors(xi_y, order, float_dtype)
    v_x = vectors(xi_x, order, float_dtype)

    return v_y, v_x


def compress(
    frame: np.ndarray,
    coeff: Tuple[np.ndarray, np.ndarray],
):
    '''
    A function to apply matrix from sampled Hermite-Gauss polynomials
    for dimensionality reduction

    Parameters
    ----------

    frame
        Input matrix in higher dimensional space

    coeff
        Matrix from sampled Hermite-Gauss polynomials
        for both x and y direction
    Returns
    -------
        Lower dimensional space of the frame

    '''

    v_y, v_x = coeff
    x_result = frame @ v_x
    low_dim = x_result.T @ v_y

    return low_dim


def decompress(
    compressed: np.ndarray,
    coeff: Tuple[np.ndarray, np.ndarray]
):
    '''
    A function to reconstruct the data into original ambient dimension
    Parameters
    ----------

    compressed
        Input matrix in lower dimensional space

    coeff
        Matrix from sampled Hermite-Gauss polynomials
        for both x and y direction
    Returns
    -------
        Higher dimensional space of the frame
    '''
    v_y, v_x = coeff
    reconstruct_x = v_x @ compressed
    return v_y @ reconstruct_x.T
