import numpy as np
from typing import Tuple
from libertem.udf import UDF
from ptychography40.reconstruction.wdd import dim_reduct
import numba
import typing
if typing.TYPE_CHECKING:
    import numpy.typing as nt


@numba.njit(fastmath=True, parallel=False)
def get_frame_contribution_to_cut_rowcol_exp(
    row_exp: np.ndarray,
    col_exp: np.ndarray,
    frame_compressed: np.ndarray,
    y: int,
    x: int,
    nav_shape: Tuple,
    wiener_filter_compressed: np.ndarray,
    scan_idx: np.ndarray,
    complex_dtype: 'nt.DTypeLike',
):

    """
    A Function to process data per frame and calculate the spatial frequency

    Parameters
    ----------
    row_exp
        Sampled Fourier basis on the row dimension
    col_exp
        Sampled Fourier basis on the column dimension
    frame_compressed
        Frame diffraction patterns after dimensionality reduction
    y
        Certain scanning point on y dimension
    x
        Certain scanning point on x dimension
    nav_shape
        Dimension of scanning points
    wiener_filter
        Four-dimensional Wiener filter after dimensionality reduction
    scan_idx
        Non zero contribution on the scanning points
    complex_dtype
        Pre defined complex dtype

    Returns
    -------
    cut
        Reconsruction of the specimen transfer function on Fourier space
        per scanning point

    """

    cut = np.zeros(nav_shape, dtype=complex_dtype)
    for nn_idx in range(len(scan_idx)):
        q = scan_idx[nn_idx][0]
        p = scan_idx[nn_idx][1]
        # assuming we have isotropic sampling,
        # np.allclose(row_exp, col_exp) holds:
        # If scan dim is not rectangular we cant use row_exp*row_exp!
        exp_factor = row_exp[y, q] * col_exp[x, p]
        acc = np.zeros((1), dtype=complex_dtype)[0]
        wiener_qp = wiener_filter_compressed[q, p]
        # NOTE: we still can cut this in ~half, as `
        # frame_compressed` should be truncated to
        # zero in the lower-right triangle
        for yy in range(frame_compressed.shape[0]):
            for xx in range(frame_compressed.shape[1]):
                acc += frame_compressed[yy, xx] * wiener_qp[yy, xx]
        cut[q, p] = (acc * exp_factor)
    return cut


def wdd_per_frame_combined(
    idp: np.ndarray,
    coeff: Tuple[np.ndarray, np.ndarray],
    wiener_filter_compressed: np.ndarray,
    scan_idx: np.ndarray,
    row_exp: np.ndarray,
    col_exp: np.ndarray,
    complex_dtype: 'nt.DTypeLike',
):
    """
    WDD in harmonic compressed space and process the data per frame

    Parameters
    ----------
    idp
        Intensity of diffraction patterns of 4D-STEM

    wiener_filter_compressed
        Four-dimensional compressed Wiener filter

    scan_idx
        Non-zero index position

    row_exp
        Sampled Fourier basis on the row dimension
    col_exp
        Sampled Fourier basis on the column dimension

    coeff
        Matrix from sampled Hermite-Gauss
        polynomials for both x and y direction
    complex_dtype
        Pre defined complex dtype
    """
    nav_shape = idp.shape[:2]
    cut = np.zeros((nav_shape[0], nav_shape[1]), dtype=complex_dtype)

    for y in range(nav_shape[0]):
        for x in range(nav_shape[1]):
            idp_compressed = dim_reduct.compress(idp[y, x], coeff)
            cut += get_frame_contribution_to_cut_rowcol_exp(
                row_exp, col_exp, idp_compressed, y, x, nav_shape,
                wiener_filter_compressed, scan_idx, complex_dtype,
            )
    real_cut = np.fft.ifft2((cut)).astype(complex_dtype)
    real_cut = real_cut/np.max(np.abs(real_cut))
    return real_cut.conj()


class WDDUDF(UDF):
    """
    Class that use UDF for live processing method,
    the implementation uses LiberTEM UDF

    Parameters
    ----------
    wiener_filter
        Pre computed Wiener filter for deconvolution process
    wiener_roi
        Region of interest of the scanning points
    row_exp
        Construction of row element of Fourier matrix
    col_exp
        Construction of column element of Fourier matrix
    complex_dtype
        Complex dtype for array

    """

    def __init__(self, recon_parameters,
                 complex_dtype: "nt.DTypeLike" = np.complex64):
        super().__init__(
            recon_parameters=recon_parameters,
            complex_dtype=complex_dtype
        )

    def get_result_buffers(self):
        """
        Method for preparation of variable output

        """
        return {
            'cut': self.buffer(
                kind='single',
                extra_shape=self.meta.dataset_shape.nav,
                dtype=self.params.complex_dtype
            ),
            'reconstructed': self.buffer(
                kind='nav',
                use='result_only',
                dtype=self.params.complex_dtype
            ),
        }

    def process_frame(self, frame):

        """
        Method for processing frame per scan, since modification of WDD
        we can model the Fourier transformation into summation
        """
        y, x = self.meta.coordinates[0]
        frame_compressed = dim_reduct.compress(
            frame,
            self.params.recon_parameters.coeff)
        self.results.cut[:] += get_frame_contribution_to_cut_rowcol_exp(
            self.params.recon_parameters.row_exp,
            self.params.recon_parameters.col_exp,
            frame_compressed,
            y,
            x,
            tuple(self.meta.dataset_shape.nav),
            self.params.recon_parameters.wiener_filter_compressed,
            self.params.recon_parameters.wiener_roi,
            self.params.complex_dtype
        )

    def merge(self, dest, src):
        """
        Merge result
        """
        dest.cut[:] += src.cut

    def get_results(self):
        """
        Inverse Fourier transform of the result in order to get into real space

        """
        real_cut = np.fft.ifft2(
            (self.results.cut)
            ).astype(self.params.complex_dtype)
        real_cut = real_cut/np.max(np.abs(real_cut))
        return {
            'cut': self.results.cut,
            'reconstructed': real_cut.conj(),
        }
