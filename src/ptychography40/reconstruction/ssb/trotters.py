import functools

import numpy as np
import sparse
import numba

from libertem.masks import circular
from libertem.corrections.coordinates import identity

from ptychography40.reconstruction.common import get_shifted


@functools.lru_cache(None)
def empty_mask(mask_shape, dtype):
    '''
    Return an empty sparse mask

    Improve readability
    '''
    return sparse.zeros(mask_shape, dtype=dtype)


def mask_pair_subpix(cy, cx, sy, sx, filter_center, semiconv_pix):
    '''
    Calculate positive and negative trotter mask for circular illumination
    using a method with subpixel capability.

    Parameters
    ----------

    cy, cx : float
        Position of the optical axis on the detector in px, center of illumination
    sy, sx : float
        Trotter shift value in px
    filter_center : numpy.ndarray
        Center illumination, i.e. zero order disk. This has to be circular and match the radius
        semiconv_pix. It is just passed as an argument for efficientcy to avoid unnecessary
        recalculation.
    semiconv_pix : float
        Semiconvergence angle in measured in detector pixel, i.e. radius of the zero order disk.

    Returns
    -------

    mask_positive, mask_negative : numpy.ndarray
        Positive and negative trotter mask
    '''
    mask_shape = filter_center.shape

    filter_positive = circular(
        centerX=cx+sx, centerY=cy+sy,
        imageSizeX=mask_shape[1], imageSizeY=mask_shape[0],
        radius=semiconv_pix,
        antialiased=True
    )

    filter_negative = circular(
        centerX=cx-sx, centerY=cy-sy,
        imageSizeX=mask_shape[1], imageSizeY=mask_shape[0],
        radius=semiconv_pix,
        antialiased=True
    )
    mask_positive = filter_center * filter_positive * (filter_negative == 0)
    mask_negative = filter_center * filter_negative * (filter_positive == 0)
    return mask_positive, mask_negative


@numba.njit
def mask_tile_pair(center_tile, tile_origin, tile_shape, filter_center, sy, sx):
    '''
    Numerical work horse for :meth:`mask_pair_shift`, including tiling support.

    The tiling support could be used to calculate the mask stack on the fly,
    including support for UDF.process_tile().

    Parameters
    ----------

    center_tile : numpy.ndarray
        Tile cut out from :code:`filter_center` for re-use to increase efficiency
    tile_origin : tuple
        Origin of the tile to calculate
    tile_shape : tuple
        Shape of the tile
    filter_center : numpy.ndarray
        Center illumination, i.e. zero order disk.
    sy, sx : float
        Trotter shift value in px

    Returns
    -------
    mask_positive : numpy.ndarray
        Positive trotter tile
    target_tup_p : numpy.ndarray of int
        Start and stop indices per axis that were used for shifting the positive trotter tile.
    offsets_p : numpy.ndarray
        Offsets per axis that were used for shifting the positive trotter tile.
    mask_negative : numpy.ndarray
        Negative trotter tile
    target_tup_n : numpy.ndarray
        Start and stop indices per axis that were used for shifting the negative trotter tile.
    offsets_n : numpy.ndarray
        Offsets per axis that were used for shifting the negative trotter tile.
    '''
    sy, sx, = np.int(np.round(sy)), np.int(np.round(sx))
    positive_tile = np.zeros_like(center_tile)
    negative_tile = np.zeros_like(center_tile)
    # We get from negative coordinates,
    # that means it looks like shifted to positive
    target_tup_p, offsets_p = get_shifted(
        arr_shape=np.array(filter_center.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((-sy, -sx))
    )
    # We get from positive coordinates,
    # that means it looks like shifted to negative
    target_tup_n, offsets_n = get_shifted(
        arr_shape=np.array(filter_center.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((sy, sx))
    )

    sta_y, sto_y, sta_x, sto_x = target_tup_p.flatten()
    off_y, off_x = offsets_p
    positive_tile[sta_y:sto_y, sta_x:sto_x] = filter_center[
        sta_y+off_y:sto_y+off_y,
        sta_x+off_x:sto_x+off_x
    ]
    sta_y, sto_y, sta_x, sto_x = target_tup_n.flatten()
    off_y, off_x = offsets_n
    negative_tile[sta_y:sto_y, sta_x:sto_x] = filter_center[
        sta_y+off_y:sto_y+off_y,
        sta_x+off_x:sto_x+off_x
    ]

    mask_positive = center_tile * positive_tile * (negative_tile == 0)
    mask_negative = center_tile * negative_tile * (positive_tile == 0)

    return (mask_positive, target_tup_p, offsets_p, mask_negative, target_tup_n, offsets_n)


def mask_pair_shift(cy, cx, sy, sx, filter_center, semiconv_pix):
    '''
    Calculate positive and negative trotter mask using a fast shifting method.

    It has the same signature as :meth:`mask_pair_subpix` for easy changing
    between the methods. That means several parameters that are only relevant
    for the subpix method are ignored in this function.

    Parameters
    ----------

    cy, cx : float
        Ignored, given implicitly by filter_center
    sy, sx : float
        Trotter shift value in px
    filter_center : numpy.ndarray
        Center illumination, i.e. zero order disk.
    semiconv_pix : float
        Ignored, given implicitly by filter_center

    Returns
    -------

    mask_positive, mask_negative : numpy.ndarray
        Positive and negative trotter mask
    '''
    (mask_positive, target_tup_p, offsets_p,
    mask_negative, target_tup_n, offsets_n) = mask_tile_pair(
        center_tile=np.array(filter_center),
        tile_origin=np.array((0, 0)),
        tile_shape=np.array(filter_center.shape),

        filter_center=np.array(filter_center),
        sy=sy,
        sx=sx
    )
    return mask_positive, mask_negative


def generate_mask(cy, cx, sy, sx, filter_center, semiconv_pix,
                  cutoff, dtype, method='subpix'):
    '''
    Generate the trotter mask for a specific shift sy, sx

    Parameters
    ----------

    cy, cx : float
        Position of the optical axis on the detector in px, center of illumination
    sy, sx : float
        Trotter shift value in px
    filter_center : numpy.ndarray
        Center illumination, i.e. zero order disk. This has to be circular and match the radius
        semiconv_pix if :code:`method=subpix`. It is passed for re-use to avoid unnecessary
        recalculation
    semiconv_pix : float
        Semiconvergence angle in measured in detector pixel, i.e. radius of the zero order disk.
    cutoff : int
        Minimum number of pixels in the positive and negative trotter. This can be used to purge
        very small trotters to reduce noise.
    dtype : numpy dtype
        dtype to use for the mask
    method : str, optional
        Can be :code:`'subpix'`(default) or :code:`'shift'` to switch between
        :meth:`mask_pair_subpix` and :meth:`mask_pair_shift` to generate the trotter pair.

    Returns
    -------
    mask : sparse.COO
        Mask in sparse.pydata.org COO format

    '''
    mask_shape = filter_center.shape
    # 1st diffraction order and primary beam don't overlap
    if sx**2 + sy**2 > 4*np.sum(semiconv_pix**2):
        return empty_mask(mask_shape, dtype=dtype)

    if np.allclose((sy, sx), (0, 0)):
        # The zero order component (0, 0) is special, comes out zero with above code
        m_0 = filter_center / filter_center.sum()
        return sparse.COO(m_0.astype(dtype))

    params = dict(
        cy=cy, cx=cx, sy=sy, sx=sx,
        filter_center=filter_center,
        semiconv_pix=semiconv_pix,
    )

    if method == 'subpix':
        mask_positive, mask_negative = mask_pair_subpix(**params)
    elif method == 'shift':
        mask_positive, mask_negative = mask_pair_shift(**params)
    else:
        raise ValueError(f"Unsupported method {method}. Allowed are 'subpix' and 'shift'")

    non_zero_positive = mask_positive.sum()
    non_zero_negative = mask_negative.sum()

    if non_zero_positive >= cutoff and non_zero_negative >= cutoff:
        m = (
            mask_positive / non_zero_positive
            - mask_negative / non_zero_negative
        ) / 2
        m_dt = m.astype(dtype)
        m_sparse = sparse.COO(m_dt)
        return m_sparse
    else:
        # Exclude small, missing or unbalanced trotters
        return empty_mask(mask_shape, dtype=dtype)


def generate_masks(reconstruct_shape, mask_shape, dtype, lamb, dpix, semiconv,
        semiconv_pix, transformation=None, cy=None, cx=None, cutoff=1,
        cutoff_freq=np.float32('inf'), method='subpix'):
    '''
    Generate the trotter mask stack.

    The y dimension is trimmed to size(y)//2 + 1 to exploit the inherent
    symmetry of the mask stack.

    Parameters
    ----------

    reconstruct_shape : tuple(int)
        Shape of the reconstructed area
    mask_shape : tuple(int)
        Shape of the detector
    dtype : numpy dtype
        dtype to use for the mask stack
    lamb : float
        Wavelength of the illuminating radiation in m
    dpix : float or (float, float)
        Scan step in m. Tuple (y, x) in case scan step is different in x and y direction.
    semiconv : float
        Semiconvergence angle of the illumination in radians
    semiconv_pix : float
        Semiconvergence angle in measured in detector pixel, i.e. radius of the zero order disk.
    transformation : numpy.ndarray, optional
        Matrix for affine transformation from the scan coordinate directions
        to the detector coordinate directions. This does not include the scale, which is handled by
        dpix, lamb, semiconv and semiconv_pix. It should only be used to rotate and flip
        the coordinate system as necessary. See also
        https://github.com/LiberTEM/LiberTEM/blob/master/src/libertem/corrections/coordinates.py
    cy, cx : float, optional
        Position of the optical axis on the detector in px, center of illumination.
        Default: Center of the detector
    cutoff : int, optional
        Minimum number of pixels in the positive and negative trotter. This can be used to purge
        very small trotters to reduce noise. Default is 1, i.e. no cutoff unless one trotter is
        empty.
    cutoff_freq: float
        Trotters belonging to a spatial frequency higher than this value in reciprocal pixel
        coordinates will be cut off.
    method : str, optional
        Can be :code:`'subpix'`(default) or :code:`'shift'` to switch between
        :meth:`mask_pair_subpix` and :meth:`mask_pair_shift` to generate a trotter pair.

    Returns
    -------
    masks : sparse.COO
        Masks in sparse.pydata.org COO format. y and x frequency index are FFT shifted, i.e. the
        zero frequency is at (0,0) and negative frequencies are in the far quadrant and reversed.
        The y frequency index is cut in half with size(y)//2 + 1 to exploit the inherent symmetry
        of a real-valued Fourier transform. The y and x index are then flattened to make it
        suitable for using it with MaskContainer.
    '''
    reconstruct_shape = np.array(reconstruct_shape)

    dpix = np.array(dpix)

    d_Kf = np.sin(semiconv)/lamb/semiconv_pix
    d_Qp = 1/dpix/reconstruct_shape

    if cy is None:
        cy = mask_shape[0] / 2
    if cx is None:
        cx = mask_shape[1] / 2

    if transformation is None:
        transformation = identity()

    filter_center = circular(
        centerX=cx, centerY=cy,
        imageSizeX=mask_shape[1], imageSizeY=mask_shape[0],
        radius=semiconv_pix,
        antialiased=True
    )

    half_reconstruct = (reconstruct_shape[0]//2 + 1, reconstruct_shape[1])
    masks = []

    for row in range(half_reconstruct[0]):
        for column in range(half_reconstruct[1]):
            # Do an fftshift of q and p
            qp = np.array((row, column))
            flip = qp > (reconstruct_shape / 2)
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - reconstruct_shape[flip]

            if np.sum(real_qp**2) > cutoff_freq**2:
                masks.append(empty_mask(mask_shape, dtype=dtype))
                continue

            # Shift of diffraction order relative to zero order
            # without rotation in physical coordinates
            real_sy_phys, real_sx_phys = real_qp * d_Qp
            # We apply the transformation backwards to go
            # from physical orientation to detector orientation,
            # while the forward direction in center of mass analysis
            # goes from detector coordinates to physical coordinates
            # Afterwards, we transform from physical detector coordinates
            # to pixel coordinates
            sy, sx = ((real_sy_phys, real_sx_phys) @ transformation) / d_Kf

            masks.append(generate_mask(
                cy=cy, cx=cx, sy=sy, sx=sx,
                filter_center=filter_center,
                semiconv_pix=semiconv_pix,
                cutoff=cutoff,
                dtype=dtype,
                method=method,
            ))

    # Since we go through masks in order, this gives a mask stack with
    # flattened (q, p) dimension to work with dot product and mask container
    masks = sparse.stack(masks)
    return masks
