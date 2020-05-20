import math

import numpy as np
import numba
import scipy.constants as const

from libertem.common import Slice, Shape


@numba.njit(fastmath=True)
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


# Calculation of the relativistic electron wavelength in meters
def wavelength(U):
    e = const.elementary_charge  # Elementary charge  !!! 1.602176634×10−19
    h = const.Planck  # Planck constant    !!! 6.62607004 × 10-34
    c = const.speed_of_light  # Speed of light
    m_0 = const.electron_mass  # Electron rest mass

    T = e*U*1000
    lambda_e = h*c/(math.sqrt(T**2+2*T*m_0*(c**2)))
    return lambda_e


@numba.njit
def offset(s1, s2):
    o1, ss1 = s1
    o2, ss2 = s2
    return o2 - o1


@numba.njit
def shift_by(sl, shift):
    origin, shape = sl
    return (
        origin + shift,
        shape
    )


@numba.njit
def shift_to(s1, origin):
    o1, ss1 = s1
    return (
        origin,
        ss1
    )


@numba.njit
def intersection(s1, s2):
    o1, ss1 = s1
    o2, ss2 = s2
    # Adapted from libertem.common.slice
    new_origin = np.maximum(o1, o2)
    new_shape = np.minimum(
        (o1 + ss1) - new_origin,
        (o2 + ss2) - new_origin,
    )
    new_shape = np.maximum(0, new_shape)
    return (new_origin, new_shape)


@numba.njit
def get_shifted(arr_shape: tuple, tile_origin: tuple, tile_shape: tuple, shift: tuple):
    # TODO this could be adapted for full sig, nav, n-D etc support
    # and included as a method in Slice?
    full_slice = (np.array((0, 0)), arr_shape)
    tileslice = (tile_origin, tile_shape)
    shifted = shift_by(tileslice, shift)
    isect = intersection(full_slice, shifted)
    if np.prod(isect[1]) == 0:
        return (
            np.array([(0, 0), (0, 0)]),
            np.array([0, 0])
        )
    # We measure by how much we have clipped the zero point
    # This is zero if we didn't shift into the negative region beyond the original array
    clip = offset(shifted, isect)
    # Now we move the intersection to (0, 0) plus the amount we clipped
    # so that the overlap region is moved by the correct amount, in total
    targetslice = shift_by(shift_to(isect, np.array((0, 0))), clip)
    start = targetslice[0]
    length = targetslice[1]
    target_tup = np.stack((start, start+length), axis=1)
    offsets = isect[0] - targetslice[0]
    return (target_tup, offsets)


def to_slices(target_tup, offsets):
    target_slice = tuple(slice(s[0], s[1]) for s in target_tup)
    source_slice = tuple(slice(s[0] + o, s[1] + o) for (s, o) in zip(target_tup, offsets))
    return (target_slice, source_slice)


def bounding_box(array):
    # Based on https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    if np.any(rows):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return np.array(((y_min, y_max), (x_min, x_max)))
    else:
        return np.array([(0, 0), (0, 0)])
