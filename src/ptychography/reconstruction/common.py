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


def offset(s1: Slice, s2: Slice):
    return (right - left for (left, right) in zip(s1.origin, s2.origin))


def shift_by(slice: Slice, shift):
    return Slice(
        origin=tuple(our_coord + shift for (our_coord, shift) in zip(slice.origin, shift)),
        shape=slice.shape
    )


def shift_to(slice: Slice, origin):
    return Slice(
        origin=origin,
        shape=slice.shape
    )


def get_shifted(arr_shape: tuple, tile_origin: tuple, tile_shape: tuple, shift: tuple):
    # TODO this could be adapted for full sig, nav, n-D etc support
    # and included as a method in Slice?
    full_slice = Slice(origin=(0,)*len(tile_origin), shape=Shape(arr_shape, sig_dims=len(arr_shape)))
    tileslice = Slice(origin=tile_origin, shape=Shape(tile_shape, sig_dims=len(tile_shape)))
    shifted = shift_by(tileslice, shift)
    intersection = full_slice.intersection_with(shifted)
    if intersection.is_null():
        return (((0, 0), ) * len(tile_shape), (0, ) * len(tile_shape))
    # We measure by how much we have clipped the zero point
    # This is zero if we didn't shift into the negative region beyond the original array
    clip = offset(shifted, intersection)
    # Now we move the intersection to (0, 0) plus the amount we clipped
    # so that the overlap region is moved by the correct amount, in total
    targetslice = shift_by(shift_to(intersection, (0, 0)), clip)
    target_tup = tuple((s.start, s.stop) for s in targetslice.get())
    offsets = tuple(s.start - target[0] for (target, s) in zip(target_tup, intersection.get()))
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
        return ((y_min, y_max), (x_min, x_max))
    else:
        return ((0, 0), (0, 0))
