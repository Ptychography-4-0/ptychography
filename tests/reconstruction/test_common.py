import numpy as np
from scipy.sparse import csc_matrix

import pytest

from libertem.common import Shape, Slice

from ptychography.reconstruction.common import (
    rotate_sysx, dot_product_transposed, wavelength, offset, get_shifted, to_slices,
    bounding_box
)


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


def test_get_shifted_base():
    data = np.random.random((6, 7))
    tile_origin = (0, 0)
    tile_shape = (6, 7)
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(0,0)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)
    assert np.all(res == data)
    assert res.dtype == data.dtype
    assert res.shape == data.shape


def test_get_shifted_plus():
    data_shape = (4, 5)
    data = np.random.random(data_shape)
    tile_origin = (0, 0)
    tile_shape = data_shape
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(1,2)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)
    assert np.all(res[:-1, :-2] == data[1:, 2:])
    assert np.all(res[-1:] == 17)
    assert np.all(res[:, -2:] == 17)
    assert res.dtype == data.dtype
    assert res.shape == data.shape


def test_get_shifted_minus():
    data_shape = (4, 5)
    data = np.random.random(data_shape)
    tile_origin = (0, 0)
    tile_shape = data_shape
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(-2,-3)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)

    assert np.all(res[2:, 3:] == data[:-2, :-3])
    assert np.all(res[:2] == 17)
    assert np.all(res[:, :3] == 17)
    assert res.dtype == data.dtype
    assert res.shape == data.shape


def test_get_shifted_partial():
    data_shape = (6, 7)
    data = np.random.random(data_shape)
    tile_origin = (1, 2)
    tile_shape = (3, 4)
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(0,0)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)

    assert np.all(res == data[1:4, 2:6])
    assert res.dtype == data.dtype


def test_get_shifted_plus_partial():
    data_shape = (6, 7)
    data = np.random.random(data_shape)
    tile_origin = (1, 2)
    tile_shape = (3, 4)
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(1,1)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)

    assert np.all(res == data[2:5, 3:7])
    assert res.dtype == data.dtype


def test_get_shifted_plusplus_partial():
    data_shape = (6, 7)
    data = np.random.random(data_shape)
    tile_origin = (1, 2)
    tile_shape = (3, 4)
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(3,4)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)

    assert np.all(res[:2, :1] == data[4:, 6:])
    assert np.all(res[2:] == 17)
    assert np.all(res[:, 1:] == 17)
    assert res.dtype == data.dtype


def test_get_shifted_minus_partial():
    data_shape = (6, 7)
    data = np.random.random(data_shape)
    tile_origin = (1, 2)
    tile_shape = (3, 4)
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(-1,-2)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)

    assert np.all(res == data[:3, :4])
    assert res.dtype == data.dtype


def test_get_shifted_minusminus_partial():
    data_shape = (6, 7)
    data = np.random.random(data_shape)
    tile_origin = (1, 2)
    tile_shape = (3, 4)
    target_tup, offsets = get_shifted(
        arr_shape=data.shape,
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=(-2,-4)
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)

    assert np.all(res[1:, 2:] == data[:2, :2])
    assert np.all(res[:1] == 17)
    assert np.all(res[:, :2] == 17)
    assert res.dtype == data.dtype


def test_bounding_box():
    data = np.zeros((6, 7))
    data[1, 2] = 1
    data[3, 4] = 1
    ((y_min, y_max), (x_min, x_max)) = bounding_box(data)

    assert y_min == 1
    assert y_max == 3
    assert x_min == 2
    assert x_max == 4


def test_bounding_full():
    data = np.ones((6, 7))
    ((y_min, y_max), (x_min, x_max)) = bounding_box(data)

    assert y_min == 0
    assert y_max == 5
    assert x_min == 0
    assert x_max == 6


def test_bounding_box_empty():
    data = np.zeros((6, 7))
    ((y_min, y_max), (x_min, x_max)) = bounding_box(data)

    assert y_min == 0
    assert y_max == 0
    assert x_min == 0
    assert x_max == 0


def test_bounding_box_single():
    data = np.zeros((6, 7))
    data[3, 4] = 1
    ((y_min, y_max), (x_min, x_max)) = bounding_box(data)

    assert y_min == 3
    assert y_max == 3
    assert x_min == 4
    assert x_max == 4
