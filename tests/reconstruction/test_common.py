import numpy as np
import scipy.ndimage

import pytest
from libertem.corrections.coordinates import rotate_deg
import libertem.api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.utils.devices import detect, has_cupy

from ptychography40.reconstruction.common import (
    diffraction_to_detector, wavelength, get_shifted, to_slices,
    bounding_box, ifftshift_coords, fftshift_coords, image_transformation_matrix, apply_matrix,
    shifted_probes,
    rolled_object_probe_product_cpu, rolled_object_aggregation_cpu,
    rolled_object_probe_product_cuda, rolled_object_aggregation_cuda
)


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


@pytest.mark.with_numba
def test_get_shifted_base():
    data = np.random.random((6, 7))
    tile_origin = np.array((0, 0))
    tile_shape = np.array((6, 7))
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((0, 0))
    )
    print(target_tup)
    print(offsets)
    (target_slice, source_slice) = to_slices(target_tup, offsets)
    print(target_slice, source_slice)
    res = np.full(tile_shape, 17, dtype=data.dtype)
    res[target_slice] = data[source_slice]
    print("result:", res)
    print("data:", data)
    assert np.all(res == data)
    assert res.dtype == data.dtype
    assert res.shape == data.shape


def test_get_shifted_plus():
    data_shape = np.array((4, 5))
    data = np.random.random(data_shape)
    tile_origin = np.array((0, 0))
    tile_shape = data_shape
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((1, 2))
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
    data_shape = np.array((4, 5))
    data = np.random.random(data_shape)
    tile_origin = np.array((0, 0))
    tile_shape = data_shape
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((-2, -3))
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
    data_shape = np.array((6, 7))
    data = np.random.random(data_shape)
    tile_origin = np.array((1, 2))
    tile_shape = np.array((3, 4))
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((0, 0))
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
    data_shape = np.array((6, 7))
    data = np.random.random(data_shape)
    tile_origin = np.array((1, 2))
    tile_shape = np.array((3, 4))
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((1, 1))
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
    data_shape = np.array((6, 7))
    data = np.random.random(data_shape)
    tile_origin = np.array((1, 2))
    tile_shape = np.array((3, 4))
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((3, 4))
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
    data_shape = np.array((6, 7))
    data = np.random.random(data_shape)
    tile_origin = np.array((1, 2))
    tile_shape = np.array((3, 4))
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((-1, -2))
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
    data_shape = np.array((6, 7))
    data = np.random.random(data_shape)
    tile_origin = np.array((1, 2))
    tile_shape = np.array((3, 4))
    target_tup, offsets = get_shifted(
        arr_shape=np.array(data.shape),
        tile_origin=tile_origin,
        tile_shape=tile_shape,
        shift=np.array((-2, -4))
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
    assert y_max == 4
    assert x_min == 2
    assert x_max == 5


def test_bounding_full():
    data = np.ones((6, 7))
    ((y_min, y_max), (x_min, x_max)) = bounding_box(data)

    assert y_min == 0
    assert y_max == 6
    assert x_min == 0
    assert x_max == 7


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
    assert y_max == 4
    assert x_min == 4
    assert x_max == 5


@pytest.mark.with_numba
def test_transformation_identity():
    data_shape = np.random.randint(1, 77, 3, dtype=int)
    data = np.random.random(data_shape)
    source_shape = data_shape[1:]
    target_shape = data_shape[1:]
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=lambda x: x
    )
    assert np.allclose(data, apply_matrix(data, m, target_shape))


def test_transformation_transpose():
    data_shape = np.random.randint(1, 77, 3, dtype=int)
    data = np.random.random(data_shape)
    source_shape = data_shape[1:]
    target_shape = tuple(reversed(data_shape[1:]))
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=lambda x: np.flip(x, axis=-1)
    )
    assert np.allclose(np.swapaxes(data, 1, 2), apply_matrix(data, m, target_shape))


def test_transformation_rot90():
    data_shape = np.random.randint(1, 77, 3, dtype=int)
    data = np.random.random(data_shape)
    source_shape = data_shape[1:]
    target_shape = tuple(reversed(data_shape[1:]))
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=lambda x: x @ rotate_deg(90),
        pre_transform=lambda x: x - np.array(target_shape) / 2 + 0.5,
        post_transform=lambda x: np.round(x + np.array(source_shape) / 2 - 0.5).astype(int)
    )
    res = apply_matrix(data, m, target_shape)
    # positive rotations are clockwise, upper right corner
    # is now lower right corner
    assert np.allclose(data[:, 0, -1], res[:, -1, -1])
    # alternative rotation: transpose and flip
    assert np.allclose(np.flip(np.transpose(data, axes=(0, 2, 1)), axis=2), res)


def test_transformation_scale():
    scale = np.random.randint(1, 7)
    data_shape = np.random.randint(1, 17, 3, dtype=int) * scale
    data = np.random.random(data_shape)
    source_shape = data_shape[1:]
    target_shape = tuple(np.array(source_shape) // scale)
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=lambda x: x * scale,
    )
    print(data_shape, source_shape, target_shape)
    res = apply_matrix(data, m, target_shape)
    # Binning, not accumulating intensity but keeping same absolute values
    ref = data.reshape(
        (data.shape[0], target_shape[0], scale, target_shape[1], scale)
    ).mean(axis=(2, 4))
    assert np.allclose(ref, res)


def test_ifftshift():
    data_shape = (12, 13)
    data = np.random.random(data_shape)
    shifted_data = np.fft.ifftshift(data)
    for y in range(data_shape[0]):
        for x in range(data_shape[1]):
            shifted_coords = ifftshift_coords(data_shape)((y, x))
            assert data[shifted_coords[0], shifted_coords[1]] == shifted_data[y, x]


def test_fftshift():
    data_shape = (12, 13)
    data = np.random.random(data_shape)
    shifted_data = np.fft.fftshift(data)
    for y in range(data_shape[0]):
        for x in range(data_shape[1]):
            shifted_coords = fftshift_coords(data_shape)((y, x))
            assert data[shifted_coords[0], shifted_coords[1]] == shifted_data[y, x]


def test_fftshift_matrix():
    data_shape = np.random.randint(1, 77, 3, dtype=int)
    data = np.random.random(data_shape)
    source_shape = data_shape[1:]
    target_shape = data_shape[1:]
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=lambda x: x,
        pre_transform=fftshift_coords(target_shape)
    )
    assert np.allclose(np.fft.fftshift(data, axes=(1, 2)), apply_matrix(data, m, target_shape))


def test_difftodect_identity():
    data_shape = np.random.randint(1, 77, 3, dtype=int)
    data = np.random.random(data_shape)
    source_shape = data_shape[1:]
    target_shape = data_shape[1:]

    f = diffraction_to_detector(
        lamb=1,
        diffraction_shape=target_shape,
        pixel_size_real=1,
        pixel_size_detector=1/(np.array(target_shape)),
        cy=source_shape[0] / 2,
        cx=source_shape[1] / 2,
        flip_y=False,
        scan_rotation=0.
    )
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=f,
    )
    assert np.allclose(data, apply_matrix(data, m, target_shape))


def test_difftodect_flip():
    data_shape = np.random.randint(1, 77, 3, dtype=int)
    data = np.random.random(data_shape)
    source_shape = data_shape[1:]
    target_shape = data_shape[1:]

    f = diffraction_to_detector(
        lamb=1,
        diffraction_shape=target_shape,
        pixel_size_real=1,
        pixel_size_detector=1/(np.array(target_shape)),
        cy=source_shape[0] / 2 - 1,
        cx=source_shape[1] / 2,
        flip_y=True,
        scan_rotation=0.
    )
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=f,
    )
    assert np.allclose(np.flip(data, axis=1), apply_matrix(data, m, target_shape))


@pytest.mark.parametrize(
    'dim', (16, 17)
)
def test_difftodect_com_flip(dim):
    lt_ctx = lt.Context(InlineJobExecutor())
    data_shape = (2, 2, dim, dim)
    data = np.zeros(data_shape)
    data[0, 0, 7, 7] = 1
    data[0, 1, 7, 8] = 1
    data[1, 1, 8, 8] = 1
    data[1, 0, 8, 7] = 1
    source_shape = data_shape[2:]
    target_shape = data_shape[2:]

    f = diffraction_to_detector(
        lamb=1,
        diffraction_shape=target_shape,
        pixel_size_real=1,
        pixel_size_detector=1/(np.array(target_shape)),
        cy=source_shape[0] / 2,
        cx=source_shape[1] / 2,
        flip_y=True,
        scan_rotation=0.
    )
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=f,
    )
    transformed_data = apply_matrix(data, m, target_shape)
    ds = lt_ctx.load('memory', data=data, sig_dims=2)
    transformed_ds = lt_ctx.load('memory', data=transformed_data, sig_dims=2)
    com_a = lt_ctx.create_com_analysis(
        dataset=ds, mask_radius=np.inf, flip_y=True, scan_rotation=0.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )

    com_res = lt_ctx.run(com_a)

    trans_com_a = lt_ctx.create_com_analysis(
        dataset=transformed_ds, mask_radius=np.inf, flip_y=False, scan_rotation=0.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )
    trans_com_res = lt_ctx.run(trans_com_a)

    assert np.allclose(com_res.field.raw_data, trans_com_res.field.raw_data)


@pytest.mark.parametrize(
    'dim', (16, 17)
)
def test_difftodect_com_rot(dim):
    lt_ctx = lt.Context(InlineJobExecutor())
    data_shape = (2, 2, dim, dim)
    data = np.zeros(data_shape)
    data[0, 0, 7, 7] = 1
    data[0, 1, 7, 8] = 1
    data[1, 1, 8, 8] = 1
    data[1, 0, 8, 7] = 1
    source_shape = data_shape[2:]
    target_shape = data_shape[2:]

    f = diffraction_to_detector(
        lamb=1,
        diffraction_shape=target_shape,
        pixel_size_real=1,
        pixel_size_detector=1/(np.array(target_shape)),
        cy=source_shape[0] / 2,
        cx=source_shape[1] / 2,
        flip_y=False,
        scan_rotation=90.
    )
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=f,
    )
    transformed_data = apply_matrix(data, m, target_shape)
    ds = lt_ctx.load('memory', data=data, sig_dims=2)
    transformed_ds = lt_ctx.load('memory', data=transformed_data, sig_dims=2)
    com_a = lt_ctx.create_com_analysis(
        dataset=ds, mask_radius=np.inf, flip_y=False, scan_rotation=90.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )

    com_res = lt_ctx.run(com_a)

    trans_com_a = lt_ctx.create_com_analysis(
        dataset=transformed_ds, mask_radius=np.inf, flip_y=False, scan_rotation=0.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )
    trans_com_res = lt_ctx.run(trans_com_a)

    assert np.allclose(com_res.field.raw_data, trans_com_res.field.raw_data)


@pytest.mark.parametrize(
    'dim', (16, 17)
)
def test_difftodect_com_scale(dim):
    lt_ctx = lt.Context(InlineJobExecutor())
    data_shape = (2, 2, dim, dim)
    data = np.zeros(data_shape)
    data[0, 0, 7, 7] = 1
    data[0, 1, 7, 8] = 1
    data[1, 1, 8, 8] = 1
    data[1, 0, 8, 7] = 1
    source_shape = data_shape[2:]
    target_shape = data_shape[2:]

    f = diffraction_to_detector(
        lamb=1,
        diffraction_shape=target_shape,
        pixel_size_real=1,
        pixel_size_detector=1/(np.array(target_shape))*4,
        cy=source_shape[0] / 2,
        cx=source_shape[1] / 2,
        flip_y=False,
        scan_rotation=0.
    )
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=f,
    )
    transformed_data = apply_matrix(data, m, target_shape)
    ds = lt_ctx.load('memory', data=data, sig_dims=2)
    transformed_ds = lt_ctx.load('memory', data=transformed_data, sig_dims=2)
    com_a = lt_ctx.create_com_analysis(
        dataset=ds, mask_radius=np.inf, flip_y=False, scan_rotation=0.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )

    com_res = lt_ctx.run(com_a)

    trans_com_a = lt_ctx.create_com_analysis(
        dataset=transformed_ds, mask_radius=np.inf, flip_y=False, scan_rotation=0.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )
    trans_com_res = lt_ctx.run(trans_com_a)
    print(com_res.field.raw_data)
    print(trans_com_res.field.raw_data)

    assert np.allclose(com_res.field.raw_data, np.array(trans_com_res.field.raw_data)/4)


@pytest.mark.parametrize(
    'dim', (16, 17)
)
def test_difftodect_com_flip_rot_scale(dim):
    lt_ctx = lt.Context(InlineJobExecutor())
    data_shape = (2, 2, dim, dim)
    data = np.zeros(data_shape)
    data[0, 0, 7, 7] = 1
    data[0, 1, 7, 8] = 1
    data[1, 1, 8, 8] = 1
    data[1, 0, 8, 7] = 1
    source_shape = data_shape[2:]
    target_shape = data_shape[2:]

    f = diffraction_to_detector(
        lamb=1,
        diffraction_shape=target_shape,
        pixel_size_real=1,
        pixel_size_detector=1/(np.array(target_shape))*4,
        cy=source_shape[0] / 2,
        cx=source_shape[1] / 2,
        flip_y=True,
        scan_rotation=-90.
    )
    m = image_transformation_matrix(
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=f,
    )
    transformed_data = apply_matrix(data, m, target_shape)
    ds = lt_ctx.load('memory', data=data, sig_dims=2)
    transformed_ds = lt_ctx.load('memory', data=transformed_data, sig_dims=2)
    com_a = lt_ctx.create_com_analysis(
        dataset=ds, mask_radius=np.inf, flip_y=True, scan_rotation=-90.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )

    com_res = lt_ctx.run(com_a)

    trans_com_a = lt_ctx.create_com_analysis(
        dataset=transformed_ds, mask_radius=np.inf, flip_y=False, scan_rotation=0.,
        cy=target_shape[0] / 2,
        cx=target_shape[1] / 2
    )
    trans_com_res = lt_ctx.run(trans_com_a)
    print(com_res.field.raw_data)
    print(trans_com_res.field.raw_data)

    assert np.allclose(com_res.field.raw_data, np.array(trans_com_res.field.raw_data)/4)


def test_shifted_probe():
    probe_shape = np.random.randint(1, 10, 2)
    y_subpixels, x_subpixels = np.random.randint(1, 11, 2)

    probe = np.random.random(tuple(probe_shape)) + 1j*np.random.random(tuple(probe_shape))
    probes = shifted_probes(probe, (y_subpixels, x_subpixels))
    for y in range(y_subpixels):
        for x in range(x_subpixels):
            assert np.allclose(
                probes[y, x].real,
                scipy.ndimage.shift(probe.real, ((y/y_subpixels, x/x_subpixels)))
            )
            if np.iscomplexobj(probe):
                assert np.allclose(
                    probes[y, x].imag,
                    scipy.ndimage.shift(probe.imag, ((y/y_subpixels, x/x_subpixels)))
                )


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'ifftshift', (False, True)
)
def test_rolled_object_probe_product_cpu(ifftshift):
    obj_shape = np.random.randint(1, 23, 2)
    probe_shape = np.array([np.random.randint(1, obj_axis+1, 1)[0] for obj_axis in obj_shape])
    print(probe_shape)

    obj = np.linspace(0, 1, np.prod(obj_shape)).reshape(obj_shape)
    y_subpixels, x_subpixels = np.random.randint(1, 11, 2)
    probe = np.random.random(tuple(probe_shape))
    probes = shifted_probes(probe, (y_subpixels, x_subpixels))
    count = 23
    result = np.zeros((count, ) + probes.shape[2:], dtype=np.result_type(obj, probes))
    shifts = np.random.randint(
        -np.max(obj_shape)*y_subpixels, np.max(obj_shape)*x_subpixels,
        (count, 2)
    ) / (y_subpixels, x_subpixels)
    subpixel_indices = rolled_object_probe_product_cpu(
        obj, probes, shifts, result, ifftshift=ifftshift
    )
    for i in range(count):
        subpixel_y = (shifts[i, 0] * y_subpixels).astype(int) % y_subpixels
        subpixel_x = (shifts[i, 1] * x_subpixels).astype(int) % x_subpixels
        shift_y = int(shifts[i, 0])
        shift_x = int(shifts[i, 1])
        ref = np.roll(
            obj, (-shift_y, -shift_x), axis=(0, 1)
        )[:probe_shape[-2], :probe_shape[-1]]*probes[subpixel_y, subpixel_x]
        if ifftshift:
            ref = np.fft.ifftshift(ref)
        assert np.allclose(result[i], ref)
        assert np.all(subpixel_indices[i] == (subpixel_y, subpixel_x))


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'fftshift', (False, True)
)
def test_rolled_object_aggregation_cpu(fftshift):
    obj_shape = np.random.randint(1, 23, 2)
    probe_shape = np.array([np.random.randint(1, obj_axis+1, 1)[0] for obj_axis in obj_shape])
    obj = np.zeros(obj_shape, dtype=np.complex64)
    count = 23
    updates = np.random.random((count, ) + tuple(probe_shape)) + 1j
    shifts = np.random.randint(-np.max(obj_shape), np.max(obj_shape), (count, 2))
    rolled_object_aggregation_cpu(obj, updates, shifts, fftshift=fftshift)
    obj_ref = np.zeros_like(obj)
    if fftshift:
        updates_ref = np.fft.fftshift(updates, axes=(1, 2))
    else:
        updates_ref = updates
    for i in range(count):
        rolled_obj = np.roll(obj_ref, (-shifts[i, 0], -shifts[i, 1]), axis=(0, 1))
        rolled_obj[:probe_shape[0], :probe_shape[1]] += updates_ref[i]
        obj_ref = np.roll(rolled_obj, (shifts[i, 0], shifts[i, 1]), axis=(0, 1))
    assert np.allclose(obj, obj_ref)


@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.skipif(not has_cupy(), reason="No functional CuPy")
@pytest.mark.parametrize(
    'ifftshift', (False, True)
)
def test_rolled_object_probe_product_cuda(ifftshift):
    import cupy
    obj_shape = np.random.randint(1, 500, 2)
    probe_shape = np.array([np.random.randint(1, obj_axis+1, 1)[0] for obj_axis in obj_shape])
    print(probe_shape)

    obj = np.linspace(0, 1, np.prod(obj_shape)).reshape(obj_shape)
    y_subpixels, x_subpixels = np.random.randint(1, 11, 2)
    probe = np.random.random((y_subpixels, x_subpixels) + tuple(probe_shape))
    count = np.random.randint(1, 1000)
    result_ref = np.zeros((count, ) + probe.shape[2:], dtype=np.result_type(obj, probe))
    result = cupy.array(result_ref)
    shifts = np.random.randint(
        -np.max(obj_shape)*y_subpixels, np.max(obj_shape)*x_subpixels, (count, 2)
    ) / (y_subpixels, x_subpixels)
    subpixel_indices_cpu = rolled_object_probe_product_cpu(
        obj, probe, shifts, result_ref, ifftshift
    )
    subpixel_indices_cuda = rolled_object_probe_product_cuda(
        cupy.array(obj),
        cupy.array(probe),
        cupy.array(shifts),
        result,
        ifftshift
    )
    assert np.allclose(result.get(), result_ref)
    assert np.all(subpixel_indices_cpu == subpixel_indices_cuda.get())


@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.skipif(not has_cupy(), reason="No functional CuPy")
@pytest.mark.parametrize(
    'fftshift', (False, True)
)
@pytest.mark.parametrize(
    'obj_dtype, update_dtype', [
        (np.complex64, np.complex64), (np.complex64, np.float32), (np.float32, np.float32)
    ]
)
def test_rolled_object_aggregation_cuda(fftshift, obj_dtype, update_dtype):
    import cupy
    obj_shape = np.random.randint(1, 500, 2)
    probe_shape = np.array([np.random.randint(1, obj_axis+1, 1)[0] for obj_axis in obj_shape])

    obj_ref = np.zeros(obj_shape, dtype=obj_dtype)
    obj = cupy.zeros(obj_shape, dtype=obj_dtype)
    count = np.random.randint(1, 1000)
    updates = np.random.random((count, ) + tuple(probe_shape)).astype(update_dtype)
    shifts = np.random.randint(0, np.max(obj_shape), (count, 2))
    rolled_object_aggregation_cpu(obj_ref, updates, shifts, fftshift)
    rolled_object_aggregation_cuda(
        obj,
        cupy.array(updates),
        cupy.array(shifts),
        fftshift
    )
    print(np.max(np.abs(obj.get() - obj_ref)))
    assert np.allclose(obj.get(), obj_ref)
