import pytest
import numpy as np

from libertem.utils.devices import detect, has_cupy

from ptychography40.reconstruction.common import (
    image_transformation_matrix, rolled_object_probe_product_cpu, rolled_object_probe_product_cuda,
    rolled_object_aggregation_cpu, rolled_object_aggregation_cuda
)


@pytest.mark.benchmark(
    group='create matrix'
)
def test_transformation_scale(benchmark):
    source_shape = (256, 256)
    target_shape = (128, 128)
    benchmark(
        image_transformation_matrix,
        source_shape=source_shape,
        target_shape=target_shape,
        affine_transformation=lambda x: x * 2
    )


@pytest.mark.benchmark(
    group='create matrix'
)
def test_transformation_identity(benchmark):
    shape = (256, 256)
    benchmark(
        image_transformation_matrix,
        source_shape=shape,
        target_shape=shape,
        affine_transformation=lambda x: x
    )


@pytest.mark.benchmark(
    group='rolled'
)
@pytest.mark.parametrize(
    'ifftshift', (False, True)
)
def test_rolled_object_probe_product_cpu(benchmark, ifftshift):
    obj_shape = (1024, 1024)
    probe_shape = (64, 64)

    obj = np.ones(obj_shape)
    probe = np.ones(probe_shape)
    count = 128
    result = np.zeros((count, ) + probe.shape, dtype=np.result_type(obj, probe))
    shifts = np.full((128, 2), 1000)

    def my_bench():
        rolled_object_probe_product_cpu(obj, probe, shifts, result, ifftshift)
        return result.sum()

    benchmark(my_bench)


@pytest.mark.benchmark(
    group='rolled'
)
@pytest.mark.parametrize(
    'fftshift', (False, True)
)
def test_rolled_object_aggregation_cpu(benchmark, fftshift):
    obj_shape = (1024, 1024)
    probe_shape = (64, 64)

    obj = np.zeros(obj_shape)
    count = 128
    updates = np.ones((count, ) + probe_shape, dtype=obj.dtype)
    shifts = np.full((128, 2), 1000)

    def my_bench():
        rolled_object_aggregation_cpu(obj, updates, shifts, fftshift)
        return obj.sum()

    benchmark(my_bench)


@pytest.mark.benchmark(
    group='rolled'
)
@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.skipif(not has_cupy(), reason="No functional CuPy")
@pytest.mark.parametrize(
    'ifftshift', (False, True)
)
def test_rolled_object_probe_product_cuda(benchmark, ifftshift):
    import cupy
    obj_shape = (1024, 1024)
    probe_shape = (64, 64)

    obj = cupy.ones(obj_shape)
    probe = cupy.ones(probe_shape)
    count = 128
    result = cupy.zeros((count, ) + probe.shape, dtype=np.result_type(obj, probe))
    shifts = cupy.full((128, 2), 1000)

    def my_bench():
        rolled_object_probe_product_cuda(obj, probe, shifts, result, ifftshift)
        return result.sum()

    benchmark(my_bench)


@pytest.mark.benchmark(
    group='rolled'
)
@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.skipif(not has_cupy(), reason="No functional CuPy")
@pytest.mark.parametrize(
    'fftshift', (False, True)
)
def test_rolled_object_aggregation_cuda(benchmark, fftshift):
    import cupy
    obj_shape = (1024, 1024)
    probe_shape = (64, 64)

    obj = cupy.zeros(obj_shape)
    count = 128
    updates = cupy.ones((count, ) + tuple(probe_shape))
    shifts = cupy.full((128, 2), 1000)

    def my_bench():
        obj[:] = 0
        rolled_object_aggregation_cuda(
            obj,
            updates,
            shifts,
            fftshift
        )
        return obj.sum()

    benchmark(my_bench)
