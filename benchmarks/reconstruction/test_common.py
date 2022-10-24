import pytest
import numpy as np

from libertem.utils.devices import detect, has_cupy

from ptychography40.reconstruction.common import (
    image_transformation_matrix, rolled_object_probe_product_cpu, rolled_object_probe_product_cuda,
    rolled_object_aggregation_cpu, rolled_object_aggregation_cuda, shifted_probes,
    trunc_divide_cpu, trunc_divide_cuda
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
    group='rolled product'
)
@pytest.mark.parametrize(
    'ifftshift', (False, True)
)
def test_rolled_object_probe_product_cpu(benchmark, ifftshift):
    obj_shape = (1024, 1024)
    probe_shape = (64, 64)

    obj = np.ones(obj_shape, dtype=np.complex64)
    probe = shifted_probes(np.ones(probe_shape, dtype=np.complex64), 4)
    count = 128
    result = np.zeros((count, ) + probe_shape, dtype=np.result_type(obj, probe))
    shifts = np.random.randint(-4*1024, 4*1024, (128, 2))/4

    def my_bench():
        rolled_object_probe_product_cpu(obj, probe, shifts, result, ifftshift)
        return result.sum()

    benchmark(my_bench)


@pytest.mark.benchmark(
    group='rolled aggregation'
)
@pytest.mark.parametrize(
    'fftshift', (False, True)
)
@pytest.mark.parametrize(
    'obj_shape', ((128, 128), (1024, 1024), (4096, 4096))
)
def test_rolled_object_aggregation_cpu(benchmark, fftshift, obj_shape):
    probe_shape = (64, 64)

    obj = np.zeros(obj_shape)
    count = 128
    updates = np.ones((count, ) + probe_shape, dtype=obj.dtype)
    shifts = np.full((128, 2), 1000)

    def my_bench():
        rolled_object_aggregation_cpu(obj, updates, shifts, fftshift)
        return obj[0, 0]

    benchmark(my_bench)


@pytest.mark.benchmark(
    group='rolled product'
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

    obj = cupy.ones(obj_shape, dtype=np.complex64)
    probe = cupy.array(
        shifted_probes(
            np.ones(probe_shape, dtype=np.complex64),
            4
        )
    )
    count = 128
    result = cupy.zeros((count, ) + probe_shape, dtype=np.result_type(obj, probe))
    shifts = cupy.random.randint(-4*1024, 4*1024, (128, 2))/4

    def my_bench():
        rolled_object_probe_product_cuda(obj, probe, shifts, result, ifftshift)
        return result.sum()

    benchmark(my_bench)


@pytest.mark.benchmark(
    group='rolled aggregation'
)
@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.skipif(not has_cupy(), reason="No functional CuPy")
@pytest.mark.parametrize(
    'fftshift', (False, True)
)
@pytest.mark.parametrize(
    'obj_shape', ((128, 128), (1024, 1024), (4096, 4096))
)
def test_rolled_object_aggregation_cuda(benchmark, fftshift, obj_shape):
    import cupy
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
        return obj[0, 0]

    benchmark(my_bench)


@pytest.mark.benchmark(
    group='truncated divide'
)
@pytest.mark.parametrize(
    'which', ('baseline', 'test', 'cuda_compatible')
)
def test_divide_cpu(benchmark, which):
    numerator = np.random.random((128, 256, 256)) + 1j*np.random.random((128, 256, 256))
    denominator = np.random.random((128, 256, 256)) + 1j*np.random.random((128, 256, 256))
    denominator[:, :256, :3] = 0
    out = np.zeros_like(numerator)
    if which == 'test':
        benchmark(
            trunc_divide_cpu,
            numerator=numerator,
            denominator=denominator,
            out=out
        )
    elif which == 'baseline':
        def _do_test():
            return np.divide(
                numerator,
                denominator,
                out=out,
                where=np.abs(denominator) > 1e-6
            )
        benchmark(_do_test)
    elif which == 'cuda_compatible':
        def _do_test():
            np.divide(
                numerator,
                denominator,
                out=out,
                where=np.abs(denominator) > 1e-6
            )
            out[np.abs(denominator) <= 1e-6] = 0
        benchmark(_do_test)
    else:
        raise ValueError()


@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.skipif(not has_cupy(), reason="No functional CuPy")
@pytest.mark.benchmark(
    group='truncated divide'
)
@pytest.mark.parametrize(
    'which', ('baseline', 'test')
)
def test_divide_cuda(benchmark, which):
    import cupy
    numerator = cupy.random.random((128, 256, 256)) + 1j*cupy.random.random((128, 256, 256))
    denominator = cupy.random.random((128, 256, 256)) + 1j*cupy.random.random((128, 256, 256))
    denominator[:, :256, :3] = 0
    out = cupy.zeros_like(numerator)
    if which == 'test':
        def do_test():
            trunc_divide_cuda(
                numerator=numerator,
                denominator=denominator,
                out=out
            )
            return out.sum().get()
    elif which == 'baseline':
        def do_test():
            cupy.divide(
                numerator,
                denominator,
                out=out,
            )
            out[cupy.abs(denominator) <= 1e-6] = 0
            return out.sum().get()
    else:
        raise ValueError()
    benchmark(do_test)
