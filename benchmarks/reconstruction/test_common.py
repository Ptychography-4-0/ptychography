import pytest

from ptychography40.reconstruction.common import image_transformation_matrix


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
