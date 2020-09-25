from os.path import isfile

import pytest
import numpy as np
from scipy.io import loadmat
import json

from libertem import api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.memory import MemoryDataSet
from libertem.masks import circular
from libertem.corrections.coordinates import identity, rotate_deg
from libertem.common.container import MaskContainer

from ptychography.reconstruction.ssb import SSB_UDF, generate_masks, mask_tile_pair
from ptychography.reconstruction.common import wavelength


DATA_PATH = "/storage/holo/clausen/testdata/ER-C-1/projects/ptycho-4.0/data/RefData/slice_00002_thick_0.312293_nm_blocksz.mat"
PARAM_PATH = "/storage/holo/clausen/testdata/ER-C-1/projects/ptycho-4.0/data/slice_00002_thick_0.312293_nm_blocksz.params.json"

has_real = isfile(DATA_PATH) and isfile(PARAM_PATH)

if has_real:
    @pytest.fixture(scope="session")
    def real_params():
        with open(PARAM_PATH) as f:
            params = json.load(f)
        params['transformation'] = np.array(params['transformation'])
        return params

    @pytest.fixture(scope="session")
    def real_data():
        return loadmat(DATA_PATH, squeeze_me=True)

    @pytest.fixture(scope="session")
    def real_complex_diffract(real_data, real_params):
        return np.moveaxis(real_data['imgstack_cdp'], (2,), (0,)).reshape(real_params["shape"])

    @pytest.fixture(scope="session")
    def real_intensity(real_data, real_params):
        return np.moveaxis(real_data['imgstack_idp'], (2,), (0,)).reshape(real_params["shape"])

    @pytest.fixture(scope="session")
    def real_plane_wave(real_complex_diffract, real_params):
        cy, cx = real_params["cy"], real_params["cx"]
        return real_complex_diffract[:, :, cy, cx]

    @pytest.fixture(scope="session")
    def lt_ctx():
        return lt.Context(executor=InlineJobExecutor())

    @pytest.fixture(scope="session")
    def real_intensity_ds(real_intensity, lt_ctx):
        return lt_ctx.load("memory", data=real_intensity, sig_dims=2)

    @pytest.fixture(scope="session")
    def real_reference_ssb(real_params, real_intensity):
        return reference_ssb(
            real_intensity,
            U=real_params["U"],
            dpix=real_params["dpix"],
            semiconv=real_params["semiconv"],
            semiconv_pix=real_params["semiconv_pix"],
            cy=real_params["cy"],
            cx=real_params["cx"]
        )


def test_mask_tile_pair_within():
    shape = (13, 17)
    filter_center = circular(
        centerX=6,
        centerY=7,
        imageSizeX=shape[1],
        imageSizeY=shape[0],
        radius=4
    )

    reference_pair = mask_tile_pair(
        center_tile=filter_center,
        tile_origin=np.array((0, 0)),
        tile_shape=np.array(shape),
        filter_center=filter_center,
        sy=3,
        sx=3,
    )
    trimmed_pair = mask_tile_pair(
        center_tile=filter_center[1:11, 2:15],
        tile_origin=np.array((1, 2)),
        tile_shape=np.array((10, 13)),
        filter_center=filter_center,
        sy=3,
        sx=3,
    )
    assert trimmed_pair[0].shape == (10, 13)
    assert np.allclose(trimmed_pair[0], reference_pair[0][1:11, 2:15])
    assert np.allclose(trimmed_pair[3], reference_pair[3][1:11, 2:15])


@pytest.mark.parametrize(
    # dpix: STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    # Second one tests for different y and x dpix
    'dpix', (0.5654/50*1e-9, (0.5654/50*1e-9, 0.5654/49*1e-9))
)
def test_ssb(dpix, lt_ctx):
    dtype = np.float64

    scaling = 4
    shape = (29, 30, 189 // scaling, 197 // scaling)

    # The acceleration voltage U in keV
    U = 300

    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling

    cy = 93 // scaling
    cx = 97 // scaling

    input_data = (
        np.random.uniform(0, 1, np.prod(shape))
        * np.linspace(1.0, 1000.0, num=np.prod(shape))
    )
    input_data = input_data.astype(np.float64).reshape(shape)

    udf = SSB_UDF(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                  dtype=dtype, center=(cy, cx), method='subpix')

    dataset = MemoryDataSet(
        data=input_data, tileshape=(20, shape[2], shape[3]), num_partitions=2, sig_dims=2,
    )

    result = lt_ctx.run_udf(udf=udf, dataset=dataset)

    result_f, reference_masks = reference_ssb(input_data, U=U, dpix=dpix, semiconv=semiconv,
                             semiconv_pix=semiconv_pix, cy=cy, cx=cx)

    task_data = udf.get_task_data()

    udf_masks = task_data['masks'].computed_masks

    half_y = shape[0] // 2 + 1
    # Use symmetry and reshape like generate_masks()
    reference_masks = reference_masks[:half_y].reshape((half_y*shape[1], shape[2], shape[3]))

    print(np.max(np.abs(udf_masks.todense() - reference_masks)))

    print(np.max(np.abs(result['pixels'].data - result_f)))

    assert np.allclose(result['pixels'].data, result_f)


@pytest.mark.parametrize(
    # dpix: STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    # Second one tests for different y and x dpix
    'dpix', (0.5654/50*1e-9, (0.5654/50*1e-9, 0.5654/49*1e-9))
)
def test_ssb_container(dpix, lt_ctx):
    dtype = np.float64

    scaling = 4
    shape = (29, 30, 189 // scaling, 197 // scaling)

    # The acceleration voltage U in keV
    U = 300

    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling

    cy = 93 // scaling
    cx = 97 // scaling

    input_data = (
        np.random.uniform(0, 1, np.prod(shape))
        * np.linspace(1.0, 1000.0, num=np.prod(shape))
    )
    input_data = input_data.astype(np.float64).reshape(shape)

    masks = generate_masks(
        reconstruct_shape=shape[:2],
        mask_shape=shape[2:],
        dtype=dtype,
        lamb=wavelength(U),
        dpix=dpix,
        semiconv=semiconv,
        semiconv_pix=semiconv_pix,
        center=(cy, cx),
        method='subpix'
    )

    mask_container = MaskContainer(
        mask_factories=lambda: masks, dtype=masks.dtype,
        use_sparse='scipy.sparse.csc', count=masks.shape[0],
    )

    udf = SSB_UDF(
        U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
        dtype=dtype, center=(cy, cx), mask_container=mask_container
    )

    dataset = MemoryDataSet(
        data=input_data, tileshape=(20, shape[2], shape[3]), num_partitions=2, sig_dims=2,
    )

    result = lt_ctx.run_udf(udf=udf, dataset=dataset)

    result_f, reference_masks = reference_ssb(input_data, U=U, dpix=dpix, semiconv=semiconv,
                             semiconv_pix=semiconv_pix, cy=cy, cx=cx)

    task_data = udf.get_task_data()

    udf_masks = task_data['masks'].computed_masks

    half_y = shape[0] // 2 + 1
    # Use symmetry and reshape like generate_masks()
    reference_masks = reference_masks[:half_y].reshape((half_y*shape[1], shape[2], shape[3]))

    print(np.max(np.abs(udf_masks.todense() - reference_masks)))

    print(np.max(np.abs(result['pixels'].data - result_f)))

    assert np.allclose(result['pixels'].data, result_f)


def test_ssb_rotate():
    ctx = lt.Context(executor=InlineJobExecutor())
    dtype = np.float64

    scaling = 4
    det = 45
    shape = (29, 30, det, det)
    #  ? shape = np.random.uniform(1, 300, (4,1,))

    # The acceleration voltage U in keV
    U = 300
    # STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    dpix = 0.5654/50*1e-9
    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling

    cy = det // 2
    cx = det // 2

    input_data = (
        np.random.uniform(0, 1, np.prod(shape))
        * np.linspace(1.0, 1000.0, num=np.prod(shape))
    )
    input_data = input_data.astype(np.float64).reshape(shape)

    data_90deg = np.zeros_like(input_data)

    # Rotate 90 degrees clockwise
    for y in range(det):
        for x in range(det):
            data_90deg[:, :, x, det-1-y] = input_data[:, :, y, x]

    udf = SSB_UDF(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                  dtype=dtype, center=(cy, cx), transformation=rotate_deg(-90.))

    dataset = MemoryDataSet(
        data=data_90deg, tileshape=(20, shape[2], shape[3]), num_partitions=2, sig_dims=2,
    )

    result = ctx.run_udf(udf=udf, dataset=dataset)

    result_f, _ = reference_ssb(input_data, U=U, dpix=dpix, semiconv=semiconv,
                             semiconv_pix=semiconv_pix, cy=cy, cx=cx)

    assert np.allclose(result['pixels'].data, result_f)


def test_ssb_roi():
    ctx = lt.Context(executor=InlineJobExecutor())
    dtype = np.float64

    scaling = 4
    shape = (29, 30, 189 // scaling, 197 // scaling)
    #  ? shape = np.random.uniform(1, 300, (4,1,))

    # The acceleration voltage U in keV
    U = 300
    # STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    dpix = 0.5654/50*1e-9
    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling

    cy = 93 // scaling
    cx = 97 // scaling

    input_data = (
        np.random.uniform(0, 1, np.prod(shape))
        * np.linspace(1.0, 1000.0, num=np.prod(shape))
    )
    input_data = input_data.astype(np.float64).reshape(shape)

    udf = SSB_UDF(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                  dtype=dtype, center=(cy, cx))

    dataset = MemoryDataSet(
        data=input_data, tileshape=(20, shape[2], shape[3]), num_partitions=2, sig_dims=2,
    )

    roi_1 = np.random.choice([True, False], shape[:2])
    roi_2 = np.invert(roi_1)

    result_1 = ctx.run_udf(udf=udf, dataset=dataset, roi=roi_1)
    result_2 = ctx.run_udf(udf=udf, dataset=dataset, roi=roi_2)

    result_f, _ = reference_ssb(input_data, U=U, dpix=dpix, semiconv=semiconv,
                             semiconv_pix=semiconv_pix, cy=cy, cx=cx)

    assert np.allclose(result_1['pixels'].data + result_2['pixels'].data, result_f)


def test_masks():
    scaling = 4
    dtype = np.float64
    shape = (29, 30, 189 // scaling, 197 // scaling)
    #  ? shape = np.random.uniform(1, 300, (4,1,))

    # The acceleration voltage U in keV
    U = 300
    lambda_e = wavelength(U)
    # STEM pixel size in m, here 50 STEM pixels on 0.5654 nm
    dpix = 0.5654/50*1e-9
    # STEM semiconvergence angle in radians
    semiconv = 25e-3
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = 78.6649 / scaling

    cy = 93 // scaling
    cx = 97 // scaling

    input_data = np.random.uniform(0, 1, shape)

    _, reference_masks = reference_ssb(
        input_data, U=U, dpix=dpix, semiconv=semiconv,
        semiconv_pix=semiconv_pix,
        cy=cy, cx=cx
    )

    # print(np.max(np.abs(np.abs(result['pixels']) - np.abs(result_f))))
    half_y = shape[0] // 2 + 1
    # Use symmetry and reshape like generate_masks()
    reference_masks = reference_masks[:half_y].reshape((half_y*shape[1], shape[2], shape[3]))

    masks = generate_masks(
        reconstruct_shape=shape[:2],
        mask_shape=shape[2:],
        dtype=dtype,
        lamb=lambda_e,
        dpix=dpix,
        semiconv=semiconv,
        semiconv_pix=semiconv_pix,
        center=(cy, cx),
        transformation=identity(),
        method="subpix"
    ).todense()

    assert reference_masks.shape == masks.shape
    print(reference_masks)
    print(masks)
    print(reference_masks - masks)
    print("maximum difference: ", np.max(np.abs(reference_masks - masks)))
    print(np.where(reference_masks != masks))
    assert np.any(reference_masks != 0)
    assert np.allclose(reference_masks, masks)


@pytest.mark.skipif(not has_real, reason="Real data not found at expected path")
@pytest.mark.slow
def test_validate_com(real_params, real_intensity_ds, lt_ctx):
    '''
    Make sure that the input data is sensible
    by testing COM analysis on it
    '''
    a = lt_ctx.create_com_analysis(
        dataset=real_intensity_ds,
        cy=real_params['cy'],
        cx=real_params['cx'],
        mask_radius=real_params['semiconv_pix'],
    )
    res = lt_ctx.run(a)

    # Make sure that there is no constant offset, which would indicate that
    # the center is off
    assert res.x.raw_data.sum() < 0.01*np.abs(res.x.raw_data).sum()
    assert res.y.raw_data.sum() < 0.01*np.abs(res.y.raw_data).sum()

    # Make sure the field from center of mass analysis is dominated by divergence
    # and has very little curl -- purely electrostatic
    assert (
        np.abs(res.curl.raw_data[1:-1, 1:-1]).sum()
        < 0.05*np.abs(res.divergence.raw_data[1:-1, 1:-1]).sum()
    )


@pytest.mark.skipif(not has_real, reason="Real data not found at expected path")
@pytest.mark.slow
@pytest.mark.parametrize(
    "method,external_container", (
        ('subpix', True),
        ('subpix', False),
        ('shift', True),
        ('shift', False),
    )
)
def test_validate_ssb(real_params, real_intensity_ds, real_plane_wave,
                      real_reference_ssb, lt_ctx, method, external_container):
    '''
    The mask generation methods can produce slightly different masks.

    Since SSB strongly suppresses noise, including any features
    where real space and diffraction space don't properly align,
    slight differences in the mask stack can lead to amplifying errors
    if the input data contains no actual features and the signal sums up to nearly zero.

    For that reason the correctness of mask generation functions shoud be tested on
    simulated data that contains a pronounced signal.

    Furthermore, this allows to compare the reconstruction with a "ground truth" phase.
    '''
    dtype = np.float64

    shape = real_intensity_ds.shape

    # The acceleration voltage U in keV
    U = real_params["U"]

    # STEM semiconvergence angle in radians
    semiconv = real_params["semiconv"]
    # Diameter of the primary beam in the diffraction pattern in pixels
    semiconv_pix = real_params["semiconv_pix"]

    cy = real_params["cy"]
    cx = real_params["cx"]

    dpix = real_params["dpix"]

    transformation = real_params["transformation"]

    if external_container:
        masks = generate_masks(
            reconstruct_shape=shape[:2],
            mask_shape=shape[2:],
            dtype=dtype,
            lamb=wavelength(U),
            dpix=dpix,
            semiconv=semiconv,
            semiconv_pix=semiconv_pix,
            center=(cy, cx),
            transformation=transformation,
            method=method,
            cutoff=1,
        )

        mask_container = MaskContainer(
            mask_factories=lambda: masks, dtype=masks.dtype,
            use_sparse='scipy.sparse.csc', count=masks.shape[0],
        )
    else:
        mask_container = None

    udf = SSB_UDF(
        U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
        dtype=dtype, center=(cy, cx), mask_container=mask_container, method=method,
        cutoff=1,
    )

    result = lt_ctx.run_udf(udf=udf, dataset=real_intensity_ds)

    result_f, reference_masks = real_reference_ssb

    ssb_res = np.fft.ifft2(result["pixels"].data)
    reference_ssb_res = np.fft.ifft2(result_f)

    ssb_phase = np.angle(ssb_res)
    ref_phase = np.angle(real_plane_wave)

    # The phases are usually shifted by a constant offset
    # Looking at Std removes the offset
    # TODO the current data is at the limit of SSB reconstruction. Better data should be simulated.
    # TODO work towards 100 % correspondence with suitable test dataset
    assert np.std(ssb_phase - ref_phase) < 0.1 * np.std(ssb_phase)

    # Make sure the methods are at least reasonably comparable
    # TODO work towards 100 % correspondence with suitable test dataset
    # TODO make the amplitude of the reconstruction match
    assert np.max(np.abs(ssb_res - reference_ssb_res)) < 0.01*np.max(np.abs(ssb_res))
    assert np.std(ssb_res - reference_ssb_res) < 0.01*np.std(ssb_res)


def reference_ssb(data, U, dpix, semiconv, semiconv_pix, cy=None, cx=None):

    # 'U' - The acceleration voltage U in keV
    # 'dpix' - STEM pixel size in m
    # 'semiconv' -  STEM semiconvergence angle in radians
    # 'semiconv_pix' - Diameter of the primary beam in the diffraction pattern in pixels
    dpix = np.array(dpix)

    reordered = np.moveaxis(data, (0, 1), (2, 3))
    ffts = np.fft.fft2(reordered)
    rearranged_ffts = np.moveaxis(ffts, (2, 3), (0, 1))

    Nblock = np.array(data.shape[0:2])
    Nscatter = np.array(data.shape[2:4])

    # electron wavelength in m
    lamb = wavelength(U)
    # spatial freq. step size in scattering space
    d_Kf = np.sin(semiconv)/lamb/semiconv_pix
    # spatial freq. step size according to probe raster
    d_Qp = 1/dpix/Nblock

    result_f = np.zeros(data.shape[:2], dtype=rearranged_ffts.dtype)

    masks = np.zeros_like(data)

    if cx is None:
        cx = data.shape[-1] / 2
    if cy is None:
        cy = data.shape[-2] / 2

    y, x = np.ogrid[0:Nscatter[0], 0:Nscatter[1]]
    filter_center = circular(
        centerX=cx, centerY=cy,
        imageSizeX=Nscatter[1], imageSizeY=Nscatter[0],
        radius=semiconv_pix,
        antialiased=True
    ).astype(np.float64)

    for row in range(Nblock[0]):
        for column in range(Nblock[1]):
            qp = np.array((row, column))
            flip = qp > Nblock / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - Nblock[flip]

            sy, sx = real_qp * d_Qp / d_Kf

            filter_positive = circular(
                centerX=cx+sx, centerY=cy+sy,
                imageSizeX=Nscatter[1], imageSizeY=Nscatter[0],
                radius=semiconv_pix,
                antialiased=True
            ).astype(np.float64)

            filter_negative = circular(
                centerX=cx-sx, centerY=cy-sy,
                imageSizeX=Nscatter[1], imageSizeY=Nscatter[0],
                radius=semiconv_pix,
                antialiased=True
            ).astype(np.float64)
            mask_positive = filter_center * filter_positive * (filter_negative == 0)
            mask_negative = filter_center * filter_negative * (filter_positive == 0)

            non_zero_positive = mask_positive.sum()
            non_zero_negative = mask_negative.sum()

            f = rearranged_ffts[row, column]

            if non_zero_positive >= 1 and non_zero_negative >= 1:
                tmp = (
                    (f * mask_positive).sum() / non_zero_positive
                    - (f * mask_negative).sum() / non_zero_negative
                ) / 2
                result_f[row, column] = tmp
                masks[row, column] = ((mask_positive / non_zero_positive) - (
                               mask_negative / non_zero_negative)) / 2
                assert np.allclose(result_f[row, column], (f*masks[row, column]).sum())
            else:
                assert non_zero_positive < 1
                assert non_zero_negative < 1

    result_f[0, 0] = (rearranged_ffts[0, 0] * filter_center).sum() / filter_center.sum()
    masks[0, 0] = filter_center / filter_center.sum()

    return result_f, masks
