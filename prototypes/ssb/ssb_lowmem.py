import numpy as np

from libertem.udf import UDF
from libertem.common.container import MaskContainer
from libertem.masks import circular
from libertem.corrections.coordinates import identity

from .common import wavelength, bounding_box
from .ssb import mask_tile_pair


def generate_skyline(reconstruct_shape, mask_shape, dtype, wavelength, dpix, semiconv,
        semiconv_pix, tiling_scheme, filter_center, debug_masks,
        transformation=None, center=None, cutoff=1):
    reconstruct_shape = np.array(reconstruct_shape)
    # print("mask shape", filter_center.shape)
    d_Kf = np.sin(semiconv)/wavelength/semiconv_pix
    d_Qp = 1/dpix/reconstruct_shape

    if center is None:
        center = np.array(mask_shape) / 2

    if transformation is None:
        transformation = identity()

    cy, cx = center

    half_reconstruct = (reconstruct_shape[0]//2 + 1, reconstruct_shape[1])

    target_ranges_p = np.zeros(
        (
            len(tiling_scheme),
            half_reconstruct[0],
            half_reconstruct[1],
            4  # y_start, y_stop, x_start, x_stop
        ),
        dtype=int
    )
    bbox_p = np.zeros_like(target_ranges_p)
    offsets_p = np.zeros(
        (
            len(tiling_scheme),
            half_reconstruct[0],
            half_reconstruct[1],
            2  # y, x
        ),
        dtype=int
    )
    nnz_p = np.zeros(
        (
            half_reconstruct[0],
            half_reconstruct[1],
        ),
        dtype=float
    )

    shift_map = np.zeros(half_reconstruct + (2, ))

    target_ranges_n = np.zeros_like(target_ranges_p)
    bbox_n = np.zeros_like(bbox_p)
    nnz_n = np.zeros_like(nnz_p)
    offsets_n = np.zeros_like(offsets_p)

    for tile_index, tile_slice in enumerate(tiling_scheme):
        tile_slice = tile_slice.discard_nav()
        center_tile = tile_slice.get(filter_center, sig_only=True)
        # print(tile_slice.origin)
        # print(tile_slice.shape)
        # print("cutoff generate skyline", cutoff)
        for row in range(half_reconstruct[0]):
            for column in range(half_reconstruct[1]):
                # Do an fftshift of q and p
                qp = np.array((row, column))
                flip = qp > (reconstruct_shape / 2)
                real_qp = qp.copy()
                real_qp[flip] = qp[flip] - reconstruct_shape[flip]

                # Shift of diffraction order relative to zero order
                # without rotation
                real_sy, real_sx = real_qp * d_Qp / d_Kf

                # We apply the transformation backwards to go
                # from physical coordinates to detector coordinates,
                # while the forward direction in center of mass analysis
                # goes from detector coordinates to physical coordinates
                sy, sx = (real_sy, real_sx) @ transformation

                shift_map[row, column] = (sy, sx)

                (mask_positive, target_tup_p, o_p,
                mask_negative, target_tup_n, o_n) = mask_tile_pair(
                    center_tile=center_tile,
                    tile_origin=np.array(tile_slice.origin),
                    tile_shape=np.array(tile_slice.shape),
                    filter_center=filter_center,
                    sy=sy,
                    sx=sx
                )
                if row != 0 or column != 0:
                    # m = (
                    #     mask_positive# / skyline["nnz_p"][row, column]
                    #     # - mask_negative# / skyline["nnz_n"][row, column]
                    # ) / 2
                    # print(row, column)
                    # print(np.where(np.abs(m - dbm) > 1e-7))
                    # print(np.max(np.abs(m - dbm)))
                    # print(dbm.shape)
                    # assert np.allclose(m, dbm)
                    nnz_p[row, column] += mask_positive.sum()
                    nnz_n[row, column] += mask_negative.sum()
                    target_ranges_p[tile_index, row, column] = target_tup_p.flatten()
                    target_ranges_n[tile_index, row, column] = target_tup_n.flatten()
                    offsets_p[tile_index, row, column] = o_p.flatten()
                    offsets_n[tile_index, row, column] = o_n.flatten()
                    bbox_p[tile_index, row, column] = bounding_box(mask_positive).flatten()
                    bbox_n[tile_index, row, column] = bounding_box(mask_negative).flatten()
                else:
                    c = center_tile*center_tile
                    # We will divide by 2 later in the skyline dot, have to 
                    # compensate for that
                    nnz_p[0, 0] += c.sum() / 2
                    nnz_n[0, 0] += c.sum() / 2  # nnz_n remains to avoid culling
                    target_ranges_p[tile_index, 0, 0] = target_tup_p.flatten()
                    # target_ranges_n remains empty (0)
                    # offsets remain zero
                    bbox_p[tile_index, 0, 0] = bounding_box(c).flatten()
                    # Bounding box negative remains empty (0)
                    # m = center_tile
                # dbm = tile_slice.get(debug_masks[row, column], sig_only=True)
                # print(row, column)
                # print(np.where(np.abs(m - dbm) > 1e-7))
                # print(np.max(np.abs(m - dbm)))
                # print(dbm.shape)
                # assert np.allclose(m, dbm)
    for row in range(half_reconstruct[0]):
        for column in range(half_reconstruct[1]):
            if nnz_p[row, column] <= cutoff or nnz_n[row, column] <= cutoff:
                nnz_p[row, column] = 0
                nnz_n[row, column] = 0
                target_ranges_p[:, row, column] = 0
                target_ranges_n[:, row, column] = 0
                offsets_p[:, row, column] = 0
                offsets_n[:, row, column] = 0
                bbox_p[:, row, column] = 0
                bbox_n[:, row, column] = 0
    return {
        "target_ranges_p": target_ranges_p,
        "target_ranges_n": target_ranges_n,
        "nnz_p": nnz_p,
        "nnz_n": nnz_n,
        "offsets_p": offsets_p,
        "offsets_n": offsets_n,
        "bbox_p": bbox_p,
        "bbox_n": bbox_n,
        "shift_map": shift_map
    }


def skyline_dot(tile, filter_center, skyline, debug_masks):
    half_reconstruct = skyline["nnz_p"].shape
    result = np.zeros(
        (tile.shape[0], half_reconstruct[0], half_reconstruct[1]),
        dtype=np.result_type(tile.dtype, filter_center.dtype)
    )
    tile_slice = tile.tile_slice.discard_nav()
    tile_index = tile.scheme_idx
    center_tile = tile_slice.get(arr=filter_center, sig_only=True)
    positive_tile = np.zeros_like(center_tile)
    negative_tile = np.zeros_like(center_tile)
    for row in range(half_reconstruct[0]):
        for column in range(half_reconstruct[1]):
            # That means nnz_n is zero, too
            if skyline["nnz_p"][row, column] == 0:
                # print("skipping", row, column)
                continue
            positive_tile[:] = 0
            negative_tile[:] = 0
            sta_y, sto_y, sta_x, sto_x = skyline["target_ranges_p"][tile_index, row, column]
            off_y, off_x = skyline["offsets_p"][tile_index, row, column]
            positive_tile[sta_y:sto_y, sta_x:sto_x] = filter_center[
                sta_y+off_y:sto_y+off_y,
                sta_x+off_x:sto_x+off_x
            ]
            sta_y, sto_y, sta_x, sto_x = skyline["target_ranges_n"][tile_index, row, column]
            off_y, off_x = skyline["offsets_n"][tile_index, row, column]
            negative_tile[sta_y:sto_y, sta_x:sto_x] = filter_center[
                sta_y+off_y:sto_y+off_y,
                sta_x+off_x:sto_x+off_x
            ]
            mask_positive = center_tile * positive_tile * (negative_tile == 0)
            mask_negative = center_tile * negative_tile * (positive_tile == 0)

            # 
            sta_y, sto_y, sta_x, sto_x = skyline["bbox_p"][tile_index, row, column]
            result[:, row, column] = (
                tile[:, sta_y:sto_y, sta_x:sto_x]
                * mask_positive[np.newaxis, sta_y:sto_y, sta_x:sto_x]
            ).sum(axis=(1, 2)) / skyline["nnz_p"][row, column]
            sta_y, sto_y, sta_x, sto_x = skyline["bbox_n"][tile_index, row, column]
            # ...and here we subtract
            result[:, row, column] -= (
                tile[:, sta_y:sto_y, sta_x:sto_x]
                * mask_negative[sta_y:sto_y, sta_x:sto_x]
            ).sum(axis=(1, 2)) / skyline["nnz_n"][row, column]

            if row == 0 and column == 0:
                neg = (
                    tile[:, sta_y:sto_y, sta_x:sto_x]
                    * mask_negative[sta_y:sto_y, sta_x:sto_x]
                ).sum(axis=(1, 2)) / skyline["nnz_n"][row, column]
                # print(neg)
                assert np.allclose(neg, 0)
                # fig, axes = plt.subplots(2)
                # axes[0].imshow(mask_positive)
                # axes[1].imshow(mask_positive - center_tile**2)
                assert np.allclose(mask_positive, center_tile**2)
                
            result[:, row, column] /= 2

            # assert np.allclose(mask_positive[sta_y:sto_y, sta_x:sto_x].sum(), mask_positive.sum())
            
            # result[:, row, column] = (
            #     tile * debug_masks[row, column]
            # ).sum(axis=(1, 2))
            if row != 0 or column != 0:
                m = (
                    mask_positive / skyline["nnz_p"][row, column]
                    - mask_negative / skyline["nnz_n"][row, column]
                ) / 2
                assert np.allclose(result[:, row, column], (tile * m).sum(axis=(1, 2)))
            else:
                m = center_tile**2 / skyline["nnz_p"][row, column] / 2
                reference = (
                    tile * m
                ).sum(axis=(1, 2)) 
                assert np.allclose(result[:, row, column], reference)
                # result[:, row, column] = reference

            # assert np.allclose(mask_negative[sta_y:sto_y, sta_x:sto_x].sum(), mask_negative.sum())
            # print(row, column)
            # print(np.where(np.abs(m - debug_masks[row, column]) > 1e-7))
            # print(np.max(np.abs(m - debug_masks[row, column])))
            # print("skyline target p", skyline["target_ranges_p"][tile_index, row, column])
            # print("skyline offset p", skyline["offsets_p"][tile_index, row, column])
            # print("skyline shift map", skyline["shift_map"][row, column])
            assert np.allclose(m, debug_masks[row, column])

            # Treatment of (0, 0): The skyline is doctored to
            # give the correct result as well.
    return result


class SSB_UDF_Lowmem(UDF):
    def __init__(self, U, dpix, semiconv, semiconv_pix,
                 dtype=np.float32, center=None, filter_center=None,
                 transformation=None, cutoff=1):
        '''
        Parameters

        U: float
            The acceleration voltage U in kV

        center: (float, float)

        dpix: float
            STEM pixel size in m

        semiconv: float
            STEM semiconvergence angle in radians

        dtype: np.dtype
            dtype to perform the calculation in

        semiconv_pix: float
            Diameter of the primary beam in the diffraction pattern in pixels

        TODO copy & paste from normal ssb

        mask_container: MaskContainer
            Hack to pass in a precomputed mask stack when using with single thread live data
            or with an inline executor.
            The proper fix is https://github.com/LiberTEM/LiberTEM/issues/335
        '''
        super().__init__(U=U, dpix=dpix, semiconv=semiconv, semiconv_pix=semiconv_pix,
                         dtype=dtype, center=center, transformation=transformation,
                         filter_center=filter_center, cutoff=cutoff)

    def get_result_buffers(self):
        dtype = np.result_type(np.complex64, self.params.dtype)
        return {
            'pixels': self.buffer(
                kind="single", dtype=dtype, extra_shape=self.reconstruct_shape,
                where='device'
             )
        }

    @property
    def reconstruct_shape(self):
        return tuple(self.meta.dataset_shape.nav)

    def get_task_data(self):
        # shorthand, cupy or numpy
        xp = self.xp
        # Hack to pass a fixed external container
        # In particular useful for single-process live processing
        # or inline executor
        ds_nav = tuple(self.meta.dataset_shape.nav)
        y_positions, x_positions = np.mgrid[0:ds_nav[0], 0:ds_nav[1]]

        # Precalculated values for Fourier transform
        row_steps = -2j*np.pi*np.linspace(0, 1, self.reconstruct_shape[0], endpoint=False)
        col_steps = -2j*np.pi*np.linspace(0, 1, self.reconstruct_shape[1], endpoint=False)

        if self.meta.roi is None:
            y_map = y_positions.flatten()
            x_map = x_positions.flatten()
        else:
            y_map = y_positions[self.meta.roi]
            x_map = x_positions[self.meta.roi]
        if self.params.filter_center is None:
            cy, cx = self.params.center
            mask_shape = tuple(self.meta.dataset_shape.sig)
            filter_center = circular(
                centerX=cx, centerY=cy,
                imageSizeX=mask_shape[1], imageSizeY=mask_shape[0],
                radius=self.params.semiconv_pix,
                antialiased=True
            ).astype(self.params.dtype)
        else:
            filter_center = self.params.filter_center.astype(self.params.dtype)

        steps_dtype = np.result_type(np.complex64, self.params.dtype)

        masks = generate_masks(
            reconstruct_shape=self.reconstruct_shape,
            mask_shape=tuple(self.meta.dataset_shape.sig),
            dtype=self.params.dtype,
            wavelength=wavelength(self.params.U),
            dpix=self.params.dpix,
            semiconv=self.params.semiconv,
            semiconv_pix=self.params.semiconv_pix,
            center=self.params.center,
            transformation=self.params.transformation,
            cutoff=self.params.cutoff,
            filter_center=filter_center
        )

        skyline = generate_skyline(
            reconstruct_shape=self.reconstruct_shape,
            mask_shape=tuple(self.meta.dataset_shape.sig),
            dtype=self.params.dtype,
            wavelength=wavelength(self.params.U),
            dpix=self.params.dpix,
            semiconv=self.params.semiconv,
            semiconv_pix=self.params.semiconv_pix,
            tiling_scheme=self.meta.tiling_scheme,
            filter_center=filter_center,
            center=self.params.center,
            transformation=self.params.transformation,
            cutoff=self.params.cutoff,
            debug_masks=masks.reshape((
                self.reconstruct_shape[0]//2 + 1,
                self.reconstruct_shape[1],
                *tuple(self.meta.dataset_shape.sig)
            )).todense()
        )
        container = MaskContainer(
            mask_factories=lambda: masks, dtype=masks.dtype,
            use_sparse='scipy.sparse.csc', count=masks.shape[0], backend=self.meta.backend
        )
        return {
            # Frame positions in the dataset masked by ROI
            # to easily access position in dataset when
            # processing with ROI applied
            "skyline": skyline,
            "masks": container,
            "filter_center": xp.array(filter_center),
            "y_map": xp.array(y_map),
            "x_map": xp.array(x_map),
            "row_steps": xp.array(row_steps.astype(steps_dtype)),
            "col_steps": xp.array(col_steps.astype(steps_dtype)),
        }

    def merge(self, dest, src):
        dest['pixels'][:] = dest['pixels'] + src['pixels']

    def process_tile(self, tile):
        # shorthand, cupy or numpy
        xp = self.xp

        tile_start = self.meta.slice.origin[0]
        tile_depth = tile.shape[0]

        buffer_frame = xp.zeros_like(self.results.pixels)
        half_y = buffer_frame.shape[0] // 2 + 1

        y_indices = self.task_data.y_map[tile_start:tile_start+tile_depth]
        x_indices = self.task_data.x_map[tile_start:tile_start+tile_depth]

        factors_dtype = np.result_type(np.complex64, self.params.dtype)

        fourier_factors_row = xp.exp(
            y_indices[:, np.newaxis, np.newaxis]
            * self.task_data.row_steps[np.newaxis, :half_y, np.newaxis]
        ).astype(factors_dtype)
        fourier_factors_col = xp.exp(
            x_indices[:, np.newaxis, np.newaxis]
            * self.task_data.col_steps[np.newaxis, np.newaxis, :]
        ).astype(factors_dtype)
        
        masks = self.task_data.masks.get(self.meta.slice, transpose=True, backend=self.meta.backend)
        
        dot_result = skyline_dot(
            tile=tile,
            filter_center=self.task_data.filter_center,
            skyline=self.task_data.skyline,
            debug_masks=masks.T.toarray().reshape((half_y, buffer_frame.shape[1], *tile.tile_slice.shape[1:]))
        )
        
        tile_flat = tile.reshape((tile.shape[0], -1))
        mask_dot_result = tile_flat * masks
        mask_dot_result = mask_dot_result.reshape((tile_depth, half_y, buffer_frame.shape[1]))

        for row in range(half_y):
            for column in range(buffer_frame.shape[1]):
                if row == 0 and column == 0:
                    continue
                m = dot_result[:, row, column]
                rm = mask_dot_result[:, row, column]
                # print("checking dot result")
                # print(row, column)
                # print(np.where(np.abs(m - rm) > 1e-7)
                # print(np.max(np.abs(m - rm)))
                assert np.allclose(m, rm)

        assert np.allclose(dot_result, mask_dot_result)

        buffer_frame[:half_y] = (dot_result*fourier_factors_row*fourier_factors_col).sum(axis=0)
        # patch accounts for even and odd sizes
        # FIXME make sure this is correct using an example that transmits also
        # the high spatial frequencies
        patch = buffer_frame.shape[0] % 2
        # We skip the first row since it would be outside the FOV
        extracted = buffer_frame[1:buffer_frame.shape[0] // 2 + patch]
        # The coordinates of the bottom half are inverted and
        # the zero column is rolled around to the front
        # The real part is inverted
        buffer_frame[half_y:] = -xp.conj(
            xp.roll(xp.flip(xp.flip(extracted, axis=0), axis=1), shift=1, axis=1)
        )

        self.results.pixels[:] += buffer_frame

    def get_backends(self):
        return ('numpy', 'cupy')

    def get_tiling_preferences(self):
        dtype = np.result_type(np.complex64, self.params.dtype)
        result_size = np.prod(self.reconstruct_shape) * dtype.itemsize
        if self.meta.backend == 'cupy':
            total_size = 10e6
            good_depth = max(1, total_size / result_size)
            return {
                "depth": good_depth,
                "total_size": total_size,
            }
        else:
            good_depth = max(1, 1e6 / result_size)
            return {
                "depth": int(good_depth),
                "total_size": 1e6,
            }
