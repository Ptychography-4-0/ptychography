import numpy as np
import numba
import sparse

from libertem.udf import UDF
from libertem.common.container import MaskContainer
from libertem.masks import circular
from libertem.corrections.coordinates import identity

from .common import wavelength, get_shifted, to_slices, bounding_box


@numba.njit
def mask_tile_pair(center_tile, tile_origin, tile_shape, filter_center, sy, sx):

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


def generate_skyline(reconstruct_shape, mask_shape, dtype, wavelength, dpix, semiconv,
        semiconv_pix, tiling_scheme, filter_center,
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
                    nnz_p[row, column] += mask_positive.sum()
                    nnz_n[row, column] += mask_negative.sum()
                    target_ranges_p[tile_index, row, column] = target_tup_p.flatten()
                    target_ranges_n[tile_index, row, column] = target_tup_n.flatten()
                    offsets_p[tile_index, row, column] = o_p.flatten()
                    offsets_n[tile_index, row, column] = o_n.flatten()
                    bbox_p[tile_index, row, column] = bounding_box(mask_positive).flatten()
                    bbox_n[tile_index, row, column] = bounding_box(mask_negative).flatten()
                else:
                    nnz_p[0, 0] += center_tile.sum()
                    nnz_p[0, 0] = 1  # nnz_n remains 1 to avoid div0
                    target_ranges_p[tile_index, 0, 0] = target_tup_p.flatten()
                    # target_ranges_n remains empty (0)
                    # offsets remain zero
                    bbox_p[tile_index, 0, 0] = bounding_box(center_tile).flatten()
                    # Bounding box negative remains empty (0)
    for row in range(half_reconstruct[0]):
        for column in range(half_reconstruct[1]):
            if nnz_p[row, column] < cutoff or nnz_n[row, column] < cutoff:
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


def skyline_dot(tile, filter_center, skyline):
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
            # That means nnz_p is zero, too
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
            sta_y, sto_y, sta_x, sto_x = skyline["bbox_p"][tile_index, row, column]

            result[:, row, column] = (
                tile[:, sta_y:sto_y, sta_x:sto_x]
                * mask_positive[sta_y:sto_y, sta_x:sto_x]
            ).sum(axis=(1, 2)) / skyline["nnz_p"][row, column]

            mask_negative = center_tile * negative_tile * (positive_tile == 0)
            sta_y, sto_y, sta_x, sto_x = skyline["bbox_n"][tile_index, row, column]
            # ...and here we subtract
            result[:, row, column] -= (
                tile[:, sta_y:sto_y, sta_x:sto_x]
                * mask_negative[sta_y:sto_y, sta_x:sto_x]
            ).sum(axis=(1, 2)) / skyline["nnz_n"][row, column]

            result[:, row, column] /= 2
            sy, sx = skyline["shift_map"][row, column]
            (m_p, target_tup_p, o_p, m_n, target_tup_n, o_n) = mask_tile_pair(
                center_tile=center_tile,
                tile_origin=np.array(tile_slice.origin[1:]),
                tile_shape=np.array(tile_slice.shape[1:]),
                filter_center=filter_center,
                sy=sy,
                sx=sx,
            )

            if row != 0 or column != 0:
                assert np.allclose(m_n, mask_negative)
                assert np.allclose(m_p, mask_positive)

            # Treatment of (0, 0): The skyline is doctored to
            # give the correct result as well.
    return result


def generate_masks(reconstruct_shape, mask_shape, dtype, wavelength, dpix, semiconv,
        semiconv_pix, transformation=None, center=None, cutoff=1):
    reconstruct_shape = np.array(reconstruct_shape)

    d_Kf = np.sin(semiconv)/wavelength/semiconv_pix
    d_Qp = 1/dpix/reconstruct_shape

    if center is None:
        center = np.array(mask_shape) / 2

    if transformation is None:
        transformation = identity()

    cy, cx = center

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

            # Shift of diffraction order relative to zero order
            # without rotation
            real_sy, real_sx = real_qp * d_Qp / d_Kf

            # We apply the transformation backwards to go
            # from physical coordinates to detector coordinates,
            # while the forward direction in center of mass analysis
            # goes from detector coordinates to physical coordinates
            sy, sx = (real_sy, real_sx) @ transformation

            (mask_positive, target_tup_p, offsets_p,
            mask_negative, target_tup_n, offsets_n) = mask_tile_pair(
                center_tile=np.array(filter_center),
                tile_origin=np.array((0, 0)),
                tile_shape=np.array(mask_shape),

                filter_center=np.array(filter_center),
                sy=sy,
                sx=sx
            )

            non_zero_positive = mask_positive.sum()
            non_zero_negative = mask_negative.sum()

            if non_zero_positive >= cutoff and non_zero_negative >= cutoff:
                m = (
                    mask_positive / non_zero_positive
                    - mask_negative / non_zero_negative
                ) / 2
                masks.append(sparse.COO(m.astype(dtype)))
            else:
                # Exclude small, missing or unbalanced trotters
                masks.append(sparse.zeros(mask_shape, dtype=dtype))

    # The zero order component (0, 0) is special, comes out zero with above code
    m_0 = filter_center / filter_center.sum()
    masks[0] = sparse.COO(m_0.astype(dtype))

    # Since we go through masks in order, this gives a mask stack with
    # flattened (q, p) dimension to work with dot product and mask container
    masks = sparse.stack(masks)
    return masks


class SSB_UDF(UDF):

    def __init__(self, U, dpix, semiconv, semiconv_pix,
                 dtype=np.float32, center=None, filter_center=None,
                 transformation=None, cutoff=0):
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
            cutoff=self.params.cutoff
        )

        return {
            # Frame positions in the dataset masked by ROI
            # to easily access position in dataset when
            # processing with ROI applied
            "skyline": skyline,
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

        dot_result = skyline_dot(
            tile=tile,
            filter_center=self.task_data.filter_center,
            skyline=self.task_data.skyline
        )

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
