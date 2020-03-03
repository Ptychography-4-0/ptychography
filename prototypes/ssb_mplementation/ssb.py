import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np  # NOQA: E402
import matplotlib.pyplot as plt  # NOQA: E402
from matplotlib.colors import LogNorm  # NOQA: E402

from libertem.udf import UDF  # NOQA: E402
from libertem import api as lt  # NOQA: E402
from libertem.executor.inline import InlineJobExecutor  # NOQA: E402

from scipy.sparse import csr_matrix  # NOQA: E402


def main():
    path = r'C:\Users\lesnic\Nextcloud\Dieter\cGaN_sim_300kV\DPs\CBED_MSAP.raw'
    shape = (50, 50, 189, 189)
    ctx = lt.Context(executor=InlineJobExecutor())
    data_s = ctx.load(
        "raw", path=path, dtype="float32",
        scan_size=shape[:2], detector_size=shape[-2:]
    )
    result = ctx.run_udf(udf=SSB_UDF(), dataset=data_s)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(np.abs(result['pixelsum']), norm=LogNorm())
    axes[1].imshow(np.angle(np.fft.ifft2(result['pixelsum'])))

    input("press return to continue")


class SSB_UDF(UDF):

    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="single", dtype="complex128", extra_shape=(50, 50,)
             )
        }

    def merge(self, dest, src):
        dest['pixelsum'][:] = dest['pixelsum'] + src['pixelsum']

    def process_tile(self, tile):

        tile_slice_i = self.meta.slice.origin[0]

        Nblock = np.array(self.meta.dataset_shape[0:2])

        filter_center = self.task_data['filter_center']
        non0_filt_c = np.count_nonzero(filter_center)
        point_0_0 = 0
        point_0_0 = (
            np.sum(tile.astype(np.float32)[..., filter_center])/non0_filt_c
        )

        masks = self.task_data['masks']
        tile_flat = tile.reshape(tile.shape[0], -1)
        intermediate_result = masks.dot(tile_flat.T)

        result = np.zeros(intermediate_result.shape[0], dtype=np.complex128)
        k = range(tile_slice_i, tile_slice_i + tile_flat.shape[0])
        j = np.ogrid[0:Nblock[0]*Nblock[1]]

        for i in k:

            m_n = divmod(i, Nblock[0])
            k_l = divmod(j, Nblock[1])
            M = 1/Nblock[0]
            N = 1/Nblock[1]
            Fourier_exponent = np.exp(-2j*np.pi*(m_n[0]*k_l[0]*M + m_n[1]*k_l[1]*N))
            result += intermediate_result[:, i-tile_slice_i]*Fourier_exponent

        result = result.reshape(Nblock[0], Nblock[1])
        self.results.pixelsum[:] = self.results.pixelsum + result
        self.results.pixelsum[0, 0] = self.results.pixelsum[0, 0] + point_0_0

    def get_task_data(self):

        Nblock = np.array(self.meta.dataset_shape[0:2])
        Nscatter = np.array(self.meta.dataset_shape[2:4])

        dpix = 0.5654/50*1e-9
        semiconv = 25e-3
        semiconv_pix = 78.6649
        lamb = 1.96e-12
        d_Kf = np.sin(semiconv)/lamb/semiconv_pix
        d_Qp = 1/dpix/Nblock
        cy, cx = np.array(self.meta.dataset_shape)[-2:]//2

        y, x = np.ogrid[0:Nscatter[0], 0:Nscatter[1]]
        filter_center = (y - cy)**2 + (x - cx)**2 < semiconv_pix**2

        size = self.meta.dataset_shape
        masks = np.zeros(size, dtype=np.float32)

        m = 0
        for n in range(Nblock[1]):

            qp = np.array((m, n))
            flip = qp > Nblock / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - Nblock[flip]
            sx, sy = real_qp * d_Qp / d_Kf

            filter_positive = (
                (y - cy - sy)**2 + (x - cx - sx)**2 < semiconv_pix**2
            )
            filter_negative = (
                (y - cy + sy)**2 + (x - cx + sx)**2 < semiconv_pix**2
            )
            mask_positive = np.all(
                (filter_center, filter_positive, np.invert(filter_negative)),
                axis=0
            )
            mask_negative = np.all(
                (filter_center, filter_negative, np.invert(filter_positive)),
                axis=0
            )

            n1 = 2*np.count_nonzero(mask_positive)
            if n1 > 0 and n > 0:
                masks[0, n] = np.subtract(mask_positive.astype(np.float32),
                                          mask_negative.astype(np.float32))/n1
            else:
                boundary_2 = n + 1
                boundary_1 = Nblock[0] - boundary_2 + 1

        cm = list(range(1, boundary_1))
        cn = list(range(boundary_1))+list(range(boundary_2, Nblock[0]))

        for m in cm:
            for n in cn:
                qp = np.array((m, n))
                flip = qp > Nblock / 2
                real_qp = qp.copy()
                real_qp[flip] = qp[flip] - Nblock[flip]
                sx, sy = real_qp * d_Qp / d_Kf

                filter_positive = (
                    (y - cy - sy)**2 + (x - cx - sx)**2 < semiconv_pix**2
                )
                filter_negative = (
                    (y - cy + sy)**2 + (x - cx + sx)**2 < semiconv_pix**2
                )

                mask_positive = np.all(
                    (filter_center, filter_positive, np.invert(filter_negative)),
                    axis=0
                )
                mask_negative = np.all(
                    (filter_center, filter_negative, np.invert(filter_positive)),
                    axis=0
                )

                nonzero_masks = 2*np.count_nonzero(mask_positive)
                if nonzero_masks > 0:
                    masks[m, n] = (
                        np.subtract(mask_positive.astype(np.float32),
                                    mask_negative.astype(np.float32))/nonzero_masks
                    )

        inversed_masks = masks[1:boundary_1, :, :, :]*(-1)
        masks[boundary_2:Nblock[0], 0, :, :] = inversed_masks[:: -1, 0, :, :]
        inversed_rows = inversed_masks[:: -1, 1:Nblock[1], :, :]
        masks[boundary_2:Nblock[0], 1:Nblock[1], :, :] = inversed_rows[:, :: -1, :, :]

        masks = csr_matrix(
            masks.reshape(Nblock[0]*Nblock[1], Nscatter[0]*Nscatter[1])
        )

        return {
            "masks": masks,
            "filter_center": filter_center
        }


if __name__ == '__main__':
    main()
