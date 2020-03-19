import torch


def torch_fftshift(x, axes=None):
    """
    Adapted from Numpy.
    :param x: Input Tensor
    :param axes: Shifting axes
    :return: Shifted Tensor.
    """

    if axes is None:
        axes = tuple(range(len(x.shape)))
        shift = [dim // 2 for dim in x.shape]
    else:
        shift = [x.shape[ax] // 2 for ax in axes]
    return torch.roll(x, shift, axes)


def torch_ifftshift(x, axes=None):
    """
    Adapted from Numpy.
    :param x: Input Tensor
    :param axes: Shifting
    :return: Shifted Tensor

    """

    if axes is None:
        axes = tuple(range(len(x.shape)))
        shift = [-(dim // 2) for dim in x.shape]
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]
    return torch.roll(x, shift, axes)


def torch_angle(z, deg=False):
    zimag = z[:, :, 1]
    zreal = z[:, :, 0]
    a = torch.atan2(zimag, zreal)
    if deg:
        a *= 180 / pi
    return a
