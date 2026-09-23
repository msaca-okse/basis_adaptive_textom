import matplotlib.pyplot as plt
import numpy as np


def dark(fontsize=28):
    plt.style.use("dark_background")
    ticksize = fontsize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["font.family"] = "Nimbus Roman"


def publication(fontsize=28):
    plt.style.use("default")
    ticksize = fontsize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["font.family"] = "Nimbus Roman"


def crop_to_mask(
    variable,
    mask,
    padding=1,
):
    """Crop a variable to the mask, with a padding of the specified number of pixels.

    This crops the input array to the bounding box defined by the mask.

    Args:
        variable (:obj:`np.ndarray`): The ndarray to crop, typically an image.
        mask (:obj:`np.ndarray`): The boolean mask to crop to (i.e crop to content of the mask).
        padding (int, optional): The number of pixels to pad the crop with. Defaults to 1.

    Returns:
        :obj:`np.ndarray`: The cropped variable.
    """
    ii, jj = np.nonzero(mask)
    i_min, i_max = np.min(ii), np.max(ii)
    j_min, j_max = np.min(jj), np.max(jj)
    return variable[
        i_min - padding : i_max + padding + 1, j_min - padding : j_max + padding + 1
    ]
