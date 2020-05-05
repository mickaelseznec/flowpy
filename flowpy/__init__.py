""" flowpy

    Contains multiple utilities for working with optical flow fields and manipulate .flo files
"""


import matplotlib.pyplot as plt
import numpy as np

from .flowpy import *
from .flow_io import flow_read, flow_write


def _get_polar(u, v):
    """ Transforms a cartesian representation of the flow to a polar representation."""

    radius = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, u)

    return radius, angle


def show_flow_color(u, v, min_is_black=True, max_norm=None):
    """ Displays flow as a RGB image."""

    flow_rgb = flow_to_color(u, v, min_is_black=min_is_black, max_norm=max_norm)

    plt.figure()
    plt.imshow(flow_rgb)
    plt.title("Color representation of the flow")
    plt.tight_layout()
    plt.show()


def show_flow_polar(u, v):
    """ Displays flow in its polar coordinates."""

    u, v = _nan_to_zero(u, v)
    radius, angle = _get_polar(u, v)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(radius)
    plt.title("Flow radius in pixels")

    plt.subplot(1, 2, 2)
    plt.imshow(angle * (180 / np.pi))
    plt.title("Flow angle in degrees")

    plt.tight_layout()
    plt.show()


def show_flow_arrows(u, v, true_scale=False, min_is_black=True, max_norm=None, sub_factors=None):
    """ Displays flow with arrows over its RGB representation.
    Args:
        u: numpy.ndarray
            2D image of the displacement field along the x axis
        v: numpy.ndarray
            2D image of the displacement field along the y axis
        true_scale: bool
            Should the arrows correspond to the real distances?
        min_is_black: bool
            Is a null vector represented as black or white?
        max_norm: float
            Used for normalizing vectors' norm
        sub_factors: int or tuple of ints
            The subsampling factor in the arrows representation. A subsampling factor of 10
            means an arrows every 10 pixels. If this is a tuple, a different subsampling
            factor for every axes.
    """

    height, width = u.shape
    y_grid, x_grid = np.mgrid[:height, :width]

    if sub_factors is None:
        sub_factor_x = int(height/25.5)
        sub_factor_y = int(width/25.5)
    elif isinstance(sub_factors, (tuple, list, np.ndarray)):
        sub_factor_x, sub_factor_y = sub_factors
    else:
        sub_factor_x = sub_factor_y = sub_factors

    plt.figure()

    flow_rgb = flow_to_color(u, v, min_is_black=min_is_black, max_norm=max_norm)
    plt.imshow(flow_rgb)

    #y-axis is reversed to match image convention
    plt.quiver(x_grid[::sub_factor_x, ::sub_factor_y],
               y_grid[::sub_factor_x, ::sub_factor_y],
               u[::sub_factor_x, ::sub_factor_y],
               -v[::sub_factor_x, ::sub_factor_y],
               units="xy",
               scale=1 if true_scale else None,
               color="w" if min_is_black else "k")
    plt.ylim(height, 0)

    plt.title("Arrows (" + ("" if true_scale else "not ") + "true scale) over rgb representation")
    plt.tight_layout()
    plt.show()


def export_flow_arrows(filename, u, v, true_scale=False, min_is_black=True, max_norm=None, sub_factors=None):
    """ Displays flow with arrows over its RGB representation.
    Args:
        filename: str
            Name of the output file
        u: numpy.ndarray
            2D image of the displacement field along the x axis
        v: numpy.ndarray
            2D image of the displacement field along the y axis
        true_scale: bool
            Should the arrows correspond to the real distances?
        min_is_black: bool
            Is a null vector represented as black or white?
        max_norm: float
            Used for normalizing vectors' norm
        sub_factors: int or tuple of ints
            The subsampling factor in the arrows representation. A subsampling factor of 10
            means an arrows every 10 pixels. If this is a tuple, a different subsampling
            factor for every axes.
    """

    height, width = u.shape
    y_grid, x_grid = np.mgrid[:height, :width]

    if sub_factors is None:
        sub_factor_x = int(height/25.5)
        sub_factor_y = int(width/25.5)
    elif isinstance(sub_factors, (tuple, list, np.ndarray)):
        sub_factor_x, sub_factor_y = sub_factors
    else:
        sub_factor_x = sub_factor_y = sub_factors

    flow_rgb = flow_to_color(u, v, min_is_black=min_is_black, max_norm=max_norm)

    fig = plt.figure(figsize=(width, height), dpi=1)
    ax = fig.add_subplot(1, 1, 1)

    ax.axis("off")
    ax.imshow(flow_rgb)

    #y-axis is reversed to match image convention
    ax.quiver(x_grid[::sub_factor_x, ::sub_factor_y],
               y_grid[::sub_factor_x, ::sub_factor_y],
               u[::sub_factor_x, ::sub_factor_y],
               -v[::sub_factor_x, ::sub_factor_y],
               units="xy",
               scale=1 if true_scale else None,
               color="w" if min_is_black else "k")

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(filename, bbox_inches=extent, pad_inches=0)
    plt.close()


def show_flow(u, v, true_scale=False, max_norm=None, min_is_black=True, sub_factors=None):
    """ Displays both arrows over RGB representation and polar representation."""
    show_flow_arrows(u, v, true_scale=true_scale, max_norm=max_norm,
                     min_is_black=min_is_black, sub_factors=sub_factors)
    show_flow_polar(u, v)
