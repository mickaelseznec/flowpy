""" flowpy

    flowpy
    ======

    Contains several utilities for working with optical flows:
        1. flow_read and flow_write let you manipulate flows in .flo and .png format
        2. flow_to_rgb generates a RGB representation of a flow
        3. attach_arrows, attach_coord, attach_calibration_pattern provide helper functions to generate beautiful graphs with matplotlib.
        4. Warp an image according to a flow, in the direct and reverse order.

    The library handles flow in the HWF format, a numpy.ndarray with 3 dimensions of size [H, W, 2] that hold respectively the height, width and 2d displacement in the (x, y) order.

    Undefined flow is attributed a NaN value.
"""

from .flowpy import (flow_to_rgb, make_colorwheel, calibration_pattern,
                     attach_arrows, attach_coord, attach_calibration_pattern,
                     get_flow_max_radius, backward_warp, forward_warp)
from .flow_io import flow_read, flow_write
