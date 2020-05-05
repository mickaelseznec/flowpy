""" flowpy

    Contains multiple utilities for working with optical flow fields and manipulate .flo files
"""

from .flowpy import (flow_to_rgb, make_colorwheel, calibration_pattern,
                     add_arrows_to_ax, format_coord)
from .flow_io import flow_read, flow_write
