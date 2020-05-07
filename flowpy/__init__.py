""" flowpy

    Contains multiple utilities for working with optical flow fields and manipulate .flo files
"""

from .flowpy import (flow_to_rgb, make_colorwheel, calibration_pattern,
                     attach_arrows, attach_coord, attach_calibration_pattern,
                     get_flow_max_radius)
from .flow_io import flow_read, flow_write
