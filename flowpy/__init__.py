""" flowpy

    Contains some utilities for working with optical flows and manipulate associated reference file formats.
"""

from .flowpy import (flow_to_rgb, make_colorwheel, calibration_pattern,
                     attach_arrows, attach_coord, attach_calibration_pattern,
                     get_flow_max_radius)
from .flow_io import flow_read, flow_write
