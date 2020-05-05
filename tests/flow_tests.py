#! /usr/bin/env python3
import sys
sys.path.append("flowpy")

import unittest
import flowpy
import matplotlib.pyplot as plt
import tempfile
import filecmp
import numpy as np
import os

class CalibrationPatternTestCase(unittest.TestCase):
    def test_calibration_bright(self):
        width = 255
        mid_width = width // 2
        pattern = flowpy.calibration_pattern(width, background="bright")

        fig, ax = plt.subplots()
        ax.imshow(pattern)
        ax.hlines(mid_width, -.5, width-.5)
        ax.vlines(mid_width, -.5, width-.5)
        circle = plt.Circle((mid_width, mid_width), mid_width, fill=False)
        ax.add_artist(circle)
        ax.set_title("Calibration Pattern (Bright)")
        plt.show()

    def test_calibration_dark(self):
        width = 255
        mid_width = width // 2
        pattern = flowpy.calibration_pattern(width, background="dark")

        fig, ax = plt.subplots()
        ax.imshow(pattern)
        ax.hlines(mid_width, -.5, width-.5)
        ax.vlines(mid_width, -.5, width-.5)
        circle = plt.Circle((mid_width, mid_width), mid_width, fill=False)
        ax.add_artist(circle)
        ax.set_title("Calibration Pattern (Dark)")
        plt.show()


class FlowInputOutput(unittest.TestCase):
    def test_read_write_flo(self):
        input_filepath = "tests/data/Dimetrodon.flo"
        flow = flowpy.flow_read(input_filepath)

        with tempfile.NamedTemporaryFile("wb", suffix=".flo") as f:
            flowpy.flow_write(f, flow)
            self.assertTrue(filecmp.cmp(f.name, input_filepath))

    def test_read_write_png_occ(self):
        input_filepath = "tests/data/kitti_occ_000010_10.png"

        flow = flowpy.flow_read(input_filepath)

        _, output_filename = tempfile.mkstemp(suffix=".png")

        try:
            flowpy.flow_write(output_filename, flow)
            new_flow = flowpy.flow_read(output_filename)
            np.testing.assert_equal(new_flow, flow)
        finally:
            os.remove(output_filename)

    def test_read_write_png_noc(self):
        input_filepath = "tests/data/kitti_noc_000010_10.png"

        flow = flowpy.flow_read(input_filepath)

        _, output_filename = tempfile.mkstemp(suffix=".png")

        try:
            flowpy.flow_write(output_filename, flow)
            new_flow = flowpy.flow_read(output_filename)
            np.testing.assert_equal(new_flow, flow)
        finally:
            os.remove(output_filename)


if __name__ == "__main__":
    unittest.main()
