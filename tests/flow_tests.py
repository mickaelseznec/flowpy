#! /usr/bin/env python3
import sys
sys.path.append("flowpy")

import unittest
import flowpy
import matplotlib.pyplot as plt

class CalibrationWheelTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
