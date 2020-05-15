#! /usr/bin/env python3
import filecmp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tempfile
import unittest

from PIL import Image

import flowpy


class CalibrationPatternTestCase(unittest.TestCase):
    def test_calibration_bright(self):
        fig, ax = plt.subplots()
        _, flow = flowpy.calibration_pattern()

        flowpy.attach_calibration_pattern(
            ax, pixel_size=255, flow_max_radius=5, background="bright"
        )

        ax.set_title("Calibration Pattern (Bright)")

        plt.show()

    def test_calibration_with_arrows(self):
        pattern, flow = flowpy.calibration_pattern()

        fig, ax = plt.subplots()
        ax.imshow(pattern)
        flowpy.attach_arrows(ax, flow)
        plt.show()

    def test_calibration_dark(self):
        width = 255
        mid_width = width // 2
        pattern, _ = flowpy.calibration_pattern(width, background="dark")

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
        input_filepath = "static/Dimetrodon.flo"
        flow = flowpy.flow_read(input_filepath)

        with tempfile.NamedTemporaryFile("wb", suffix=".flo") as f:
            flowpy.flow_write(f, flow)
            self.assertTrue(filecmp.cmp(f.name, input_filepath))

    def test_read_write_png_occ(self):
        input_filepath = "static/kitti_occ_000010_10.png"

        flow = flowpy.flow_read(input_filepath)

        _, output_filename = tempfile.mkstemp(suffix=".png")

        try:
            flowpy.flow_write(output_filename, flow)
            new_flow = flowpy.flow_read(output_filename)
            np.testing.assert_equal(new_flow, flow)
        finally:
            os.remove(output_filename)

    def test_read_write_png_noc(self):
        input_filepath = "static/kitti_noc_000010_10.png"

        flow = flowpy.flow_read(input_filepath)

        _, output_filename = tempfile.mkstemp(suffix=".png")

        try:
            flowpy.flow_write(output_filename, flow)
            new_flow = flowpy.flow_read(output_filename)
            np.testing.assert_equal(new_flow, flow)
        finally:
            os.remove(output_filename)


class FlowDisplay(unittest.TestCase):
    def test_flow_to_rgb(self):
        flow = flowpy.flow_read("static/Dimetrodon.flo")
        plt.imshow(flowpy.flow_to_rgb(flow))
        plt.show()

    def test_flow_with_arrows(self):
        flow = flowpy.flow_read("static/kitti_occ_000010_10.png")

        fig, ax = plt.subplots()
        ax.imshow(flowpy.flow_to_rgb(flow))
        flowpy.attach_arrows(ax, flow, xy_steps=(20, 20), scale=1)

        plt.show()

    def test_flow_arrows_and_coord(self):
        flow = flowpy.flow_read("static/Dimetrodon.flo")

        fig, ax = plt.subplots()
        ax.imshow(flowpy.flow_to_rgb(flow))
        flowpy.attach_arrows(ax, flow)
        flowpy.attach_coord(ax, flow)

        plt.show()

    def test_flow_arrows_coord_and_calibration_pattern(self):
        flow = flowpy.flow_read("static/Dimetrodon.flo")
        height, width, _ = flow.shape
        image_ratio = height / width

        max_radius = flowpy.get_flow_max_radius(flow)

        fig, (ax_1, ax_2) = plt.subplots(1, 2,
                                         gridspec_kw={"width_ratios": [1, image_ratio]})

        ax_1.imshow(flowpy.flow_to_rgb(flow))
        flowpy.attach_arrows(ax_1, flow)
        flowpy.attach_coord(ax_1, flow)

        flowpy.attach_calibration_pattern(ax_2, flow_max_radius=max_radius)

        plt.show()


class FlowWarp(unittest.TestCase):
    def test_backward_warp_greyscale(self):
        flow = flowpy.flow_read("static/kitti_occ_000010_10.png")
        first_image = np.asarray(Image.open("static/kitti_000010_10.png").convert("L"))
        second_image = np.asarray(Image.open("static/kitti_000010_11.png").convert("L"))

        flow[np.isnan(flow)] = 0
        warped_first_image = flowpy.backward_warp(second_image, flow)

        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 1)
        ax_1.imshow(first_image)
        ax_2.imshow(second_image)
        ax_3.imshow(warped_first_image)

        plt.show()

    def test_backward_warp_rgb(self):
        flow = flowpy.flow_read("static/kitti_occ_000010_10.png")
        first_image = np.asarray(Image.open("static/kitti_000010_10.png"))
        second_image = np.asarray(Image.open("static/kitti_000010_11.png"))

        flow[np.isnan(flow)] = 0
        warped_first_image = flowpy.backward_warp(second_image, flow)

        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 1)
        ax_1.imshow(first_image)
        ax_2.imshow(second_image)
        ax_3.imshow(warped_first_image)

        plt.show()

    def test_forward_warp_rgb(self):
        flow = flowpy.flow_read("static/kitti_occ_000010_10.png")
        first_image = np.asarray(Image.open("static/kitti_000010_10.png"))
        second_image = np.asarray(Image.open("static/kitti_000010_11.png"))

        flow[np.isnan(flow)] = 0
        warped_second_image = flowpy.forward_warp(first_image, flow, k=1)

        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 1)
        ax_1.imshow(first_image)
        ax_2.imshow(flowpy.flow_to_rgb(flow))
        ax_3.imshow(warped_second_image)

        plt.show()

    def test_forward_warp_greyscale(self):
        flow = flowpy.flow_read("static/kitti_occ_000010_10.png")
        first_image = np.asarray(Image.open("static/kitti_000010_10.png").convert("L"))
        second_image = np.asarray(Image.open("static/kitti_000010_11.png").convert("L"))

        flow[np.isnan(flow)] = 0
        warped_second_image = flowpy.forward_warp(first_image, flow, k=4)

        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 1)
        ax_1.imshow(first_image)
        ax_2.imshow(flowpy.flow_to_rgb(flow))
        ax_3.imshow(warped_second_image)

        plt.show()

if __name__ == "__main__":
    unittest.main()
