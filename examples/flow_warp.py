import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import flowpy

flow = flowpy.flow_read("static/kitti_occ_000010_10.png")
first_image = np.asarray(Image.open("static/kitti_000010_10.png"))
second_image = np.asarray(Image.open("static/kitti_000010_11.png"))

flow[np.isnan(flow)] = 0
warped_first_image = flowpy.backward_warp(second_image, flow)

fig, axes = plt.subplots(3, 1)
for ax, image, title in zip(axes, (first_image, second_image, warped_first_image),
                            ("First Image", "Second Image", "Second image warped to first image")):
    ax.imshow(image)
    ax.set_title(title)
    ax.set_axis_off()

plt.show()
