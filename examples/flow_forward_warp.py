import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import flowpy

flow = flowpy.flow_read("static/kitti_occ_000010_10.png")
first_image = np.asarray(Image.open("static/kitti_000010_10.png"))
second_image = np.asarray(Image.open("static/kitti_000010_11.png"))

flow[np.isnan(flow)] = 0
warped_second_image = flowpy.forward_warp(first_image, flow)

fig, axes = plt.subplots(2, 1)
for ax, image, title in zip(axes, (first_image, warped_second_image),
                            ("First Image", "First image warped to the second")):
    ax.imshow(image)
    ax.set_title(title)
    ax.set_axis_off()

plt.show()

