import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import flowpy

flow = flowpy.flow_read("static/kitti_occ_000010_10.png")
first_image = np.asarray(Image.open("static/kitti_000010_10.png"))
second_image = np.asarray(Image.open("static/kitti_000010_11.png"))

flow[np.isnan(flow)] = 0
warped_second_image = flowpy.forward_warp(first_image, flow)

fig, ax = plt.subplots()

ax.imshow(warped_second_image)
ax.set_title( "First image warped to the second")
ax.set_axis_off()

plt.show()

