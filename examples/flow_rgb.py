import flowpy
import matplotlib.pyplot as plt

flow = flowpy.flow_read("static/kitti_occ_000010_10.png")

fig, ax = plt.subplots()
ax.imshow(flowpy.flow_to_rgb(flow))
plt.show()
