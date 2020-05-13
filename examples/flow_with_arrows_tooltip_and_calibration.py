import flowpy
import matplotlib.pyplot as plt

flow = flowpy.flow_read("static/Dimetrodon.flo")
height, width, _ = flow.shape

image_ratio = height / width
max_radius = flowpy.get_flow_max_radius(flow)

fig, (ax_1, ax_2) = plt.subplots(
    1, 2, gridspec_kw={"width_ratios": [1, image_ratio]}
)

ax_1.imshow(flowpy.flow_to_rgb(flow))
flowpy.attach_arrows(ax_1, flow)
flowpy.attach_coord(ax_1, flow)

flowpy.attach_calibration_pattern(ax_2, flow_max_radius=max_radius)

plt.show()
