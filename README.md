# flowpy 💾 - A python package for working with optical flows

Optical flow is the displacement map of pixels between two frames. It is a low-level analysis used in many computer vision programs.

Working with optical flow may be cumbersome:
- It is quite hard to represent it in a comprehensible manner.
- Multiple formats exist for storing it.

Flowpy provides tools to work with optical flow more easily in python.

## Installing

We recommend using pip:
```bash
pip install flowpy
```

## Features

The main features of flowpy are:
- Reading and writing optical flows in two formats:
    - **.flo** (as defined [here](http://vision.middlebury.edu/flow/))
    - **.png** (as defined [here](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow))
- Visualizing optical flows with matplotlib

## Examples

### A simple RGB plot

This is the simplest example of how to use flowpy, it:
- Reads a file using *flowpy.flow_read*.
- Transforms the flow as an rgb image with *flowpy.flow_to_rgb* and shows it with matplotlib

#### Code:
```python
import flowpy
import matplotlib.pyplot as plt

flow = flowpy.flow_read("tests/data/kitti_occ_000010_10.flo")

fig, ax = plt.subplots()
ax.imshow(flowpy.flow_to_rgb(flow))
plt.show()
```

#### Result:
![simple_example]

### Plotting arrows, showing flow values and a calibration pattern

Flowpy comes with more than just RGB plots, the main features here are:
    - Arrows to quickly visualize the flow
    - The flow values below cursor showing in the tooltips
    - A calibration pattern side by side as a legend for your graph

#### Code:
```python
import flowpy
import matplotlib.pyplot as plt

flow = flowpy.flow_read("tests/data/Dimetrodon.flo")
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
```

#### Result:
![complex_example]

### More

You can find more examples in the `tests` folder.
If you encounter a bug or have an idea for a new feature, feel free to open an issue.

Most of the visualization and io handling has been translated from matlab and c code from the [Middlebury flow code](http://vision.middlebury.edu/flow/code/flow-code/).
I would like to thank Simon Baker, Daniel Scharste, J. P. Lewis, Stefan Roth, Michael J. Blackand Richard Szeliski.

[simple_example]: https://raw.githubusercontent.com/mickaelseznec/flowpy/master/static/example_0.png "Displaying an optical flow as an RGB image"
[complex_example]: https://raw.githubusercontent.com/mickaelseznec/flowpy/master/static/example_1.png "Displaying an optical flow as an RGB image with arrows, tooltip and legend"
