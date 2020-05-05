from itertools import accumulate
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

def replace_nans(array, value=0):
    nan_mask = np.isnan(array)
    array[nan_mask] = value

    return array, nan_mask


def flow_to_rgb(flow, flow_max_radius=None, background="bright", custom_colorwheel=None):
    """ Returns a RGB image that represents the flow field.

    Args:
        flow: numpy.ndarray
            Displacement array in the HWD format, where D stands for the direction of the flow.
            flow[..., 0] must contain the x-displacement
            flow[..., 1] must contain the y-displacement

        background: str
            States if zero-valued flow should look 'bright' or 'dark'

        flow_max_radius: float
            Set the radius that gives the maximum color intensity.
            Useful for comparing different flows.
            Clip the input flow whose radius is bigger than the provided value.

            By default, no clipping is performed and the normalization is based
            on the flow maximum radius.

        custom_colorwheel: numpy.ndarray
            Use a custom colorwheel to change the hue transitions.
            By default, the default transitions are used.
            See: make_colorwheel

    Returns:
        img: numpy.ndarray
            A 2D RGB image that represents the flow
    """

    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError("background should be one the following: {}, not {}".format(
            valid_backgrounds, background))

    wheel = make_colorwheel() if custom_colorwheel is None else custom_colorwheel

    flow_height, flow_width, _ = flow.shape

    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    complex_flow, nan_mask = replace_nans(complex_flow)

    radius, angle = np.abs(complex_flow), np.angle(complex_flow)

    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    else:
        radius = np.minimum(flow_max_radius, radius)

    if flow_max_radius > 0:
        radius /= flow_max_radius

    ncols = len(wheel)

    # Map the angles from (-pi, pi] to [0, ncols - 1)
    angle = (-angle + np.pi) * ((ncols - 1) / (2 * np.pi))

    color_interpoler = interp1d(np.arange(ncols), wheel, axis=0)

    float_hue = color_interpoler(angle.flatten())

    if background == "dark":
        # Mutiplying by a factor in [0, 1] plays on the Value (in HSV)
        colors = float_hue * radius.reshape((-1, 1))
        colors[nan_mask.flatten()] = np.array([255., 255., 255.])
    else:
        # Taking the complement of the complement multiplied by a factor in [0, 1]
        # plays on the Saturation (in HSV)
        colors = 255. - radius.reshape((-1, 1)) * (255. - float_hue)
        colors[nan_mask.flatten()] = np.array([0., 0., 0.])

    return colors.astype(np.uint8).reshape((flow_height, flow_width, 3))


def make_colorwheel(transitions=DEFAULT_TRANSITIONS):
    """ Creates the color wheel.

    Think of it as a circular buffer. On each index of the circular buffer lies a RGB value giving the hue.
    It is generated as linear interpolation between 6 primitives hues (here stated with their RGB values): Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).

    Args:
        transitions: sequence_like
            Contains the length of the six transitions.
            Defaults to (15, 6, 4, 11, 13, 6), based on humain perception.

    Returns:
        colorwheel: numpy.ndarray
            The RGB values of the transitions in the color space.
    """

    colorwheel_length = sum(transitions)

    # The red hue is repeated to make the color wheel cyclic
    base_hues = map(np.array,
                    ([255, 0, 0], [255, 255, 0], [0, 255, 0],
                     [0, 255, 255], [0, 0, 255], [255, 0, 255],
                     [255, 0, 0]))

    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index

        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index

    return colorwheel


def calibration_pattern(width=151, **flow_to_rgb_args):
    """ Generates a test pattern.

    Useful to add a legend to your graphs.

    Args:
        width: int
            Radius of the square test pattern.
        flow_to_rgb_args: kwargs
            Arguments passed to the flow_to_rgb function

    Returns:
        img: numpy.ndarray
            A 2D image of the test pattern.
    """
    hw = width // 2

    x_grid, y_grid = np.mgrid[:width, :width]

    u = x_grid / hw - 1
    v = y_grid / hw - 1

    flow = np.zeros((width, width, 2))
    flow[..., 0] = u
    flow[..., 1] = v

    if "flow_max_radius" not in flow_to_rgb_args:
        flow_to_rgb_args["flow_max_radius"] = 1

    img = flow_to_rgb(flow, **flow_to_rgb_args)

    return img
