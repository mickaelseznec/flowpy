import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from itertools import accumulate
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

def get_flow_max_radius(flow):
    return np.sqrt(np.nanmax(np.sum(flow ** 2, axis=2)))


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
    complex_flow, nan_mask = _replace_nans(complex_flow)

    radius, angle = np.abs(complex_flow), np.angle(complex_flow)

    if flow_max_radius is None:
        flow_max_radius = np.max(radius)

    if flow_max_radius > 0:
        radius /= flow_max_radius

    ncols = len(wheel)

    # Map the angles from (-pi, pi] to [0, ncols - 1)
    angle = (-angle + np.pi) * ((ncols - 1) / (2 * np.pi))

    color_interpoler = interp1d(np.arange(ncols), wheel, axis=0)

    float_hue = color_interpoler(angle.flatten())
    radius = radius.reshape((-1, 1))

    ColorizationArgs = namedtuple("ColorizationArgs", [
        'move_hue_valid_radius',
        'move_hue_oversized_radius',
        'invalid_color'])

    move_hue_on_V_axis = lambda hue, factor: hue * factor
    move_hue_on_S_axis = lambda hue, factor: 255. - factor * (255. - hue)

    if background == "dark":
        parameters = ColorizationArgs(move_hue_on_V_axis, move_hue_on_S_axis,
                                      np.array([255, 255, 255], dtype=np.float))
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis,
                                      np.array([0, 0, 0], dtype=np.float))

    colors = parameters.move_hue_valid_radius(float_hue, radius)

    oversized_radius_mask = radius.flatten() > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    colors[nan_mask.flatten()] = parameters.invalid_color

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


def calibration_pattern(pixel_size=151, flow_max_radius=1, **flow_to_rgb_args):
    """ Generates a test pattern.

    Useful to add a legend to your graphs.

    Args:
        pixel_size: int
            Radius of the square test pattern.
        flow_to_rgb_args: kwargs
            Arguments passed to the flow_to_rgb function

    Returns:
        img: numpy.ndarray
            A 2D image of the test pattern.
    """
    half_width = pixel_size // 2

    y_grid, x_grid = np.mgrid[:pixel_size, :pixel_size]

    u = flow_max_radius * (x_grid / half_width - 1)
    v = flow_max_radius * (y_grid / half_width - 1)

    flow = np.zeros((pixel_size, pixel_size, 2))
    flow[..., 0] = u
    flow[..., 1] = v

    flow_to_rgb_args["flow_max_radius"] = flow_max_radius
    img = flow_to_rgb(flow, **flow_to_rgb_args)

    return img, flow


def attach_arrows(ax, flow, xy_steps=(20, 20), units="xy", color="w", **kwargs):
    """ Displays flow with arrows over its RGB representation.
    Args:
        flow: numpy.ndarray
            3D image of the displacement field
        true_scale: bool
            Should the arrows correspond to the real distances?
        min_is_black: bool
            Is a null vector represented as black or white?
        max_norm: float
            Used for normalizing vectors' norm
        sub_factors: int or tuple of ints
            The subsampling factor in the arrows representation. A subsampling factor of 10
            means an arrows every 10 pixels. If this is a tuple, a different subsampling
            factor for every axes.
    """
    height, width, _ = flow.shape

    y_grid, x_grid = np.mgrid[:height, :width]

    step_x, step_y = xy_steps
    half_step_x, half_step_y = step_x // 2, step_y // 2

    #y-axis is reversed to match image convention
    ax.quiver(
        x_grid[half_step_x::step_x, half_step_y::step_y],
        y_grid[half_step_x::step_x, half_step_y::step_y],
        flow[half_step_x::step_x, half_step_y::step_y, 0],
        -flow[half_step_x::step_x, half_step_y::step_y, 1],
        units=units, color=color, **kwargs,
    )


def attach_coord(ax, flow, extent=None):
    height, width, _ = flow.shape
    base_format = ax.format_coord

    if extent is not None:
        left, right, bottom, top = extent
        x_ratio = width / (right - left)
        y_ratio = height / (top - bottom)

    def new_format_coord(x, y):
        if extent is None:
            int_x = int(x + 0.5)
            int_y = int(y + 0.5)
        else:
            int_x = int((x - left) * x_ratio)
            int_y = int((y - bottom) * y_ratio)


        if 0 <= int_x < width and 0 <= int_y < height:
            format_string = "Coord: x={}, y={} / Flow: ".format(int_x, int_y)

            u, v = flow[int_y, int_x, :]
            if np.isnan(u) or np.isnan(v):
                format_string += "invalid"
            else:
                complex_flow = u - 1j * v
                r, h = np.abs(complex_flow), np.angle(complex_flow, deg=True)
                format_string += ("u={:.2f}, v={:.2f} (cartesian) ρ={:.2f}, θ={:.2f}° (polar)"
                                  .format(u, v, r, h))
            return format_string
        else:
            return base_format(x, y)

    ax.format_coord = new_format_coord


def attach_calibration_pattern(ax, **calibration_pattern_args):
    pattern, flow = calibration_pattern(**calibration_pattern_args)
    flow_max_radius = calibration_pattern_args.get("flow_max_radius", 1)

    extent = (-flow_max_radius, flow_max_radius) * 2

    ax.imshow(pattern, extent=extent)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine in ("bottom", "left"):
        ax.spines[spine].set_position("zero")
        ax.spines[spine].set_linewidth(1)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    attach_coord(ax, flow, extent=extent)

    circle = plt.Circle((0, 0), flow_max_radius, fill=False, lw=1)
    ax.add_artist(circle)


def _replace_nans(array, value=0):
    nan_mask = np.isnan(array)
    array[nan_mask] = value

    return array, nan_mask
