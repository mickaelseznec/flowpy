import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from itertools import accumulate
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def flow_to_rgb(flow, flow_max_radius=None, background="bright", custom_colorwheel=None):
    """
    Creates a RGB representation of an optical flow.

    Parameters
    ----------
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement

    flow_max_radius: float, optionnal
        Set the radius that gives the maximum color intensity, useful for comparing different flows.
        Default: The normalization is based on the input flow maximum radius.

    background: str, optionnal
        States if zero-valued flow should look 'bright' or 'dark'
        Default: "bright"

    custom_colorwheel: numpy.ndarray
        Use a custom colorwheel for specific hue transition lengths.
        By default, the default transition lengths are used.

    Returns
    -------
    rgb_image: numpy.ndarray
        A 2D RGB image that represents the flow

    See Also
    --------
    make_colorwheel

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

    if flow_max_radius > 0:
        radius /= flow_max_radius

    ncols = len(wheel)

    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))

    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))

    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (wheel[angle_floor.astype(np.int)] * (1 - angle_fractional) +
                 wheel[angle_ceil.astype(np.int)] * angle_fractional)

    ColorizationArgs = namedtuple("ColorizationArgs", [
        'move_hue_valid_radius',
        'move_hue_oversized_radius',
        'invalid_color'])

    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)

    def move_hue_on_S_axis(hues, factors):
        return 255. - np.expand_dims(factors, -1) * (255. - hues)

    if background == "dark":
        parameters = ColorizationArgs(move_hue_on_V_axis, move_hue_on_S_axis,
                                      np.array([255, 255, 255], dtype=np.float))
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis,
                                      np.array([0, 0, 0], dtype=np.float))

    colors = parameters.move_hue_valid_radius(float_hue, radius)

    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    colors[nan_mask] = parameters.invalid_color

    return colors.astype(np.uint8)


def make_colorwheel(transitions=DEFAULT_TRANSITIONS):
    """
    Creates a color wheel.

    A color wheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).

    Parameters
    ----------
    transitions: sequence_like
        Contains the length of the six transitions.
        Defaults to (15, 6, 4, 11, 13, 6), based on humain perception.

    Returns
    -------
    colorwheel: numpy.ndarray
        The RGB values of the transitions in the color space.

    Notes
    -----
    For more information, take a look at
    https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm

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
    """
    Generates a calibration pattern.

    Useful to add a legend to your optical flow plots.

    Parameters
    ----------
    pixel_size: int
        Radius of the square test pattern.
    flow_max_radius: float
        The maximum radius value represented by the calibration pattern.
    flow_to_rgb_args: kwargs
        Arguments passed to the flow_to_rgb function

    Returns
    -------
    calibration_img: numpy.ndarray
        The RGB image representation of the calibration pattern.
    calibration_flow: numpy.ndarray
        The flow represented in the calibration_pattern. In HWF layout

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


def attach_arrows(ax, flow, xy_steps=(20, 20),
                  units="xy", color="w", angles="xy", **quiver_kwargs):
    """
    Attach the flow arrows to a matplotlib axes using quiver.

    Parameters:
    -----------
    ax: matplotlib.axes
        The axes the arrows should be plotted on.
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement
    xy_steps: sequence_like
        The arrows are plotted every xy_steps[0] in the x-dimension and xy_steps[1] in the y-dimension

    Quiver Parameters:
    ------------------
    The following parameters are here to override matplotlib.quiver's defaults.
    units: str
        See matplotlib.quiver documentation.
    color: str
        See matplotlib.quiver documentation.
    angles: str
        See matplotlib.quiver documentation.
    quiver_kwargs: kwargs
        Other parameters passed to matplotlib.quiver
        See matplotlib.quiver documentation.

    Returns
    -------
    quiver_artist: matplotlib.artist
        See matplotlib.quiver documentation
        Useful for removing the arrows from the figure

    """
    height, width, _ = flow.shape

    y_grid, x_grid = np.mgrid[:height, :width]

    step_x, step_y = xy_steps
    half_step_x, half_step_y = step_x // 2, step_y // 2

    return ax.quiver(
        x_grid[half_step_x::step_x, half_step_y::step_y],
        y_grid[half_step_x::step_x, half_step_y::step_y],
        flow[half_step_x::step_x, half_step_y::step_y, 0],
        flow[half_step_x::step_x, half_step_y::step_y, 1],
        angles=angles,
        units=units, color=color, **quiver_kwargs,
    )


def attach_coord(ax, flow, extent=None):
    """
    Attach the flow value to the coordinate tooltip.

    It allows you to see on the same figure, the RGB value of the pixel and the underlying value of the flow.
    Shows cartesian and polar coordinates.

    Parameters:
    -----------
    ax: matplotlib.axes
        The axes the arrows should be plotted on.
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement
    extent: sequence_like, optional
        Use this parameters in combination with matplotlib.imshow to resize the RGB plot.
        See matplotlib.imshow extent parameter.
        See attach_calibration_pattern

    """
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


def attach_calibration_pattern(ax, **calibration_pattern_kwargs):
    """
    Attach a calibration pattern to axes.

    This function uses calibration_pattern to generate a figure and shows it as nicely as possible.

    Parameters:
    -----------
    calibration_pattern_kwargs: kwargs, optional
        Parameters to be given to the calibration_pattern function.

    See Also:
    ---------
    calibration_pattern

    Returns
    -------
    image_axes: matplotlib.AxesImage
        See matplotlib.imshow documentation
        Useful for changing the image dynamically
    circle_artist: matplotlib.artist
        See matplotlib.circle documentation
        Useful for removing the circle from the figure

    """
    pattern, flow = calibration_pattern(**calibration_pattern_kwargs)
    flow_max_radius = calibration_pattern_kwargs.get("flow_max_radius", 1)

    extent = (-flow_max_radius, flow_max_radius) * 2

    image = ax.imshow(pattern, extent=extent)
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

    return image, circle


def backward_warp(second_image, flow, **map_coordinates_kwargs):
    """
    Compute the backwarp warp of an image.

    Given second_image and the flow from first_image to second_image, it warps the second_image to something close to the first image if the flow is accurate.

    Parameters:
    -----------
    second_image: numpy.ndarray
        Image of the form [H, W] or [H, W, C] for greyscale or RGB images
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout, from first_image to second_image.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement
    map_coordinates_kwargs: kwargs
        Keyword arguments passed to scipy.ndimage.map_coordinates
        Most important ones are *mode* for out-of-bound handling (defaults to nearest),
        and "order" to set the quality of the interpolation.
        see scipy.ndimage.map_coordinates

    Returns
    -------
    first_image: numpy.ndarray
        The warped image with same dimensions as second_image.
    """
    height, width, *_ = second_image.shape
    coord = np.mgrid[:height, :width]

    gx = (coord[1] + flow[..., 0])
    gy = (coord[0] + flow[..., 1])

    if "mode" not in map_coordinates_kwargs:
        map_coordinates_kwargs["mode"] = "nearest"

    first_image = np.zeros_like(second_image)
    if second_image.ndim == 3:
        for dim in range(second_image.shape[2]):
            map_coordinates(second_image[..., dim], (gy, gx), first_image[..., dim], **map_coordinates_kwargs)
    else:
        map_coordinates(second_image, (gy, gx), first_image, **map_coordinates_kwargs)
    return first_image


def forward_warp(first_image, flow, k=4):
    """
    Compute the forward warp of an image.

    Given first_image and the flow from first_image to second_image, it warps the first_image to something close to the first image if the flow is accurate.

    It uses a k-nearest neighbors search to perform an interpolation.

    Parameters:
    -----------
    first_image: numpy.ndarray
        Image of the form [H, W] or [H, W, C] for greyscale or RGB images
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout, from first_image to second_image.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement
    k: int, optional
        How many neighbors should be taken into account to interpolate.

    Returns
    -------
    second_image: numpy.ndarray
        The warped image with same dimensions as first_image.
    """
    first_image_3d = first_image[..., np.newaxis] if first_image.ndim == 2 else first_image
    height, width, channels = first_image_3d.shape

    coord = np.mgrid[:height, :width]
    grid = coord.transpose(1, 2, 0).reshape((width * height, 2))

    gx = (coord[1] + flow[..., 0])
    gy = (coord[0] + flow[..., 1])

    warped_points = np.asarray((gy.flatten(), gx.flatten())).T
    kdt = cKDTree(warped_points)

    distance, neighbor = kdt.query(grid, k=k)

    y, x = neighbor // width, neighbor % width

    neigbor_values = first_image_3d[(y, x)]

    if k == 1:
        second_image_flat = neigbor_values
    else:
        weights = np.exp(-distance[..., np.newaxis])
        normalizer = np.sum(weights, axis=1)

        second_image_flat = np.sum(neigbor_values * weights, axis=1)
        second_image_flat = (second_image_flat / normalizer).astype(first_image.dtype)

    return second_image_flat.reshape(first_image.shape)


def replace_nans(array, value=0):
    nan_mask = np.isnan(array)
    array[nan_mask] = value

    return array, nan_mask


def get_flow_max_radius(flow):
    return np.sqrt(np.nanmax(np.sum(flow ** 2, axis=2)))
