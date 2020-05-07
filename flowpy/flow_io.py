import numpy as np
import png
import struct

from pathlib import Path
from warnings import warn


def flow_write(output_file, flow, format=None):
    """
    Writes optical flow to file.

    Parameters
    ----------
    output_file: {str, pathlib.Path, file}
        Path of the file to write or file object.
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement
    format: str, optional
        Specify in what format the flow is written, accepted formats: "png" or "flo"
        If None, it is guessed on the file extension

    See Also
    --------
    flow_read

    """

    supported_formats = ("png", "flo")

    output_format = guess_extension(output_file, override=format)

    with FileManager(output_file, "wb") as f:
        if output_format == "png":
            flow_write_png(f, flow)
        else:
            flow_write_flo(f, flow)


def flow_read(input_file, format=None):
    """
    Reads optical flow from file

    Parameters
    ----------
    output_file: {str, pathlib.Path, file}
        Path of the file to read or file object.
    format: str, optional
        Specify in what format the flow is raed, accepted formats: "png" or "flo"
        If None, it is guess on the file extension

    Returns
    -------
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] is the x-displacement
        flow[..., 1] is the y-displacement

    Notes
    -----

    The flo format is dedicated to optical flow and was first used in Middlebury optical flow database.
    The original defition can be found here: http://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp

    The png format uses 16-bit RGB png to store optical flows.
    It was developped along with the KITTI Vision Benchmark Suite.
    More information can be found here: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow

    The both handle flow with invalid ``invalid'' values, to deal with occlusion for example.
    We convert such invalid values to NaN.

    See Also
    --------
    flow_write

    """

    input_format = guess_extension(input_file, override=format)

    with FileManager(input_file, "rb") as f:
        if input_format == "png":
            output = flow_read_png(f)
        else:
            output = flow_read_flo(f)

    return output


def flow_read_flo(f):
    if (f.read(4) != b'PIEH'):
        warn("{} does not have a .flo file signature".format(f.name))

    width, height = struct.unpack("II", f.read(8))
    result = np.fromfile(f, dtype="float32").reshape((height, width, 2))

    # Set invalid flows to NaN
    mask_u = np.greater(np.abs(result[..., 0]), 1e9, where=(~np.isnan(result[..., 0])))
    mask_v = np.greater(np.abs(result[..., 1]), 1e9, where=(~np.isnan(result[..., 1])))

    result[mask_u | mask_v] = np.NaN

    return result


def flow_write_flo(f, flow):
    SENTINEL = 1666666800.0  # Only here to look like Middlebury original files
    height, width, _ = flow.shape

    image = flow.copy()
    image[np.isnan(image)] = SENTINEL

    f.write(b'PIEH')
    f.write(struct.pack("II", width, height))
    image.astype(np.float32).tofile(f)


def flow_read_png(f):
    width, height, stream, *_ = png.Reader(f).read()

    file_content = np.concatenate(list(stream)).reshape((height, width, 3))
    flow, valid = file_content[..., 0:2], file_content[..., 2]

    flow = (flow.astype(np.float) - 2 ** 15) / 64.

    flow[~valid.astype(np.bool)] = np.NaN

    return flow


def flow_write_png(f, flow):
    SENTINEL = 0.  # Only here to look like original KITTI files
    height, width, _ = flow.shape
    flow_copy = flow.copy()

    valid = ~(np.isnan(flow[..., 0]) | np.isnan(flow[..., 1]))
    flow_copy[~valid] = SENTINEL

    flow_copy = (flow_copy * 64. + 2 ** 15).astype(np.uint16)
    image = np.dstack((flow_copy, valid))

    writer = png.Writer(width, height, bitdepth=16, greyscale=False)
    writer.write(f, image.reshape((height, 3 * width)))


class FileManager:
    def __init__(self, abstract_file, mode):
        self.abstract_file = abstract_file
        self.opened_file = None
        self.mode = mode

    def __enter__(self):
        if isinstance(self.abstract_file, str):
            self.opened_file = open(self.abstract_file, self.mode)
        elif isinstance(self.abstract_file, Path):
            self.opened_file = self.abstract_file.open(self.mode)
        else:
            return self.abstract_file

        return self.opened_file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opened_file is not None:
            self.opened_file.close()


def guess_extension(abstract_file, override=None):
    if override is not None:
        return override

    if isinstance(abstract_file, str):
        return Path(abstract_file).suffix[1:]
    elif isinstance(abstract_file, Path):
        return abstract_file.suffix[1:]

    return Path(abstract_file.name).suffix[1:]
