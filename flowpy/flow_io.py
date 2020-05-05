import png
import numpy as np
import struct

from sys import stderr
from io import BufferedIOBase
from contextlib import AbstractContextManager
from pathlib import Path


class FileManager(AbstractContextManager):
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


def flow_write(output_file, flow, format=None):
    """ Writes an optical flow field to a flo file on disk

    Args:
        output_file: {str, pathlib.Path, file}
            Relative path to the file to write
        u: array_like
            2D image of the displacement field along the x axis
        v: array_like
            2D image of the displacement field along the y axis
        format: str
            In what format the flow is read, accepted formats: "png" or "flo"
            If None, guess on the file extension
    """

    supported_formats = ("png", "flo")

    output_format = guess_extension(output_file, override=format)

    with FileManager(output_file, "wb") as f:
        if output_format == "png":
            _flow_write_png(f, flow)
        else:
            _flow_write_flo(f, flow)


def flow_read(input_file, format=None):
    """ Reads a flow field from a file in the .flo format

    Args:
        input_file: str
            Relative path to the file to read

    Returns:
        u: numpy.ndarray
            2D image of the displacement field along the x axis
        v: numpy.ndarray
            2D image of the displacement field along the y axis
        format: str
            In what format the flow is written, accepted formats: "png" or "flo"
            If None, guess on the file extension
    """

    input_format = guess_extension(input_file, override=format)

    with FileManager(input_file, "rb") as f:
        if input_format == "png":
            output = _flow_read_png(f)
        else:
            output = _flow_read_flo(f)

    return output


def _flow_read_flo(f):
    if (f.read(4) != b'PIEH'):
        print("WARNING: {} does not have the flo signature", file=stderr)

    width, height = struct.unpack("II", f.read(8))
    result = np.fromfile(f, dtype="float32").reshape((height, width, 2))

    # Directly set invalid flows to NaN
    mask_u = np.greater(np.abs(result[..., 0]), 1e9, where=(~np.isnan(result[..., 0])))
    mask_v = np.greater(np.abs(result[..., 1]), 1e9, where=(~np.isnan(result[..., 1])))

    result[mask_u | mask_v] = np.NaN

    return result


def _flow_write_flo(f, flow):
    SENTINEL = 1666666800.0
    height, width, _ = flow.shape

    image = flow.copy()
    image[np.isnan(image)] = SENTINEL

    f.write(b'PIEH')
    f.write(struct.pack("II", width, height))
    image.astype(np.float32).tofile(f)


def _flow_read_png(f):
    width, height, stream, *_ = png.Reader(f).read()

    file_content = np.concatenate(list(stream)).reshape((height, width, 3))
    flow, valid = file_content[..., 0:2], file_content[..., 2]

    flow = (flow.astype(np.float) - 2 ** 15) / 64.

    flow[~valid.astype(np.bool)] = np.NaN

    return flow


def _flow_write_png(f, flow):
    SENTINEL = 0.
    height, width, _ = flow.shape
    flow_copy = flow.copy()

    valid = ~(np.isnan(flow[..., 0]) | np.isnan(flow[..., 1]))
    flow_copy[~valid] = SENTINEL

    flow_copy = (flow_copy * 64. + 2 ** 15).astype(np.uint16)
    image = np.dstack((flow_copy, valid))

    writer = png.Writer(width, height, bitdepth=16, greyscale=False)
    writer.write(f, image.reshape((height, 3 * width)))
