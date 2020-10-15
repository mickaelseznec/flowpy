import flowpy
import numpy as np
import tifffile


class InputManager():
    def __init__(self, file_paths):
        self.input_paths = file_paths

    def open(self):
        if len(self.input_paths) == 1:
            self._data = np.asarray([flowpy.flow_read(self.input_paths[0])])
        elif len(self.input_paths) == 2:
            self._data = tifffile.imread(self.input_paths)
            if self._data.ndim == 3:
                self._data = np.asarray([self._data])
            self._data = self._data.transpose((1, 2, 3, 0))
        return self

    def close(self):
        pass

    def get_shape(self):
        return self._data.shape

    def get_data(self, index):
        return self._data[index]

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, tb):
        self.close()
