from typing import Literal, Tuple, Optional, List, Dict, Any
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import tifffile

from syd import Viewer


class HistologyViewer(Viewer):
    mice_names: list[str] = ["ATL061", "ATL062", "ATL063", "ATL064", "ATL065"]
    resolutions: list[str] = ["10", "25", "50"]
    planes: list[str] = ["coronal", "transverse", "sagittal"]

    _current_volume: np.ndarray = None

    def __init__(self):
        self.add_selection("mouse_name", value=self.mice_names[0], options=self.mice_names)
        self.add_selection("resolution", value=self.resolutions[-1], options=self.resolutions)
        self.add_selection("plane", value=self.planes[0], options=self.planes)
        self.add_integer("slice", value=0, min=0, max=1)
        self.add_float("vmax", value=1.0, min=0, max=1.0, step=0.001)
        self.add_float("redgreen_balance", value=0.5, min=0, max=1.0)
        self.on_change(["mouse_name", "resolution"], self.update_volume)
        self.on_change("plane", self.update_slice_limits)
        self.update_volume(self.state)

    def get_histology_folder(self, mouse_name: str, resolution: Literal["10", "25", "50"], server: str = "zortex"):
        root_name = rf"\\{server}.cortexlab.net\Subjects\{mouse_name}\Histology\downsampled_stacks\0{resolution}_micron"
        return root_name

    def get_histology_files(self, mouse_name, resolution, server="zortex"):
        root_name = Path(self.get_histology_folder(mouse_name, resolution, server))
        red_file = list(root_name.glob("*red.tif"))[0]
        green_file = list(root_name.glob("*green.tif"))[0]
        red = np.array(tifffile.imread(red_file), dtype=np.float32)
        green = np.array(tifffile.imread(green_file), dtype=np.float32)
        return red, green

    def build_rgb_volume(self, red, green):
        blue = np.zeros_like(red)
        rgb = np.stack([red, green, blue], axis=-1)
        return rgb

    def update_volume(self, state):
        mouse_name = state["mouse_name"]
        resolution = state["resolution"]
        red, green = self.get_histology_files(mouse_name, resolution)
        rgb = self.build_rgb_volume(red, green)
        self._current_volume = rgb
        self.update_slice_limits(self.state)

    def update_slice_limits(self, state):
        dim = self.planes.index(state["plane"])
        self.update_integer("slice", max=self._current_volume.shape[dim] - 1)

    def get_slice(self, state):
        slice = state["slice"]
        dim = self.planes.index(state["plane"])
        c_slice = np.take(self._current_volume, slice, axis=dim)
        if dim == 2:
            c_slice = np.transpose(c_slice, (1, 0, 2))
        if dim == 1:
            c_slice = c_slice[::-1]
        c_slice = c_slice / np.max(c_slice)
        c_slice[:, :, 0] = c_slice[:, :, 0] * state["redgreen_balance"]
        c_slice[:, :, 1] = c_slice[:, :, 1] * (1 - state["redgreen_balance"])
        return c_slice

    def plot(self, state):
        vmax = state["vmax"]
        c_slice = self.get_slice(state)
        c_slice = np.clip(c_slice / vmax, 0, 1)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(c_slice, aspect="equal")
        return fig
