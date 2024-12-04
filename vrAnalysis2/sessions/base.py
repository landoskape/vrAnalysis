from warnings import warn
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union
from types import SimpleNamespace
import numpy as np
import json
from ..helpers import PrettyDatetime
from ..files import local_data_path
from numpyencoder import NumpyEncoder


class LoadingRecipe:
    RECIPE_MARKER = "__LOADING_RECIPE__"

    def __init__(self, loader_type: str, source_arg: str, transforms: list = None, **kwargs):
        """
        Represents instructions for loading data.

        Use case:
        Save a recipe to load data from an existing location instead of resaving the data with a new name.
        Will save memory on the device and generally have a very small overhead.

        Args:
            loader_type: String identifying the loading method (e.g., 'numpy', 's2p')
            source_arg: String identifying the source data (e.g. 'F', 'ops')
            transforms: List of transformation operations to apply
            **kwargs: Additional arguments for the loader
        """
        self.loader_type = loader_type
        self.source_arg = source_arg
        self.transforms = transforms or []
        self.kwargs = kwargs

    def to_dict(self) -> dict:
        """Convert recipe to dictionary for serialization."""
        return {
            "marker": self.RECIPE_MARKER,
            "loader_type": self.loader_type,
            "source_arg": self.source_arg,
            "transforms": self.transforms,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoadingRecipe":
        """Create recipe from dictionary."""
        return cls(data["loader_type"], data["source_arg"], data["transforms"], **data["kwargs"])

    @classmethod
    def is_recipe(cls, data: np.ndarray) -> bool:
        """Check if the loaded data is actually a recipe."""
        try:
            dict_data = data.item()
            return isinstance(dict_data, dict) and dict_data.get("marker", None) == cls.RECIPE_MARKER
        except (AttributeError, ValueError):
            return False


@dataclass
class vrSession(ABC):
    """Top-level data class to store and manage behavioral and neural data from a single session."""

    # session identifiers
    mouse_name: str
    date: PrettyDatetime
    session_id: Optional[str] = None

    # additional fields
    values: SimpleNamespace = field(default_factory=SimpleNamespace, repr=False)

    # immutable subclass properties
    data_path: str = field(init=False, repr=False)

    # mutable subclass properties
    spks_type: str = field(init=False, repr=False)

    def __post_init__(self):
        """Post-initialization method to set all properties specific to subclasses"""
        self.data_path = self._init_data_path()
        self.spks_type = self.set_spks_type()

        if not isinstance(self.data_path, Path):
            raise TypeError(f"data_path must be a Path-like object, received: {type(self.data_path)}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        # additional loading for subclasses
        self._additional_loading()

    def get_value(self, key: str) -> object:
        """Get additional values from the session"""
        return getattr(self.values, key, None)

    def set_value(self, key: str, value: object) -> None:
        """Set additional values to the session"""
        setattr(self.values, key, value)

    def session_print(self, joinby="/") -> str:
        """Useful function for generating string of session name for useful feedback to user"""
        return joinby.join(self.session_name)

    @property
    def session_name(self) -> tuple[str]:
        """Function for returning mouse name, date string, and session id"""
        if self.session_id:
            return self.mouse_name, self.date, self.session_id
        else:
            return self.mouse_name, self.date

    @abstractmethod
    def _init_data_path(self) -> str:
        """Set the data path for the session"""

    def _additional_loading(self) -> None:
        """Additional loading for subclasses.

        Called in the __post_init__ method. Allows subclasses to perform additional
        loading steps. Default is to do nothing.
        """
        pass

    # neural data properties
    @property
    def spks(self) -> np.ndarray:
        """Load spks with the abstract method _get_spks"""
        return self._get_spks()

    @abstractmethod
    def _get_spks(self) -> np.ndarray:
        """Load spks data"""


class OneSession(vrSession):
    """Top-level class to manage data that uses the onedata format to save & load data"""

    # a cache for onedata
    one_cache: dict = field(default_factory=dict, repr=False)

    @property
    @abstractmethod
    def recipe_loaders(self) -> dict:
        """Dictionary of loaders for loading data from recipes"""

    @property
    @abstractmethod
    def recipe_transforms(self) -> dict:
        """Dictionary of transforms for applying to data when loading recipes"""

    @property
    def one_path(self):
        """Path to oneData directory"""
        return self.data_path / "oneData"

    def get_saved_one(self):
        """Get all saved oneData files"""
        return self.one_path.glob("*.npy")

    def print_saved_one(self):
        """Print all saved oneData files"""
        return [name.stem for name in self.get_saved_one()]

    def clear_one_data(self, one_file_names=None, certainty=False):
        """Clear oneData files from the session directory"""
        if not certainty:
            print(f"You have to be certain to clear oneData! (This means set kwarg certainty=True).")
            return None
        one_files = self.get_saved_one()
        if one_file_names:
            one_files = [file for file in one_files if file.stem in one_file_names]
        for file in one_files:
            file.unlink()
        print(f"Cleared oneData from session: {self.session_print()}")

    def get_one_filename(self, *names):
        """create one filename given an arbitrary length list of names"""
        return ".".join(names) + ".npy"

    def saveone(self, data: Union[np.ndarray, LoadingRecipe], *names: str) -> None:
        """
        Save data directly or as a loading recipe.

        Args:
            data: Either numpy array or LoadingRecipe
            names: sequence of strings to join into filename (e.g., "mpci", "roiActivityF" -> "mpci.roiActivityF")
        """
        file_name = self.get_one_filename(*names)
        path = self.one_path / file_name
        if isinstance(data, LoadingRecipe) or (
            hasattr(data, "to_dict") and (hasattr(data, "RECIPE_MARKER") and data.RECIPE_MARKER == LoadingRecipe.RECIPE_MARKER)
        ):
            # Save recipe as a numpy array containing a dictionary
            recipe_dict = data.to_dict()
            np.save(path, np.array(recipe_dict, dtype=object))
        else:
            # Save data directly to one file
            self.one_cache[file_name] = data  # (standard practice is to buffer the data for efficient data handling)
            np.save(path, data)

    def loadone(self, *names: str, force=False) -> np.ndarray:
        """
        Load data, either directly, from the buffer, or by following a recipe.

        Args:
            names: Sequence of strings to join into filename (e.g., "mpci", "roiActivityF" -> "mpci.roiActivityF")
            force: If True, reload data even if it is already in the buffer

        Returns:
            Loaded and potentially transformed data
        """
        file_name = self.get_one_filename(*names)
        if not force and file_name in self.one_cache.keys():
            return self.one_cache[file_name]
        else:
            path = self.one_path / file_name
            if not (path.exists()):
                print(f"In session {self.session_print}, the one file {file_name} doesn't exist. Here is a list of saved oneData files:")
                for oneFile in self.print_saved_one():
                    print(oneFile)
                raise ValueError("oneData requested is not available")

            # Load saved numpy array at savepath
            data = np.load(path, allow_pickle=True)

            if LoadingRecipe.is_recipe(data):
                # Get loading recipe
                recipe = LoadingRecipe.from_dict(data.item())

                # Load data using appropriate loader
                if recipe.source_arg is not None:
                    data = self.recipe_loaders[recipe.loader_type](recipe.source_arg, **recipe.kwargs)
                else:
                    data = self.recipe_loaders[recipe.loader_type](**recipe.kwargs)

                # Apply transforms
                for transform in recipe.transforms:
                    data = self.recipe_transforms[transform](data)

            # add to buffer
            self.one_cache[file_name] = data
            return data

    def clear_cache(self):
        """Clear cached data to free memory"""
        self.one_cache = {}


@dataclass
class B2Session(OneSession):
    opts: dict = field(default_factory=dict, repr=False)
    preprocessing: list[str] = field(default_factory=list, repr=False)

    @property
    def s2p_path(self):
        """Path to suite2p directory"""
        return self.data_path / "suite2p"

    @property
    def recipe_loaders(self):
        return {"S2P": self.load_s2p, "roiPosition": self.get_roi_position}

    @property
    def recipe_transforms(self):
        return {"transpose": lambda x: x.T, "idx_column1": lambda x: x[:, 1]}

    def _get_spks(self) -> np.ndarray:
        """Load spks data"""
        return self.loadone(self.spks_type)

    def set_spks_type(self) -> str:
        """Set spks_type, will determine which onefile to load spks data from"""
        return "mpci.roiActivityDeconvolvedOasis"

    def _init_data_path(self) -> str:
        """Set the data path for the session"""
        return local_data_path() / self.mouse_name / self.date / self.session_id

    def _additional_loading(self):
        """Load registered experiment data"""
        if not (self.data_path / "vrExperimentOptions.json").exists():
            raise ValueError("session json files were not found! you need to register the session first.")
        opts = json.load(open(self.data_path / "vrExperimentOptions.json"))
        preprocessing = json.load(open(self.data_path / "vrExperimentPreprocessing.json"))
        values = json.load(open(self.data_path / "vrExperimentValues.json"))
        self.opts = opts
        self.preprocessing = preprocessing
        for key, val in values.items():
            self.set_value(key, val)

    def save_session_prms(self):
        """Save registered session parameters"""
        with open(self.data_path / "vrExperimentOptions.json", "w") as file:
            json.dump(self.opts, file, ensure_ascii=False)
        with open(self.data_path / "vrExperimentPreprocessing.json", "w") as file:
            json.dump(self.preprocessing, file, ensure_ascii=False)
        with open(self.data_path / "vrExperimentValues.json", "w") as file:
            json.dump(self.values, file, ensure_ascii=False, cls=NumpyEncoder)

    # =============================================================================
    # Suite2p loading functions
    # =============================================================================
    def loadfcorr(self, mean_adjusted=True, try_from_one=True):
        # corrected fluorescence requires a special loading function because it isn't saved directly
        if try_from_one:
            F = self.loadone("mpci.roiActivityF").T
            Fneu = self.loadone("mpci.roiNeuropilActivityF").T
        else:
            F = self.load_s2p("F")
            Fneu = self.load_s2p("Fneu")
        meanFneu = np.mean(Fneu, axis=1, keepdims=True) if mean_adjusted else np.zeros((np.sum(self.get_value("roiPerPlane")), 1))
        return F - self.opts["neuropilCoefficient"] * (Fneu - meanFneu)

    def load_s2p(self, varName, concatenate=True):
        # load S2P variable from suite2p folders
        assert varName in self.get_value("available"), f"{varName} is not available in the suite2p folders for {self.session_print()}"
        frame_vars = ["F", "F_chan2", "Fneu", "Fneu_chan2", "spks"]

        if varName == "ops":
            concatenate = False

        var = [np.load(self.s2p_path / planeName / f"{varName}.npy", allow_pickle=True) for planeName in self.get_value("planeNames")]
        if varName == "ops":
            var = [cvar.item() for cvar in var]
            return var

        if concatenate:
            # if concatenation is requested, then concatenate each plane across the ROIs axis so we have just one ndarray of shape: (allROIs, allFrames)
            if varName in frame_vars:
                var = [v[:, : self.get_value("numFrames")] for v in var]  # trim if necesary so each plane has the same number of frames
            var = np.concatenate(var, axis=0)

        return var

    def get_plane_idx(self):
        """Return the plane index for each ROI (concatenated across planes)."""
        planeIDs = self.get_value("planeIDs")
        roiPerPlane = self.get_value("roiPerPlane")
        return np.repeat(planeIDs, roiPerPlane).astype(np.uint8)

    def get_roi_position(self, mode="weightedmean"):
        """Return the x & y positions and plane index for all ROIs.

        Parameters
        ----------
        mode : str, optional
            Method for calculating the position of the ROI, by default "weightedmean"
            but can also use median which ignores the intensity (lam) values.

        Returns
        -------
        np.ndarray
            Array of shape (nROIs, 3) with columns: x-position, y-position, planeIdx
        """
        planeIdx = self.get_plane_idx()
        stat = self.load_s2p("stat")
        lam = [s["lam"] for s in stat]
        ypix = [s["ypix"] for s in stat]
        xpix = [s["xpix"] for s in stat]
        if mode == "weightedmean":
            yc = np.array([np.sum(l * y) / np.sum(l) for l, y in zip(lam, ypix)])
            xc = np.array([np.sum(l * x) / np.sum(l) for l, x in zip(lam, xpix)])
        elif mode == "median":
            yc = np.array([np.median(y) for y in ypix])
            xc = np.array([np.median(x) for x in xpix])
        stackPosition = np.stack((xc, yc, planeIdx)).T
        return stackPosition
