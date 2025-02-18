from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csc_array, save_npz, load_npz
from ..helpers import PrettyDatetime


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
class SessionData(ABC):
    """Base class for session data"""

    # session identifiers
    mouse_name: str
    date: Union[str, datetime, PrettyDatetime]
    session_id: Optional[str] = None

    # session data path -- identifies where the session data is stored and loaded from
    data_path: Union[Path, str] = field(init=False, repr=False)

    # Values: namespace for storing arbitrary data -- useful for flexible storing of (small bytes) data
    values: SimpleNamespace = field(default_factory=SimpleNamespace, repr=False, init=False)

    # A cache for buffering onedata files
    one_cache: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        """Post-initialization method to set all properties specific to subclasses"""
        # Enable additional loading for subclasses if required
        # usually to put things in the values namespace
        self.date = str(PrettyDatetime.make_pretty(self.date))
        self.data_path = self._init_data_path()
        self._additional_loading()

    def _additional_loading(self) -> None:
        """Additional loading for subclasses.

        Called in the __post_init__ method. Allows subclasses to perform additional
        loading steps. Default is to do nothing.
        """
        pass

    @abstractmethod
    def _init_data_path(self) -> Union[Path, str]:
        """Defines the data path for the session data

        This is required for all subclasses of SessionData to define!
        """
        pass

    @property
    @abstractmethod
    def spks(self) -> np.ndarray:
        """Neural spks data -- always required!"""
        pass

    @property
    def session_name(self) -> tuple[str]:
        """Function for returning the session identifier as a tuple"""
        if self.session_id:
            return self.mouse_name, self.date, self.session_id
        else:
            return self.mouse_name, self.date

    def session_print(self, joinby="/") -> str:
        """Useful function for generating string of session name"""
        return joinby.join(self.session_name)

    def get_value(self, key: str) -> object:
        """Get values from the session stored in the values namespace"""
        return getattr(self.values, key, None)

    def set_value(self, key: str, value: object) -> None:
        """Set values to the session stored in the values namespace"""
        setattr(self.values, key, value)

    @property
    def recipe_loaders(self) -> dict:
        """Dictionary of loaders for loading data from recipes.

        This is to enable specialized loading of onedata stored with recipes.
        If not specified, attempting to load a recipe will raise an error.
        """
        raise NotImplementedError(f"Attempting to load a recipe from {self.session_print()} but no recipe_loaders are specified.")

    @property
    def recipe_transforms(self) -> dict:
        """Dictionary of transforms for applying to data when loading recipes.

        This is to enable specialized transforms of onedata stored with recipes.
        If not specified, attempting to load a recipe that requires a transform
        will raise an error.
        """
        raise NotImplementedError(f"Attempting to transform a loaded recipe from {self.session_print()} but no recipe_transforms are specified.")

    @property
    def one_path(self) -> Path:
        """Path to onedata directory"""
        return self.data_path / "onedata"

    def get_saved_one(self) -> list[Path]:
        """Get all saved onedata files"""
        return list(self.one_path.glob("*.npy"))

    def print_saved_one(self, include_path: bool = False, include_extension: bool = False) -> list[str]:
        """Print all saved onedata files"""

        def _format_name(name: Path):
            onename = name.stem
            if include_extension:
                onename = onename + name.suffix
            if include_path:
                onename = name.parent / onename
            return onename

        return [_format_name(name) for name in self.get_saved_one()]

    def clear_one_data(self, one_file_names: List[str] = None, certainty: bool = False) -> None:
        """Clear onedata files from the session directory"""
        if not certainty:
            print(f"You have to be certain to clear onedata! (This means set kwarg certainty=True).")
            return None
        one_files = self.get_saved_one()
        if one_file_names:
            one_files = [file for file in one_files if file.stem in one_file_names]
        for file in one_files:
            file.unlink()
        print(f"Cleared onedata from session: {self.session_print()}")

    def get_one_filename(self, *names) -> str:
        """create one filename given an arbitrary length list of names"""
        return ".".join(names) + ".npy"

    def saveone(self, data: Union[np.ndarray, LoadingRecipe], *names: str, sparse: bool = False) -> None:
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
        elif sparse:
            path = path.with_suffix(".npz")
            if isinstance(data, csc_array):
                save_npz(path, data)
            else:
                raise ValueError("Data is not a scipy.sparse.csc_array, not supported for saving with sparse=True")
        else:
            # Save data directly to one file
            self.one_cache[file_name] = data  # (standard practice is to buffer the data for efficient data handling)
            np.save(path, data)

    def loadone(self, *names: str, force: bool = False, sparse: bool = False, keep_sparse: bool = False) -> np.ndarray:
        """
        Load data, either directly or by following a recipe.

        Args:
            names: Sequence of strings to join into filename (e.g., "mpci", "roiActivityF" -> "mpci.roiActivityF")
            force: If True, reload data even if it is already in the buffer
            sparse: If True, return a sparse array
            keep_sparse: If True, return a sparse array when the data is sparse -- otherwise returns a dense array

        Returns:
            Loaded and potentially transformed data
        """
        file_name = self.get_one_filename(*names)
        path = self.one_path / file_name
        if sparse:
            path = path.with_suffix(".npz")
            file_name = path.stem + path.suffix

        if not force and file_name in self.one_cache.keys():
            return self.one_cache[file_name]

        else:
            if not (path.exists()):
                print(f"In session {self.session_print()}, the one file {file_name} doesn't exist. Here is a list of saved oneData files:")
                for one_file in self.print_saved_one():
                    print(one_file)
                raise ValueError("onedata requested is not available")

            # Load saved numpy array at savepath (or sparse array if sparse=True)
            if sparse:
                data = load_npz(path)
                if not keep_sparse:
                    data = data.toarray()
            else:
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

            self.one_cache[file_name] = data
            return data

    def clear_cache(self, file_names: Optional[List[str]] = None) -> None:
        """Clear cached data to free memory"""
        if file_names is None:
            self.one_cache = {}
        else:
            for file_name in file_names:
                self.one_cache.pop(file_name, None)
