from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Any, List
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csc_array, save_npz, load_npz
from ..helpers import PrettyDatetime


class LoadingRecipe:
    RECIPE_MARKER = "__LOADING_RECIPE__"

    def __init__(self, loader_type: str, source_arg: str, transforms: list = None, **kwargs):
        """
        Represents instructions for loading data.

        Use case: Save a recipe to load data from an existing location instead of
        resaving the data with a new name. Will save memory on the device and
        generally have a very small overhead.

        Parameters
        ----------
        loader_type : str
            String identifying the loading method (e.g., 'numpy', 's2p').
        source_arg : str
            String identifying the source data (e.g. 'F', 'ops').
        transforms : list, optional
            List of transformation operations to apply. Default is None.
        **kwargs
            Additional arguments for the loader.
        """
        self.loader_type = loader_type
        self.source_arg = source_arg
        self.transforms = transforms or []
        self.kwargs = kwargs

    def to_dict(self) -> dict:
        """
        Convert recipe to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation of the recipe with marker, loader_type,
            source_arg, transforms, and kwargs.
        """
        return {
            "marker": self.RECIPE_MARKER,
            "loader_type": self.loader_type,
            "source_arg": self.source_arg,
            "transforms": self.transforms,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoadingRecipe":
        """
        Create recipe from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing recipe information with keys: loader_type,
            source_arg, transforms, and kwargs.

        Returns
        -------
        LoadingRecipe
            LoadingRecipe instance created from the dictionary.
        """
        return cls(data["loader_type"], data["source_arg"], data["transforms"], **data["kwargs"])

    @classmethod
    def is_recipe(cls, data: np.ndarray) -> bool:
        """
        Check if the loaded data is actually a recipe.

        Parameters
        ----------
        data : np.ndarray
            Numpy array that may contain a recipe dictionary.

        Returns
        -------
        bool
            True if the data contains a recipe marker, False otherwise.
        """
        try:
            dict_data = data.item()
            return isinstance(dict_data, dict) and dict_data.get("marker", None) == cls.RECIPE_MARKER
        except (AttributeError, ValueError):
            return False


@dataclass
class SessionData(ABC):
    """
    Base class for session data.

    This abstract base class provides the core functionality for managing VR
    session data, including loading and saving onedata files, managing session
    identifiers, and providing a flexible values namespace for storing
    arbitrary data.

    Attributes
    ----------
    mouse_name : str
        Name of the mouse for this session.
    date : str, datetime, or PrettyDatetime
        Date of the session. Will be converted to string format.
    session_id : str, optional
        Optional session identifier. Default is None.
    data_path : Path or str
        Path to the session data directory. Set automatically during initialization.
    values : SimpleNamespace
        Namespace for storing arbitrary data. Useful for flexible storing of
        small bytes data.
    one_cache : dict
        Cache for buffering onedata files to avoid repeated disk reads.
    """

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
        """
        Post-initialization method to set all properties specific to subclasses.

        This method is called automatically after dataclass initialization.
        It formats the date, initializes the data path, and calls additional
        loading steps for subclasses.
        """
        # Enable additional loading for subclasses if required
        # usually to put things in the values namespace
        self.date = str(PrettyDatetime.make_pretty(self.date))
        self.data_path = self._init_data_path()
        self._additional_loading()

    def _additional_loading(self) -> None:
        """
        Additional loading for subclasses.

        Called in the __post_init__ method. Allows subclasses to perform
        additional loading steps. Default is to do nothing.

        Notes
        -----
        Subclasses should override this method to perform any additional
        initialization steps, such as loading configuration files or setting
        up specialized data structures.
        """
        pass

    @abstractmethod
    def _init_data_path(self) -> Union[Path, str]:
        """
        Define the data path for the session data.

        This is required for all subclasses of SessionData to define.

        Returns
        -------
        Path or str
            Path to the session data directory.

        Notes
        -----
        This method must be implemented by all subclasses. It should return
        the path where session data is stored and loaded from.
        """
        pass

    @property
    @abstractmethod
    def spks(self) -> np.ndarray:
        """
        Neural spks data.

        This property must be implemented by all subclasses. It should return
        the neural spike data for the session.

        Returns
        -------
        np.ndarray
            Neural spike data array. Shape and format depend on the subclass
            implementation.

        Notes
        -----
        This is always required for all session data objects.
        """
        pass

    @property
    def session_name(self) -> tuple[str]:
        """
        Return the session identifier as a tuple.

        Returns
        -------
        tuple[str]
            Tuple containing (mouse_name, date) or (mouse_name, date, session_id)
            if session_id is provided.
        """
        if self.session_id:
            return self.mouse_name, self.date, self.session_id
        else:
            return self.mouse_name, self.date

    def session_print(self, joinby: str = "/") -> str:
        """
        Generate string representation of session name.

        Parameters
        ----------
        joinby : str, optional
            String to join session name components. Default is "/".

        Returns
        -------
        str
            String representation of the session name with components joined
            by the specified separator.
        """
        return joinby.join(self.session_name)

    def get_value(self, key: str) -> Any:
        """
        Get value from the session stored in the values namespace.

        Parameters
        ----------
        key : str
            Key name of the value to retrieve.

        Returns
        -------
        Any
            Value stored under the specified key in the values namespace.

        Raises
        ------
        AttributeError
            If the key does not exist in the values namespace.
        """
        return getattr(self.values, key)

    def set_value(self, key: str, value: Any) -> None:
        """
        Set value in the session stored in the values namespace.

        Parameters
        ----------
        key : str
            Key name to store the value under.
        value : Any
            Value to store in the values namespace.
        """
        setattr(self.values, key, value)

    @property
    def recipe_loaders(self) -> dict:
        """
        Dictionary of loaders for loading data from recipes.

        This property enables specialized loading of onedata stored with recipes.
        If not specified, attempting to load a recipe will raise an error.

        Returns
        -------
        dict
            Dictionary mapping loader type strings to loader functions.

        Raises
        ------
        NotImplementedError
            If this property is not overridden by a subclass that uses recipes.

        Notes
        -----
        Subclasses that use LoadingRecipe objects should override this property
        to provide a dictionary of loader functions. Each loader function should
        accept a source_arg and optional kwargs.
        """
        raise NotImplementedError(f"Attempting to load a recipe from {self.session_print()} but no recipe_loaders are specified.")

    @property
    def recipe_transforms(self) -> dict:
        """
        Dictionary of transforms for applying to data when loading recipes.

        This property enables specialized transforms of onedata stored with
        recipes. If not specified, attempting to load a recipe that requires
        a transform will raise an error.

        Returns
        -------
        dict
            Dictionary mapping transform names to transform functions.

        Raises
        ------
        NotImplementedError
            If this property is not overridden by a subclass that uses recipes
            with transforms.

        Notes
        -----
        Subclasses that use LoadingRecipe objects with transforms should
        override this property to provide a dictionary of transform functions.
        Each transform function should accept data and return transformed data.
        """
        raise NotImplementedError(f"Attempting to transform a loaded recipe from {self.session_print()} but no recipe_transforms are specified.")

    @property
    def one_path(self) -> Path:
        """
        Path to onedata directory.

        Returns
        -------
        Path
            Path object pointing to the onedata subdirectory within the
            session data path.
        """
        return self.data_path / "onedata"

    def get_saved_one(self) -> list[Path]:
        """
        Get all saved onedata files.

        Returns
        -------
        list[Path]
            List of Path objects for all .npy and .npz files in the onedata
            directory.
        """
        return list(self.one_path.glob("*.npy")) + list(self.one_path.glob("*.npz"))

    def print_saved_one(self, include_path: bool = False, include_extension: bool = False) -> list[str]:
        """
        Get formatted list of all saved onedata files.

        Parameters
        ----------
        include_path : bool, optional
            If True, include the full path in the output. Default is False.
        include_extension : bool, optional
            If True, include the file extension in the output. Default is False.

        Returns
        -------
        list[str]
            List of formatted file names (and optionally paths) for all saved
            onedata files.
        """

        def _format_name(name: Path):
            onename = name.stem
            if include_extension:
                onename = onename + name.suffix
            if include_path:
                onename = name.parent / onename
            return onename

        return [_format_name(name) for name in self.get_saved_one()]

    def clear_one_data(self, one_file_names: List[str] = None, certainty: bool = False) -> None:
        """
        Clear onedata files from the session directory.

        Parameters
        ----------
        one_file_names : list[str], optional
            List of specific onedata file names (without extension) to clear.
            If None, all onedata files will be cleared. Default is None.
        certainty : bool, optional
            Safety flag that must be set to True to actually delete files.
            Default is False.

        Notes
        -----
        This operation is destructive and cannot be undone. The certainty flag
        must be set to True to prevent accidental deletions.
        """
        if not certainty:
            print(f"You have to be certain to clear onedata! (This means set kwarg certainty=True).")
            return None
        one_files = self.get_saved_one()
        if one_file_names:
            one_files = [file for file in one_files if file.stem in one_file_names]
        for file in one_files:
            file.unlink()
        print(f"Cleared onedata from session: {self.session_print()}")

    def get_one_filename(self, *names: str) -> str:
        """
        Create onedata filename from an arbitrary length list of names.

        Parameters
        ----------
        *names : str
            Variable number of strings to join into a filename.

        Returns
        -------
        str
            Filename with components joined by "." and ".npy" extension added.
            Example: get_one_filename("mpci", "roiActivityF") returns
            "mpci.roiActivityF.npy".
        """
        return ".".join(names) + ".npy"

    def saveone(self, data: Union[np.ndarray, LoadingRecipe], *names: str, sparse: bool = False) -> None:
        """
        Save data directly or as a loading recipe.

        Parameters
        ----------
        data : np.ndarray or LoadingRecipe
            Data to save. Can be a numpy array or a LoadingRecipe object.
        *names : str
            Sequence of strings to join into filename (e.g., "mpci", "roiActivityF"
            -> "mpci.roiActivityF.npy").
        sparse : bool, optional
            If True, save as sparse array (.npz format). Data must be a
            scipy.sparse.csc_array. Default is False.

        Raises
        ------
        ValueError
            If sparse=True but data is not a scipy.sparse.csc_array.

        Notes
        -----
        When saving a numpy array (not sparse), the data is also cached in
        one_cache for efficient subsequent access. LoadingRecipe objects are
        saved as numpy arrays containing dictionaries with a special marker.
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

        Parameters
        ----------
        *names : str
            Sequence of strings to join into filename (e.g., "mpci", "roiActivityF"
            -> "mpci.roiActivityF.npy").
        force : bool, optional
            If True, reload data even if it is already in the cache. Default
            is False.
        sparse : bool, optional
            If True, load as sparse array (.npz format). Default is False.
        keep_sparse : bool, optional
            If True and data is sparse, return a sparse array. Otherwise,
            convert sparse arrays to dense. Default is False.

        Returns
        -------
        np.ndarray
            Loaded data. May be transformed if loaded from a recipe.

        Raises
        ------
        ValueError
            If the requested onedata file does not exist.
        NotImplementedError
            If loading a recipe but recipe_loaders or recipe_transforms are
            not properly configured.

        Notes
        -----
        Data is cached in one_cache after loading to avoid repeated disk reads.
        If a LoadingRecipe is encountered, it will be executed using the
        recipe_loaders and recipe_transforms properties.
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
        """
        Clear cached data to free memory.

        Parameters
        ----------
        file_names : list[str], optional
            List of specific file names (with extension) to remove from cache.
            If None, the entire cache is cleared. Default is None.

        Notes
        -----
        This method helps manage memory usage by removing cached onedata from
        memory. The files themselves are not deleted, only the in-memory cache.
        """
        if file_names is None:
            self.one_cache = {}
        else:
            for file_name in file_names:
                self.one_cache.pop(file_name, None)
