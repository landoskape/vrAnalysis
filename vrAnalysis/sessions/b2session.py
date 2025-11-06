from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import joblib
import numpy as np
import json
from numpyencoder import NumpyEncoder
import speedystats as ss
from ..files import local_data_path
from .base import SessionData
from roicat_support.classifier import load_classifier
from roicat_support.classifier import get_results_path as get_classifier_results_path


def create_b2session(
    mouse_name: str,
    date: str,
    session_id: str,
    params: "B2SessionParams" | Dict[str, Any] | None = None,
) -> "B2Session":
    """Create a B2Session object and (optionally) specify the parameters

    Parameters
    ----------
    mouse_name: str
        The name of the mouse
    date: str
        The date of the session
    session_id: str
        The id of the session
    params: B2SessionParams, dict, or None
        The parameters to use for the session. If None, the default parameters will be used.
        If a dictionary, it can contain the keys:
            - spks_type: str (which kind of spks data to load)
            - keep_planes: list[int] (which planes to keep)
            - good_labels: list[str] (which labels to keep from the roicat classifier analysis)
            - fraction_filled_threshold: float (threshold for the fraction of the ROI that is filled -- based on local concavity analysis)
            - footprint_size_threshold: int (threshold for the size of the ROI)
    """
    if params is None:
        params = B2SessionParams()
    elif isinstance(params, dict):
        params = B2SessionParams.from_dict(params)
    elif isinstance(params, B2SessionParams):
        pass
    else:
        raise ValueError(f"params must be a B2SessionParams object or a dictionary")
    return B2Session(mouse_name, date, session_id, params)


@dataclass
class B2RegistrationOpts:
    """
    Options for B2 session registration.

    Attributes
    ----------
    vrBehaviorVersion : int, optional
        Version of VR behavior data format. Default is 1.
    facecam : bool, optional
        Whether facecam data is included. Default is False.
    imaging : bool, optional
        Whether imaging data is included. Default is True.
    oasis : bool, optional
        Whether OASIS deconvolution was performed. Default is True.
    redCellProcessing : bool, optional
        Whether red cell processing was performed. Default is True.
    clearOne : bool, optional
        Whether to clear onedata during registration. Default is True.
    neuropilCoefficient : float, optional
        Coefficient for neuropil subtraction. Default is 0.7.
    tau : float, optional
        Time constant for deconvolution. Default is 1.5.
    fs : int, optional
        Sampling frequency in Hz. Default is 6.
    """

    vrBehaviorVersion: int = 1
    facecam: bool = False
    imaging: bool = True
    oasis: bool = True
    redCellProcessing: bool = True
    clearOne: bool = True
    neuropilCoefficient: float = 0.7
    tau: float = 1.5
    fs: int = 6
    moveRawData: bool = field(
        default=False,
        metadata={
            "deprecated": True,
            "note": "For backwards compatibility only; this option is never used in current code.",
        },
    )


@dataclass
class B2SessionParams:
    """
    Parameters for configuring B2Session data loading and filtering.

    Attributes
    ----------
    spks_type : str, optional
        Type of spike data to load. Options: "oasis", "deconvolved", "raw",
        "neuropil", "significant", "corrected". Default is "significant".
    keep_planes : list[int], optional
        List of plane indices to keep. If None, all planes are kept.
        Default is None.
    good_labels : list[str], optional
        List of ROICaT classifier labels to keep. Default is ["c", "d"].
    fraction_filled_threshold : float, optional
        Minimum fraction filled threshold for ROI filtering based on local
        concavity analysis. If None, no filtering is applied. Default is None.
    footprint_size_threshold : int, optional
        Minimum footprint size threshold for ROI filtering. If None, no
        filtering is applied. Default is None.
    exclude_silent_rois : bool, optional
        If True, exclude ROIs with zero variance. Default is True.
    neuropil_coefficient : float, optional
        Coefficient for neuropil subtraction when computing corrected
        fluorescence. If None, uses value from B2RegistrationOpts.
        Default is None.
    exclude_redundant_rois : bool, optional
        If True, exclude redundant ROIs based on clustering analysis.
        Default is True.
    """

    spks_type: str = "significant"
    keep_planes: list[int] | None = None
    good_labels: list[str] = field(default_factory=lambda: ["c", "d"])
    fraction_filled_threshold: float | None = None
    footprint_size_threshold: int | None = None
    exclude_silent_rois: bool = True
    neuropil_coefficient: float | None = None
    exclude_redundant_rois: bool = True

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "B2SessionParams":
        """
        Create B2SessionParams from a dictionary.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary containing parameter values.

        Returns
        -------
        B2SessionParams
            B2SessionParams instance created from the dictionary.
        """
        return cls(**params)

    def __post_init__(self) -> None:
        """
        Post-initialization method to load classifier label mapping.

        This method is called automatically after dataclass initialization.
        It loads the ROICaT classifier to get the label_to_id mapping for
        validating good_labels.
        """
        classifier = load_classifier()
        self._label_to_id = classifier["label_to_id"]

    def update(self, **kwargs) -> None:
        """
        Update the parameters for the session.

        Parameters
        ----------
        **kwargs
            Parameter names and values to update. Can include any parameter
            from B2SessionParams. Special handling for "good_labels" which
            validates against the classifier.
        """
        for key, val in kwargs.items():
            if key == "good_labels":
                self.set_good_labels(val)
            else:
                setattr(self, key, val)

    @property
    def good_label_idx(self) -> list[int] | None:
        """
        Get the good label indices for the session.

        Returns
        -------
        list[int] or None
            List of label indices corresponding to good_labels, or None if
            good_labels is None.
        """
        if self.good_labels is None:
            return None
        else:
            return [self._label_to_id[label] for label in self.good_labels]

    def set_good_labels(self, good_labels: list[str] | None) -> list[str]:
        """
        Set the good labels for the session.

        Parameters
        ----------
        good_labels : list[str] or None
            List of ROICaT classifier labels to keep. If None, all labels
            are kept.

        Returns
        -------
        list[str]
            The set good_labels list.

        Raises
        ------
        ValueError
            If any label in good_labels is not found in the classifier.
        """
        if good_labels is None:
            self.good_labels = None
        else:
            if any(label not in self._label_to_id for label in good_labels):
                raise ValueError(f"Not all labels in good_labels are found in the classifier: {good_labels}")
            self.good_labels = good_labels


@dataclass
class B2Session(SessionData):
    """
    B2Session represents a registered VR imaging session with suite2p and ROICaT data.

    This class extends SessionData to provide specialized functionality for
    loading and managing B2 format session data, including suite2p outputs,
    ROICaT classifier results, and behavioral data.

    Attributes
    ----------
    opts : B2RegistrationOpts
        Registration options for the session.
    preprocessing : list[str]
        List of preprocessing steps that were applied during registration.
    params : B2SessionParams
        Parameters for configuring data loading and ROI filtering.
    """

    opts: B2RegistrationOpts = field(default_factory=B2RegistrationOpts, repr=False, init=False)
    preprocessing: list[str] = field(default_factory=list, repr=False, init=False)
    params: B2SessionParams = field(default_factory=B2SessionParams, repr=False)

    @property
    def s2p_path(self) -> Path:
        """
        Path to suite2p directory.

        Returns
        -------
        Path
            Path object pointing to the suite2p subdirectory within the
            session data path.
        """
        return self.data_path / "suite2p"

    @property
    def roicat_path(self) -> Path:
        """
        Path to roicat directory.

        Returns
        -------
        Path
            Path object pointing to the roicat subdirectory within the
            session data path.
        """
        return self.data_path / "roicat"

    @property
    def recipe_loaders(self) -> dict:
        """
        Dictionary of loaders for loading data from recipes.

        Returns
        -------
        dict
            Dictionary mapping loader type strings to loader functions.
            Available loaders: "S2P", "roiPosition".
        """
        return {"S2P": self.load_s2p, "roiPosition": self.get_roi_position}

    @property
    def recipe_transforms(self) -> dict:
        """
        Dictionary of transforms for applying to data when loading recipes.

        Returns
        -------
        dict
            Dictionary mapping transform names to transform functions.
            Available transforms: "transpose", "idx_column1".
        """
        return {"transpose": lambda x: x.T, "idx_column1": lambda x: x[:, 1]}

    @classmethod
    def spks_types(cls) -> list[str]:
        """
        Get list of available spike data types.

        Returns
        -------
        list[str]
            List of valid spks_type values: "oasis", "deconvolved", "raw",
            "neuropil", "significant", "corrected".
        """
        return ["oasis", "deconvolved", "raw", "neuropil", "significant", "corrected"]

    def _load_spks(self, spks_type: str = None) -> np.ndarray:
        """
        Load spike data of the specified type.

        Parameters
        ----------
        spks_type : str, optional
            Type of spike data to load. If None, uses params.spks_type.
            Options: "oasis", "deconvolved", "raw", "neuropil", "significant",
            "corrected".

        Returns
        -------
        np.ndarray
            Spike data array. Shape is (n_frames, n_rois) for most types.

        Raises
        ------
        ValueError
            If spks_type is not recognized.
        """
        if spks_type == "oasis":
            return self.loadone("mpci.roiActivityDeconvolvedOasis")
        elif spks_type == "deconvolved":
            return self.loadone("mpci.roiActivityDeconvolved")
        elif spks_type == "raw":
            return self.loadone("mpci.roiActivityF")
        elif spks_type == "neuropil":
            return self.loadone("mpci.roiNeuropilActivityF")
        elif spks_type == "significant":
            return self.loadone("mpci.roiSignificantFluorescence", sparse=True, keep_sparse=False)
        elif spks_type == "corrected":
            return self.loadfcorr().T
        else:
            raise ValueError(f"spks_type {spks_type} not recognized")

    def _are_spks_zero_baseline(self, spks_type: str) -> bool:
        """
        Check if the spike data type has zero baseline (nonnegative values).

        Parameters
        ----------
        spks_type : str
            Type of spike data to check.

        Returns
        -------
        bool
            True if the spks_type has zero baseline (e.g., deconvolved,
            significant), False otherwise (e.g., raw fluorescence).

        Raises
        ------
        ValueError
            If spks_type is not recognized.
        """
        if spks_type == "oasis":
            return True
        elif spks_type == "deconvolved":
            return True
        elif spks_type == "raw":
            return False
        elif spks_type == "neuropil":
            return False
        elif spks_type == "significant":
            return True
        elif spks_type == "corrected":
            return False
        else:
            raise ValueError(f"spks_type {spks_type} not recognized")

    def get_spks(self, spks_type: Optional[str] = None) -> np.ndarray:
        """
        Get spike data for the session.

        Parameters
        ----------
        spks_type : str, optional
            Type of spike data to load. If None, uses params.spks_type.
            Default is None.

        Returns
        -------
        np.ndarray
            Spike data array of the specified type.
        """
        spks_type = spks_type or self.params.spks_type
        return self._load_spks(spks_type)

    @property
    def spks(self) -> np.ndarray:
        """
        Neural spike data for the session.

        Returns
        -------
        np.ndarray
            Spike data array using the configured spks_type from params.
        """
        return self.get_spks(self.params.spks_type)

    @property
    def spks_type(self) -> str:
        """
        Current spike data type.

        Returns
        -------
        str
            The spks_type parameter value.
        """
        return self.params.spks_type

    @property
    def zero_baseline_spks(self) -> bool:
        """
        Whether the current spike data type has zero baseline.

        Returns
        -------
        bool
            True if the current spks_type has zero baseline, False otherwise.
        """
        return self._are_spks_zero_baseline(self.params.spks_type)

    @property
    def timestamps(self) -> np.ndarray:
        """
        Imaging timestamps.

        Returns
        -------
        np.ndarray
            Array of timestamps for each imaging frame.
        """
        return self.loadone("mpci.times")

    @property
    def env_length(self) -> np.ndarray:
        """
        Environment length for each trial.

        Returns
        -------
        np.ndarray
            Array of environment lengths (room lengths) for each trial.

        Notes
        -----
        Part of SessionToSpkmapProtocol.
        """
        return self.loadone("trials.roomLength")

    @property
    def positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the position of the mouse during the VR experiment and timestamps.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - timestamps: Array of timestamps for each position sample
            - position: Array of positions (typically 1D position along track)
            - trial_numbers: Array of trial numbers for each position sample
            - idx_behave_to_frame: Array mapping behavioral samples to imaging frames

        Notes
        -----
        Part of SessionToSpkmapProtocol.
        """
        timestamps = self.loadone("positionTracking.times")
        position = self.loadone("positionTracking.position")
        idx_behave_to_frame = self.loadone("positionTracking.mpci")
        trial_start_index = self.loadone("trials.positionTracking")
        num_samples = len(position)
        trial_numbers = np.arange(len(trial_start_index))
        trial_lengths = np.append(np.diff(trial_start_index), num_samples - trial_start_index[-1])
        trial_numbers = np.repeat(trial_numbers, trial_lengths)
        return timestamps, position, trial_numbers, idx_behave_to_frame

    @property
    def trial_environment(self) -> np.ndarray:
        """
        Environment index for each trial.

        Returns
        -------
        np.ndarray
            Array of environment indices for each trial.
        """
        return self.loadone("trials.environmentIndex")

    @property
    def environments(self) -> np.ndarray:
        """
        Unique environments used in the session.

        Returns
        -------
        np.ndarray
            Array of unique environment indices present in the session.
        """
        return np.unique(self.trial_environment)

    @property
    def num_trials(self) -> int:
        """
        Number of trials in the session.

        Returns
        -------
        int
            Total number of trials in the session.

        Notes
        -----
        Part of SessionToSpkmapProtocol.
        """
        return self.get_value("numTrials")

    @property
    def idx_rois(self) -> np.ndarray:
        """
        Boolean indices of ROIs to load based on filtering criteria.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_rois,) indicating which ROIs pass all
            filtering criteria (plane, label, fill fraction, footprint size,
            activity, redundancy).
        """
        num_rois = self.get_value("numROIs")
        idx_rois = np.ones(num_rois, dtype=bool)

        # Filter ROIs by which plane they are in
        if self.params.keep_planes is not None:
            idx_rois &= self.valid_plane_idx()

        # Filter ROIs by the results of the ROICaT classifier analysis
        if self.roicat_classifier is not None and (
            self.params.good_label_idx is not None
            or self.params.fraction_filled_threshold is not None
            or self.params.footprint_size_threshold is not None
        ):
            valid_label, valid_fill_fraction, valid_footprint_size = self.valid_mask_idx()

            if valid_label is not None:
                idx_rois &= valid_label

            if valid_fill_fraction is not None:
                idx_rois &= valid_fill_fraction

            if valid_footprint_size is not None:
                idx_rois &= valid_footprint_size

        if self.params.exclude_silent_rois:
            idx_rois &= self.valid_activity_idx()

        if self.params.exclude_redundant_rois:
            idx_rois &= self.valid_redundancy_idx()

        return idx_rois

    def valid_plane_idx(self) -> np.ndarray:
        """
        Boolean indices of ROIs in the specified planes.

        Returns
        -------
        np.ndarray
            Boolean array indicating which ROIs are in the planes specified
            by params.keep_planes. If keep_planes is None, all ROIs are valid.
        """
        if self.params.keep_planes is not None:
            plane_idx = self.get_plane_idx()
            return np.isin(plane_idx, self.params.keep_planes)
        else:
            return np.ones(self.get_value("numROIs"), dtype=bool)

    def valid_mask_idx(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter ROIs by the results of the ROICaT classifier analysis.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing three boolean arrays:
            - valid_label: ROIs with acceptable classifier labels
            - valid_fill_fraction: ROIs above fraction_filled_threshold
            - valid_footprint_size: ROIs above footprint_size_threshold

        Notes
        -----
        If ROICaT classifier is not available or thresholds are not set,
        all arrays will be True (all ROIs valid).
        """
        if self.roicat_classifier is not None and (
            self.params.good_label_idx is not None
            or self.params.fraction_filled_threshold is not None
            or self.params.footprint_size_threshold is not None
        ):
            class_predictions = self.roicat_classifier["class_predictions"]
            fill_fraction = self.roicat_classifier["fill_fraction"]
            footprint_size = self.roicat_classifier["footprint_size"]

            if self.params.good_label_idx is not None:
                valid_label = np.isin(class_predictions, self.params.good_label_idx)
            else:
                valid_label = np.ones(self.get_value("numROIs"), dtype=bool)

            if self.params.fraction_filled_threshold is not None:
                valid_fill_fraction = fill_fraction > self.params.fraction_filled_threshold
            else:
                valid_fill_fraction = np.ones(self.get_value("numROIs"), dtype=bool)

            if self.params.footprint_size_threshold is not None:
                valid_footprint_size = footprint_size > self.params.footprint_size_threshold
            else:
                valid_footprint_size = np.ones(self.get_value("numROIs"), dtype=bool)
        else:
            valid_label = np.ones(self.get_value("numROIs"), dtype=bool)
            valid_fill_fraction = np.ones(self.get_value("numROIs"), dtype=bool)
            valid_footprint_size = np.ones(self.get_value("numROIs"), dtype=bool)

        return valid_label, valid_fill_fraction, valid_footprint_size

    def valid_activity_idx(self) -> np.ndarray:
        """
        Filter ROIs by activity (non-silent ROIs).

        Returns
        -------
        np.ndarray
            Boolean array indicating which ROIs have non-zero variance.
            If exclude_silent_rois is False, all ROIs are valid.
        """
        if self.params.exclude_silent_rois:
            valid_activity = ss.var(self.spks, axis=0) != 0
        else:
            valid_activity = np.ones(self.get_value("numROIs"), dtype=bool)
        return valid_activity

    def valid_redundancy_idx(self) -> np.ndarray:
        """
        Filter ROIs by redundancy (non-redundant ROIs).

        Returns
        -------
        np.ndarray
            Boolean array indicating which ROIs are not redundant.
            If exclude_redundant_rois is False, all ROIs are valid.
        """
        if self.params.exclude_redundant_rois:
            valid_redundancy = ~self.loadone("mpciROIs.redundant")
        else:
            valid_redundancy = np.ones(self.get_value("numROIs"), dtype=bool)
        return valid_redundancy

    def get_validity_indices(self) -> dict[str, np.ndarray]:
        """
        Get all validity indices for ROI filtering.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing boolean arrays for each filtering criterion:
            - "plane_idx": Valid planes
            - "mask_idx": Valid labels
            - "fill_fraction_idx": Valid fill fractions
            - "footprint_size_idx": Valid footprint sizes
            - "activity_idx": Valid activity (non-silent)
            - "redundancy_idx": Valid redundancy (non-redundant)
        """
        valid_plane = self.valid_plane_idx()
        valid_label, valid_fill_fraction, valid_footprint_size = self.valid_mask_idx()
        valid_activity = self.valid_activity_idx()
        valid_redundancy = self.valid_redundancy_idx()
        return {
            "plane_idx": valid_plane,
            "mask_idx": valid_label,
            "fill_fraction_idx": valid_fill_fraction,
            "footprint_size_idx": valid_footprint_size,
            "activity_idx": valid_activity,
            "redundancy_idx": valid_redundancy,
        }

    def get_red_idx(self) -> np.ndarray:
        """
        Get the indices of the red ROIs.

        Returns
        -------
        np.ndarray
            Boolean array indicating which ROIs are red cells.

        Notes
        -----
        redCellIdxCoherent is a consolidated red cell index array that uses
        tracking information to determine which cells are red in a coherent
        manner. The roicat_support.tracking module builds this array. When not
        available, uses the standard redCellIdx array (session wasn't tracked).
        """
        if "mpciROIs.redCellIdxCoherent" in self.print_saved_one():
            return self.loadone("mpciROIs.redCellIdxCoherent")
        else:
            return self.loadone("mpciROIs.redCellIdx")

    def update_params(self, **kwargs) -> None:
        """
        Update the parameters for the session.

        Parameters
        ----------
        **kwargs
            The parameters to update, can be any of the parameters in
            B2SessionParams (except for good_label_idx, which is set automatically).
            Including:
                - spks_type: str
                - keep_planes: list[int]
                - good_labels: list[str]
                - fraction_filled_threshold: float
                - footprint_size_threshold: int
                - exclude_silent_rois: bool
                - neuropil_coefficient: float
                - exclude_redundant_rois: bool
        """
        self.params.update(**kwargs)

    def _init_data_path(self) -> Path:
        """
        Set the data path for the session.

        Returns
        -------
        Path
            Path to the session data directory: local_data_path / mouse_name / date / session_id.
        """
        return local_data_path() / self.mouse_name / self.date / self.session_id

    def _additional_loading(self) -> None:
        """
        Load registered experiment data.

        This method loads session configuration files, preprocessing steps,
        stored values, and ROICaT classifier results if available.

        Raises
        ------
        ValueError
            If session JSON files are not found (session not registered).
        """
        super()._additional_loading()
        if not (self.data_path / "vrExperimentOptions.json").exists():
            raise ValueError("session json files were not found! you need to register the session first.")

        # Load options and preprocessing steps
        self.opts = B2RegistrationOpts(**json.load(open(self.data_path / "vrExperimentOptions.json")))
        self.preprocessing = json.load(open(self.data_path / "vrExperimentPreprocessing.json"))

        # Load stored values
        values = json.load(open(self.data_path / "vrExperimentValues.json"))
        for key, val in values.items():
            self.set_value(key, val)
        # Also load ROICaT Classifier Results if they exist
        results_path = get_classifier_results_path(self)
        if results_path.exists():
            self.roicat_classifier = joblib.load(results_path)
        else:
            self.roicat_classifier = None

    def save_session_prms(self) -> None:
        """
        Save registered session parameters to JSON files.

        Saves opts, preprocessing steps, and values to their respective
        JSON files in the session data directory.
        """
        with open(self.data_path / "vrExperimentOptions.json", "w") as file:
            json.dump(vars(self.opts), file, ensure_ascii=False)
        with open(self.data_path / "vrExperimentPreprocessing.json", "w") as file:
            json.dump(self.preprocessing, file, ensure_ascii=False)
        with open(self.data_path / "vrExperimentValues.json", "w") as file:
            json.dump(vars(self.values), file, ensure_ascii=False, cls=NumpyEncoder)

    # =============================================================================
    # Suite2p loading functions
    # =============================================================================
    def loadfcorr(self, mean_adjusted: bool = True, try_from_one: bool = True) -> np.ndarray:
        """
        Load corrected fluorescence data.

        Corrected fluorescence is computed as F - neuropil_coefficient * (Fneu - meanFneu).
        This requires a special loading function because it isn't saved directly.

        Parameters
        ----------
        mean_adjusted : bool, optional
            If True, subtract mean neuropil activity per ROI. Default is True.
        try_from_one : bool, optional
            If True, try loading from onedata first, otherwise load from suite2p.
            Default is True.

        Returns
        -------
        np.ndarray
            Corrected fluorescence array of shape (n_rois, n_frames).
        """
        if try_from_one:
            F = self.loadone("mpci.roiActivityF").T
            Fneu = self.loadone("mpci.roiNeuropilActivityF").T
        else:
            F = self.load_s2p("F")
            Fneu = self.load_s2p("Fneu")
        meanFneu = np.mean(Fneu, axis=1, keepdims=True) if mean_adjusted else np.zeros((np.sum(self.get_value("roiPerPlane")), 1))
        neuropil_coefficient = self.params.neuropil_coefficient or self.opts.neuropilCoefficient
        return F - neuropil_coefficient * (Fneu - meanFneu)

    def load_s2p(self, varName: str, concatenate: bool = True) -> Union[np.ndarray, list]:
        """
        Load suite2p variable from suite2p folders.

        Parameters
        ----------
        varName : str
            Name of the variable to load (e.g., "F", "Fneu", "spks", "ops").
            Must be in the available variables list.
        concatenate : bool, optional
            If True, concatenate data across planes. If False, return list
            of arrays per plane. For "ops", concatenate is always False.
            Default is True.

        Returns
        -------
        np.ndarray or list
            If concatenate=True, returns concatenated array across all planes.
            If concatenate=False or varName="ops", returns list of arrays
            (one per plane).

        Raises
        ------
        AssertionError
            If varName is not available in the suite2p folders.
        """
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

    def get_plane_idx(self) -> np.ndarray:
        """
        Return the plane index for each ROI (concatenated across planes).

        Returns
        -------
        np.ndarray
            Array of plane indices for each ROI, with shape (n_rois,).
            Values are uint8.
        """
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
        stack_position = np.stack((xc, yc, planeIdx)).T
        return stack_position

    #
    # =============================================================================
    # Behavior processing functions
    # =============================================================================
    def get_behave_trial_idx(self, trial_start_frame: np.ndarray) -> np.ndarray:
        """
        Get the trial index for each behavioral sample.

        Parameters
        ----------
        trial_start_frame : np.ndarray
            Array of frame indices where each trial starts.

        Returns
        -------
        np.ndarray
            Array of trial indices for each behavioral sample, with shape
            (n_behave_samples,). Values are uint64.
        """
        nspt = np.array([*np.diff(trial_start_frame), self.get_value("numBehaveTimestamps") - trial_start_frame[-1]])
        return np.concatenate([tidx * np.ones(ns) for (tidx, ns) in enumerate(nspt)]).astype(np.uint64)

    def group_behave_by_trial(self, data: np.ndarray, trial_start_frame: np.ndarray) -> list[np.ndarray]:
        """
        Group behavioral data by trial.

        Parameters
        ----------
        data : np.ndarray
            Behavioral data array to group, with shape (n_behave_samples, ...).
        trial_start_frame : np.ndarray
            Array of frame indices where each trial starts.

        Returns
        -------
        list[np.ndarray]
            List of arrays, one per trial, containing the behavioral data
            for that trial.
        """
        trial_index = self.get_behave_trial_idx(trial_start_frame)
        return [data[trial_index == tidx] for tidx in range(len(trial_start_frame))]

    # =============================================================================
    # Equality and Hashing
    # =============================================================================
    def __eq__(self, other: Any) -> bool:
        """
        Check if two sessions are equal.

        Parameters
        ----------
        other : Any
            Object to compare with.

        Returns
        -------
        bool
            True if other is a B2Session with the same session_name, False otherwise.
        """
        if not isinstance(other, B2Session):
            return False
        if hash(self) == hash(other):
            return True
        else:
            return False

    def __hash__(self) -> int:
        """
        Hash the session based on session name.

        Returns
        -------
        int
            Hash value based on the session_name tuple.
        """
        return hash(self.session_name)
