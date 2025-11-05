from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import joblib
import numpy as np
import json
from numpyencoder import NumpyEncoder
import speedystats as ss
from ..files import local_data_path
from .base import SessionData
from roicat_support.classifier import load_classifier
from roicat_support.classifier import get_results_path as get_classifier_results_path


def create_b2session(mouse_name: str, date: str, session_id: str, params: "B2SessionParams" | Dict[str, Any] | None = None) -> "B2Session":
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
    vrBehaviorVersion: int = 1
    facecam: bool = False
    imaging: bool = True
    oasis: bool = True
    moveRawData: bool = False
    redCellProcessing: bool = True
    clearOne: bool = True
    neuropilCoefficient: float = 0.7
    tau: float = 1.5
    fs: int = 6


@dataclass
class B2SessionParams:
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
        return cls(**params)

    def __post_init__(self):
        classifier = load_classifier()
        self._label_to_id = classifier["label_to_id"]

    def update(self, **kwargs):
        """Update the parameters for the session"""
        for key, val in kwargs.items():
            if key == "good_labels":
                self.set_good_labels(val)
            else:
                setattr(self, key, val)

    @property
    def good_label_idx(self) -> list[int] | None:
        """Get the good label indices for the session"""
        if self.good_labels is None:
            return None
        else:
            return [self._label_to_id[label] for label in self.good_labels]

    def set_good_labels(self, good_labels: list[str] | None) -> list[str]:
        """Set the good labels for the session"""
        if good_labels is None:
            self.good_labels = None
        else:
            if any(label not in self._label_to_id for label in good_labels):
                raise ValueError(f"Not all labels in good_labels are found in the classifier: {good_labels}")
            self.good_labels = good_labels


@dataclass
class B2Session(SessionData):
    opts: B2RegistrationOpts = field(default_factory=B2RegistrationOpts, repr=False, init=False)
    preprocessing: list[str] = field(default_factory=list, repr=False, init=False)
    params: B2SessionParams = field(default_factory=B2SessionParams, repr=False)

    @property
    def s2p_path(self):
        """Path to suite2p directory"""
        return self.data_path / "suite2p"

    @property
    def roicat_path(self):
        """Path to roicat directory"""
        return self.data_path / "roicat"

    @property
    def recipe_loaders(self):
        return {"S2P": self.load_s2p, "roiPosition": self.get_roi_position}

    @property
    def recipe_transforms(self):
        return {"transpose": lambda x: x.T, "idx_column1": lambda x: x[:, 1]}

    @classmethod
    def spks_types(cls):
        return ["oasis", "deconvolved", "raw", "neuropil", "significant", "corrected"]

    def _load_spks(self, spks_type: str = None):
        """Lookup the spks_type in the session"""
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
        """Check if the spks are nonnegative"""
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

    def get_spks(self, spks_type: Optional[str] = None):
        """Get spks data for the session. Optionally specify the spks_type to load."""
        spks_type = spks_type or self.params.spks_type
        return self._load_spks(spks_type)

    @property
    def spks(self):
        return self.get_spks(self.params.spks_type)

    @property
    def spks_type(self):
        return self.params.spks_type

    @property
    def zero_baseline_spks(self) -> bool:
        return self._are_spks_zero_baseline(self.params.spks_type)

    @property
    def timestamps(self):
        return self.loadone("mpci.times")

    @property
    def env_length(self):
        """Return the env length for each trial

        Part of SessionToSpkmapProtocol: Needs documentation
        """
        return self.loadone("trials.roomLength")

    @property
    def positions(self):
        """Return the position of the mouse during the VR experiment and timestamps

        Part of SessionToSpkmapProtocol: Needs documentation
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
    def trial_environment(self):
        """Return the environment for each trial"""
        return self.loadone("trials.environmentIndex")

    @property
    def environments(self):
        """Return the environments used in the session"""
        return np.unique(self.trial_environment)

    @property
    def num_trials(self):
        """Return the number of trials in the session

        Part of SessionToSpkmapProtocol: Needs documentation
        """
        return self.get_value("numTrials")

    @property
    def idx_rois(self):
        """Return the indices of the ROIs to load"""
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

    def valid_plane_idx(self):
        if self.params.keep_planes is not None:
            plane_idx = self.get_plane_idx()
            return np.isin(plane_idx, self.params.keep_planes)
        else:
            return np.ones(self.get_value("numROIs"), dtype=bool)

    def valid_mask_idx(self):
        """Filter ROIs by the results of the ROICaT classifier analysis"""
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

    def valid_activity_idx(self):
        """Filter ROIs by the activity of the ROIs"""
        if self.params.exclude_silent_rois:
            valid_activity = ss.var(self.spks, axis=0) != 0
        else:
            valid_activity = np.ones(self.get_value("numROIs"), dtype=bool)
        return valid_activity

    def valid_redundancy_idx(self):
        """Filter ROIs by the cluster of the ROIs"""
        if self.params.exclude_redundant_rois:
            valid_redundancy = ~self.loadone("mpciROIs.redundant")
        else:
            valid_redundancy = np.ones(self.get_value("numROIs"), dtype=bool)
        return valid_redundancy

    def get_validity_indices(self):
        """Get the indices of the ROIs that are valid for the session"""
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

    def get_red_idx(self):
        """Get the indices of the red ROIs.

        redCellIdxCoherent is a consolidated red cell index array that uses tracking information
        to determine which cells are red in a coherent manner. The roicat_support.tracking module
        builds this array. When not available, we just use the standard redCellIdx array because
        it means the session wasn't tracked.
        """
        if "mpciROIs.redCellIdxCoherent" in self.print_saved_one():
            return self.loadone("mpciROIs.redCellIdxCoherent")
        else:
            return self.loadone("mpciROIs.redCellIdx")

    def update_params(self, **kwargs):
        """Update the parameters for the session

        Parameters
        ----------
        **kwargs: dict
            The parameters to update, can be any of the parameters in B2SessionParams
            (except for good_label_idx, which is set automatically)
            Including:
                - spks_type: str
                - keep_planes: list[int]
                - good_labels: list[str]
                - fraction_filled_threshold: float
                - footprint_size_threshold: int
        """
        self.params.update(**kwargs)

    def _init_data_path(self) -> str:
        """Set the data path for the session"""
        return local_data_path() / self.mouse_name / self.date / self.session_id

    def _additional_loading(self):
        """Load registered experiment data"""
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

    def save_session_prms(self):
        """Save registered session parameters"""
        with open(self.data_path / "vrExperimentOptions.json", "w") as file:
            json.dump(vars(self.opts), file, ensure_ascii=False)
        with open(self.data_path / "vrExperimentPreprocessing.json", "w") as file:
            json.dump(self.preprocessing, file, ensure_ascii=False)
        with open(self.data_path / "vrExperimentValues.json", "w") as file:
            json.dump(vars(self.values), file, ensure_ascii=False, cls=NumpyEncoder)

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
        neuropil_coefficient = self.params.neuropil_coefficient or self.opts.neuropilCoefficient
        return F - neuropil_coefficient * (Fneu - meanFneu)

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
        stack_position = np.stack((xc, yc, planeIdx)).T
        return stack_position

    #
    # =============================================================================
    # Behavior processing functions
    # =============================================================================
    def get_behave_trial_idx(self, trial_start_frame):
        """Get the trial index for each behavioral sample"""
        nspt = np.array([*np.diff(trial_start_frame), self.get_value("numBehaveTimestamps") - trial_start_frame[-1]])
        return np.concatenate([tidx * np.ones(ns) for (tidx, ns) in enumerate(nspt)]).astype(np.uint64)

    def group_behave_by_trial(self, data, trial_start_frame):
        trial_index = self.get_behave_trial_idx(trial_start_frame)
        return [data[trial_index == tidx] for tidx in range(len(trial_start_frame))]

    # =============================================================================
    # Equality and Hashing
    # =============================================================================
    def __eq__(self, other: Any) -> bool:
        """Check if two sessions are equal"""
        if not isinstance(other, B2Session):
            return False
        if hash(self) == hash(other):
            return True
        else:
            return False

    def __hash__(self) -> int:
        """Hash the session"""
        return hash(self.session_name)
