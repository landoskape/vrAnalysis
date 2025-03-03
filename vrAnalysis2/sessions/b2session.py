from dataclasses import dataclass, field
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
class B2SessionParams:
    spks_type: str = "significant"
    keep_planes: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
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

    def update_parameters(self, **kwargs):
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
    opts: dict = field(default_factory=dict, repr=False, init=False)
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
            plane_idx = self.get_plane_idx()
            idx_rois &= np.isin(plane_idx, self.params.keep_planes)

        # Filter ROIs by the results of the ROICaT classifier analysis
        if self.roicat_classifier is not None and (
            self.params.good_label_idx is not None
            or self.params.fraction_filled_threshold is not None
            or self.params.footprint_size_threshold is not None
        ):
            class_predictions = self.roicat_classifier["class_predictions"]
            fill_fraction = self.roicat_classifier["fill_fraction"]
            footprint_size = self.roicat_classifier["footprint_size"]

            if self.params.good_label_idx is not None:
                idx_rois &= np.isin(class_predictions, self.params.good_label_idx)

            if self.params.fraction_filled_threshold is not None:
                idx_rois &= fill_fraction > self.params.fraction_filled_threshold

            if self.params.footprint_size_threshold is not None:
                idx_rois &= footprint_size > self.params.footprint_size_threshold

        if self.params.exclude_silent_rois:
            idx_rois &= ss.var(self.spks, axis=0) != 0

        if self.params.exclude_redundant_rois:
            idx_rois &= ~self.loadone("mpciROIs.redundant")

        return idx_rois

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
        self.params.update_parameters(**kwargs)

    def _init_data_path(self) -> str:
        """Set the data path for the session"""
        return local_data_path() / self.mouse_name / self.date / self.session_id

    def _additional_loading(self):
        """Load registered experiment data"""
        super()._additional_loading()
        if not (self.data_path / "vrExperimentOptions.json").exists():
            raise ValueError("session json files were not found! you need to register the session first.")
        opts = json.load(open(self.data_path / "vrExperimentOptions.json"))
        preprocessing = json.load(open(self.data_path / "vrExperimentPreprocessing.json"))
        values = json.load(open(self.data_path / "vrExperimentValues.json"))
        self.opts = opts
        self.preprocessing = preprocessing
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
        neuropil_coefficient = self.params.neuropil_coefficient or self.opts["neuropilCoefficient"]
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
        stackPosition = np.stack((xc, yc, planeIdx)).T
        return stackPosition
