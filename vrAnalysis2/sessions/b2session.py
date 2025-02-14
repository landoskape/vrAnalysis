from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json
from numpyencoder import NumpyEncoder
from ..files import local_data_path
from .base import SessionData


def create_b2session(mouse_name: str, date: str, session_id: str, spks_type: Optional[str] = None) -> "B2Session":
    if spks_type is None:
        return B2Session(mouse_name, date, session_id)
    else:
        return B2Session(mouse_name, date, session_id, spks_type)


@dataclass
class B2Session(SessionData):
    spks_type: str = field(default="oasis", repr=False)
    opts: dict = field(default_factory=dict, repr=False, init=False)
    preprocessing: list[str] = field(default_factory=list, repr=False, init=False)

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
        spks_type = spks_type or self.spks_type
        return self._load_spks(spks_type)

    @property
    def spks(self):
        return self.get_spks(self.spks_type)

    @property
    def zero_baseline_spks(self) -> bool:
        return self._are_spks_zero_baseline(self.spks_type)

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
    def num_trials(self):
        """Return the number of trials in the session

        Part of SessionToSpkmapProtocol: Needs documentation
        """
        return self.get_value("numTrials")

    def set_spks_type(self, spks_type: str) -> str:
        """Set spks_type, will determine which onefile to load spks data from"""
        self.spks_type = spks_type

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
