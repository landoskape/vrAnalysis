from dataclasses import dataclass, field
import numpy as np
from .. import helpers
from ..sessions.base import SessionData
import json
from typing import Dict, Any


"""
This is what is performed in placeCellSingleSession when it loads data to get spkmaps

        self.trial_envnum = self.vrexp.loadone("trials.environmentIndex")
        self.environments = np.unique(self.trial_envnum)
        self.numEnvironments = len(self.environments)
        
        if distStep is not None:
            self.distStep = distStep
        if speedThreshold is not None:
            self.speedThreshold = speedThreshold
        if full_trial_flexibility is not None:
            self.full_trial_flexibility = full_trial_flexibility

        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {
            "distStep": self.distStep,
            "speedThreshold": self.speedThreshold,
            "speedSmoothing": self.smoothWidth,
            "get_spkmap": False,
        }
        self.occmap, self.speedmap, _, _, self.sample_counts, self.distedges = helpers.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)
        self.distcenters = helpers.edge2center(self.distedges)

        self.numTrials = self.occmap.shape[0]

        # find out which trials the mouse explored the whole environment
        if self.full_trial_flexibility is None:
            # if full trial flexiblity is None, then they need to have visited every bin
            idx_to_required_bins = np.arange(self.occmap.shape[1])
        else:
            start_idx = np.where(self.distedges >= self.full_trial_flexibility)[0][0]
            end_idx = np.where(self.distedges <= self.distedges[-1] - self.full_trial_flexibility)[0][-1]
            idx_to_required_bins = np.arange(start_idx, end_idx)

        self.boolFullTrials = np.all(~np.isnan(self.occmap[:, idx_to_required_bins]), axis=1)
        self.idxFullTrials = np.where(self.boolFullTrials)[0]
        self.idxFullTrialEachEnv = [np.where(self.boolFullTrials & (self.trial_envnum == env))[0] for env in self.environments]

        # update onefile if using a different measure of activity
        if onefile is not None:
            self.onefile = onefile

        # update analysis parameters if requested
        if distStep is not None:
            self.distStep = distStep
        if speedThreshold is not None:
            self.speedThreshold = speedThreshold
        if numcv is not None:
            self.numcv = numcv
        if keep_planes is not None:
            self.keep_planes = keep_planes
        if full_trial_flexibility is not None:
            self.full_trial_flexibility = full_trial_flexibility

        _ = self.get_plane_idx(keep_planes=self.keep_planes)

        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {
            "distStep": self.distStep,
            "onefile": self.onefile,
            "speedThreshold": self.speedThreshold,
            "standardizeSpks": self.standardizeSpks,
            "idxROIs": self.idxUseROI if not self.use_all_rois else None,
            "speedSmoothing": self.smoothWidth,
        }
        self.occmap, self.speedmap, _, self.rawspkmap, self.sample_counts, self.distedges = helpers.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)

        self.distcenters = helpers.edge2center(self.distedges)

        self.numTrials = self.occmap.shape[0]

        self.boolFullTrials, self.idxFullTrials, self.idxFullTrialEachEnv = self._return_trial_indices(
            self.occmap, self.distedges, self.full_trial_flexibility
        )

        # report that data has been loaded
        self.dataloaded = True

        # measure reliability
        self.measure_reliability(new_split=new_split, with_test=with_test)
"""


@dataclass
class SpkmapProcessor:
    """Class for processing and caching spike maps from session data

    NOTES ON ENGINEERING:
    I want the variables required for processing spkmaps to be properties (@property)
    that have hidden attributes for caching. Therefore, we can use the property method
    to get the attribute and each property method can do whatever processing is needed
    for that attribute. (Uh, duh). Time to get modern. lol.
    """

    session: SessionData
    data_cache: dict = field(default_factory=dict, repr=False, init=False)

    def _create_param_hash(self, params: Dict[str, Any]) -> str:
        """Create a deterministic hash from parameters.

        Converts parameters to a sorted, canonical string representation
        that can be used to check if parameters have changed.

        Args:
            params: Dictionary of processing parameters

        Returns:
            String hash representing the parameter set
        """
        # Define default parameters
        default_params = {
            "dist_step": 2,
            "speed_threshold": 2,
            "speed_smoothing": 5,
            "full_trial_flexibility": None,
            "standardize_spks": False,
            "use_all_rois": False,
        }

        # Update defaults with provided params
        full_params = default_params.copy()
        full_params.update(params)

        # Sort parameters by key for consistency
        param_items = sorted(full_params.items())

        # Create string representation
        param_str = json.dumps(param_items)

        return param_str

    def check_params_match(self, new_params: Dict[str, Any]) -> bool:
        """Check if new parameters match those used in saved data.

        Args:
            new_params: Dictionary of processing parameters to check

        Returns:
            True if parameters match saved parameters, False otherwise
        """
        try:
            saved_hash = self.session.loadone("spkmaps.param_hash")
            new_hash = self._create_param_hash(new_params)
            return saved_hash == new_hash
        except ValueError:
            return False

    def register_spkmaps(self, params: dict, save_to_one: bool = False) -> None:
        """Register spkmaps from session data.

        Replicates placeCellSingleSession.load_data() functionality by:
        1. Loading trial environment data
        2. Computing occupancy/speed maps
        3. Computing spike maps
        4. Identifying full trials
        5. Saving processed data to session's onedata

        Args:
            params: Dictionary containing processing parameters:
                - dist_step: Distance bin size (cm)
                - speed_threshold: Minimum speed threshold (cm/s)
                - speed_smoothing: Smoothing width for speed (cm)
                - full_trial_flexibility: Distance from track ends that must be visited (cm)
                - standardize_spks: Whether to z-score spikes
                - use_all_rois: Whether to use all ROIs or only selected ones
            save_to_one: Whether to save results to onedata files
        """
        # Load trial environment data
        trial_envnum = self.session.loadone("trials.environmentIndex")
        environments = np.unique(trial_envnum)

        # First pass - get behavior maps without spikes
        kwargs = {
            "distStep": params.get("dist_step", 2),
            "speedThreshold": params.get("speed_threshold", 2),
            "speedSmoothing": params.get("speed_smoothing", 5),
            "get_spkmap": False,
        }

        # Get behavior maps
        occmap, speedmap, _, _, sample_counts, distedges = helpers.getBehaviorAndSpikeMaps(self.session, **kwargs)
        distcenters = helpers.edge2center(distedges)

        # Identify full trials
        full_trial_flexibility = params.get("full_trial_flexibility", None)
        if full_trial_flexibility is None:
            # Must visit every bin
            idx_required = np.arange(occmap.shape[1])
        else:
            # Must visit bins within flexibility distance of track ends
            start_idx = np.where(distedges >= full_trial_flexibility)[0][0]
            end_idx = np.where(distedges <= distedges[-1] - full_trial_flexibility)[0][-1]
            idx_required = np.arange(start_idx, end_idx)

        bool_full_trials = np.all(~np.isnan(occmap[:, idx_required]), axis=1)
        idx_full_trials = np.where(bool_full_trials)[0]
        idx_full_trial_each_env = [np.where(bool_full_trials & (trial_envnum == env))[0] for env in environments]

        # Second pass - get spike maps
        kwargs.update(
            {
                "get_spkmap": True,
                "standardizeSpks": params.get("standardize_spks", False),
                "idxROIs": None if params.get("use_all_rois", False) else self.session.get_value("use_roi_idx"),
            }
        )

        # Get spike maps
        occmap, speedmap, _, rawspkmap, sample_counts, distedges = helpers.getBehaviorAndSpikeMaps(self.session, **kwargs)

        # Registered data
        registered_data = dict(
            trial_envnum=trial_envnum,
            environments=environments,
            distedges=distedges,
            occmap=occmap,
            speedmap=speedmap,
            rawspkmap=rawspkmap,
            bool_full_trials=bool_full_trials,
            idx_full_trials=idx_full_trials,
        )

        self.data_cache.update(registered_data)

        if save_to_one:
            # Create parameter hash
            param_hash = self._create_param_hash(params)

            # Save processed data to onedata
            self.session.saveone(occmap, "spkmaps.occupancy")
            self.session.saveone(speedmap, "spkmaps.speed")
            self.session.saveone(rawspkmap, "spkmaps.raw")
            self.session.saveone(sample_counts, "spkmaps.sample_counts")
            self.session.saveone(distedges, "spkmaps.dist_edges")
            self.session.saveone(bool_full_trials, "spkmaps.bool_full_trials")
            self.session.saveone(idx_full_trials, "spkmaps.idx_full_trials")
            self.session.saveone(idx_full_trial_each_env, "spkmaps.idx_full_trial_each_env")
            self.session.saveone(param_hash, "spkmaps.param_hash")
