"""LocPrediction - a config for analyzing estimates of location from activity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from tqdm import tqdm
from vrAnalysis.sessions import B2Session
from ..registry import PopulationRegistry
from vrAnalysis.processors.placefields import get_placefield, get_placefield_prediction, get_frame_behavior, convert_position_to_bins
from vrAnalysis.processors.placefields import Placefield, FrameBehavior
from vrAnalysis.processors.support import smooth
from vrAnalysis.processors.em import _estep, _mstep, _compute_ce_score
from vrAnalysis.helpers import cross_validate_trials, uniq_val_filter, reliability_loo
from vrAnalysis.metrics import FractionActive
from dimilibi.metrics import mse, measure_r2, measure_rms
from ..pipeline.base import AnalysisConfigBase


def _true_position_bins(frame_behavior: FrameBehavior, placefield: Placefield) -> np.ndarray:
    num_bins = len(placefield.dist_edges) - 1
    env_to_idx = {env: i for i, env in enumerate(placefield.environment)}
    true_env_idx = np.array([env_to_idx[e] for e in frame_behavior.environment])
    true_pos_idx = convert_position_to_bins(frame_behavior.position, placefield.dist_edges, check_invalid=False)
    return true_env_idx * num_bins + true_pos_idx


@dataclass(frozen=True)
class LocPredConfig(AnalysisConfigBase):
    """Configuration for estimating position from neural activity and placefields.

    Parameters
    ----------
    norm_method : str
        Method for normalizing spike counts. Currently only supports "zero-one" (divide by max
        value for each cell).
    speed_threshold : float
        Minimum speed for including timepoints in the analysis.
    num_bins : int
        Number of spatial bins for place field computation.
    train_test_split : tuple[float, float]
        Proportion of trials to use for training and testing, respectively. Must sum to 1.
    smooth_width : float or None
        Gaussian smoothing width for place fields. None means no smoothing.
    reliability_cutoff : float
        Minimum reliability for including cells in the analysis.
    fraction_active_cutoff : float
        Minimum fraction active for including cells in the analysis. This uses the participation ratio
        method on the FractionActive metric.
    objective : str
        E-step objective for position assignment. "mse" uses expanded ||x-y||^2; "angle" uses
        negative cosine similarity (minimum angle between spike vector and placefield vector).
    # uniq_val_window : int
    #     Window size for counting unique values in assigned position bins. This is used to measure the diversity
    #     of assigned positions across EM steps. Larger values will count more bins as the same "unique value".
    # uniq_val_interp_points : int
    #     Number of points to interpolate unique value counts across EM steps.
    """

    schema_version: str = "v1"
    data_config_name: str = "default"

    norm_method: str = "zero-one"
    speed_threshold: float = 1.0
    num_bins: int = 100
    train_test_split: tuple[float, float] = (0.8, 0.2)
    smooth_width: float | None = 0.25
    num_steps: int = 10
    reliability_cutoff: float = 0.1
    fraction_active_cutoff: float = 0.1
    objective: str = "mse"
    display_name: ClassVar[str] = "locprediction"

    @staticmethod
    def _param_grid() -> dict:
        return {}

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"norm_method={self.norm_method}",
            f"speed_threshold={self.speed_threshold}",
            f"num_bins={self.num_bins}",
            f"train_test_split={self.train_test_split}",
            f"smooth_width={self.smooth_width}",
            f"num_steps={self.num_steps}",
            f"reliability_cutoff={self.reliability_cutoff}",
            f"fraction_active_cutoff={self.fraction_active_cutoff}",
            f"objective={self.objective}",
            self.schema_version,
        ]
        return "_".join(parts)

    def _select_rois(self, session: B2Session, spks: np.ndarray, frame_behavior: FrameBehavior, dist_edges: np.ndarray) -> np.ndarray:
        """Select reliable and active ROIs based on leave-one-out reliability and fraction active."""
        _all_trials = get_placefield(
            spks,
            frame_behavior,
            dist_edges,
            average=False,
            use_fast_sampling=True,
            session=session,
        )
        pf_data = np.transpose(_all_trials.placefield, (2, 0, 1))
        idx_reliable = reliability_loo(pf_data) >= self.reliability_cutoff
        fraction_active = (
            FractionActive.compute(
                pf_data,
                activity_axis=2,
                fraction_axis=1,
                activity_method="rms",
                fraction_method="participation",
            )
            >= self.fraction_active_cutoff
        )

        idx_keep_rois = idx_reliable & fraction_active
        return idx_keep_rois

    def _normalize_spks(self, spks: np.ndarray) -> np.ndarray:
        """Normalize spike counts according to the specified method."""
        if self.norm_method == "zero-one":
            norm_value = np.max(spks, axis=0)
            spks = spks / norm_value
        return spks

    def process(self, session: B2Session, registry: PopulationRegistry, return_models: bool = False, verbose: bool = False) -> dict:
        """Run Location Prediction analysis on a session."""
        # Start by defining a train/test split with the frame behavior data
        frame_behavior = get_frame_behavior(session)
        idx_valid = frame_behavior.valid_frames()
        idx_fast = frame_behavior.speed >= self.speed_threshold
        idx_filter = idx_valid & idx_fast
        frame_behavior = frame_behavior.filter(idx_filter)

        # Do train/test split via whole trials (balancing environment trials across folds)
        trial_folds = cross_validate_trials(session.trial_environment, self.train_test_split)
        idx_train = np.isin(frame_behavior.trial, trial_folds[0])
        idx_test = np.isin(frame_behavior.trial, trial_folds[1])

        # Select frame_behavior for train/test timepoints
        frame_behavior_tr = frame_behavior.filter(idx_train)
        frame_behavior_te = frame_behavior.filter(idx_test)

        # Now get calcium data, do normalization
        spks = session.spks[idx_filter][:, session.idx_rois]  # Always filter by session filters (saves computation)
        spks = self._normalize_spks(spks)

        # Divide by train/test split, subselect cells again based on reliability / fraction active on training data
        spks_tr = spks[idx_train]
        spks_te = spks[idx_test]

        dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        idx_keep_rois = self._select_rois(session, spks_tr, frame_behavior_tr, dist_edges)
        spks_tr = spks_tr[:, idx_keep_rois]
        spks_te = spks_te[:, idx_keep_rois]

        # Now let's do a proper estimation procedure:
        # - Estimate probability of each position bin in testing data (using a valid likelhiood computation)
        # - Measure loss in test set prediction of bins from true bins
        # - Both likelihood and loss should be configurable by config params so we can compare different approaches
        # - But since the heavy computation is in the preparation, let's just use a list of params and do all of them
        #   in this same process method and save the results of all valid combos....
        # ---- Including, poisson, exponential, gaussian likelihoods
        # ---- Loss could be cross-entropy, MSE, rank ordering, hinge loss...

        # Claude, go from here!

        # # Make training and testing placefields
        # dist_edges = np.linspace(0, session.env_length[0], self.num_bins + 1)
        # placefield_kwargs = dict(
        #     dist_edges=dist_edges,
        #     speed_threshold=self.speed_threshold,
        #     smooth_width=self.smooth_width,
        #     use_fast_sampling=True,
        #     session=session,
        # )
        # placefield_tr = get_placefield(spks_tr, frame_behavior_tr, average=True, **placefield_kwargs)
        # placefield_te = get_placefield(spks_te, frame_behavior_te, average=True, **placefield_kwargs)

        # num_steps = self.num_steps
        # fb_e = [frame_behavior_tr]
        # pf_m = [placefield_tr]
        # for i in tqdm(range(num_steps - 1), desc="EM steps", disable=not verbose):
        #     fb_next = _estep(spks_tr, pf_m[-1], fb_e[-1], objective=self.objective)
        #     pf_next = _mstep(spks_tr, fb_next, dist_edges, self.smooth_width)
        #     fb_e.append(fb_next)
        #     pf_m.append(pf_next)

        # # Measure improvement in performance of EM model over iterations
        # step_mse = []
        # step_r2 = []
        # step_rms = []
        # for step in tqdm(range(num_steps), desc="Performance steps", disable=not verbose):
        #     _step_pred = get_placefield_prediction(pf_m[step], fb_e[step])[0]
        #     step_mse.append(mse(_step_pred, spks_tr, dim=0, reduce="mean"))
        #     step_r2.append(measure_r2(_step_pred, spks_tr, dim=0, reduce="mean"))
        #     step_rms.append(measure_rms(_step_pred, spks_tr, dim=0, reduce="mean"))

        # best_step = np.argmin(step_mse)

        # # Compare EM model to null model (empirical placefield) on testing timepoints
        # _test_pred = get_placefield_prediction(pf_m[best_step], frame_behavior_te)[0]
        # _null_pred = get_placefield_prediction(placefield_te, frame_behavior_te)[0]
        # em_test_r2 = measure_r2(_test_pred, spks_te, dim=0, reduce="mean")
        # em_null_r2 = measure_r2(_null_pred, spks_te, dim=0, reduce="mean")
        # em_test_rms = measure_rms(_test_pred, spks_te, dim=0, reduce="mean")
        # em_null_rms = measure_rms(_null_pred, spks_te, dim=0, reduce="mean")
        # em_test_mse = mse(_test_pred, spks_te, dim=0, reduce="mean")
        # em_null_mse = mse(_null_pred, spks_te, dim=0, reduce="mean")

        # # Cross-entropy: empirical placefield decoded on test data using objective
        # true_bins_te = _true_position_bins(frame_behavior_te, placefield_tr, self.num_bins)
        # pred_bins_te, ce_score = _compute_ce_score(spks_te, placefield_tr, true_bins_te, self.objective)

        # # Measure diversity of assigned positions
        # step_uniq_val_count = []
        # step_pos_est_delta = []
        # step_pos_envswap_fraction = []
        # uniq_val_centers = None
        # pos_by_env0 = fb_e[0].position_by_environment()
        # pbe_bins0 = convert_position_to_bins(pos_by_env0, pf_m[0].dist_edges, check_invalid=False) * 1.0
        # pbe_bins0[np.isnan(pos_by_env0)] = np.nan
        # offset = np.arange(pbe_bins0.shape[0])[:, None] * (np.nanmax(pbe_bins0) + 1)
        # pbe_bins0 = pbe_bins0 + offset

        # for step in tqdm(range(num_steps), desc="Diversity steps", disable=not verbose):
        #     _step_pos_by_env = fb_e[step].position_by_environment()
        #     pbe_bins = convert_position_to_bins(_step_pos_by_env, pf_m[step].dist_edges, check_invalid=False) * 1.0
        #     pbe_bins[np.isnan(_step_pos_by_env)] = np.nan
        #     pbe_bins = pbe_bins + offset
        #     pbe_bins_1d = np.nansum(pbe_bins, axis=0)
        #     _uniq_val, _uniq_centers = uniq_val_filter(pbe_bins_1d, width=self.uniq_val_window)
        #     if uniq_val_centers is None:
        #         uniq_val_centers = _uniq_centers
        #     else:
        #         if not np.array_equal(uniq_val_centers, _uniq_centers):
        #             raise ValueError("Unique value centers should be the same across steps")

        #     pbe_difference = pbe_bins - pbe_bins0
        #     _idx_nan_diff = np.all(np.isnan(pbe_difference), axis=0)
        #     pbe_difference = np.abs(np.nansum(pbe_difference, axis=0))
        #     pbe_difference[_idx_nan_diff] = np.nan

        #     # filter pbe_difference
        #     pbe_difference_smooth = smooth(pbe_difference, range(len(pbe_difference)), width=self.uniq_val_window)
        #     idx_nan_filtered = smooth(_idx_nan_diff * 1.0, range(len(pbe_difference)), width=self.uniq_val_window)

        #     steps_norm = np.linspace(0, 1, len(_uniq_centers))
        #     steps_full = np.linspace(0, 1, len(pbe_difference_smooth))
        #     xi = np.linspace(0, 1, self.uniq_val_interp_points)
        #     uniq_val_interp = np.interp(xi, steps_norm, _uniq_val)
        #     pbe_difference_smooth_interp = np.interp(xi, steps_full, pbe_difference_smooth)
        #     idx_nan_filtered_interp = np.interp(xi, steps_full, idx_nan_filtered)
        #     step_uniq_val_count.append(uniq_val_interp)
        #     step_pos_est_delta.append(pbe_difference_smooth_interp)
        #     step_pos_envswap_fraction.append(idx_nan_filtered_interp)

        # results = dict(
        #     em_test_r2=em_test_r2,
        #     em_null_r2=em_null_r2,
        #     em_test_rms=em_test_rms,
        #     em_null_rms=em_null_rms,
        #     em_test_mse=em_test_mse,
        #     em_null_mse=em_null_mse,
        #     step_mse=np.array(step_mse),
        #     step_r2=np.array(step_r2),
        #     step_rms=np.array(step_rms),
        #     step_uniq_val_count=np.stack(step_uniq_val_count),
        #     step_pos_est_delta=np.stack(step_pos_est_delta),
        #     step_pos_envswap_fraction=np.stack(step_pos_envswap_fraction),
        #     true_position_bins_te=true_bins_te,
        #     pred_position_bins_te=pred_bins_te,
        #     ce_score=ce_score,
        # )

        # if return_models:
        #     extras = dict(
        #         frame_behavior_est=fb_e,
        #         placefield_est=pf_m,
        #         idx_keep_rois=idx_keep_rois,
        #     )
        #     return results, extras

        return None
