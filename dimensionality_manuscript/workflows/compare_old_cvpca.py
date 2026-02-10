from typing import Optional, Dict, Any
from pathlib import Path
import itertools
import gc
import joblib
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from syd import Viewer
from vrAnalysis.database import get_database
from vrAnalysis.helpers import cross_validate_trials
from vrAnalysis.helpers.plotting import errorPlot
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield
from dimilibi.cvpca import CVPCA, LegacyCVPCA
from dimensionality_manuscript.registry import PopulationRegistry

# Import old method helpers
from _old_vrAnalysis.analysis import placeCellMultiSession, placeCellSingleSession
from _old_vrAnalysis import tracking


# get session database
sessiondb = get_database("vrSessions")

# get population registry and models
registry = PopulationRegistry()


def nanmax(tensor: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """Get max value handling NaNs."""
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim).values
    return output


def normalize_by_max(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Normalize tensor by max value along dimension."""
    max_value = nanmax(tensor, dim=dim, keepdim=True)
    max_value[max_value == 0] = 1
    return tensor / max_value


def get_legacy_cvpca(session: B2Session, best_env_idx: Optional[int] = None) -> Dict[str, Any]:
    """
    Get original cvPCA data computed with the legacy old_VrAnalysis code from saved pickle files.

    This function loads the pre-computed cvPCA data that was saved by the old variance_structure.py
    scripts. The data is organized by mouse, so it finds the session within the mouse's saved data.

    Parameters
    ----------
    session : B2Session
        Session to get cvPCA data for
    best_env_idx: int
        If provided, will filter the cvPCA data for the best environment

    Returns
    -------
    dict
        Dictionary containing:
        - 'cv_by_env_all': List of cvPCA spectra for each environment (all cells)
        - 'cv_by_env_rel': List of cvPCA spectra for each environment (reliable cells)
        - 'cv_across_all': cvPCA spectrum for concatenated environments (all cells)
        - 'cv_across_rel': cvPCA spectrum for concatenated environments (reliable cells)
        - 'session_idx': Index of this session in the saved data
        - 'session_name': Name of this session
        - 'all_data': Full spectra_data dictionary (for reference)
    """
    # Convert B2Session to old session format to get session name
    vrexp = session.to_old_session()
    session_name = vrexp.sessionPrint()

    # Get mouse name
    mouse_name = session.mouse_name

    # Create tracker and placeCellMultiSession to access saveDirectory
    track = tracking.tracker(mouse_name)
    pcm = placeCellMultiSession(track, autoload=False)

    # Path to saved spectra data
    spectra_file = pcm.saveDirectory("temp") / f"{mouse_name}_spectra_data.pkl"

    if not spectra_file.exists():
        raise FileNotFoundError(
            f"Legacy cvPCA data not found at {spectra_file}. " f"Run the old variance_structure.py scripts first to generate this data."
        )

    # Load the saved data
    with open(spectra_file, "rb") as f:
        spectra_data = pickle.load(f)

    # Find this session in the saved data
    try:
        session_idx = spectra_data["names"].index(session_name)
    except ValueError:
        available_sessions = "\n".join(spectra_data["names"])
        raise ValueError(f"Session '{session_name}' not found in saved data for mouse '{mouse_name}'. " f"Available sessions:\n{available_sessions}")

    # Extract cvPCA data for this session
    result = {
        "cv_by_env_all": spectra_data["cv_by_env_all"][session_idx],
        "cv_by_env_rel": spectra_data["cv_by_env_rel"][session_idx],
        "cv_across_all": spectra_data["cv_across_all"][session_idx],
        "cv_across_rel": spectra_data["cv_across_rel"][session_idx],
        "session_idx": session_idx,
        "session_name": session_name,
        "all_data": spectra_data,  # Keep full data for reference
    }

    if best_env_idx is not None:
        result["cv_by_env_all"] = result["cv_by_env_all"][best_env_idx]
        result["cv_by_env_rel"] = result["cv_by_env_rel"][best_env_idx]

    return result


def cvpca_integrated_comparison(
    session: B2Session,
    spks_type: SpksTypes,
    num_bins: int = 100,
    smooth_width: Optional[float] = None,
    normalize: bool = True,
    center: bool = False,
    shuffle_fraction: float = 0.0,
) -> Dict[str, Any]:

    # Find best environment (most trials)
    num_per_env = {i: np.sum(session.trial_environment == i) for i in session.environments}
    best_env = max(num_per_env, key=num_per_env.get)
    best_env_idx = np.where(session.environments == best_env)[0][0]

    env_length = session.env_length[0]
    dist_edges = np.linspace(0, env_length, num_bins + 1)
    population, frame_behavior = registry.get_population(session, spks_type)
    data = np.array(population.data[session.idx_rois][:, population.idx_samples]).T

    trial_folds = cross_validate_trials(session.trial_environment, [1, 1, 1])
    placefields = [
        get_placefield(data, frame_behavior, dist_edges, trial_filter=trial_fold, smooth_width=smooth_width, zero_to_nan=True)
        for trial_fold in trial_folds
    ]
    pf_data = [pf.placefield[best_env_idx].T for pf in placefields]

    #  Filter out positions with zero counts
    if any([np.any(pf.count[best_env_idx] == 0) for pf in placefields]):
        bad_locations = [np.where(pf.count[best_env_idx] == 0)[0] for pf in placefields]
        bad_locations = np.unique(np.concatenate(bad_locations))
        good_idx = np.setdiff1d(np.arange(num_bins), bad_locations)
        pf_data = [pf_pf[:, good_idx] for pf_pf in pf_data]

    if normalize:
        _max_neuron = np.max(np.concatenate(pf_data, axis=1), axis=1, keepdims=True)
        _max_neuron[_max_neuron == 0] = 1
        pf_data = [pfd / _max_neuron for pfd in pf_data]

    pf_data = [torch.from_numpy(pfd) for pfd in pf_data]

    org_covariances = []
    leg_covariances = []
    true_leg_covariances = []
    for ref_fold in range(len(trial_folds)):
        c_repeat_0 = pf_data[ref_fold]
        c_repeat_1 = pf_data[(ref_fold + 1) % len(trial_folds)]
        c_repeat_2 = pf_data[(ref_fold + 2) % len(trial_folds)]

        # Original CVPCA
        cvpca = CVPCA(use_svd=not center).fit(c_repeat_0)
        org_covariance_1 = cvpca.score(c_repeat_0, c_repeat_1)
        org_covariance_2 = cvpca.score(c_repeat_0, c_repeat_2)
        org_covariance = np.mean(np.stack([org_covariance_1, org_covariance_2], axis=0), axis=0)
        org_covariances.append(org_covariance)

        # Legacy CVPCA
        leg_cvpca = LegacyCVPCA(use_svd=not center, shuffle_fraction=shuffle_fraction)
        leg_covariance = leg_cvpca.fit_score(c_repeat_0, c_repeat_1)
        leg_covariances.append(leg_covariance)

        # True Legacy CVPCA
        true_leg_cvpca = LegacyCVPCA(use_svd=not center, shuffle_fraction=shuffle_fraction, true_legacy=True)
        true_leg_covariance = true_leg_cvpca.fit_score(c_repeat_0, c_repeat_1)
        true_leg_covariances.append(true_leg_covariance)

    results = dict(
        org_covariances=np.mean(np.stack(org_covariances, axis=0), axis=0),
        leg_covariances=np.mean(np.stack(leg_covariances, axis=0), axis=0),
        true_leg_covariances=np.mean(np.stack(true_leg_covariances, axis=0), axis=0),
        best_env_idx=best_env_idx,
    )
    return results


def gather_cvpca_comparisons(
    session: B2Session,
    spks_type: SpksTypes,
    num_bins: int = 100,
    smooth_width: Optional[float] = None,
    speed_threshold: Optional[float] = None,
    pre_average: bool = True,
    normalize: bool = True,
    center: bool = False,
    shuffle_fraction: float = 0.5,
) -> Dict[str, Any]:
    results = dict()
    intermediate_results = dict()

    # Find best environment (most trials)
    num_per_env = {i: np.sum(session.trial_environment == i) for i in session.environments}
    best_env = max(num_per_env, key=num_per_env.get)
    best_env_idx = np.where(session.environments == best_env)[0][0]

    # Get saved legacy cvPCA data (if available)
    try:
        legacy_cvpca_data = get_legacy_cvpca(session, best_env_idx=best_env_idx)
        saved_legacy_covariances = legacy_cvpca_data["cv_by_env_all"]
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load legacy cvPCA data for session {session.session_print()}: {e}")
        saved_legacy_covariances = None

    env_length = session.env_length[0]
    dist_edges = np.linspace(0, env_length, num_bins + 1)
    population, frame_behavior = registry.get_population(session, spks_type)
    data = np.array(population.data[session.idx_rois][:, population.idx_samples]).T

    trials_in_fb = np.unique(frame_behavior.trial).astype(int)
    trial_environment_in_fb = session.trial_environment[trials_in_fb]
    trial_folds = cross_validate_trials(trial_environment_in_fb, [1, 1, 1])
    placefields = [
        get_placefield(
            data,
            frame_behavior,
            dist_edges,
            speed_threshold=speed_threshold,
            average=pre_average,
            trial_filter=trial_fold,
            smooth_width=smooth_width,
            zero_to_nan=True,
        )
        for trial_fold in trial_folds
    ]
    placefields_fast = [
        get_placefield(
            data,
            frame_behavior,
            dist_edges,
            speed_threshold=speed_threshold,
            average=pre_average,
            trial_filter=trial_fold,
            smooth_width=smooth_width,
            zero_to_nan=True,
            use_fast_sampling=True,
            by_sample_duration=False,
            session=session,
        )
        for trial_fold in trial_folds
    ]
    if pre_average:
        # Then we've already averaged over trials within environment
        # So the first axis is simply the average for that environment
        pf_data = [pf.placefield[best_env_idx].T for pf in placefields]
        pf_data_fast = [pf.placefield[best_env_idx].T for pf in placefields_fast]
    else:
        # If not averaged, we need to filter out the trials for the best environment
        _idx_trials_best_env = [np.where(session.trial_environment[pf.trials] == best_env)[0].astype(int) for pf in placefields]
        pf_data = [np.transpose(pf.placefield[_itbe], (2, 0, 1)) for pf, _itbe in zip(placefields, _idx_trials_best_env)]
        pf_data_fast = [np.transpose(pf.placefield[_itbe], (2, 0, 1)) for pf, _itbe in zip(placefields_fast, _idx_trials_best_env)]
        intermediate_results["pf_data_from_placefields"] = pf_data
        intermediate_results["pf_data_from_placefields_fast"] = pf_data_fast
        pf_data = [np.nanmean(pfd, axis=1) for pfd in pf_data]
        pf_data_fast = [np.nanmean(pfd, axis=1) for pfd in pf_data_fast]

    # Now do the same for old version of vrAnalysis session code
    dist_step = dist_edges[1] - dist_edges[0]
    vrexp = session.to_old_session()
    pcss = placeCellSingleSession(vrexp, distStep=dist_step, speedThreshold=speed_threshold, autoload=True, keep_planes=[0, 1, 2, 3, 4])
    spkmaps = [
        pcss.get_spkmap(envnum=best_env, trials=np.array(trial_fold), smooth=smooth_width, average=pre_average, pop_nan=False)[0][session.idx_rois]
        for trial_fold in trial_folds
    ]

    intermediate_results["spkmap_from_pcss"] = spkmaps

    if not pre_average:
        # Get trials kept in placefields
        kept_trials = [pf.trials[_itbe] for pf, _itbe in zip(placefields, _idx_trials_best_env)]
        _idx_trials_kept_spkmap = [np.where(session.trial_environment[np.array(trial_fold)] == best_env)[0].astype(int) for trial_fold in trial_folds]
        kept_in_spkmap = [np.array(trial_fold)[itks] for trial_fold, itks in zip(trial_folds, _idx_trials_kept_spkmap)]
        idx_filter_spkmap = [np.isin(kis, kt) for kis, kt in zip(kept_in_spkmap, kept_trials)]
        spkmaps = [np.nanmean(spkmap[:, ifs], axis=1) for spkmap, ifs in zip(spkmaps, idx_filter_spkmap)]

    #  Filter out positions with zero counts
    if any([np.any(pf.count[best_env_idx] == 0) for pf in placefields]):
        nan_pos_pf_data = np.any(np.stack([np.all(np.isnan(pfd), axis=0) for pfd in pf_data]), axis=0)
        nan_pos_pf_data_fast = np.any(np.stack([np.all(np.isnan(pfd), axis=0) for pfd in pf_data_fast]), axis=0)
        nan_pos_spkmaps = np.any(np.stack([np.all(np.isnan(spkmap), axis=0) for spkmap in spkmaps]), axis=0)
        bad_locations = np.where(nan_pos_pf_data | nan_pos_pf_data_fast | nan_pos_spkmaps)[0]
        good_idx = np.setdiff1d(np.arange(num_bins), bad_locations)
        pf_data = [pf_pf[:, good_idx] for pf_pf in pf_data]
        pf_data_fast = [pf_pf_fast[:, good_idx] for pf_pf_fast in pf_data_fast]
        spkmaps = [spkmap[:, good_idx] for spkmap in spkmaps]

    if normalize:
        _max_neuron = np.max(np.concatenate(pf_data, axis=1), axis=1, keepdims=True)
        _max_neuron[_max_neuron == 0] = 1
        pf_data = [pfd / _max_neuron for pfd in pf_data]

        _max_neuron = np.max(np.concatenate(pf_data_fast, axis=1), axis=1, keepdims=True)
        _max_neuron[_max_neuron == 0] = 1
        pf_data_fast = [pfd_fast / _max_neuron for pfd_fast in pf_data_fast]

        _max_neuron = np.max(np.concatenate(spkmaps, axis=1), axis=1, keepdims=True)
        _max_neuron[_max_neuron == 0] = 1
        spkmaps = [spkmap / _max_neuron for spkmap in spkmaps]

    for pfd, prefix in [(pf_data, "pf_data"), (pf_data_fast, "pf_data_fast"), (spkmaps, "spkmap")]:
        pfd = [torch.from_numpy(pf) for pf in pfd]

        org_covariances = []
        leg_covariances = []
        for ref_fold in range(len(trial_folds)):
            c_repeat_0 = pfd[ref_fold]
            c_repeat_1 = pfd[(ref_fold + 1) % len(trial_folds)]
            c_repeat_2 = pfd[(ref_fold + 2) % len(trial_folds)]

            # Original CVPCA
            cvpca = CVPCA(use_svd=not center).fit(c_repeat_0)
            org_covariance_1 = cvpca.score(c_repeat_0, c_repeat_1)
            org_covariance_2 = cvpca.score(c_repeat_0, c_repeat_2)
            org_covariance = np.mean(np.stack([org_covariance_1, org_covariance_2], axis=0), axis=0)
            org_covariances.append(org_covariance)

            # Legacy CVPCA
            leg_cvpca = LegacyCVPCA(use_svd=not center, shuffle_fraction=shuffle_fraction)
            leg_covariance = leg_cvpca.fit_score(c_repeat_0, c_repeat_1)
            leg_covariances.append(leg_covariance)

        results[f"{prefix}_org_covariances"] = np.mean(np.stack(org_covariances, axis=0), axis=0)
        results[f"{prefix}_leg_covariances"] = np.mean(np.stack(leg_covariances, axis=0), axis=0)

    # Use saved legacy cvPCA data (same for both, so only store once)
    results["true_leg_covariances"] = saved_legacy_covariances

    return results, intermediate_results


def get_filepath(
    session: B2Session,
    num_bins: int = 100,
    smooth_width: Optional[float] = None,
    speed_threshold: Optional[float] = None,
    pre_average: bool = True,
    normalize: bool = True,
    center: bool = False,
    shuffle_fraction: float = 0.5,
) -> Path:
    """Get the filepath for the results of a session."""
    name = session.session_print(joinby=".")
    # Include parameters in filename to distinguish different configurations
    param_parts = [f"bins{num_bins}"]
    if smooth_width is not None:
        param_parts.append(f"smooth{smooth_width}")
    if speed_threshold is not None:
        # Format as float to ensure consistent naming (e.g., 1.0 not 1)
        param_parts.append(f"speed{float(speed_threshold)}")
    if not pre_average:
        param_parts.append("noavg")
    if not normalize:
        param_parts.append("nonorm")
    if center:
        param_parts.append("center")
    if shuffle_fraction != 0.5:
        param_parts.append(f"shuffle{shuffle_fraction}")
    if param_parts:
        name += "_" + "_".join(param_parts)
    return registry.registry_paths.compare_cvpca_path / f"{name}.pkl"


def _process_single_session(
    session: B2Session,
    spks_type: SpksTypes,
    num_bins: int,
    smooth_width: Optional[float],
    speed_threshold: Optional[float],
    pre_average: bool,
    normalize: bool,
    center: bool,
    shuffle_fraction: float,
    force_remake: bool,
) -> dict:
    """Process a single session - designed to be parallelized.

    This function handles the computation for one session. File checking is done
    in the main process to avoid race conditions, but we double-check here in
    case another worker created the file.
    """
    results_path = get_filepath(
        session,
        num_bins=num_bins,
        smooth_width=smooth_width,
        speed_threshold=speed_threshold,
        pre_average=pre_average,
        normalize=normalize,
        center=center,
        shuffle_fraction=shuffle_fraction,
    )

    # Double-check if file exists (another worker might have created it)
    if not force_remake and results_path.exists():
        return {"status": "skipped", "session": session.session_print()}

    try:
        results, _ = gather_cvpca_comparisons(
            session,
            spks_type=spks_type,
            num_bins=num_bins,
            smooth_width=smooth_width,
            speed_threshold=speed_threshold,
            pre_average=pre_average,
            normalize=normalize,
            center=center,
            shuffle_fraction=shuffle_fraction,
        )

        joblib.dump(results, results_path)
        return {"status": "success", "session": session.session_print()}
    except Exception as e:
        error_msg = f"Error processing session {session.session_print()}: {e}"
        print(error_msg)
        return {"status": "error", "session": session.session_print(), "error": str(e)}
    finally:
        # Clean up session cache and GPU memory
        session.clear_cache()
        torch.cuda.empty_cache()
        gc.collect()


# Hardcoded parameters
process_sessions = False
force_remake = False
clear_cache = False
validate_results = False
check_sessions = False

# See below for completed runs
n_jobs = 4  # Switch to 4 for parallel processing (1 for debugging)
spks_type = "oasis"
num_bins = 100
smooth_width_list = [None, 0.1, 1.0]
speed_threshold_list = [1.0]
pre_average_list = [False, True]
normalize_list = [True]
center_list = [True]
shuffle_fraction_list = [0.5]

if __name__ == "__main__":
    # Iterate through all parameter combinations
    param_combinations = list(
        itertools.product(
            smooth_width_list,
            speed_threshold_list,
            pre_average_list,
            normalize_list,
            center_list,
            shuffle_fraction_list,
        )
    )

    total_combinations = len(param_combinations)
    print(f"Processing {total_combinations} parameter combinations...")

    for combo_idx, (smooth_width, speed_threshold, pre_average, normalize, center, shuffle_fraction) in enumerate(
        tqdm(param_combinations, desc="Parameter combinations")
    ):
        print(f"\n--- Parameter combination {combo_idx + 1}/{total_combinations} ---")
        print(
            f"  smooth_width={smooth_width}, speed_threshold={speed_threshold}, "
            f"pre_average={pre_average}, normalize={normalize}, "
            f"center={center}, shuffle_fraction={shuffle_fraction}"
        )

        # Collect all sessions that need processing for this parameter combination
        sessions_to_process = []
        for session in tqdm(sessiondb.iter_sessions(imaging=True), desc="Checking sessions", leave=False):
            results_path = get_filepath(
                session,
                num_bins=num_bins,
                smooth_width=smooth_width,
                speed_threshold=speed_threshold,
                pre_average=pre_average,
                normalize=normalize,
                center=center,
                shuffle_fraction=shuffle_fraction,
            )

            # Pre-check: clear cache if requested
            if clear_cache and results_path.exists():
                results_path.unlink()

            # Pre-check: validate results if requested
            if (validate_results or check_sessions) and results_path.exists():
                try:
                    loaded_results = joblib.load(results_path)
                    if loaded_results is not None:
                        expected_result_keys = [
                            "pf_data_org_covariances",
                            "pf_data_leg_covariances",
                            "pf_data_fast_org_covariances",
                            "pf_data_fast_leg_covariances",
                            "spkmap_org_covariances",
                            "spkmap_leg_covariances",
                            "true_leg_covariances",
                        ]
                        results_valid = all(key in loaded_results for key in expected_result_keys)
                        if results_valid and check_sessions:
                            # print(f"  Results valid for session {session.session_print()}")
                            pass
                        if not results_valid:
                            if validate_results:
                                results_path.unlink()
                            if check_sessions:
                                print(f"  Results invalid for session {session.session_print()}")
                                pass

                except Exception:
                    if results_path.exists() and validate_results:
                        results_path.unlink()
                    if check_sessions:
                        print(f"  Error loading results for session {session.session_print()}")

            else:
                if check_sessions:
                    print(f"  Results not found for session {session.session_print()}")

            # Add to processing list if needed
            if process_sessions and (force_remake or not results_path.exists()):
                sessions_to_process.append(session)

        # Process sessions in parallel for this parameter combination
        if sessions_to_process:
            num_workers = n_jobs if n_jobs > 0 else "all"
            print(f"  Processing {len(sessions_to_process)} sessions with {num_workers} workers...")
            results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
                joblib.delayed(_process_single_session)(
                    session,
                    spks_type,
                    num_bins,
                    smooth_width,
                    speed_threshold,
                    pre_average,
                    normalize,
                    center,
                    shuffle_fraction,
                    force_remake,
                )
                for session in tqdm(sessions_to_process, desc="Comparing CVPCA", leave=False)
            )

            # Report summary for this parameter combination
            success_count = sum(1 for r in results if r["status"] == "success")
            error_count = sum(1 for r in results if r["status"] == "error")
            skipped_count = sum(1 for r in results if r["status"] == "skipped")
            print(f"  Completed: {success_count} successful, {error_count} errors, {skipped_count} skipped")
        else:
            print(f"  All sessions already processed for this parameter combination.")

    print("\nDone!")


class CVPCAComparisonViewer(Viewer):
    """
    Syd-based viewer for cvPCA comparison results.
    """

    def __init__(
        self,
        smooth_width_list=None,
        speed_threshold_list=None,
        pre_average_list=None,
        normalize_list=None,
        center_list=None,
        shuffle_fraction_list=None,
        num_bins=100,
        spks_type="oasis",
    ):
        self.num_bins = num_bins
        self.spks_type = spks_type

        # Get parameter lists from module if not provided
        import dimensionality_manuscript.workflows.compare_old_cvpca as compare_module

        if smooth_width_list is None:
            smooth_width_list = getattr(compare_module, "smooth_width_list", [None, 0.1, 1.0])
        if speed_threshold_list is None:
            speed_threshold_list = getattr(compare_module, "speed_threshold_list", [1.0])
        if pre_average_list is None:
            pre_average_list = getattr(compare_module, "pre_average_list", [False, True])
        if normalize_list is None:
            normalize_list = getattr(compare_module, "normalize_list", [True])
        if center_list is None:
            center_list = getattr(compare_module, "center_list", [True])
        if shuffle_fraction_list is None:
            shuffle_fraction_list = getattr(compare_module, "shuffle_fraction_list", [0.5])

        # Store parameter lists
        self.smooth_width_list = smooth_width_list
        self.speed_threshold_list = speed_threshold_list
        self.pre_average_list = pre_average_list
        self.normalize_list = normalize_list
        self.center_list = center_list
        self.shuffle_fraction_list = shuffle_fraction_list

        # Get default values
        self.default_smooth_width = smooth_width_list[0] if smooth_width_list else None
        self.default_speed_threshold = speed_threshold_list[0] if speed_threshold_list else 1.0
        self.default_pre_average = pre_average_list[0] if pre_average_list else True
        self.default_normalize = normalize_list[0] if normalize_list else True
        self.default_center = center_list[0] if center_list else False
        self.default_shuffle_fraction = shuffle_fraction_list[0] if shuffle_fraction_list else 0.5

        # Get available mice and sessions, and build session lookup dict
        self.mice_sessions = {}
        self.session_lookup = {}  # (mouse_name, session_name) -> session object
        for session in sessiondb.iter_sessions(imaging=True):
            mouse_name = session.mouse_name
            session_name = session.session_print(joinby=".")
            if mouse_name not in self.mice_sessions:
                self.mice_sessions[mouse_name] = []
            self.mice_sessions[mouse_name].append(session_name)
            self.session_lookup[(mouse_name, session_name)] = session

        # Sort mice and sessions
        self.mice_list = sorted(self.mice_sessions.keys())
        for mouse in self.mice_list:
            self.mice_sessions[mouse].sort()

        # Add mouse selection
        self.add_selection("mouse", options=self.mice_list, value=self.mice_list[0] if self.mice_list else None)

        # Add session selection (will be updated by callback)
        self.add_selection("session", options=["session1"])

        # Add parameter controls (only for parameters with multiple options)
        if len(smooth_width_list) > 1:
            smooth_options = ["None" if sw is None else str(sw) for sw in smooth_width_list]
            self.add_selection(
                "smooth_width", options=smooth_options, value="None" if self.default_smooth_width is None else str(self.default_smooth_width)
            )

        if len(pre_average_list) > 1:
            self.add_selection("pre_average", options=[str(pa) for pa in pre_average_list], value=str(self.default_pre_average))

        if len(normalize_list) > 1:
            self.add_boolean("normalize", value=self.default_normalize)

        if len(center_list) > 1:
            self.add_boolean("center", value=self.default_center)

        if len(speed_threshold_list) > 1:
            self.add_selection("speed_threshold", options=[str(st) for st in speed_threshold_list], value=str(self.default_speed_threshold))

        if len(shuffle_fraction_list) > 1:
            self.add_selection("shuffle_fraction", options=[str(sf) for sf in shuffle_fraction_list], value=str(self.default_shuffle_fraction))

        # Add scale options
        self.add_selection("xscale", options=["linear", "log"], value="log")
        self.add_selection("yscale", options=["linear", "log"], value="log")

        # Add ylims control (stored in log scale: -20 to 2)
        # Default: -5 to 0 (which is log10(1e-5) to log10(1))
        self.add_float_range("ylims", min=-20.0, max=2.0, value=(-5.0, 0.0))
        self.add_boolean("show_all_data", value=True)
        self.add_boolean("show_new_by_frame", value=True)
        self.add_boolean("show_new_fast_samples", value=True)
        self.add_boolean("show_old_fast_samples", value=False)
        self.add_boolean("average_across_mice", value=True)

        # Initialize averages dictionary for lazy loading
        self.averages = {}

        # Set up callbacks
        self.on_change("mouse", self._update_mouse)
        self.on_change("yscale", self._update_ylims)

        # Initialize session options
        if self.mice_list:
            self._update_mouse(self.state)

    def _update_mouse(self, state):
        """Update session options when mouse changes."""
        mouse_name = state["mouse"]
        if mouse_name and mouse_name in self.mice_sessions:
            session_options = self.mice_sessions[mouse_name]
            self.update_selection("session", options=session_options, value=session_options[0] if session_options else None)
        else:
            self.update_selection("session", options=[])

    def _update_ylims(self, state):
        """Update ylims slider range and value when yscale changes."""
        yscale = state["yscale"]
        current_ylims = state["ylims"]

        ymin_current, ymax_current = current_ylims

        # Convert based on direction of change
        if yscale == "linear":
            # Switching from log to linear: current values are in log, convert to linear
            ymin_linear = 10.0**ymin_current
            ymax_linear = 10.0**ymax_current
            self.update_float_range("ylims", min=1e-20, max=10.0, value=(ymin_linear, ymax_linear))
        else:
            # Switching from linear to log: current values are in linear, convert to log
            ymin_log = np.log10(max(ymin_current, 1e-20))
            ymax_log = np.log10(min(ymax_current, 10.0))
            self.update_float_range("ylims", min=-20.0, max=1.0, value=(ymin_log, ymax_log))

    def _get_param_combo_key(self, state):
        """Generate a unique key for the current parameter combination."""
        # Parse parameters
        if len(self.smooth_width_list) > 1:
            smooth_width_str = state["smooth_width"]
            smooth_width = None if smooth_width_str == "None" else float(smooth_width_str)
        else:
            smooth_width = self.default_smooth_width

        if len(self.pre_average_list) > 1:
            pre_average_str = state["pre_average"]
            pre_average = pre_average_str == "True"
        else:
            pre_average = self.default_pre_average

        if len(self.normalize_list) > 1:
            normalize = state["normalize"]
        else:
            normalize = self.default_normalize

        if len(self.center_list) > 1:
            center = state["center"]
        else:
            center = self.default_center

        if len(self.speed_threshold_list) > 1:
            speed_threshold_str = state["speed_threshold"]
            speed_threshold = float(speed_threshold_str)
        else:
            speed_threshold = self.default_speed_threshold

        if len(self.shuffle_fraction_list) > 1:
            shuffle_fraction_str = state["shuffle_fraction"]
            shuffle_fraction = float(shuffle_fraction_str)
        else:
            shuffle_fraction = self.default_shuffle_fraction

        # Create a tuple key for this parameter combination
        param_key = (
            smooth_width,
            speed_threshold,
            pre_average,
            normalize,
            center,
            shuffle_fraction,
            self.num_bins,
        )
        return param_key

    def _load_single_session_results(self, mouse_name, session_name, param_key):
        """Load results for a single session. Designed for parallel execution."""
        try:
            # Use lookup dict instead of iterating
            session = self.session_lookup.get((mouse_name, session_name))
            if session is None:
                return None

            results_path = get_filepath(
                session,
                num_bins=self.num_bins,
                smooth_width=param_key[0],
                speed_threshold=param_key[1],
                pre_average=param_key[2],
                normalize=param_key[3],
                center=param_key[4],
                shuffle_fraction=param_key[5],
            )

            if not results_path.exists():
                return None

            return joblib.load(results_path)
        except Exception:
            return None

    def _get_averages_filepath(self, param_key):
        """Get filepath for cached averages."""
        # Create a hash of the parameter key for the filename
        import hashlib

        param_str = str(param_key)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return registry.registry_paths.compare_cvpca_path / f"averages_{param_hash}.pkl"

    def _get_averages(self, state):
        """Lazy load averages for current parameter combination."""
        param_key = self._get_param_combo_key(state)

        # Result names we need to average (include both normal and fast)
        result_names = [
            "pf_data_org_covariances",
            "pf_data_leg_covariances",
            "pf_data_fast_org_covariances",
            "pf_data_fast_leg_covariances",
            "spkmap_org_covariances",
            "spkmap_leg_covariances",
            "true_leg_covariances",
        ]

        def _validate_averages(averages):
            valid_averages = True
            for result_name in result_names:
                if result_name not in averages:
                    valid_averages = False
                    break
            return valid_averages

        # Check if we already have averages in memory for this parameter combination
        if param_key in self.averages:
            if _validate_averages(self.averages[param_key]):
                return self.averages[param_key]
            else:
                # Averages are invalid, so clear them
                self.averages.pop(param_key)

        # Check if we have pre-computed averages on disk
        averages_filepath = self._get_averages_filepath(param_key)
        if averages_filepath.exists():
            try:
                averages = joblib.load(averages_filepath)
                if _validate_averages(averages):
                    self.averages[param_key] = averages
                    return averages
            except Exception:
                # If loading fails, continue to compute fresh
                pass

        # Initialize structure for this parameter combination
        averages = {}
        for result_name in result_names:
            averages[result_name] = {}

        # Prepare list of (mouse_name, session_name) tuples for parallel loading
        session_tasks = []
        for mouse_name in self.mice_list:
            for session_name in self.mice_sessions[mouse_name]:
                session_tasks.append((mouse_name, session_name))

        # Load all sessions in parallel
        debug = False
        results_list = joblib.Parallel(n_jobs=1 if debug else 4, verbose=1 if debug else 0, backend="sequential" if debug else None)(
            joblib.delayed(self._load_single_session_results)(mouse_name, session_name, param_key) for mouse_name, session_name in session_tasks
        )

        # Organize results by mouse and result name
        for (mouse_name, session_name), results in zip(session_tasks, results_list):
            if results is None:
                continue

            # Initialize mouse dicts if needed
            for result_name in result_names:
                if mouse_name not in averages[result_name]:
                    averages[result_name][mouse_name] = []

            # Store results for each result name
            for result_name in result_names:
                if result_name in results:
                    averages[result_name][mouse_name].append(results[result_name])

        # Store averages for this parameter combination
        if not _validate_averages(averages):
            raise ValueError(f"Averages are invalid for parameter combination {param_key}")

        self.averages[param_key] = averages

        # Save to disk for future use
        try:
            joblib.dump(averages, averages_filepath)
        except Exception:
            # If saving fails, continue anyway
            pass

        return averages

    def _iter_param_combos(self):
        """Yield (state_dict, param_key) for each parameter combination from the lists."""
        smooth_widths = self.smooth_width_list if len(self.smooth_width_list) > 1 else [self.default_smooth_width]
        speed_thresholds = self.speed_threshold_list if len(self.speed_threshold_list) > 1 else [self.default_speed_threshold]
        pre_averages = self.pre_average_list if len(self.pre_average_list) > 1 else [self.default_pre_average]
        normalizes = self.normalize_list if len(self.normalize_list) > 1 else [self.default_normalize]
        centers = self.center_list if len(self.center_list) > 1 else [self.default_center]
        shuffle_fractions = self.shuffle_fraction_list if len(self.shuffle_fraction_list) > 1 else [self.default_shuffle_fraction]

        for smooth_width, speed_threshold, pre_average, normalize, center, shuffle_fraction in itertools.product(
            smooth_widths, speed_thresholds, pre_averages, normalizes, centers, shuffle_fractions
        ):
            state = {}
            if len(self.smooth_width_list) > 1:
                state["smooth_width"] = "None" if smooth_width is None else str(smooth_width)
            if len(self.pre_average_list) > 1:
                state["pre_average"] = str(pre_average)
            if len(self.normalize_list) > 1:
                state["normalize"] = normalize
            if len(self.center_list) > 1:
                state["center"] = center
            if len(self.speed_threshold_list) > 1:
                state["speed_threshold"] = str(speed_threshold)
            if len(self.shuffle_fraction_list) > 1:
                state["shuffle_fraction"] = str(shuffle_fraction)
            param_key = (
                smooth_width,
                speed_threshold,
                pre_average,
                normalize,
                center,
                shuffle_fraction,
                self.num_bins,
            )
            yield state, param_key

    def load_all_averages(self):
        """
        Load and cache averages for every parameter combination from the lists.

        Iterates over all combos (smooth_width, pre_average, normalize, center,
        speed_threshold, shuffle_fraction) via _iter_param_combos, passes each
        state to _get_averages so results are in memory and on disk.
        """
        for state, _ in tqdm(self._iter_param_combos(), desc="Loading averages"):
            self._get_averages(state)

    def plot(self, state):
        """Create the matplotlib figure with cvPCA comparison plots."""
        mouse_name = state["mouse"]
        session_name = state["session"]

        # Parse parameters - use defaults for single-option parameters
        if len(self.smooth_width_list) > 1:
            smooth_width_str = state["smooth_width"]
            smooth_width = None if smooth_width_str == "None" else float(smooth_width_str)
        else:
            smooth_width = self.default_smooth_width

        if len(self.pre_average_list) > 1:
            pre_average_str = state["pre_average"]
            pre_average = pre_average_str == "True"
        else:
            pre_average = self.default_pre_average

        if len(self.normalize_list) > 1:
            normalize = state["normalize"]
        else:
            normalize = self.default_normalize

        if len(self.center_list) > 1:
            center = state["center"]
        else:
            center = self.default_center

        if len(self.speed_threshold_list) > 1:
            speed_threshold_str = state["speed_threshold"]
            speed_threshold = float(speed_threshold_str)
        else:
            speed_threshold = self.default_speed_threshold

        if len(self.shuffle_fraction_list) > 1:
            shuffle_fraction_str = state["shuffle_fraction"]
            shuffle_fraction = float(shuffle_fraction_str)
        else:
            shuffle_fraction = self.default_shuffle_fraction

        xscale = state["xscale"]
        yscale = state["yscale"]
        ylims = state["ylims"]
        if yscale == "log":
            ylims = (10.0 ** ylims[0], 10.0 ** ylims[1])
        show_all_data = state["show_all_data"]

        # Check if we should show averaged data
        if show_all_data:
            averages = self._get_averages(state)
            return self._plot_averaged(state, averages, xscale, yscale, ylims)

        # Find the session
        session = None
        for s in sessiondb.gen_sessions(imaging=True):
            if s.mouse_name == mouse_name and s.session_print(joinby=".") == session_name:
                session = s
                break

        if session is None:
            fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
            for ax in axes:
                ax.text(0.5, 0.5, f"Session {session_name} not found", ha="center", va="center", transform=ax.transAxes)
            return fig

        # Get filepath and load results
        results_path = get_filepath(
            session,
            num_bins=self.num_bins,
            smooth_width=smooth_width,
            speed_threshold=speed_threshold,
            pre_average=pre_average,
            normalize=normalize,
            center=center,
            shuffle_fraction=shuffle_fraction,
        )

        results = joblib.load(results_path)

        required_keys = [
            "pf_data_org_covariances",
            "pf_data_leg_covariances",
            "pf_data_fast_org_covariances",
            "pf_data_fast_leg_covariances",
            "spkmap_org_covariances",
            "spkmap_leg_covariances",
            "true_leg_covariances",
        ]

        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
            for ax in axes:
                ax.text(0.5, 0.5, f"Missing keys:\n{missing_keys}", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            return fig

        # Normalize function
        def norm(x):
            if x is None:
                return np.array([])
            x = np.array(x)
            if len(x) == 0:
                return x
            x_sum = np.sum(x)
            if x_sum == 0:
                return x
            return x / x_sum

        # Create plots: slow new (black), fast new (blue), old (red)
        pf_data_org_slow = results["pf_data_org_covariances"]
        pf_data_leg_slow = results["pf_data_leg_covariances"]
        pf_data_org_fast = results["pf_data_fast_org_covariances"]
        pf_data_leg_fast = results["pf_data_fast_leg_covariances"]
        spkmap_org = results["spkmap_org_covariances"]
        spkmap_leg = results["spkmap_leg_covariances"]
        true_leg = results["true_leg_covariances"]

        num_bins_new = len(pf_data_org_slow)
        if pf_data_org_fast is not None and len(pf_data_org_fast) > 0:
            num_bins_new = max(num_bins_new, len(pf_data_org_fast))
        xvals = np.arange(num_bins_new) + 1

        if true_leg is not None and len(true_leg) > 0:
            xvals_leg = np.arange(len(true_leg)) + 1
        else:
            xvals_leg = np.array([])

        # Normalize
        pf_data_org_slow_norm = norm(pf_data_org_slow)
        pf_data_leg_slow_norm = norm(pf_data_leg_slow)
        pf_data_org_fast_norm = norm(pf_data_org_fast)
        pf_data_leg_fast_norm = norm(pf_data_leg_fast)
        spkmap_org_norm = norm(spkmap_org)
        spkmap_leg_norm = norm(spkmap_leg)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        fig.suptitle(f"{mouse_name} - {session_name}", fontsize=14, fontweight="bold")

        ylim_min, ylim_max = ylims

        # Plot 1: Original CVPCA — slow new (black), fast new (blue), old (red)
        ax = axes[0]
        if state["show_new_by_frame"]:
            ax.plot(xvals[: len(pf_data_org_slow_norm)], pf_data_org_slow_norm, color="k", label="New (slow)", linewidth=2)
        if state["show_new_fast_samples"]:
            ax.plot(xvals[: len(pf_data_org_fast_norm)], pf_data_org_fast_norm, color="b", label="New (fast)", linewidth=2)
        if state["show_old_fast_samples"]:
            ax.plot(xvals[: len(spkmap_org_norm)], spkmap_org_norm, color="r", label="Old", linewidth=2)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_title("New")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Plot 2: Legacy CVPCA (shuffle) — slow (black), fast (blue), old (red)
        ax = axes[1]
        if state["show_new_by_frame"]:
            ax.plot(xvals[: len(pf_data_leg_slow_norm)], pf_data_leg_slow_norm, color="k", linewidth=2)
        if state["show_new_fast_samples"]:
            ax.plot(xvals[: len(pf_data_leg_fast_norm)], pf_data_leg_fast_norm, color="b", linewidth=2)
        if state["show_old_fast_samples"]:
            ax.plot(xvals[: len(spkmap_leg_norm)], spkmap_leg_norm, color="r", linewidth=2)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_title("Shuffle")
        ax.grid(True, alpha=0.3)

        # Plot 3: True Legacy CVPCA (black)
        ax = axes[2]
        if true_leg is not None and len(true_leg) > 0:
            true_leg_norm = norm(true_leg)
            if len(true_leg_norm) > 0 and len(xvals_leg) > 0:
                ax.plot(xvals_leg, true_leg_norm, color="k", linewidth=2)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylim(ylim_min, ylim_max)
        legacy_details = "SW=0.1, PA=∅"
        ax.set_title(f"Legacy\n{legacy_details}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_averaged(self, state, averages, xscale, yscale, ylims):
        """Plot averaged data across sessions, one curve per mouse."""

        # Normalize function
        def norm(x):
            if x is None:
                return np.array([])
            x = np.array(x)
            if len(x) == 0:
                return x
            x_sum = np.sum(x)
            if x_sum == 0:
                return x
            return x / x_sum

        def average_with_padding(arrays_list):
            """Average arrays of potentially different sizes by padding with NaN."""
            if len(arrays_list) == 0:
                return None

            # Filter out None values
            valid_arrays = [np.array(arr) for arr in arrays_list if arr is not None]
            if len(valid_arrays) == 0:
                return None

            # Find maximum size
            max_size = max(len(arr) for arr in valid_arrays)

            # Pad all arrays to max_size with NaN
            padded_arrays = []
            for arr in valid_arrays:
                if len(arr) < max_size:
                    padded = np.full(max_size, np.nan)
                    padded[: len(arr)] = arr
                    padded_arrays.append(padded)
                else:
                    padded_arrays.append(arr)

            # Use nanmean to average (ignores NaN values)
            return np.nanmean(padded_arrays, axis=0)

        def stack_per_mouse_curves(avg_key):
            """Get one normalized curve per mouse, pad to common length, stack (n_mice, n_bins)."""
            curves = []
            for mouse_name in self.mice_list:
                arr_list = averages[avg_key].get(mouse_name, [])
                avg = average_with_padding(arr_list) if arr_list else None
                if avg is not None:
                    c = norm(avg)
                    if len(c) > 0:
                        curves.append(c)
            if not curves:
                return None, None
            max_len = max(len(c) for c in curves)
            stacked = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                stacked[i, : len(c)] = c
            xvals = np.arange(max_len) + 1
            return xvals, stacked

        average_across_mice = state["average_across_mice"]
        slow_color, fast_color, old_color = "k", "b", "r"

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        fig.suptitle("Averaged across sessions", fontsize=14, fontweight="bold")

        ylim_min, ylim_max = ylims

        # Plot 1: Original CVPCA — slow new (black), fast new (blue), old (red)
        ax = axes[0]
        if average_across_mice:
            xvals_slow, data_slow = stack_per_mouse_curves("pf_data_org_covariances")
            if state["show_new_by_frame"] and data_slow is not None:
                errorPlot(xvals_slow, data_slow, axis=0, se=True, ax=ax, alpha=0.3, color=slow_color, label="New (slow)")
            xvals_fast, data_fast = stack_per_mouse_curves("pf_data_fast_org_covariances")
            if state["show_new_fast_samples"] and data_fast is not None:
                errorPlot(xvals_fast, data_fast, axis=0, se=True, ax=ax, alpha=0.3, color=fast_color, label="New (fast)")
            xvals_old, data_old = stack_per_mouse_curves("spkmap_org_covariances")
            if state["show_old_fast_samples"] and data_old is not None:
                errorPlot(xvals_old, data_old, axis=0, se=True, ax=ax, alpha=0.3, color=old_color, label="Old")
        else:
            need_slow_label = True
            need_fast_label = True
            need_old_label = True
            for mouse_name in self.mice_list:
                pf_data_slow_list = averages["pf_data_org_covariances"].get(mouse_name, [])
                pf_data_fast_list = averages["pf_data_fast_org_covariances"].get(mouse_name, [])
                spkmap_list = averages["spkmap_org_covariances"].get(mouse_name, [])

                if state["show_new_by_frame"] and len(pf_data_slow_list) > 0:
                    pf_data_avg = average_with_padding(pf_data_slow_list)
                    if pf_data_avg is not None:
                        pf_data_norm = norm(pf_data_avg)
                        if len(pf_data_norm) > 0:
                            xvals = np.arange(len(pf_data_norm)) + 1
                            ax.plot(xvals, pf_data_norm, color=slow_color, label="New (slow)" if need_slow_label else None, linewidth=1.0, alpha=0.7)
                            need_slow_label = False

                if state["show_new_fast_samples"] and len(pf_data_fast_list) > 0:
                    pf_data_avg = average_with_padding(pf_data_fast_list)
                    if pf_data_avg is not None:
                        pf_data_norm = norm(pf_data_avg)
                        if len(pf_data_norm) > 0:
                            xvals = np.arange(len(pf_data_norm)) + 1
                            ax.plot(xvals, pf_data_norm, color=fast_color, label="New (fast)" if need_fast_label else None, linewidth=1.0, alpha=0.7)
                            need_fast_label = False

                if state["show_old_fast_samples"] and len(spkmap_list) > 0:
                    spkmap_avg = average_with_padding(spkmap_list)
                    if spkmap_avg is not None:
                        spkmap_norm = norm(spkmap_avg)
                        if len(spkmap_norm) > 0:
                            xvals = np.arange(len(spkmap_norm)) + 1
                            ax.plot(xvals, spkmap_norm, color=old_color, label="Old" if need_old_label else None, linewidth=1.0, alpha=0.7)
                            need_old_label = False

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_title("New")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Legacy CVPCA (shuffle) — slow (black), fast (blue), old (red)
        ax = axes[1]
        if average_across_mice:
            xvals_slow, data_slow = stack_per_mouse_curves("pf_data_leg_covariances")
            if state["show_new_by_frame"] and data_slow is not None:
                errorPlot(xvals_slow, data_slow, axis=0, se=True, ax=ax, alpha=0.3, color=slow_color)
            xvals_fast, data_fast = stack_per_mouse_curves("pf_data_fast_leg_covariances")
            if state["show_new_fast_samples"] and data_fast is not None:
                errorPlot(xvals_fast, data_fast, axis=0, se=True, ax=ax, alpha=0.3, color=fast_color)
            xvals_old, data_old = stack_per_mouse_curves("spkmap_leg_covariances")
            if state["show_old_fast_samples"] and data_old is not None:
                errorPlot(xvals_old, data_old, axis=0, se=True, ax=ax, alpha=0.3, color=old_color)
        else:
            for mouse_name in self.mice_list:
                pf_data_slow_list = averages["pf_data_leg_covariances"].get(mouse_name, [])
                pf_data_fast_list = averages["pf_data_fast_leg_covariances"].get(mouse_name, [])
                spkmap_list = averages["spkmap_leg_covariances"].get(mouse_name, [])

                if state["show_new_by_frame"] and len(pf_data_slow_list) > 0:
                    pf_data_avg = average_with_padding(pf_data_slow_list)
                    if pf_data_avg is not None:
                        pf_data_norm = norm(pf_data_avg)
                        if len(pf_data_norm) > 0:
                            xvals = np.arange(len(pf_data_norm)) + 1
                            ax.plot(xvals, pf_data_norm, color=slow_color, linewidth=1.0, alpha=0.7)

                if state["show_new_fast_samples"] and len(pf_data_fast_list) > 0:
                    pf_data_avg = average_with_padding(pf_data_fast_list)
                    if pf_data_avg is not None:
                        pf_data_norm = norm(pf_data_avg)
                        if len(pf_data_norm) > 0:
                            xvals = np.arange(len(pf_data_norm)) + 1
                            ax.plot(xvals, pf_data_norm, color=fast_color, linewidth=1.0, alpha=0.7)

                if state["show_old_fast_samples"] and len(spkmap_list) > 0:
                    spkmap_avg = average_with_padding(spkmap_list)
                    if spkmap_avg is not None:
                        spkmap_norm = norm(spkmap_avg)
                        if len(spkmap_norm) > 0:
                            xvals = np.arange(len(spkmap_norm)) + 1
                            ax.plot(xvals, spkmap_norm, color=old_color, linewidth=1.0, alpha=0.7)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_title("Shuffle")
        ax.grid(True, alpha=0.3)

        # Plot 3: True Legacy CVPCA (black)
        ax = axes[2]
        if average_across_mice:
            xvals_leg, data_leg = stack_per_mouse_curves("true_leg_covariances")
            if data_leg is not None:
                errorPlot(xvals_leg, data_leg, axis=0, se=True, ax=ax, alpha=0.3, color=slow_color)
        else:
            for mouse_name in self.mice_list:
                true_leg_list = averages["true_leg_covariances"].get(mouse_name, [])

                if len(true_leg_list) > 0:
                    true_leg_avg = average_with_padding(true_leg_list)
                    if true_leg_avg is not None:
                        true_leg_norm = norm(true_leg_avg)
                        if len(true_leg_norm) > 0:
                            xvals_leg = np.arange(len(true_leg_norm)) + 1
                            ax.plot(xvals_leg, true_leg_norm, color=slow_color, linewidth=1.0, alpha=0.7)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_title("Legacy")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
