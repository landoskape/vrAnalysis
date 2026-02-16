from typing import Optional
from pathlib import Path
import gc
import joblib
import numpy as np
import torch
from tqdm import tqdm
from vrAnalysis.database import get_database
from vrAnalysis.helpers import cross_validate_trials
from vrAnalysis.sessions import B2Session, SpksTypes
from vrAnalysis.processors.placefields import get_placefield
from dimilibi import gaussian_filter
from dimilibi.pca import PCA
from dimilibi.cvpca import RegularizedCVPCA, CVPCA, LegacyCVPCA
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.workflows.compare_old_cvpca import get_legacy_cvpca

# get session database
sessiondb = get_database("vrSessions")

# get population registry and models
registry = PopulationRegistry()


def nanmax(tensor: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim).values
    return output


def process_session(session: B2Session, spks_type: SpksTypes, num_bins: int = 100, center: bool = True, normalize: bool = True):
    num_per_env = {i: np.sum(session.trial_environment == i) for i in session.environments}
    best_env = max(num_per_env, key=num_per_env.get)
    best_env_idx = np.where(session.environments == best_env)[0][0]

    env_length = session.env_length[0]
    dist_edges = np.linspace(0, env_length, num_bins + 1)
    population, frame_behavior = registry.get_population(session, spks_type)

    trial_folds = cross_validate_trials(session.trial_environment, [1, 1, 1])
    data = np.array(population.data[population.idx_neurons][:, population.idx_samples]).T
    placefields = [get_placefield(data, frame_behavior, dist_edges, trial_filter=trial_fold, zero_to_nan=True) for trial_fold in trial_folds]
    torch_pfs = [torch.tensor(pf.placefield[best_env_idx].T) for pf in placefields]

    if any([np.any(pf.count[best_env_idx] == 0) for pf in placefields]):
        bad_locations = [np.where(pf.count[best_env_idx] == 0)[0] for pf in placefields]
        bad_locations = np.unique(np.concatenate(bad_locations))
        good_idx = np.setdiff1d(np.arange(num_bins), bad_locations)
        if not np.all(np.diff(good_idx) == 1):
            print(f"Some placefields have non sequential missing counts at locations: {bad_locations}")
            raise ValueError("Some placefields have no counts in the best environment!")
        else:
            torch_pfs = [torch_pf[:, good_idx] for torch_pf in torch_pfs]

    if normalize:
        _max_neuron = nanmax(torch.concatenate(torch_pfs, dim=1), dim=1, keepdim=True)
        _max_neuron[_max_neuron == 0] = 1
        torch_pfs = [pf / _max_neuron for pf in torch_pfs]

    if any([torch.any(torch.isnan(pf)) for pf in torch_pfs]):
        print("Some placefields have NaNs!")
        raise ValueError("Some placefields have NaNs!")

    pca_covariances = []
    pca_smooth_covariances = []
    reg_covariances = []
    org_covariances = []
    org_smooth_covariances = []
    smoothing_widths = []
    for ref_fold in range(len(trial_folds)):
        c_repeat_0 = torch_pfs[ref_fold]
        c_repeat_1 = torch_pfs[(ref_fold + 1) % len(trial_folds)]
        c_repeat_2 = torch_pfs[(ref_fold + 2) % len(trial_folds)]

        reg_cvpca = RegularizedCVPCA(center=center)
        reg_cvpca = reg_cvpca.fit_smoothing(c_repeat_0, c_repeat_1, c_repeat_2)
        reg_cvpca = reg_cvpca.fit(c_repeat_0)
        reg_covariance = reg_cvpca.score(c_repeat_1, c_repeat_2)
        reg_covariances.append(reg_covariance)

        cvpca = CVPCA(center=center).fit(c_repeat_0)
        org_covariance_1 = cvpca.score(c_repeat_0, c_repeat_1)
        org_covariance_2 = cvpca.score(c_repeat_0, c_repeat_2)
        org_covariance = np.mean(np.stack([org_covariance_1, org_covariance_2], axis=0), axis=0)
        org_covariances.append(org_covariance)

        c_repeat_0_smooth = gaussian_filter(c_repeat_0, reg_cvpca.smoothing_widths, axis=1)
        c_repeat_1_smooth = gaussian_filter(c_repeat_1, reg_cvpca.smoothing_widths, axis=1)
        c_repeat_2_smooth = gaussian_filter(c_repeat_2, reg_cvpca.smoothing_widths, axis=1)

        cvpca_smooth = CVPCA(center=center).fit(c_repeat_0_smooth)
        org_smooth_covariance_1 = cvpca_smooth.score(c_repeat_0_smooth, c_repeat_1_smooth)
        org_smooth_covariance_2 = cvpca_smooth.score(c_repeat_0_smooth, c_repeat_2_smooth)
        org_smooth_covariance = np.mean(np.stack([org_smooth_covariance_1, org_smooth_covariance_2], axis=0), axis=0)
        org_smooth_covariances.append(org_smooth_covariance)

        pca = PCA(center=center).fit(c_repeat_0)
        pca_covariances.append(pca.get_eigenvalues())

        pca_smooth = PCA(center=center).fit(c_repeat_0_smooth)
        pca_smooth_covariances.append(pca_smooth.get_eigenvalues())

        # Also save smoothing widths for reproducibility
        smoothing_widths.append(reg_cvpca.smoothing_widths)

    # Get Saved Legacy CVPCA results as well
    try:
        saved_leg_result = get_legacy_cvpca(session, best_env_idx=best_env_idx)
    except Exception as e:
        print(f"Error getting saved legacy CVPCA results for session {session.session_print()}: {e}")
        saved_leg_result = None

    result = {
        "trial_folds": trial_folds,
        "reg_covariances": np.mean(np.stack(reg_covariances, axis=0), axis=0),
        "org_covariances": np.mean(np.stack(org_covariances, axis=0), axis=0),
        "org_smooth_covariances": np.mean(np.stack(org_smooth_covariances, axis=0), axis=0),
        "pca_covariances": np.mean(np.stack(pca_covariances, axis=0), axis=0),
        "pca_smooth_covariances": np.mean(np.stack(pca_smooth_covariances, axis=0), axis=0),
        "saved_leg_covariances": saved_leg_result["cv_by_env_all"] if saved_leg_result is not None else None,
        "smoothing_widths": np.mean(np.stack(smoothing_widths, axis=0), axis=0),
    }
    return result


def get_filepath(session: B2Session, center: bool = True) -> Path:
    """Get the filepath for the results of a session."""
    name = session.session_print(joinby=".")
    if not center:
        name += "_notcentered"
    return registry.registry_paths.measure_cvpca_path / f"{name}.pkl"


def _process_single_session(session: B2Session, spks_type: SpksTypes, num_bins: int, center: bool, force_remake: bool) -> dict:
    """Process a single session - designed to be parallelized.

    This function handles the computation for one session. File checking is done
    in the main process to avoid race conditions, but we double-check here in
    case another worker created the file.
    """
    results_path = get_filepath(session, center=center)

    # Double-check if file exists (another worker might have created it)
    if not force_remake and results_path.exists():
        return {"status": "skipped", "session": session.session_print()}

    try:
        result = process_session(session, spks_type=spks_type, num_bins=num_bins, center=center)
        joblib.dump(result, results_path)
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


process_sessions = True
force_remake = False
clear_cache = False
validate_results = True

n_jobs = 4
spks_type = "oasis"
num_bins = 100

if __name__ == "__main__":
    # Collect all sessions that need processing
    for center in [True, False]:
        sessions_to_process = []
        for session in tqdm(sessiondb.iter_sessions(imaging=True), desc=f"Checking sessions for center={center}"):
            results_path = get_filepath(session, center=center)

            # Pre-check: clear cache if requested
            if clear_cache and results_path.exists():
                results_path.unlink()

            # Pre-check: validate results if requested
            if validate_results and results_path.exists():
                try:
                    results = joblib.load(results_path)
                    if results is not None:
                        required_keys = [
                            "trial_folds",
                            "reg_covariances",
                            "org_covariances",
                            "org_smooth_covariances",
                            "pca_covariances",
                            "pca_smooth_covariances",
                            "saved_leg_covariances",
                            "smoothing_widths",
                        ]
                        results_valid = all(key in results for key in required_keys)
                        if not results_valid:
                            results_path.unlink()
                except Exception:
                    if results_path.exists():
                        results_path.unlink()

            # Add to processing list if needed
            if process_sessions and (force_remake or not results_path.exists()):
                sessions_to_process.append(session)

        # Process sessions in parallel
        if sessions_to_process:
            num_workers = n_jobs if n_jobs > 0 else "all"
            print(f"Processing {len(sessions_to_process)} sessions with {num_workers} workers...")
            results = joblib.Parallel(n_jobs=n_jobs, verbose=10, backend="sequential" if n_jobs == 1 else None)(
                joblib.delayed(_process_single_session)(session, spks_type, num_bins, center, force_remake)
                for session in tqdm(sessions_to_process, desc="Measuring CVPCA")
            )

            # Report summary
            success_count = sum(1 for r in results if r["status"] == "success")
            error_count = sum(1 for r in results if r["status"] == "error")
            skipped_count = sum(1 for r in results if r["status"] == "skipped")
            print(f"\nCompleted: {success_count} successful, {error_count} errors, {skipped_count} skipped")

        print("Done!")
