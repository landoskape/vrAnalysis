import gc
from pathlib import Path

import joblib
import torch
from tqdm import tqdm

from vrAnalysis.database import get_database
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors.em import process_session, ExpMaxConfig
from dimensionality_manuscript.registry import PopulationRegistry

sessiondb = get_database("vrSessions")
registry = PopulationRegistry()


def get_filepath(session: B2Session, config: ExpMaxConfig) -> Path:
    """Get the filepath for saved EM results of a session.

    Parameters
    ----------
    session : B2Session
        The session to get the filepath for.
    config : ExpMaxConfig
        The EM configuration used for processing.

    Returns
    -------
    Path
        The filepath where results are saved.
    """
    name = session.session_print(joinby=".")
    name += f"_steps{config.num_steps}"
    if config.smooth_width is not None:
        name += f"_smooth{config.smooth_width}"
    return registry.registry_paths.em_path / f"{name}.pkl"


def _process_single_session(
    session: B2Session,
    config: ExpMaxConfig,
    force_remake: bool,
) -> dict:
    """Process a single session with the EM algorithm.

    Parameters
    ----------
    session : B2Session
        The session to process.
    config : ExpMaxConfig
        The EM configuration.
    force_remake : bool
        Whether to reprocess even if results already exist.

    Returns
    -------
    dict
        Status dictionary with 'status', 'session', and optionally 'error' keys.
    """
    results_path = get_filepath(session, config)

    if not force_remake and results_path.exists():
        return {"status": "skipped", "session": session.session_print()}

    try:
        result = process_session(session, config=config)
        joblib.dump(result, results_path)
        return {"status": "success", "session": session.session_print()}
    except Exception as e:
        error_msg = f"Error processing session {session.session_print()}: {e}"
        print(error_msg)
        return {"status": "error", "session": session.session_print(), "error": str(e)}
    finally:
        session.clear_cache()
        torch.cuda.empty_cache()
        gc.collect()


process_sessions = True
force_remake = False
clear_cache = False
validate_results = True

n_jobs = 1
config = ExpMaxConfig()

REQUIRED_KEYS = [
    "em_test_r2",
    "em_null_r2",
    "em_test_rms",
    "em_null_rms",
    "em_test_mse",
    "em_null_mse",
    "step_mse",
    "step_r2",
    "step_rms",
]

if __name__ == "__main__":
    sessions_to_process = []
    for session in tqdm(sessiondb.iter_sessions(imaging=True), desc="Checking sessions"):
        results_path = get_filepath(session, config)

        if clear_cache and results_path.exists():
            results_path.unlink()

        if validate_results and results_path.exists():
            try:
                results = joblib.load(results_path)
                if results is not None:
                    results_valid = all(key in results for key in REQUIRED_KEYS)
                    if not results_valid:
                        results_path.unlink()
            except Exception:
                if results_path.exists():
                    results_path.unlink()

        if process_sessions and (force_remake or not results_path.exists()):
            sessions_to_process.append(session)

    if sessions_to_process:
        num_workers = n_jobs if n_jobs > 0 else "all"
        print(f"Processing {len(sessions_to_process)} sessions with {num_workers} workers...")
        results = joblib.Parallel(n_jobs=n_jobs, verbose=10, backend="sequential" if n_jobs == 1 else None)(
            joblib.delayed(_process_single_session)(session, config, force_remake) for session in tqdm(sessions_to_process, desc="Running EM")
        )

        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        print(f"\nCompleted: {success_count} successful, {error_count} errors, {skipped_count} skipped")

    print("Done!")
