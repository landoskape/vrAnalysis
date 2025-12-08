from pathlib import Path
import joblib
import numpy as np
import torch
from tqdm import tqdm

from vrAnalysis.database import get_database
from vrAnalysis.sessions import B2Session
from dimilibi import RidgeRegression
from dimensionality_manuscript.registry import PopulationRegistry, get_model
import optuna
import gc

# get session database
sessiondb = get_database("vrSessions")

# get population registry and models
registry = PopulationRegistry()
int_gain_model = get_model("internal_placefield_1d_gain", registry)


def get_filepath(session: B2Session) -> Path:
    """Get the filepath for the results of a session."""
    return registry.registry_paths.pf1d_internals_path / f"{session.session_print(joinby='.')}.pkl"


process_sessions = True
force_remake = False
clear_cache = False
validate_results = False

spks_type = "oasis"
bins = np.linspace(-100, 100, 101)

if __name__ == "__main__":
    for session in tqdm(sessiondb.iter_sessions(imaging=True), desc="Measuring Placefield Model Internals"):
        _clear_session_cache = False
        results_path = get_filepath(session)

        # Clear cache if requested
        if clear_cache:
            results_path.unlink()

        # Validate results if requested
        if validate_results:
            if results_path.exists():
                results = joblib.load(results_path)
                if results is not None:
                    required_keys = [
                        "dev_bin_counts",
                        "fraction_switch_env",
                        "r2_gain_target",
                        "r2_gain_source",
                        "slope_gain_source",
                        "yint_gain_source",
                        "slope_gain_target",
                        "yint_gain_target",
                    ]
                    results_valid = all(key in results for key in required_keys)
                    if not results_valid:
                        results_path.unlink()

        # Process session if requested (or not exists)
        if process_sessions:
            if force_remake or not results_path.exists():
                try:
                    _clear_session_cache = True
                    results = int_gain_model.measure_internals(session, spks_type=spks_type, dev_bin_edges=bins)
                    joblib.dump(results, results_path)
                except Exception as e:
                    print(f"Error processing session {session.session_print()}: {e}")
                    continue
                finally:
                    if _clear_session_cache:
                        session.clear_cache()
                        torch.cuda.empty_cache()
                        gc.collect()
