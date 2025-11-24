import gc
from tqdm import tqdm
import torch
from vrAnalysis.database import get_database
from vrAnalysis.sessions import SpksTypes
from dimensionality_manuscript.registry import SubspaceName, PopulationRegistry, get_subspace

clear_hyperparameters = False  # Clears hyperparameter cache
clear_scores = False  # Clears score cache

score_subspaces = True  # Scores subspace models
force_remake = False  # Remakes even if existing
force_reoptimize = False  # Re-optimizes even if existing

check_existing_scores = False  # Checks if scores already exist

# Note: there's more parameters to scores & hyperparameters, but by not setting them we use the default values
# which are the primary ones used for the manuscript. Non-defaults are primarly for testing and exploratory analysis.

SUBSPACE_NAMES: list[SubspaceName] = [
    "pca_subspace",
    "cvpca_subspace",
    "svca_subspace",
]

SPKS_TYPES: tuple[SpksTypes] = (
    "oasis",
    # "sigrebase",
    # "deconvolved",
)

METHOD = "optuna"

if __name__ == "__main__":
    sessiondb = get_database("vrSessions")
    registry = PopulationRegistry()

    for subspace_name in tqdm(SUBSPACE_NAMES, desc="Testing different subspace types"):
        subspace_model = get_subspace(subspace_name, registry)

        for spks_type in SPKS_TYPES:
            for isession, session in enumerate(tqdm(sessiondb.iter_sessions(imaging=True, session_params=dict(spks_type=spks_type)))):
                if clear_hyperparameters:
                    subspace_model.clear_cached_hyperparameter(session, spks_type=spks_type, method=METHOD)

                if clear_scores:
                    subspace_model.clear_cached_score(session, spks_type=spks_type, method=METHOD)

                if score_subspaces:
                    try:
                        _clear_cache = not subspace_model.check_existing_score(
                            session,
                            spks_type=spks_type,
                            method=METHOD,
                        )
                        _ = subspace_model.get_best_score(
                            session,
                            spks_type=spks_type,
                            force_remake=force_remake,
                            force_reoptimize=force_reoptimize,
                            method=METHOD,
                        )

                    except Exception as e:
                        error_path = registry.registry_paths.subspace_error_path / f"{subspace_name}_{session.session_print(joinby='.')}.txt"
                        error_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(error_path, "w") as f:
                            f.write(str(e))
                        print(f"Error scoring subspace {subspace_name} on session {session.session_print()}: {e}")
                        continue

                    finally:
                        # Make sure any stored data is cleared (not actually sure if torch is saving things but better clear in case)
                        if _clear_cache:
                            session.clear_cache()
                            torch.cuda.empty_cache()
                            gc.collect()

                if check_existing_scores:
                    if subspace_model.check_existing_score(session, spks_type=spks_type, method=METHOD):
                        # print(f"{isession} Score for subspace {subspace_name} on session {session.session_print()} already exists")
                        pass
                    else:
                        print(f"{isession} !!!!! Score for subspace {subspace_name} on session {session.session_print()} does not exist")
                        pass
