import gc
from tqdm import tqdm
import torch
from vrAnalysis.database import get_database
from vrAnalysis.sessions import SpksTypes
from dimensionality_manuscript.registry import SubspaceName, PopulationRegistry, get_subspace
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters

# only works on the "optimized" route, not on specific hyperparameters
clear_hyperparameters = False  # Clears hyperparameter cache
clear_scores = False  # Clears score cache

# This is for setting specific hyperparameters
score_with_specific_hyperparameters = True  # Scores subspace models with specific hyperparameters
check_existing_with_specific_hyperparameters = False  # Checks if scores already exist with specific hyperparameters
specific_hyperparameters = [
    PlaceFieldHyperparameters(num_bins=100, smooth_width=5.0),
    PlaceFieldHyperparameters(num_bins=100, smooth_width=None),
]

# For all situations where these inputs are possible
force_remake = False  # Remakes even if existing
force_reoptimize = False  # Re-optimizes even if existing


# Note: there's more parameters to scores & hyperparameters, but by not setting them we use the default values
# which are the primary ones used for the manuscript. Non-defaults are primarly for testing and exploratory analysis.

SUBSPACE_NAMES: list[SubspaceName] = [
    "svca_subspace",
    "covcov_subspace",
]

SPKS_TYPES: tuple[SpksTypes] = (
    "oasis",
    # "sigrebase",
    # "deconvolved",
)

if __name__ == "__main__":
    sessiondb = get_database("vrSessions")
    registry = PopulationRegistry()

    for subspace_name in tqdm(SUBSPACE_NAMES, desc="Testing different subspace types"):
        subspace_model = get_subspace(subspace_name, registry, match_dimensions=False)

        for spks_type in SPKS_TYPES:
            for isession, session in enumerate(tqdm(sessiondb.iter_sessions(imaging=True, session_params=dict(spks_type=spks_type)))):
                if score_with_specific_hyperparameters:
                    for specific_hyperparameter in specific_hyperparameters:
                        try:
                            _clear_cache = not subspace_model.check_existing_score_from_hyps(
                                session,
                                spks_type=spks_type,
                                hyperparameters=specific_hyperparameter,
                            )
                            _ = subspace_model.get_score(
                                session,
                                spks_type=spks_type,
                                force_remake=force_remake,
                                hyperparameters=specific_hyperparameter,
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

                if check_existing_with_specific_hyperparameters:
                    for specific_hyperparameter in specific_hyperparameters:
                        if subspace_model.check_existing_score_from_hyps(
                            session,
                            spks_type=spks_type,
                            hyperparameters=specific_hyperparameter,
                        ):
                            # print(f"{isession} Score for subspace {subspace_name} on session {session.session_print()} already exists")
                            pass
                        else:
                            print(f"{isession} !!!!! Score for subspace {subspace_name} on session {session.session_print()} does not exist")
                            pass
