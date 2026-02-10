import gc
from tqdm import tqdm
import torch
from vrAnalysis.database import get_database
from vrAnalysis.sessions import SpksTypes
from dimensionality_manuscript.registry import ModelName, PopulationRegistry, get_model

clear_hyperparameters = False  # Clears hyperparameter cache
clear_scores = False  # Clears score cache

score_models = True  # Scores models
force_remake = False  # Remakes even if existing
force_reoptimize = False  # Re-optimizes even if existing

check_existing_scores = False  # Checks if scores already exist

# Note: there's more parameters to scores & hyperparameters, but by not setting them we use the default values
# which are the primary ones used for the manuscript. Non-defaults are primarly for testing and exploratory analysis.

MODEL_NAMES: list[ModelName] = [
    # "external_placefield_1d",
    # "internal_placefield_1d",
    # "external_placefield_1d_gain",
    # "internal_placefield_1d_gain",
    "external_placefield_1d_vector_gain",
    "internal_placefield_1d_vector_gain",
    # "rbfpos_decoder_only",
    # "rbfpos",
    # "rbfpos_leak",
    # "rrr",
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

    for model_name in tqdm(MODEL_NAMES, desc="Testing different model types"):
        model = get_model(model_name, registry)

        for spks_type in SPKS_TYPES:
            for isession, session in enumerate(tqdm(sessiondb.iter_sessions(imaging=True, session_params=dict(spks_type=spks_type)))):
                if clear_hyperparameters:
                    model.clear_cached_hyperparameter(session, spks_type=spks_type, method=METHOD)

                if clear_scores:
                    model.clear_cached_score(session, spks_type=spks_type, method=METHOD)

                if score_models:
                    try:
                        _clear_cache = not model.check_existing_score(
                            session,
                            spks_type=spks_type,
                            method=METHOD,
                        )
                        _ = model.get_best_score(
                            session,
                            spks_type=spks_type,
                            force_remake=force_remake,
                            force_reoptimize=force_reoptimize,
                            method=METHOD,
                        )

                    except Exception as e:
                        error_path = registry.registry_paths.error_path / f"{model_name}_{session.session_print(joinby='.')}.txt"
                        with open(error_path, "w") as f:
                            f.write(str(e))
                        print(f"Error scoring model {model_name} on session {session.session_print()}: {e}")
                        continue

                    finally:
                        # Make sure any stored data is cleared (not actually sure if torch is saving things but better clear in case)
                        if _clear_cache:
                            session.clear_cache()
                            torch.cuda.empty_cache()
                            gc.collect()

                if check_existing_scores:
                    if model.check_existing_score(session, spks_type=spks_type, method=METHOD):
                        # print(f"{isession} Score for model {model_name} on session {session.session_print()} already exists")
                        pass
                    else:
                        print(f"{isession} !!!!! Score for model {model_name} on session {session.session_print()} does not exist")
                        pass
