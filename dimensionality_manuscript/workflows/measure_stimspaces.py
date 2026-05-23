"""Score StimSpaceSubspace models across sessions and config variations.

Mirrors ``measure_subspaces.py``: populates the subspace score cache via
``get_score`` for each ``StimSpaceConfig`` variation (or an explicit list).
"""

from __future__ import annotations

import gc

import torch
from tqdm import tqdm

from dimensionality_manuscript.configs.stimspace import StimSpaceConfig
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters
from dimensionality_manuscript.subspace_analysis.stimspace import StimSpaceSubspace
from vrAnalysis.database import get_database
from vrAnalysis.sessions import SpksTypes

clear_scores = True  # Clears score cache for each config variation
score_configs = False  # Scores stimspace models for each config variation
check_existing_configs = False  # Reports sessions missing cached scores

# When False, use STIMSPACE_CONFIGS below instead of the full param grid.
use_full_param_grid = True
STIMSPACE_CONFIGS: list[StimSpaceConfig] = [
    # Example overrides when use_full_param_grid is False:
    # StimSpaceConfig(smooth_width=5.0),
    # StimSpaceConfig(smooth_width=None, directions_from_placefield_only=True),
]

force_remake = False  # Recompute scores even when cache exists

SPKS_TYPES: tuple[SpksTypes, ...] = (
    "oasis",
    # "sigrebase",
    # "deconvolved",
)


def _hyperparameters(cfg: StimSpaceConfig) -> PlaceFieldHyperparameters:
    """Place-field hyperparameters for a config instance."""
    return PlaceFieldHyperparameters(num_bins=cfg.num_bins, smooth_width=cfg.smooth_width)


def _build_model(cfg: StimSpaceConfig, registry: PopulationRegistry) -> StimSpaceSubspace:
    """Construct a StimSpaceSubspace matching ``StimSpaceConfig.process``."""
    return StimSpaceSubspace(
        registry,
        centered=cfg.center,
        normalize=cfg.normalize,
        use_fast_sampling=cfg.use_fast_sampling,
        reliability_threshold=cfg.reliability_threshold,
        fraction_active_threshold=cfg.fraction_active_threshold,
        directions_from_placefield_only=cfg.directions_from_placefield_only,
        cross_validated_placefield_kernel=cfg.cross_validated_placefield_kernel,
    )


def _error_label(model: StimSpaceSubspace) -> str:
    """Filesystem-safe label for error logs (matches score cache model name)."""
    return model._get_model_name()


if __name__ == "__main__":
    sessiondb = get_database("vrSessions")
    registry = PopulationRegistry()

    configs = StimSpaceConfig.generate_variations() if use_full_param_grid else STIMSPACE_CONFIGS
    if not configs:
        raise ValueError("No StimSpace configs to run; set use_full_param_grid or STIMSPACE_CONFIGS.")

    for cfg in tqdm(configs, desc="StimSpace configs"):
        model = _build_model(cfg, registry)
        hyperparameters = _hyperparameters(cfg)
        label = _error_label(model)

        for spks_type in SPKS_TYPES:
            for isession, session in enumerate(
                tqdm(
                    sessiondb.iter_sessions(imaging=True, session_params=dict(spks_type=spks_type)),
                    desc=f"{label} ({spks_type})",
                    leave=False,
                )
            ):
                if clear_scores:
                    model.clear_cached_score_from_hyps(
                        session,
                        spks_type=spks_type,
                        hyperparameters=hyperparameters,
                    )

                if score_configs:
                    try:
                        _clear_cache = not model.check_existing_score_from_hyps(
                            session,
                            spks_type=spks_type,
                            hyperparameters=hyperparameters,
                        )
                        _ = model.get_score(
                            session,
                            spks_type=spks_type,
                            force_remake=force_remake,
                            hyperparameters=hyperparameters,
                        )
                    except Exception as e:
                        error_path = registry.registry_paths.subspace_error_path / f"{label}_{session.session_print(joinby='.')}.txt"
                        error_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(error_path, "w") as f:
                            f.write(str(e))
                        print(f"Error scoring stimspace {label} on session {session.session_print()}: {e}")
                        continue
                    finally:
                        if _clear_cache or force_remake:
                            session.clear_cache()
                            torch.cuda.empty_cache()
                            gc.collect()

                if check_existing_configs:
                    if model.check_existing_score_from_hyps(
                        session,
                        spks_type=spks_type,
                        hyperparameters=hyperparameters,
                    ):
                        print(f"{isession} Score for stimspace {label} on session {session.session_print()} exists")
                        pass
                    else:
                        # print(f"{isession} !!!!! Score for stimspace {label} on session " f"{session.session_print()} does not exist")
                        pass
