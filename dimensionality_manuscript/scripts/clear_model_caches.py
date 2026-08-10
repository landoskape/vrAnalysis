"""Clear the joblib hyperparameter/score caches for regression models.

Those caches are keyed by model name, session, spks type, registry params, splits, method and
activity scaling -- and by nothing about the model's internal behavior (see
``RegressionModel._get_hyperparameter_cache_key``). So when a model's fit or prediction rule
changes, the stale caches are silently reused and have to be cleared by hand.

The cache key is rebuilt through the model object itself rather than by matching filenames, so
whatever the key includes is accounted for automatically. That means the ``--data-config`` and
``--activity-parameters`` used here must match the ones the results were produced with, or a
different (and probably empty) set of files is targeted.

Usage
-----
    python -m dimensionality_manuscript.scripts.clear_model_caches
        [--models NAME ...] [--spks-types TYPE ...] [--activity-parameters NAME ...]
        [--data-config NAME] [--method NAME] [--hyperparameters-only | --scores-only]
        [--sessions-file path.json] [--dry-run]

Examples
--------
    # See what would be removed for the structured-gain models, without touching anything
    python -m dimensionality_manuscript.scripts.clear_model_caches --dry-run

    # Actually clear them
    python -m dimensionality_manuscript.scripts.clear_model_caches

    # On MYRIAD (no pyodbc/Access), read the exported session list instead of the database
    # -- same file as sge_submit.py / sge_worker.py, see MYRIAD_SETUP.md
    python -m dimensionality_manuscript.scripts.clear_model_caches --sessions-file ~/vrAnalysis/sessions.json

    # --sessions-file falls back to $DIM_MANUSCRIPT_SESSIONS_FILE if not passed explicitly
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from vrAnalysis.sessions import SpksTypes
from dimensionality_manuscript.configs.data_config import get_data_config, list_data_configs
from dimensionality_manuscript.registry import (
    ACTIVITY_PARAMETERS_NAMES,
    ModelName,
    PopulationRegistry,
    get_model,
)
from dimensionality_manuscript.scripts.run import collect_sessions_auto

#: Models cleared when ``--models`` is not given.
DEFAULT_MODELS: tuple[ModelName, ...] = (
    "external_placefield_1d_structured_gain",
    "internal_placefield_1d_structured_gain",
)

#: Spike types cleared when ``--spks-types`` is not given.
DEFAULT_SPKS_TYPES: tuple[SpksTypes, ...] = ("sigrebase",)

#: Activity scalings cleared when ``--activity-parameters`` is not given -- every one the
#: regression pipeline sweeps (``configs.regression.VALID_ACTIVITY_PARAMETERS``). Each scaling is a
#: separate cache entry, and clearing only the one you happened to think of is a silent no-op, so
#: the default covers all of them. Names with nothing cached simply report zero.
DEFAULT_ACTIVITY_PARAMETERS: tuple[str, ...] = ("default", "preserved", "std")


def clear_model_caches(
    models: list[ModelName],
    spks_types: list[SpksTypes],
    activity_parameters_names: list[str],
    data_config_name: str = "default",
    method: str = "preferred",
    clear_hyperparameters: bool = True,
    clear_scores: bool = True,
    sessions_file: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Remove cached hyperparameters and scores for every requested combination.

    Parameters
    ----------
    models : list of ModelName
        Models whose caches are cleared.
    spks_types : list of SpksTypes
        Spike types to clear. Each is a separate cache entry.
    activity_parameters_names : list of str
        Named activity scalings to clear. Each is a separate cache entry.
    data_config_name : str
        Named DataConfig whose registry parameters were used to produce the results. The registry
        parameter hash is part of the cache key, so this must match.
    method : str
        Optimization method the results were produced with, or "preferred" for the model's own.
    clear_hyperparameters : bool
        Whether to clear the hyperparameter cache.
    clear_scores : bool
        Whether to clear the score cache.
    sessions_file : Path or None
        JSON file from ``export_sessions.py`` to use instead of the vrSessions database.
    dry_run : bool
        If True, report what would be removed without removing it.

    Returns
    -------
    dict
        Counts of cache entries found, under keys ``hyperparameters`` and ``scores``.
    """
    registry = PopulationRegistry(registry_params=get_data_config(data_config_name).to_registry_params())
    counts = {"hyperparameters": 0, "scores": 0}

    for spks_type in spks_types:
        sessions = collect_sessions_auto(sessions_file, session_params=dict(spks_type=spks_type))
        for model_name in models:
            for activity_parameters_name in activity_parameters_names:
                model = get_model(model_name, registry, activity_parameters=activity_parameters_name)
                label = f"{model_name} | spks={spks_type} | ap={activity_parameters_name}"

                found = {"hyperparameters": 0, "scores": 0}
                for session in tqdm(sessions, desc=label, leave=False):
                    if clear_hyperparameters and model.check_existing_hyperparameters(session, spks_type=spks_type, method=method):
                        found["hyperparameters"] += 1
                        if not dry_run:
                            model.clear_cached_hyperparameter(session, spks_type=spks_type, method=method)

                    if clear_scores and model.check_existing_score(session, spks_type=spks_type, method=method):
                        found["scores"] += 1
                        if not dry_run:
                            model.clear_cached_score(session, spks_type=spks_type, method=method)

                verb = "would clear" if dry_run else "cleared"
                print(f"  {label}: {verb} {found['hyperparameters']} hyperparameter(s), {found['scores']} score(s) of {len(sessions)} session(s)")
                counts["hyperparameters"] += found["hyperparameters"]
                counts["scores"] += found["scores"]

    return counts


def main():
    parser = argparse.ArgumentParser(description="Clear cached hyperparameters and scores for regression models")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help=f"Model names. Default: {' '.join(DEFAULT_MODELS)}")
    parser.add_argument("--spks-types", nargs="+", default=list(DEFAULT_SPKS_TYPES), help=f"Spike types. Default: {' '.join(DEFAULT_SPKS_TYPES)}")
    parser.add_argument(
        "--activity-parameters",
        nargs="+",
        default=list(DEFAULT_ACTIVITY_PARAMETERS),
        help=f"Named activity scalings. Available: {', '.join(ACTIVITY_PARAMETERS_NAMES)}. Default: {' '.join(DEFAULT_ACTIVITY_PARAMETERS)}",
    )
    parser.add_argument(
        "--data-config",
        default="default",
        help=f"Named DataConfig the results were produced with. Available: {', '.join(list_data_configs())}. Default: default",
    )
    parser.add_argument("--method", default="preferred", help="Optimization method the results were produced with. Default: preferred")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--hyperparameters-only", action="store_true", help="Clear only the hyperparameter cache")
    cache_group.add_argument("--scores-only", action="store_true", help="Clear only the score cache")
    parser.add_argument(
        "--sessions-file",
        type=Path,
        default=None,
        help="JSON file from export_sessions.py to use instead of the vrSessions Access database (e.g. on MYRIAD)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be removed without removing it")
    args = parser.parse_args()

    counts = clear_model_caches(
        models=args.models,
        spks_types=args.spks_types,
        activity_parameters_names=args.activity_parameters,
        data_config_name=args.data_config,
        method=args.method,
        clear_hyperparameters=not args.scores_only,
        clear_scores=not args.hyperparameters_only,
        sessions_file=args.sessions_file,
        dry_run=args.dry_run,
    )

    verb = "Would clear" if args.dry_run else "Cleared"
    print(f"{verb} {counts['hyperparameters']} hyperparameter cache entrie(s) and {counts['scores']} score cache entrie(s)")


if __name__ == "__main__":
    main()
