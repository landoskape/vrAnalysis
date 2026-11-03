"""Clear the joblib hyperparameter/score caches for regression models.

Those caches are keyed by model name, session, spks type, registry params, splits, method and
activity scaling -- and by nothing about the model's internal behavior (see
``RegressionModel._get_hyperparameter_cache_key``). So when a model's fit or prediction rule
changes, the stale caches are silently reused and have to be cleared by hand.

The cache key is rebuilt through the model object itself rather than by matching filenames, so
whatever the key includes is accounted for automatically. That means the ``--data-config`` and
``--activity-parameters`` used here must match the ones the results were produced with, or a
different (and probably empty) set of files is targeted.

``--all-cache-variants`` instead scans the two cache directories for the explicitly
named models. It covers every registry (including environment/trial registries),
session, scaling, spike type, method, and implementation revision. It does not load
sessions and cannot be combined with the ordinary combination-selection flags.

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

    # All variants for an explicitly named model, including old cache-key revisions
    python -m dimensionality_manuscript.scripts.clear_model_caches --models internal_placefield_1d_structured_gain_residual --all-cache-variants --dry-run
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from vrAnalysis.sessions import SpksTypes
from dimensionality_manuscript.configs.data_config import get_data_config, list_data_configs
from dimensionality_manuscript.registry import (
    ACTIVITY_PARAMETERS_NAMES,
    MODEL_NAMES,
    ModelName,
    PopulationRegistry,
    RegistryPaths,
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


def find_model_cache_variants(models: list[ModelName], registry_paths=None):
    """Find direct joblib cache files belonging to exactly the requested model names.

    Match the longest registered model-name prefix so clearing a base model does
    not accidentally clear registered variants with a longer shared prefix.
    Population caches and ResultsStore blobs are outside the two scanned folders.
    """
    unknown = set(models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    paths = registry_paths or RegistryPaths()
    directories = {"hyperparameters": paths.hyperparameter_path, "scores": paths.score_path}
    found = {name: {kind: [] for kind in directories} for name in dict.fromkeys(models)}
    names_by_length = sorted(MODEL_NAMES, key=len, reverse=True)
    for kind, directory in directories.items():
        directory = Path(directory).resolve()
        for path in sorted(directory.glob("*.joblib")):
            owner = next((name for name in names_by_length if path.name.startswith(name + "_")), None)
            if owner not in found:
                continue
            if path.resolve().parent != directory or not path.is_file():
                raise ValueError(f"Cache path is not a file within the intended cache directory: {path}")
            found[owner][kind].append(path)
    return found


def clear_model_cache_variants(models, *, registry_paths=None, clear_hyperparameters=True, clear_scores=True, dry_run=False):
    """Clear all variants of selected models without loading sessions or registries."""
    found = find_model_cache_variants(models, registry_paths)
    enabled = {"hyperparameters": clear_hyperparameters, "scores": clear_scores}
    counts = {kind: 0 for kind in enabled}
    print("Scope: ONLY the named models; ALL sessions, registry variants, spike types, scalings, methods, and cache revisions.")
    for name, entries in found.items():
        print(name)
        for kind, files in entries.items():
            if not enabled[kind]:
                continue
            counts[kind] += len(files)
            verb = "would clear" if dry_run else "clearing"
            print(f"  {kind}: {verb} {len(files)} file(s)")
            if files:
                print(f"    Directory: {files[0].parent}")
            if not dry_run:
                for path in files:
                    path.unlink()
    return counts


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
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None, help=f"Model names. Default: {' '.join(DEFAULT_MODELS)}")
    parser.add_argument("--spks-types", nargs="+", default=None, help=f"Spike types. Default: {' '.join(DEFAULT_SPKS_TYPES)}")
    parser.add_argument(
        "--activity-parameters",
        nargs="+",
        default=None,
        help=f"Named activity scalings. Available: {', '.join(ACTIVITY_PARAMETERS_NAMES)}. Default: {' '.join(DEFAULT_ACTIVITY_PARAMETERS)}",
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help=f"Named DataConfig the results were produced with. Available: {', '.join(list_data_configs())}. Default: default",
    )
    parser.add_argument("--method", default=None, help="Optimization method the results were produced with. Default: preferred")
    parser.add_argument(
        "--all-cache-variants",
        action="store_true",
        help="Clear ALL cached variants of the explicitly named --models, including trial registries and old revisions; no session loading",
    )
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

    if args.all_cache_variants:
        if not args.models:
            parser.error("--all-cache-variants requires explicit --models")
        if any(value is not None for value in (args.spks_types, args.activity_parameters, args.data_config, args.method, args.sessions_file)):
            parser.error("--all-cache-variants cannot be combined with spike/scaling/data-config/method/session filters")
        counts = clear_model_cache_variants(
            args.models,
            clear_hyperparameters=not args.scores_only,
            clear_scores=not args.hyperparameters_only,
            dry_run=args.dry_run,
        )
    else:
        counts = clear_model_caches(
            models=args.models or list(DEFAULT_MODELS),
            spks_types=args.spks_types or list(DEFAULT_SPKS_TYPES),
            activity_parameters_names=args.activity_parameters or list(DEFAULT_ACTIVITY_PARAMETERS),
            data_config_name=args.data_config or "default",
            method=args.method or "preferred",
            clear_hyperparameters=not args.scores_only,
            clear_scores=not args.hyperparameters_only,
            sessions_file=args.sessions_file,
            dry_run=args.dry_run,
        )

    verb = "Would clear" if args.dry_run else "Cleared"
    print(f"{verb} {counts['hyperparameters']} hyperparameter cache entrie(s) and {counts['scores']} score cache entrie(s)")


if __name__ == "__main__":
    main()
