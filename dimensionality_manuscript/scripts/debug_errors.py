"""Re-run recorded pipeline errors locally for debugging.

Pulls failing (session, config) pairs straight from the error table and
re-runs ``config.process(session, registry)`` directly, printing tracebacks
as they happen and a summary table at the end. Does not touch the store
unless ``--save`` is passed.

Usage
-----
    python -m dimensionality_manuscript.scripts.debug_errors
        [--analyses subspace ...] [--param-filters key=value ...]
        [--sessions UID1 UID2 ...] [--sessions-file path.json] [--schema-version v5]
        [--max-attempts N] [--by-config | --by-session]
        [--save] [--full-traceback]

Examples
--------
    # Try one error from covcov_crossvalidated_subspace with raw activity params
    python -m dimensionality_manuscript.scripts.debug_errors --analyses subspace --param-filters subspace_name=covcov_crossvalidated_subspace activity_parameters_name=raw --max-attempts 1

    # Same, but iterate session-by-session and save successes back to the store
    python -m dimensionality_manuscript.scripts.debug_errors --analyses subspace --by-session --save

    # On Myriad (no pyodbc/Access available), use the exported session list instead
    # of the vrSessions database — same file used by sge_submit.py / sge_worker.py,
    # see MYRIAD_SETUP.md (default location ~/vrAnalysis/sessions.json):
    python -m dimensionality_manuscript.scripts.debug_errors --sessions-file ~/vrAnalysis/sessions.json --analyses subspace

    # --sessions-file falls back to $DIM_MANUSCRIPT_SESSIONS_FILE if not passed explicitly
"""

import argparse
import os
import traceback as _traceback
from collections import defaultdict
from pathlib import Path

from dimensionality_manuscript.registry import PopulationRegistry, RegistryPaths
from dimensionality_manuscript import ResultsStore
from dimensionality_manuscript.pipeline.store import _analysis_config_classes
from dimensionality_manuscript.scripts.run import (
    build_analysis_configs,
    collect_sessions,
    collect_sessions_from_file,
    _parse_param_filters,
)

REGISTRY_PATHS = RegistryPaths()


def _analysis_display_names(analyses: list[str] | None) -> list[str] | None:
    """Resolve run.py analysis keys to stored ``analysis_type`` values."""
    if analyses is None:
        return None
    configs = build_analysis_configs(include=analyses)
    return list({cfg.display_name for cfg in configs})


def _matches_param_filters(cfg, param_filters: dict | None) -> bool:
    if not param_filters:
        return True
    for key, value in param_filters.items():
        if not hasattr(cfg, key):
            return False
        if str(getattr(cfg, key)) != str(value):
            return False
    return True


def _reconstruct_jobs(
    store: ResultsStore,
    analyses: list[str] | None,
    param_filters: dict | None,
    sessions: list[str] | None,
    schema_version: str | None,
    sessions_file: Path | None,
) -> list[tuple]:
    """Return list of (session, cfg, error_row) reconstructed from the error table."""
    analysis_types = _analysis_display_names(analyses)
    errors = store.get_errors(session_ids=sessions, schema_version=schema_version)
    if analysis_types is not None:
        allowed = set(analysis_types)
        errors = [e for e in errors if e.get("analysis_type") in allowed]
    if not errors:
        return []

    config_classes = _analysis_config_classes()
    all_sessions = collect_sessions_from_file(sessions_file) if sessions_file is not None else collect_sessions()
    uid_to_session = {s.session_uid: s for s in all_sessions}

    jobs = []
    missing_sessions = set()
    bad_keys = []
    for row in errors:
        cfg_cls = config_classes.get(row["analysis_type"])
        if cfg_cls is None:
            bad_keys.append(row)
            continue
        try:
            cfg = cfg_cls.from_key(row["analysis_key"])
        except KeyError:
            bad_keys.append(row)
            continue
        if not _matches_param_filters(cfg, param_filters):
            continue
        session = uid_to_session.get(row["session_id"])
        if session is None:
            missing_sessions.add(row["session_id"])
            continue
        jobs.append((session, cfg, row))

    for sid in sorted(missing_sessions):
        print(f"  [warn] session {sid!r} not found in vrSessions database, skipping its errors")
    if bad_keys:
        print(f"  [warn] {len(bad_keys)} error row(s) had no matching config class/key, skipping")

    return jobs


def _order_jobs(jobs: list[tuple], by_session: bool) -> list[tuple]:
    """Group jobs by config or by session, then flatten in group order."""
    group_key = (lambda j: j[0].session_uid) if by_session else (lambda j: j[1].key())
    groups: dict = defaultdict(list)
    order: list = []
    for job in jobs:
        k = group_key(job)
        if k not in groups:
            order.append(k)
        groups[k].append(job)
    ordered = []
    for k in order:
        ordered.extend(groups[k])
    return ordered


def debug_errors(
    analyses: list[str] | None = None,
    param_filters: dict | None = None,
    sessions: list[str] | None = None,
    schema_version: str | None = None,
    max_attempts: int | None = None,
    by_session: bool = False,
    save: bool = False,
    full_traceback: bool = False,
    sessions_file: Path | None = None,
):
    """Re-run recorded errors and report pass/fail with tracebacks.

    Parameters
    ----------
    analyses : list of str or None
        Analysis config keys to include (same as ``run.py --analyses``). None = all.
    param_filters : dict or None
        Fixed param values to filter reconstructed configs, e.g. ``{"subspace_name": "covcov_subspace"}``.
    sessions : list of str or None
        Session uids to restrict to. None = every session present in matching error rows.
    schema_version : str or None
        Passthrough filter to ``ResultsStore.get_errors``.
    max_attempts : int or None
        Stop after attempting this many (session, config) pairs. None = no limit.
    by_session : bool
        If True, group/iterate by session first, then config. Default groups by config first.
    save : bool
        If True, write successful results back to the store and clear the error row.
    full_traceback : bool
        If True, print the full traceback immediately on each failure (in addition to the summary table).
    sessions_file : Path or None
        JSON file produced by ``export_sessions.py`` to use instead of the vrSessions Access
        database (e.g. on systems like Myriad where ``pyodbc``/Access is unavailable).
    """
    db_path = REGISTRY_PATHS.pipeline_v2_db_path
    store = ResultsStore(db_path)

    jobs = _reconstruct_jobs(store, analyses, param_filters, sessions, schema_version, sessions_file)
    if not jobs:
        print("No matching errors found.")
        return

    jobs = _order_jobs(jobs, by_session=by_session)
    if max_attempts is not None:
        jobs = jobs[:max_attempts]

    print(f"Attempting {len(jobs)} (session, config) pair(s) | order={'by-session' if by_session else 'by-config'} | save={save}\n")

    registries: dict = {}
    successes = 0
    failures: list[dict] = []

    for i, (session, cfg, row) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {session.session_uid} | {cfg.summary()} | key={cfg.key()}")
        dk = cfg.data_config_name
        if dk not in registries:
            registries[dk] = PopulationRegistry(registry_params=cfg.data_config.to_registry_params())

        try:
            session.params.spks_type = cfg.data_config.spks_type
            result = cfg.process(session, registries[dk])
            if save:
                store.put(session.session_uid, cfg, result)
                store.clear_error(session.session_uid, cfg)
            print("  OK")
            successes += 1
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            trace = _traceback.format_exc()
            last_frame = _traceback.extract_tb(e.__traceback__)[-1]
            location = f"{Path(last_frame.filename).name}:{last_frame.lineno} in {last_frame.name}"
            print(f"  FAILED: {msg}\n    at {location}")
            if full_traceback:
                print(trace)
            failures.append(
                dict(
                    session_uid=session.session_uid,
                    summary=cfg.summary(),
                    key=cfg.key(),
                    message=msg,
                    location=location,
                    traceback=trace,
                )
            )
        finally:
            session.clear_cache()

    print(f"\nDone: {successes}/{len(jobs)} succeeded, {len(failures)} failed.\n")
    if failures:
        print("Failure summary:")
        for f in failures:
            print(f"  {f['session_uid']} | {f['summary']} | key={f['key']} | {f['message']} | at {f['location']}")
            if full_traceback:
                print(f["traceback"])


def main():
    parser = argparse.ArgumentParser(description="Re-run recorded pipeline errors locally for debugging")
    parser.add_argument("--analyses", nargs="+", default=None, help="Which analysis configs to include (same keys as run.py --analyses)")
    parser.add_argument(
        "--param-filters",
        nargs="+",
        metavar="KEY=VALUE",
        help="Filter reconstructed configs by fixed param values, e.g. --param-filters subspace_name=covcov_subspace",
    )
    parser.add_argument("--sessions", nargs="+", default=None, help="Session uids to restrict to. Default: all sessions with matching errors")
    parser.add_argument(
        "--sessions-file",
        type=Path,
        default=None,
        help="JSON file from export_sessions.py to use instead of the vrSessions Access database (e.g. on Myriad)",
    )
    parser.add_argument("--schema-version", default=None, help="Filter errors by schema_version")
    parser.add_argument("--max-attempts", type=int, default=None, help="Stop after attempting this many (session, config) pairs")
    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument("--by-config", action="store_true", help="Group/iterate by config first, then session (default)")
    order_group.add_argument("--by-session", action="store_true", help="Group/iterate by session first, then config")
    parser.add_argument("--save", action="store_true", help="Write successful results back to the store and clear the error row")
    parser.add_argument("--full-traceback", action="store_true", help="Print full traceback immediately on each failure")
    args = parser.parse_args()

    sessions_file = args.sessions_file
    if sessions_file is None:
        env_val = os.environ.get("DIM_MANUSCRIPT_SESSIONS_FILE")
        if env_val:
            sessions_file = Path(env_val)

    debug_errors(
        analyses=args.analyses,
        param_filters=_parse_param_filters(args.param_filters),
        sessions=args.sessions,
        schema_version=args.schema_version,
        max_attempts=args.max_attempts,
        by_session=args.by_session,
        save=args.save,
        full_traceback=args.full_traceback,
        sessions_file=sessions_file,
    )


if __name__ == "__main__":
    main()
