"""Resume the already-authorized PF+E refit without invalidating completed work."""
from pathlib import Path
import json

from dimensionality_manuscript.configs.trial_regression import GAIN_RESIDUAL_MODELS, TrialPlacecellRegressionConfig
from dimensionality_manuscript.pipeline import AnalysisPlan, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.scripts.run import collect_sessions


def main():
    configs = [TrialPlacecellRegressionConfig(model_name=m) for m in GAIN_RESIDUAL_MODELS]
    keys = {c.key() for c in configs}
    store = ResultsStore(RegistryPaths().pipeline_v2_db_path)
    sessions = collect_sessions()
    before = {r["result_uid"]: r for r in store.summary_table() if r["analysis_key"] not in keys}
    print("Resuming missing jobs only; schema v2; no invalidation.", flush=True)
    plan = AnalysisPlan(configs)
    plan.analyze(sessions, store, dry_run=True)
    plan.analyze(sessions, store, n_jobs=4, snapshot_codebase=False)
    after = store.summary_table()
    assert before == {r["result_uid"]: r for r in after if r["analysis_key"] not in keys}
    status = {}
    for c in configs:
        rows = [r for r in after if r["analysis_key"] == c.key()]
        errors = [r for r in store.get_errors(analysis_type=c.display_name) if r["analysis_key"] == c.key()]
        status[c.model_name] = dict(results=len(rows), expected=len(sessions), errors=len(errors))
        print(c.model_name, status[c.model_name], flush=True)
    complete = all(s["results"] == s["expected"] and not s["errors"] for s in status.values())
    Path("artifacts/resume_trial_residual_status.json").write_text(json.dumps(dict(complete=complete, models=status, unrelated_results_unchanged=True), indent=2))
    assert complete, "Inspect stored errors and resume missing jobs after addressing them."
    print("Verified: all targeted results refreshed; unrelated results unchanged during resume.", flush=True)


if __name__ == "__main__":
    main()

