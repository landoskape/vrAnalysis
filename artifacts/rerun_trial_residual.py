"""One-time PF+E refresh: invalidate only gain-residual trial configs, then rerun."""

from dimensionality_manuscript.configs.trial_regression import (
    GAIN_RESIDUAL_MODELS,
    TrialPlacecellRegressionConfig,
)
from dimensionality_manuscript.pipeline import AnalysisPlan, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.scripts.clear_model_caches import clear_model_cache_variants
from dimensionality_manuscript.scripts.run import collect_sessions


def main():
    configs = [TrialPlacecellRegressionConfig(model_name=name) for name in GAIN_RESIDUAL_MODELS]
    assert all(cfg.schema_version == "v2" for cfg in configs)
    store = ResultsStore(RegistryPaths().pipeline_v2_db_path)
    sessions = collect_sessions()
    keys = {cfg.key() for cfg in configs}
    before = store.summary_table()
    unaffected = {r["result_uid"]: r for r in before if r["analysis_key"] not in keys}
    prior_sessions = {r["session_id"] for r in before if r["analysis_key"] in keys}
    assert prior_sessions <= {s.session_uid for s in sessions}
    print(f"Targeted rerun: {len(configs)} configs, {len(sessions)} sessions, schema v2", flush=True)
    for cfg in configs:
        plan = store.plan_invalidate(analysis_cfg=cfg, schema_version="v2")
        print(f"Invalidating {cfg.model_name}: {len(store.rows_matching_invalidate_plan(plan))} rows", flush=True)
    clear_model_cache_variants(list(GAIN_RESIDUAL_MODELS))
    for cfg in configs:
        print(f"Deleted {store.invalidate(analysis_cfg=cfg, schema_version='v2')} results for {cfg.model_name}", flush=True)
    AnalysisPlan(configs).analyze(sessions, store, n_jobs=4, snapshot_codebase=False)
    after = store.summary_table()
    assert unaffected == {r["result_uid"]: r for r in after if r["analysis_key"] not in keys}
    for cfg in configs:
        rows = [r for r in after if r["analysis_key"] == cfg.key()]
        errors = [r for r in store.get_errors(analysis_type=cfg.display_name) if r["analysis_key"] == cfg.key()]
        print(f"FINAL {cfg.model_name}: {len(rows)} results, {len(errors)} errors", flush=True)
        assert len(rows) == len(sessions) and not errors
    print("Verified: rerun complete and unrelated results unchanged.", flush=True)


if __name__ == "__main__":
    main()
