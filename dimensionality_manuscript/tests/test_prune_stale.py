"""Tests for ``ResultsStore.plan_prune_stale``."""

from __future__ import annotations

import pytest

from dimensionality_manuscript.configs import CVPCAConfig, PlacefieldPredictionConfig
from dimensionality_manuscript.pipeline.store import ResultsStore

SESSION = "ATL012.2023-01-25.701"


@pytest.fixture
def store(tmp_path):
    return ResultsStore(tmp_path / "results.db")


def _put(store, cfg, session=SESSION, payload=None):
    store.put(session, cfg, payload if payload is not None else {"value": 1})


def test_current_configs_are_not_stale(store):
    cfg = PlacefieldPredictionConfig()
    _put(store, cfg)
    plan = store.plan_prune_stale()
    assert plan.plans == ()
    assert not plan.stale_keys


def test_superseded_schema_version_is_stale(store):
    current = PlacefieldPredictionConfig()
    old = PlacefieldPredictionConfig(schema_version="v0-retired")
    _put(store, current)
    _put(store, old)

    plan = store.plan_prune_stale()
    assert plan.stale_keys == {"placefield_prediction": (old.key(),)}
    assert len(plan.plans) == 1


def test_prune_removes_only_the_stale_row(store):
    current = PlacefieldPredictionConfig()
    old = PlacefieldPredictionConfig(schema_version="v0-retired")
    _put(store, current, payload={"keep": True})
    _put(store, old, payload={"keep": False})

    plan = store.plan_prune_stale()
    deleted = sum(store._execute_invalidate_plan(p) for p in plan.plans)

    assert deleted == 1
    assert store.get(SESSION, current) == {"keep": True}
    assert not store.has(SESSION, old)


def test_prune_deletes_the_stale_blob(store):
    old = PlacefieldPredictionConfig(schema_version="v0-retired")
    _put(store, old)
    blob = store._blob_path(store._uid(SESSION, old))
    assert blob.exists()

    for p in store.plan_prune_stale().plans:
        store._execute_invalidate_plan(p)
    assert not blob.exists()


def test_unregistered_analysis_type_is_never_deleted(store):
    """The critical safety rule: a missing registry entry must not mean 'delete'."""
    cfg = PlacefieldPredictionConfig()
    _put(store, cfg)
    with store._connect() as conn:
        conn.execute("UPDATE results SET analysis_type='retired_analysis'")

    plan = store.plan_prune_stale()
    assert plan.plans == ()
    assert plan.unknown_types == {"retired_analysis": 1}

    # and the row survives an execute
    for p in plan.plans:
        store._execute_invalidate_plan(p)
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1


def test_null_analysis_type_is_never_deleted(store):
    cfg = PlacefieldPredictionConfig()
    _put(store, cfg)
    with store._connect() as conn:
        conn.execute("UPDATE results SET analysis_type=NULL")

    plan = store.plan_prune_stale()
    assert plan.plans == ()
    assert "" in plan.unknown_types


def test_analyses_filter_restricts_the_sweep(store):
    stale_pfpred = PlacefieldPredictionConfig(schema_version="v0-retired")
    stale_cvpca = CVPCAConfig(schema_version="v0-retired")
    _put(store, stale_pfpred)
    _put(store, stale_cvpca)

    plan = store.plan_prune_stale(analysis_types=["placefield_prediction"])
    assert set(plan.stale_keys) == {"placefield_prediction"}

    both = store.plan_prune_stale()
    assert set(both.stale_keys) == {"placefield_prediction", "cvpca"}


def test_stale_error_rows_are_planned(store):
    old = PlacefieldPredictionConfig(schema_version="v0-retired")
    store.put_error(SESSION, old, "boom")

    plan = store.plan_prune_stale()
    assert plan.stale_keys == {"placefield_prediction": (old.key(),)}
    assert len(store.errors_matching_invalidate_plan(plan.plans[0])) == 1

    store._execute_invalidate_plan(plan.plans[0])
    assert not store.has_error(SESSION, old)


def test_empty_store_plans_nothing(store):
    plan = store.plan_prune_stale()
    assert plan.plans == ()
    assert plan.unknown_types == {}
