"""End-to-end test of the namespaced-key migration on a synthetic legacy store."""

from __future__ import annotations

import pickle
import sqlite3

import pytest

from dimensionality_manuscript.configs import CrossValidatedPlacefieldsConfig, PlacefieldPredictionConfig
from dimensionality_manuscript.pipeline.base import KEY_SCHEME, namespaced_key
from dimensionality_manuscript.pipeline.store import _ERROR_SCHEMA, _SCHEMA, ResultsStore, result_uid
from dimensionality_manuscript.scripts.migrate_namespace_keys import migrate

SESSION = "ATL012.2023-01-25.701"


def _legacy_uid(session_id: str, cfg) -> str:
    return result_uid(session_id, cfg.legacy_key())


@pytest.fixture
def legacy_store(tmp_path):
    """A store written under the pre-namespacing scheme, with one blob on disk."""
    db_path = tmp_path / "results.db"
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()

    cfg = CrossValidatedPlacefieldsConfig(schema_version="v1")
    uid = _legacy_uid(SESSION, cfg)
    (blob_dir / f"{uid}.pkl").write_bytes(pickle.dumps({"even_placefield": "cvpf-payload"}))

    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    conn.execute(_ERROR_SCHEMA)
    conn.execute(
        "INSERT INTO results (result_uid, session_id, analysis_key, analysis_summary, analysis_type, "
        "schema_version, result_stored, snapshot_path, computed_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (uid, SESSION, cfg.legacy_key(), cfg.summary(), cfg.display_name, "v1", 1, None, "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    return db_path, cfg


def test_colliding_configs_share_a_legacy_uid():
    """The precondition: these two classes were indistinguishable under the old scheme."""
    cvpf = CrossValidatedPlacefieldsConfig(schema_version="v1")
    pfpred = PlacefieldPredictionConfig(schema_version="v1")
    assert cvpf.legacy_key() == pfpred.legacy_key()
    assert _legacy_uid(SESSION, cvpf) == _legacy_uid(SESSION, pfpred)


def test_namespacing_separates_them():
    cvpf = CrossValidatedPlacefieldsConfig(schema_version="v1")
    pfpred = PlacefieldPredictionConfig(schema_version="v1")
    assert cvpf.key() != pfpred.key()


def test_store_refuses_unmigrated_database(legacy_store):
    db_path, _ = legacy_store
    with pytest.raises(RuntimeError, match="migrate_namespace_keys"):
        ResultsStore(db_path)


def test_dry_run_changes_nothing(legacy_store):
    db_path, cfg = legacy_store
    before = sqlite3.connect(db_path).execute("SELECT result_uid, analysis_key FROM results").fetchall()
    blobs_before = sorted(p.name for p in (db_path.parent / "blobs").glob("*.pkl"))

    migrate(db_path=db_path, apply=False)

    after = sqlite3.connect(db_path).execute("SELECT result_uid, analysis_key FROM results").fetchall()
    assert after == before
    assert sorted(p.name for p in (db_path.parent / "blobs").glob("*.pkl")) == blobs_before


def test_migration_preserves_the_result(legacy_store):
    db_path, cfg = legacy_store
    migrate(db_path=db_path, apply=True)

    store = ResultsStore(db_path)
    assert store.has(SESSION, cfg)
    assert store.get(SESSION, cfg) == {"even_placefield": "cvpf-payload"}


def test_migration_stamps_the_scheme(legacy_store):
    db_path, _ = legacy_store
    migrate(db_path=db_path, apply=True)
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT value FROM store_meta WHERE name='key_scheme'").fetchone()[0] == KEY_SCHEME


def test_migration_rewrites_key_and_uid(legacy_store):
    db_path, cfg = legacy_store
    migrate(db_path=db_path, apply=True)

    conn = sqlite3.connect(db_path)
    uid, key = conn.execute("SELECT result_uid, analysis_key FROM results").fetchone()
    assert key == cfg.key() == namespaced_key(cfg.display_name, cfg.legacy_key())
    assert uid == result_uid(SESSION, cfg.key())
    assert (db_path.parent / "blobs" / f"{uid}.pkl").exists()


def test_previously_colliding_config_is_now_absent_then_coexists(legacy_store):
    """The bug, and its fix, in one test."""
    db_path, cvpf = legacy_store
    migrate(db_path=db_path, apply=True)
    store = ResultsStore(db_path)

    pfpred = PlacefieldPredictionConfig(schema_version="v1")
    # Before the fix this returned True and reported 100% coverage against cvpf's rows.
    assert not store.has(SESSION, pfpred)

    store.put(SESSION, pfpred, {"placefield_prediction": "pfpred-payload"})

    # Both now coexist instead of overwriting one another.
    assert store.get(SESSION, pfpred) == {"placefield_prediction": "pfpred-payload"}
    assert store.get(SESSION, cvpf) == {"even_placefield": "cvpf-payload"}


def test_migration_is_idempotent(legacy_store):
    db_path, cfg = legacy_store
    migrate(db_path=db_path, apply=True)
    migrate(db_path=db_path, apply=True)

    store = ResultsStore(db_path)
    assert store.get(SESSION, cfg) == {"even_placefield": "cvpf-payload"}
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
