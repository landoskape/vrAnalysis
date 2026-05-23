"""Tests for dimensionality_manuscript pipeline."""

import pathlib
import tempfile

from dimensionality_manuscript import (
    AnalysisPlan,
    CVPCAConfig,
    DataConfig,
    Job,
    PopulationConfig,
    RegressionConfig,
    ResultsStore,
    SubspaceConfig,
    get_data_config,
    list_data_configs,
    result_uid,
)


class FakeSession:
    session_uid = "test_session"

    class params:
        spks_type = "oasis"

    def clear_cache(self):
        pass


# -- AnalysisConfigBase / CVPCAConfig ------------------------------------------


def test_config_generation():
    configs = CVPCAConfig.generate_variations()
    assert len(configs) == 24


def test_key_stability():
    c1 = CVPCAConfig(center=True, normalize=True)
    c2 = CVPCAConfig(center=True, normalize=True)
    assert c1.key() == c2.key()


def test_key_changes_with_params():
    c1 = CVPCAConfig(center=True)
    c2 = CVPCAConfig(center=False)
    assert c1.key() != c2.key()


def test_config_summary():
    c = CVPCAConfig(center=False, normalize=False, reliability_threshold=None, fraction_active_threshold=0.05)
    s = c.summary()
    assert "center=False" in s
    assert "norm=False" in s
    assert "frac=0.05" in s
    assert "rel=None" in s
    assert "bins=100" in s


def test_from_key():
    original = CVPCAConfig(center=False, fraction_active_threshold=0.05)
    recovered = CVPCAConfig.from_key(original.key())
    assert recovered == original


def test_from_key_unknown():
    try:
        CVPCAConfig.from_key("0000000000000000")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# -- DataConfig ----------------------------------------------------------------


def test_data_config_key_and_summary():
    dc = get_data_config("default")
    assert len(dc.key()) == 16
    assert dc.summary() == "default"


def test_data_config_to_registry_params():
    dc = DataConfig(
        name="custom",
        speed_threshold=2.0,
        time_split_groups=4,
        time_split_relative_size=(4, 4, 1, 1),
        time_split_chunks_per_group=10,
        time_split_num_buffer=3,
        cell_split_force_even=False,
        spks_type="oasis",
    )
    rp = dc.to_registry_params()
    assert rp.speed_threshold == 2.0


def test_get_data_config_unknown():
    try:
        get_data_config("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_list_data_configs():
    names = list_data_configs()
    assert "default" in names


# -- result_uid ----------------------------------------------------------------


def test_result_uid_deterministic():
    uid1 = result_uid("ses1", "ak1")
    uid2 = result_uid("ses1", "ak1")
    assert uid1 == uid2
    assert len(uid1) == 16


def test_result_uid_varies():
    uid1 = result_uid("ses1", "ak1")
    uid2 = result_uid("ses1", "ak2")
    assert uid1 != uid2


# -- ResultsStore --------------------------------------------------------------


def test_store_round_trip():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        acfg = CVPCAConfig()
        sid = "test_session"

        assert not store.has(sid, acfg)
        store.put(sid, acfg, {"foo": 42}, snapshot_path="/snap.zip")
        assert store.has(sid, acfg)
        assert store.get(sid, acfg) == {"foo": 42}

        uid = store._uid(sid, acfg)
        assert store._blob_path(uid).exists()


def test_store_get_by_uid():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        acfg = CVPCAConfig()
        sid = "test_session"

        store.put(sid, acfg, {"bar": 99})
        uid = store._uid(sid, acfg)
        assert store.get_by_uid(uid) == {"bar": 99}
        assert store.get_by_uid("nonexistent") is None


def test_store_none_completion_marker():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        acfg = CVPCAConfig(center=False)
        sid = "test_session"

        store.put(sid, acfg, None)
        assert store.has(sid, acfg)
        assert store.get(sid, acfg) is None

        uid = store._uid(sid, acfg)
        assert not store._blob_path(uid).exists()


def test_store_result_stored_flag():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        sid = "test_session"

        acfg_in = CVPCAConfig()
        store.put(sid, acfg_in, {"data": 1}, result_stored=True)

        acfg_ext = CVPCAConfig(center=False)
        store.put(sid, acfg_ext, {"data": 2}, result_stored=False)

        df = store.summary_table(as_dataframe=True)
        assert "result_stored" in df.columns
        assert "result_blob" not in df.columns
        stored_vals = dict(zip(df["result_uid"], df["result_stored"]))
        uid_in = store._uid(sid, acfg_in)
        uid_ext = store._uid(sid, acfg_ext)
        assert stored_vals[uid_in] == 1
        assert stored_vals[uid_ext] == 0

        # result_stored=True writes file; result_stored=False does not
        assert store._blob_path(uid_in).exists()
        assert not store._blob_path(uid_ext).exists()


def test_store_summary_table():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        acfg1 = CVPCAConfig()
        acfg2 = CVPCAConfig(center=False)
        sid = "test_session"

        store.put(sid, acfg1, {"a": 1}, snapshot_path="/snap.zip")
        store.put(sid, acfg2, None)

        records = store.summary_table()
        assert len(records) == 2
        assert records[0]["snapshot_path"] == "/snap.zip"

        df = store.summary_table(as_dataframe=True)
        assert len(df) == 2
        assert "result_uid" in df.columns


def test_store_coverage():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        acfg1 = CVPCAConfig()
        acfg2 = CVPCAConfig(center=False)

        store.put("test_session", acfg1, {"a": 1})
        store.put("test_session", acfg2, None)

        cov = store.coverage([FakeSession()], [acfg1, acfg2])
        assert cov == 1.0


def test_store_invalidate():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        acfg = CVPCAConfig()
        sid = "test_session"

        store.put(sid, acfg, {"a": 1})
        uid = store._uid(sid, acfg)
        assert store.has(sid, acfg)
        assert store._blob_path(uid).exists()
        store.invalidate(analysis_type="cvpca")
        assert not store.has(sid, acfg)
        assert not store._blob_path(uid).exists()


def test_store_invalidate_requires_filter():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        try:
            store.invalidate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


def test_store_invalidate_all():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        store.put("s1", CVPCAConfig(), {"a": 1})
        store.put("s2", CVPCAConfig(), {"b": 2})
        store.invalidate_all()
        assert len(store.summary_table()) == 0
        assert list(store._blob_dir.glob("*.pkl")) == []


# -- Job -----------------------------------------------------------------------


def test_job_result_uid():
    job = Job(session=FakeSession(), analysis_config=CVPCAConfig())
    assert len(job.result_uid) == 16
    expected = result_uid("test_session", CVPCAConfig().key())
    assert job.result_uid == expected


def test_job_repr():
    job = Job(session=FakeSession(), analysis_config=CVPCAConfig())
    r = repr(job)
    assert "test_session" in r
    assert "cvpca" in r


# -- RegistryPaths -------------------------------------------------------------


def test_registry_paths_pipeline_v2():
    from dimensionality_manuscript.registry import RegistryPaths

    rp = RegistryPaths()
    assert rp.pipeline_v2_path.exists()
    assert str(rp.pipeline_v2_db_path).endswith("results.db")


# -- SubspaceConfig ------------------------------------------------------------


def test_subspace_config_generation():
    configs = SubspaceConfig.generate_variations()
    # 4 subspace names (excl. cvpca) × 1 spks_type × 1 num_bins × 2 smooth_widths = 8
    assert len(configs) == 8
    for c in configs:
        assert c.subspace_name != "cvpca_subspace"


def test_subspace_config_summary():
    c = SubspaceConfig()
    s = c.summary()
    assert "subspace" in s
    assert "bins=100" in s


def test_subspace_validate_rejects_cvpca():
    try:
        SubspaceConfig(subspace_name="cvpca_subspace")
        assert False
    except ValueError:
        pass


def test_subspace_from_key():
    c = SubspaceConfig(subspace_name="pca_subspace", smooth_width=5.0)
    assert SubspaceConfig.from_key(c.key()) == c


# -- PopulationConfig ----------------------------------------------------------


def test_population_config_generation():
    configs = PopulationConfig.generate_variations()
    assert len(configs) == 1


def test_population_config_key_stable():
    assert PopulationConfig().key() == PopulationConfig().key()


if __name__ == "__main__":
    import sys

    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failures += 1
    print(f"\n{len(test_funcs) - failures}/{len(test_funcs)} passed")
    sys.exit(failures)
