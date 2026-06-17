"""Tests for dimensionality_manuscript pipeline."""

import pathlib
import tempfile
import warnings

from unittest.mock import patch

import numpy as np

from dimensionality_manuscript import (
    AnalysisPlan,
    CVPCAConfig,
    DataConfig,
    Job,
    PopulationConfig,
    RegressionConfig,
    ResultsAggregator,
    ResultsStore,
    SubspaceConfig,
    average_array_by_mouse,
    average_by_mouse,
    get_data_config,
    list_data_configs,
    result_uid,
)


class FakeSession:
    session_uid = "test_session"
    mouse_name = "mouse_a"

    class params:
        spks_type = "oasis"

    def clear_cache(self):
        pass


class FakeSessionB:
    session_uid = "test_session_b"
    mouse_name = "mouse_b"

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


def test_plan_invalidate_param_filters_matches_delete():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        cfg_default = RegressionConfig(activity_parameters_name="default")
        cfg_raw = RegressionConfig(activity_parameters_name="raw")
        sid = "test_session"
        store.put(sid, cfg_default, {"score": 1})
        store.put(sid, cfg_raw, {"score": 2})

        plan = store.plan_invalidate(
            analysis_type="regression",
            param_filters={"activity_parameters_name": "raw"},
        )
        rows = store.rows_matching_invalidate_plan(plan)
        assert len(rows) == 1
        assert rows[0]["analysis_key"] == cfg_raw.key()

        n = store.invalidate(analysis_type="regression", param_filters={"activity_parameters_name": "raw"})
        assert n == 1
        assert store.has(sid, cfg_default)
        assert not store.has(sid, cfg_raw)


def test_store_invalidate_param_filters():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        cfg_default = RegressionConfig(activity_parameters_name="default")
        cfg_raw = RegressionConfig(activity_parameters_name="raw")
        sid = "test_session"
        store.put(sid, cfg_default, {"score": 1})
        store.put(sid, cfg_raw, {"score": 2})
        n = store.invalidate(
            analysis_type="regression",
            param_filters={"activity_parameters_name": "raw"},
        )
        assert n == 1
        assert store.has(sid, cfg_default)
        assert not store.has(sid, cfg_raw)
        assert not store._blob_path(store._uid(sid, cfg_raw)).exists()


def test_generate_variations_matching():
    configs = RegressionConfig.generate_variations_matching({"activity_parameters_name": "raw"})
    assert configs
    assert all(c.activity_parameters_name == "raw" for c in configs)
    assert len(configs) < len(RegressionConfig.generate_variations())
    assert len({c.key() for c in configs}) == len(configs)


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


# -- ResultsStore summary filters / cache --------------------------------------


def test_store_summary_table_analysis_type_filter():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        store.put("s1", CVPCAConfig(), {"a": 1})
        store.put("s2", SubspaceConfig(), {"b": 2})

        cvpca_rows = store.summary_table(analysis_type="cvpca")
        assert len(cvpca_rows) == 1
        assert cvpca_rows[0]["analysis_type"] == "cvpca"


def test_store_summary_table_session_ids_filter():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        store.put("s1", CVPCAConfig(), {"a": 1})
        store.put("s2", CVPCAConfig(center=False), {"a": 2})

        rows = store.summary_table(session_ids=["s1"])
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"


def test_store_blob_cache():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        acfg = CVPCAConfig()
        store.put("s1", acfg, {"x": 1})
        uid = store._uid("s1", acfg)

        with patch("dimensionality_manuscript.pipeline.store.pickle.loads", wraps=__import__("pickle").loads) as loads:
            assert store.get_by_uid(uid) == {"x": 1}
            assert store.get_by_uid(uid) == {"x": 1}
            assert loads.call_count == 1
        assert uid in store._blob_cache


# -- ResultsAggregator lazy loading --------------------------------------------


def _populate_cvpca_grid(store, sessions):
    """Two sessions × center True/False with distinct pad values."""
    for ses in sessions:
        for center in (True, False):
            cfg = CVPCAConfig(center=center)
            val = 1.0 if center else 2.0
            store.put(
                ses.session_uid,
                cfg,
                {
                    "reg_covariances": np.array([val, val + 1.0]),
                    "trial_folds": {"marker": center},
                },
            )


def test_aggregator_lazy_eager_parity():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        sessions = [FakeSession(), FakeSessionB()]
        _populate_cvpca_grid(store, sessions)

        eager = ResultsAggregator(CVPCAConfig, store, sessions, lazy=False)
        lazy = ResultsAggregator(CVPCAConfig, store, sessions, lazy=True)
        lazy.load_all(load_objects=True)

        for key in eager.arrays:
            if key in lazy.arrays:
                np.testing.assert_array_equal(eager.arrays[key], lazy.arrays[key])
        assert "trial_folds" in eager.objects
        assert "trial_folds" in lazy.objects


def test_aggregator_lazy_sel_loads_slice_only():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        sessions = [FakeSession(), FakeSessionB()]
        _populate_cvpca_grid(store, sessions)

        lazy = ResultsAggregator(CVPCAConfig, store, sessions, lazy=True, keys=["reg_covariances"])
        with patch.object(store, "get_by_uid", wraps=store.get_by_uid) as get_by_uid:
            sliced = lazy.sel(center=True)
            assert sliced["reg_covariances"].shape[0] == 2
            assert get_by_uid.call_count == 2

        with patch.object(store, "get_by_uid", wraps=store.get_by_uid) as get_by_uid:
            lazy.sel(center=True)
            assert get_by_uid.call_count == 0


def test_aggregator_skip_keys_deferred_until_sel_objects():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        sessions = [FakeSession()]
        _populate_cvpca_grid(store, sessions)

        lazy = ResultsAggregator(CVPCAConfig, store, sessions, lazy=True)
        assert len(lazy._objects_backend) == 0
        lazy.sel(center=True, keys=["reg_covariances"])
        assert len(lazy._objects_backend) == 0

        objs = lazy.sel_objects(center=True)
        assert "trial_folds" in objs
        assert objs["trial_folds"].shape[0] == 1


def test_average_array_by_mouse():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    mouse_names = ["m1", "m1", "m2", "m2"]
    averaged = average_array_by_mouse(arr, mouse_names)
    np.testing.assert_allclose(averaged[0], [2.0, 3.0])
    np.testing.assert_allclose(averaged[1], [6.0, 7.0])
def test_average_by_mouse_dict_skips_object_dtype():
    arrays = {
        "values": np.array([[1.0], [3.0], [5.0], [7.0]]),
        "ragged": np.empty((4,), dtype=object),
    }
    mouse_names = ["m1", "m1", "m2", "m2"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        averaged = average_by_mouse(arrays, mouse_names)
    assert "ragged" not in averaged
    assert len(caught) == 1
    np.testing.assert_allclose(averaged["values"][:, 0], [2.0, 6.0])


def test_aggregator_average_by_mouse():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        sessions = [FakeSession(), FakeSessionB(), FakeSession(), FakeSessionB()]
        sessions[2].session_uid = "test_session_c"
        sessions[3].session_uid = "test_session_d"
        _populate_cvpca_grid(store, sessions)

        agg = ResultsAggregator(CVPCAConfig, store, sessions, lazy=False)
        mouse_avg = agg._average_by_mouse()
        assert mouse_avg.session_ids == ["mouse_a", "mouse_b"]
        expected = average_array_by_mouse(agg.arrays["reg_covariances"], agg.mouse_names)
        np.testing.assert_allclose(mouse_avg.arrays["reg_covariances"], expected)


def test_aggregator_lazy_init_fast():
    with tempfile.TemporaryDirectory() as td:
        store = ResultsStore(pathlib.Path(td) / "test.db")
        sessions = [FakeSession(), FakeSessionB()]
        _populate_cvpca_grid(store, sessions)

        with patch.object(store, "get_by_uid", wraps=store.get_by_uid) as get_by_uid:
            ResultsAggregator(CVPCAConfig, store, sessions, lazy=True)
            assert get_by_uid.call_count == 0


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
