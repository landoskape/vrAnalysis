from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from dimensionality_manuscript.configs.trial_regression import TrialPlacecellRegressionConfig
from dimensionality_manuscript.figure_scripts.figure2 import (
    TrialModelPlacefieldResidualInsetViewer,
    TrialRegressionFamiliarityByEnvViewer,
    trial_placecell,
)
from dimensionality_manuscript.pipeline import ResultsAggregator


def _results(rows):
    sessions = {}
    records = []
    blobs = {}
    for uid, model, scores in rows:
        mouse_name, date, session_id = uid.split(".")
        sessions[uid] = SimpleNamespace(session_uid=uid, mouse_name=mouse_name, date=date, session_id=session_id)
        result_uid = len(blobs)
        records.append({
            "session_id": uid,
            "analysis_key": TrialPlacecellRegressionConfig(model_name=model).key(),
            "result_uid": result_uid,
        })
        blobs[result_uid] = {"r2": np.asarray(scores)}
    store = SimpleNamespace(summary_table=lambda **kwargs: records, get_by_uid=blobs.get)
    return ResultsAggregator(TrialPlacecellRegressionConfig(), store, sessions.values())


def test_trial_familiarity_reads_stored_r2_with_one_session():
    results = _results([
        ("m1.2024-01-01.1", "external_placefield_1d", [0.1, np.nan, 0.2]),
        ("m1.2024-01-01.1", "internal_placefield_1d_structured_gain", [0.3, np.nan, 0.4]),
    ])
    viewer = TrialRegressionFamiliarityByEnvViewer(results, min_mice=1)
    np.testing.assert_allclose(viewer._baseline_scores, [[0.1, np.nan, 0.2]])
    np.testing.assert_allclose(viewer._reference_scores, [[0.3, np.nan, 0.4]])
    fig = viewer.plot(viewer.state)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_trial_residual_reduces_slots_then_sessions_and_reuses_replay(monkeypatch):
    model = "internal_placefield_1d_structured_gain"
    results = _results([
        ("m1.2024-01-01.1", model, [2.0, 4.0, np.nan]),
        ("m1.2024-01-02.1", model, [5.0, np.nan, np.nan]),
        ("m2.2024-01-01.1", model, [9.0, 7.0, np.nan]),
    ])
    calls = []

    def replay(config, saved, session):
        calls.append((config.model_name, session.session_uid))
        return [
            {"mean_xval_within_pf_rms": value, "mean_xval_outside_pf_rms": value / 2}
            for value in saved["r2"]
        ]

    monkeypatch.setattr(trial_placecell, "_trial_residual_summaries", replay)
    viewer = TrialModelPlacefieldResidualInsetViewer(results, model_names=[model])
    np.testing.assert_allclose(viewer._scores["within"], [[4, 8]])
    np.testing.assert_allclose(viewer._scores["outside"], [[2, 4]])
    viewer.set_parameter_value("environment", "Slot 2")
    np.testing.assert_allclose(viewer._scores["within"], [[4, 7]])
    np.testing.assert_allclose(viewer._scores["outside"], [[2, 3.5]])
    assert calls == [(model, session.session_uid) for session in results.sessions]
    fig = viewer.plot(viewer.state)
    assert len(fig.axes[0].child_axes) == 1
    plt.close(fig)
