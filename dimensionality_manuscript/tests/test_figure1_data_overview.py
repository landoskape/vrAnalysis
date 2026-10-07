from types import SimpleNamespace

import numpy as np

from dimensionality_manuscript.figure_scripts.figure1.reliability import DataOverviewViewer


class FakeSession:
    def __init__(self, session_uid, mouse_name, spks_type, idx_rois):
        self.session_uid = session_uid
        self.mouse_name = mouse_name
        self.params = SimpleNamespace(spks_type=spks_type)
        self.idx_rois = np.asarray(idx_rois, dtype=bool)


class FakeResults:
    param_axes = {}

    def __init__(self, session_ids, reliability, roi_counts=None):
        self.session_ids = session_ids
        self.reliability = np.asarray(reliability, dtype=float)
        if roi_counts is None:
            roi_counts = np.sum(~np.isnan(self.reliability), axis=1)
        roi_counts = np.asarray(roi_counts)
        self.result_shapes = {"reliability": roi_counts[:, None]}

    def sel(self, **kwargs):
        return {"reliability": self.reliability}


def test_data_overview_pools_session_counts_by_mouse_and_restores_spks_type():
    sessions = [
        FakeSession("b1", "mouse_b", "oasis", [1, 1, 0]),
        FakeSession("a1", "mouse_a", "raw", [1, 1, 1, 0]),
        FakeSession("b2", "mouse_b", "oasis", [1, 0, 0, 0, 1]),
    ]
    # Deliberately use a different result order to exercise session-UID alignment.
    results = FakeResults(
        ["a1", "b2", "b1"],
        [[0.4, 0.2, np.nan], [0.6, 0.7, np.nan], [0.2, 0.9, np.nan]],
        roi_counts=[3, 2, 2],
    )

    viewer = DataOverviewViewer(sessions, results, place_cell_threshold=0.3)

    np.testing.assert_array_equal(viewer.mice, ["mouse_a", "mouse_b"])
    np.testing.assert_allclose(viewer.num_sessions, [1, 2])
    np.testing.assert_allclose(viewer.num_rois, [3, 2])
    np.testing.assert_allclose(viewer.num_place_cells, [1, 1.5])
    assert [session.params.spks_type for session in sessions] == ["oasis", "raw", "oasis"]


def test_data_overview_rejects_an_empty_session_list():
    try:
        DataOverviewViewer([], FakeResults([], np.empty((0, 0))))
    except ValueError as exc:
        assert "at least one session" in str(exc)
    else:
        raise AssertionError("Expected an empty session list to be rejected")
