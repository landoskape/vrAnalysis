from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure2.familiarity import RegressionFamiliarityViewer


class _FakeRegressionResults:
    param_axes = {
        "model_name": [
            "internal_placefield_1d",
            "internal_placefield_1d_gain",
            "rrr",
        ]
    }
    unique_mice = np.array(["m1", "m2"])
    mouse_names = np.array(["m1", "m1", "m2", "m2"])
    sessions = [
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-02", environments=np.array([0])),
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-02", environments=np.array([0])),
    ]

    def sel(self, *, model_name, avg_by_mouse, **selection):
        assert avg_by_mouse is False
        model_offset = self.param_axes["model_name"].index(model_name) * 0.05
        return {"r2": np.array([0.1, 0.2, 0.15, 0.25]) + model_offset}


@pytest.mark.parametrize(
    ("value_mode", "ylabel", "legend_labels"),
    [
        ("absolute", r"R$^2$", ["PF", "PF+Gain", "Peer Prediction"]),
        ("improvement", r"$\Delta$ R$^2$", ["PF+Gain", "Peer Prediction"]),
    ],
)
def test_regression_familiarity_single_modes_keep_one_axis(value_mode, ylabel, legend_labels):
    viewer = RegressionFamiliarityViewer(
        _FakeRegressionResults(),
        value_mode=value_mode,
        figsize=(3.5, 2.5),
        min_mice=1,
    )

    fig = viewer.plot(viewer.state)

    assert len(fig.axes) == 1
    assert fig.get_size_inches() == pytest.approx((3.5, 2.5))
    assert fig.axes[0].get_ylabel() == ylabel
    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == legend_labels
    plt.close(fig)


def test_regression_familiarity_both_composes_the_two_single_mode_axes():
    viewer = RegressionFamiliarityViewer(
        _FakeRegressionResults(),
        value_mode="both",
        figsize=(3.5, 2.5),
        min_mice=1,
    )

    fig = viewer.plot(viewer.state)
    absolute_ax, improvement_ax = fig.axes

    assert fig.get_size_inches() == pytest.approx((7.0, 2.5))
    assert [absolute_ax.get_ylabel(), improvement_ax.get_ylabel()] == [r"R$^2$", r"$\Delta$ R$^2$"]
    assert [text.get_text() for text in absolute_ax.get_legend().get_texts()] == [
        "PF",
        "PF+Gain",
        "Peer Prediction",
    ]
    assert [text.get_text() for text in improvement_ax.get_legend().get_texts()] == [
        "PF+Gain",
        "Peer Prediction",
    ]
    assert not any(line.get_linestyle() == "--" for line in absolute_ax.lines)
    assert any(line.get_linestyle() == "--" for line in improvement_ax.lines)
    plt.close(fig)
