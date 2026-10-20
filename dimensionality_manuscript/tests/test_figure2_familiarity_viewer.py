from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure2.familiarity import (
    RegressionFamiliarityByEnvViewer,
    RegressionFamiliarityViewer,
)


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


class _FakeStructuredRegressionResults(_FakeRegressionResults):
    param_axes = {
        "model_name": [
            "internal_placefield_1d",
            "internal_placefield_1d_gain",
            "internal_placefield_1d_structured_additive",
            "rrr",
        ]
    }


class _FakeByEnvRegressionResults(_FakeRegressionResults):
    param_axes = {"model_name": ["external_placefield_1d", "internal_placefield_1d_gain", "rrr"]}
    sessions = [
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-03", environments=np.array([0, 1])),
        SimpleNamespace(date="2024-01-01", environments=np.array([0])),
        SimpleNamespace(date="2024-01-03", environments=np.array([0, 1])),
    ]

    def sel(self, *, model_name, avg_by_mouse, **selection):
        assert avg_by_mouse is False
        control = np.array(
            [
                [0.10, np.nan, np.nan],
                [0.20, 0.30, np.nan],
                [0.15, np.nan, np.nan],
                [0.25, 0.35, np.nan],
            ]
        )
        offset = self.param_axes["model_name"].index(model_name) * 0.05
        return {"r2_env": control + offset}


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


@pytest.mark.parametrize(
    ("value_mode", "legend_labels"),
    [
        ("absolute", ["PF", "PF+Gain", "Peer Residual", "Peer Prediction"]),
        ("improvement", ["PF+Gain", "Peer Residual", "Peer Prediction"]),
    ],
)
def test_regression_familiarity_structured_additive_respects_value_mode(value_mode, legend_labels):
    viewer = RegressionFamiliarityViewer(
        _FakeStructuredRegressionResults(),
        include_structured_additive=True,
        value_mode=value_mode,
        min_mice=1,
    )

    assert list(viewer._scores) == [
        "internal_placefield_1d",
        "internal_placefield_1d_gain",
        "internal_placefield_1d_structured_additive",
        "rrr",
    ]
    fig = viewer.plot(viewer.state)
    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == legend_labels
    plt.close(fig)


@pytest.mark.parametrize(
    ("within_env_session", "xlabel", "env2_x"),
    [
        (True, "Environment Session #", [0.0]),
        (False, "Overall Session #", [0.0, 1.0]),
    ],
)
def test_regression_familiarity_by_env_alignment_and_improvement(
    within_env_session, xlabel, env2_x
):
    viewer = RegressionFamiliarityByEnvViewer(
        _FakeByEnvRegressionResults(),
        within_env_session=within_env_session,
        style="all",
        min_mice=1,
    )

    fig = viewer.plot(viewer.state)
    absolute_ax, improvement_ax = fig.axes
    env2_line = next(line for line in absolute_ax.lines if line.get_label() == "Env #2")
    improvement_lines = {
        line.get_label(): line for line in improvement_ax.lines if line.get_label().startswith("Env #")
    }

    assert len(fig.axes) == 2
    assert absolute_ax.get_xlabel() == improvement_ax.get_xlabel() == xlabel
    assert env2_line.get_xdata().tolist() == env2_x
    assert np.asarray(improvement_lines["Env #1"].get_ydata()) == pytest.approx([0.10, 0.10])
    baseline_line = next(line for line in absolute_ax.lines if line.get_label() == "Env #1")
    assert baseline_line.get_ydata() == pytest.approx([0.125, 0.225])
    assert [baseline_line.get_color(), env2_line.get_color()] == ["k", "blue"]
    assert [text.get_text() for text in absolute_ax.get_legend().get_texts()] == ["Env #1", "Env #2"]
    assert any(line.get_linestyle() == "--" for line in improvement_ax.lines)
    plt.close(fig)


def test_regression_familiarity_by_env_model_selection():
    viewer = RegressionFamiliarityByEnvViewer(
        _FakeByEnvRegressionResults(),
        baseline_model="rrr",
        reference_model="internal_placefield_1d_gain",
        style="all",
        min_mice=1,
    )
    fig = viewer.plot(viewer.state)
    baseline_line = next(line for line in fig.axes[0].lines if line.get_label() == "Env #1")
    delta_line = next(line for line in fig.axes[1].lines if line.get_label() == "Env #1")
    assert baseline_line.get_ydata() == pytest.approx([0.225, 0.325])
    assert delta_line.get_ydata() == pytest.approx([-0.05, -0.05])
    plt.close(fig)

    viewer.set_parameter_value("reference_model", "rrr")
    fig = viewer.plot(viewer.state)
    delta_line = next(line for line in fig.axes[1].lines if line.get_label() == "Env #1")
    assert delta_line.get_ydata() == pytest.approx([0.0, 0.0])
    plt.close(fig)

    viewer.set_parameter_value("baseline_model", "external_placefield_1d")
    fig = viewer.plot(viewer.state)
    baseline_line = next(line for line in fig.axes[0].lines if line.get_label() == "Env #1")
    delta_line = next(line for line in fig.axes[1].lines if line.get_label() == "Env #1")
    assert baseline_line.get_ydata() == pytest.approx([0.125, 0.225])
    assert delta_line.get_ydata() == pytest.approx([0.10, 0.10])
    plt.close(fig)


@pytest.mark.parametrize("parameter", ["baseline_model", "reference_model"])
def test_regression_familiarity_by_env_rejects_missing_model(parameter):
    with pytest.raises(ValueError, match=parameter):
        RegressionFamiliarityByEnvViewer(_FakeByEnvRegressionResults(), **{parameter: "missing"})


def test_regression_familiarity_by_env_requires_per_environment_results():
    with pytest.raises(ValueError, match="requires r2_env"):
        RegressionFamiliarityByEnvViewer(
            _FakeRegressionResults(),
            baseline_model="internal_placefield_1d",
        )


@pytest.mark.parametrize("style", ["all", "errorPlot"])
@pytest.mark.parametrize("within_env_session", [True, False])
def test_regression_familiarity_by_env_reference_mean(style, within_env_session):
    viewer = RegressionFamiliarityByEnvViewer(
        _FakeByEnvRegressionResults(),
        style=style,
        within_env_session=within_env_session,
        min_mice=1,
    )
    fig = viewer.plot(viewer.state)
    collection_count = len(fig.axes[0].collections)
    delta_lines = len(fig.axes[1].lines)
    assert not any(line.get_label().endswith(" reference") for line in fig.axes[0].lines)

    viewer.set_parameter_value("show_reference_mean", True)
    fig = viewer.plot(viewer.state)
    absolute_ax, delta_ax = fig.axes
    reference_lines = [line for line in absolute_ax.lines if line.get_label().endswith(" reference")]
    assert len(reference_lines) == 2
    assert reference_lines[0].get_ydata() == pytest.approx([0.225, 0.325])
    expected_env2 = [0.425] if within_env_session else [np.nan, 0.425]
    np.testing.assert_allclose(reference_lines[1].get_ydata(), expected_env2, equal_nan=True)
    assert [line.get_color() for line in reference_lines] == ["k", "blue"]
    assert all(line.get_linestyle() == "--" for line in reference_lines)
    assert all(line.get_linewidth() == 0.5 * viewer.state["linewidth"] for line in reference_lines)
    assert len(absolute_ax.collections) == collection_count
    assert len(delta_ax.lines) == delta_lines
    assert not any(line.get_label().endswith(" reference") for line in delta_ax.lines)
    assert absolute_ax.get_ylim()[1] > 0.425

    viewer._reference_scores[2:] = np.nan
    viewer.set_parameter_value("min_mice", 2)
    fig = viewer.plot(viewer.state)
    assert not any(line.get_label().endswith(" reference") for line in fig.axes[0].lines)
    plt.close(fig)
