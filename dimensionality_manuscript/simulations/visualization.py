"""
Visualization and dashboard for shared-space dimensionality simulations.

Provides a Plotly/Dash dashboard to vary SharedSpaceConfig and generation
parameters (num_samples, rotation_angle) and inspect SVD-XY and related plots.
Includes Optuna study runs with parallel coordinate and parameter importance
visualizations, plus loading best params into config and a reset button.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from plotly.subplots import make_subplots

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update
from optuna.visualization import plot_parallel_coordinate, plot_param_importances

from dimensionality_manuscript.simulations import (
    SharedSpaceConfig,
    SharedSpaceGenerator,
    sqrtm_spd,
)


# -----------------------------------------------------------------------------
# Run config and analysis result (extensible state)
# -----------------------------------------------------------------------------


@dataclass
class RunConfig:
    """
    Full configuration for a single dashboard run.

    Includes SharedSpaceConfig parameters plus generation knobs:
    num_samples (S), rotation_angle, and noise_variance.
    """

    # SharedSpaceConfig
    num_neurons: int = 100
    shared_dimensions: int = 10
    private_dim_1: int = 10
    private_dim_2: int = 10
    alpha_shared_1: float = 1.0
    alpha_shared_2: float = 1.0
    shuffle_shared: bool = True
    alpha_private_1: float = 0.1
    alpha_private_2: float = 0.1
    private_ratio: float = 1.0

    # Generation
    num_samples: int = 10000
    rotation_angle: float = np.pi / 4
    noise_variance: float = 0.1

    def to_shared_space_config(self) -> SharedSpaceConfig:
        """Build SharedSpaceConfig from this run config."""
        return SharedSpaceConfig(
            num_neurons=self.num_neurons,
            shared_dimensions=self.shared_dimensions,
            private_dimensions=(self.private_dim_1, self.private_dim_2),
            alpha_shared_1=self.alpha_shared_1,
            alpha_shared_2=self.alpha_shared_2,
            shuffle_shared=self.shuffle_shared,
            alpha_private_1=self.alpha_private_1,
            alpha_private_2=self.alpha_private_2,
            private_ratio=self.private_ratio,
        )

    @classmethod
    def from_optuna_best_params(
        cls,
        best_params: dict[str, Any],
        num_neurons: int = 100,
        shared_dimensions: int = 10,
        private_dim_1: int = 10,
        private_dim_2: int = 10,
    ) -> "RunConfig":
        """
        Build RunConfig from Optuna study best_params.

        Uses the fixed structure (num_neurons, shared_dimensions, private_dim_*)
        and fills in optimized values from the study.

        Parameters
        ----------
        best_params : dict[str, Any]
            study.best_params from an Optuna study.
        num_neurons : int, optional
        shared_dimensions : int, optional
        private_dim_1 : int, optional
        private_dim_2 : int, optional

        Returns
        -------
        RunConfig
        """
        return cls(
            num_neurons=int(num_neurons),
            shared_dimensions=int(shared_dimensions),
            private_dim_1=int(private_dim_1),
            private_dim_2=int(private_dim_2),
            alpha_shared_1=float(best_params.get("alpha_shared_1", 1.0)),
            alpha_shared_2=float(best_params.get("alpha_shared_2", 1.0)),
            shuffle_shared=bool(best_params.get("shuffle_shared", True)),
            alpha_private_1=float(best_params.get("alpha_private_1", 0.1)),
            alpha_private_2=float(best_params.get("alpha_private_2", 0.1)),
            private_ratio=float(best_params.get("private_ratio", 1.0)),
            num_samples=int(best_params.get("num_samples", 10000)),
            rotation_angle=float(best_params.get("rotation_angle", np.pi / 4)),
            noise_variance=float(best_params.get("noise_variance", 0.1)),
        )


@dataclass
class AnalysisResult:
    """Holds computed quantities from one run for use by plot builders."""

    svals_xy_root: np.ndarray
    svals_xy_test: np.ndarray
    evals_rABAr: np.ndarray
    config: RunConfig
    subspace_extras: Optional[dict[str, Any]] = None
    """Extras from generate(return_extras=True): shared1, shared2, private1, private2, spectra."""

    def xvals(self, arr: np.ndarray) -> np.ndarray:
        """1-based index for plotting."""
        return np.arange(1, len(arr) + 1, dtype=float)


def compute_analysis(cfg: RunConfig) -> AnalysisResult:
    """
    Generate data and compute SVD-XY / rABAr quantities for the given config.

    Parameters
    ----------
    cfg : RunConfig
        Full run configuration.

    Returns
    -------
    AnalysisResult
        Arrays and config for plotting.
    """
    ss_cfg = cfg.to_shared_space_config()
    generator = SharedSpaceGenerator(ss_cfg)

    S = cfg.num_samples
    data1, data2 = generator.generate(S, noise_variance=cfg.noise_variance)
    test1, test2, subspace_extras = generator.generate(
        S,
        noise_variance=cfg.noise_variance,
        rotation_angle=cfg.rotation_angle,
        return_extras=True,
    )

    A = np.cov(data1)
    B = np.cov(data2)
    rA = sqrtm_spd(A)
    rB = sqrtm_spd(B)

    evals_rABAr, _ = np.linalg.eigh(rA @ B @ rA)
    evals_rABAr = np.sqrt(np.maximum(0, evals_rABAr))[::-1]
    Uroot, svals_xyroot, Vtroot = np.linalg.svd(rA @ rB)

    Atest = np.cov(test1)
    Btest = np.cov(test2)
    rAtest = sqrtm_spd(Atest)
    rBtest = sqrtm_spd(Btest)
    svals_xy_test = np.diag(Uroot.T @ rAtest @ rBtest @ Vtroot.T)

    return AnalysisResult(
        svals_xy_root=svals_xyroot,
        svals_xy_test=svals_xy_test,
        evals_rABAr=evals_rABAr,
        config=cfg,
        subspace_extras=subspace_extras,
    )


def compute_analysis_random(
    num_neurons: int,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> AnalysisResult:
    """
    Compute SVD-XY / rABAr on completely random data (no shared structure).

    Each of data1, data2, test1, test2 is independently drawn from
    np.random.randn(num_neurons, num_samples). Serves as a null baseline.

    Parameters
    ----------
    num_neurons : int
        Number of neurons/dimensions.
    num_samples : int
        Number of samples per dataset.
    rng : np.random.Generator, optional
        Random number generator. Uses default_rng() if None.

    Returns
    -------
    AnalysisResult
        Arrays and minimal config for plotting.
    """
    rng = rng if rng is not None else np.random.default_rng()
    data1 = rng.standard_normal((num_neurons, num_samples))
    data2 = rng.standard_normal((num_neurons, num_samples))
    test1 = rng.standard_normal((num_neurons, num_samples))
    test2 = rng.standard_normal((num_neurons, num_samples))

    A = np.cov(data1)
    B = np.cov(data2)
    rA = sqrtm_spd(A)
    rB = sqrtm_spd(B)

    evals_rABAr, _ = np.linalg.eigh(rA @ B @ rA)
    evals_rABAr = np.sqrt(np.maximum(0, evals_rABAr))[::-1]
    Uroot, svals_xyroot, Vtroot = np.linalg.svd(rA @ rB)

    Atest = np.cov(test1)
    Btest = np.cov(test2)
    rAtest = sqrtm_spd(Atest)
    rBtest = sqrtm_spd(Btest)
    svals_xy_test = np.diag(Uroot.T @ rAtest @ rBtest @ Vtroot.T)

    cfg = RunConfig(
        num_neurons=num_neurons,
        num_samples=num_samples,
    )
    return AnalysisResult(
        svals_xy_root=svals_xyroot,
        svals_xy_test=svals_xy_test,
        evals_rABAr=evals_rABAr,
        config=cfg,
    )


def run_optuna_study(
    num_neurons: int = 100,
    shared_dimensions: int = 10,
    private_dim_1: int = 10,
    private_dim_2: int = 10,
    n_trials: int = 100,
) -> tuple[Any, dict[str, Any]]:
    """
    Run an Optuna study with SharedSpaceGenerator and return study plus figure data.

    Parameters
    ----------
    num_neurons : int, optional
    shared_dimensions : int, optional
    private_dim_1 : int, optional
    private_dim_2 : int, optional
    n_trials : int, optional
        Number of Optuna trials. Default 50.

    Returns
    -------
    study
        Completed Optuna study.
    figure_data : dict
        Keys: "parallel_fig", "param_importance_fig", "best_params". Values are
        plotly figure dicts (for dcc.Graph) and best_params dict.
    """
    study = SharedSpaceGenerator.optuna_study_fraction_negative_svals(
        num_neurons=num_neurons,
        shared_dimensions=shared_dimensions,
        private_dimensions=(private_dim_1, private_dim_2),
        n_trials=n_trials,
        seed=None,
    )
    fig_parallel = plot_parallel_coordinate(study)
    fig_param = plot_param_importances(study)
    return study, {
        "parallel_fig": fig_parallel.to_dict() if fig_parallel is not None else None,
        "param_importance_fig": fig_param.to_dict() if fig_param is not None else None,
        "best_params": study.best_params,
    }


# -----------------------------------------------------------------------------
# Plot builders: (name, fn) where fn(AnalysisResult) -> plotly Figure
# -----------------------------------------------------------------------------


def _plot_svd_xy_root_vs_test(result: AnalysisResult) -> Any:
    """SVD-XY singular values: root (train) vs test, log-log."""
    x_root = result.xvals(result.svals_xy_root)
    x_test = result.xvals(result.svals_xy_test)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_root,
            y=result.svals_xy_root,
            mode="lines",
            name="SVD-XY (Root)",
            line=dict(color="black"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_test,
            y=result.svals_xy_test,
            mode="lines",
            name="SVD-XY (Test)",
            line=dict(color="blue"),
        )
    )
    fig.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        title="SVD-XY singular values (root vs test)",
        xaxis_title="Index",
        yaxis_title="Singular value",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def _plot_subspace_overlaps(result: AnalysisResult) -> Any:
    """Subspace overlap matrices (shared1.T@shared2, shared/private, private1.T@private2)."""
    if result.subspace_extras is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Subspace extras not available (e.g. random baseline)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template="plotly_white", height=300)
        return fig

    ex = result.subspace_extras
    s1, s2 = ex["shared1"], ex["shared2"]
    p1, p2 = ex["private1"], ex["private2"]

    overlaps = [
        ("shared1 vs shared2", s1.T @ s2),
        ("shared1 vs private1", s1.T @ p1),
        ("shared1 vs private2", s1.T @ p2),
        ("shared2 vs private1", s2.T @ p1),
        ("shared2 vs private2", s2.T @ p2),
        ("private1 vs private2", p1.T @ p2),
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[t for t, _ in overlaps],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for i, (_, mat) in enumerate(overlaps):
        row, col = i // 3 + 1, i % 3 + 1
        fig.add_trace(
            go.Heatmap(z=mat, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1),
            row=row,
            col=col,
        )
    fig.update_layout(
        title="Subspace overlaps (test, after rotation)",
        template="plotly_white",
        height=500,
    )
    for r in range(1, 3):
        for c in range(1, 4):
            fig.update_xaxes(
                scaleanchor="y",
                scaleratio=1,
                constrain="domain",
                row=r,
                col=c,
            )
    return fig


def _plot_eigenspectra(result: AnalysisResult) -> Any:
    """Eigenspectra of shared and private subspaces (power-law)."""
    if result.subspace_extras is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Subspace extras not available (e.g. random baseline)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template="plotly_white", height=300)
        return fig

    ex = result.subspace_extras
    x_shared1 = np.arange(1, len(ex["shared_spectrum1"]) + 1, dtype=float)
    x_shared2 = np.arange(1, len(ex["shared_spectrum2"]) + 1, dtype=float)
    x_priv1 = np.arange(1, len(ex["private_spectrum1"]) + 1, dtype=float)
    x_priv2 = np.arange(1, len(ex["private_spectrum2"]) + 1, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_shared1, y=ex["shared_spectrum1"], mode="lines", name="shared (cond1)"),
    )
    fig.add_trace(
        go.Scatter(x=x_shared2, y=ex["shared_spectrum2"], mode="lines", name="shared (cond2)"),
    )
    fig.add_trace(
        go.Scatter(x=x_priv1, y=ex["private_spectrum1"], mode="lines", name="private1"),
    )
    fig.add_trace(
        go.Scatter(x=x_priv2, y=ex["private_spectrum2"], mode="lines", name="private2"),
    )
    fig.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        title="Eigenspectra (shared & private power-law)",
        xaxis_title="Index",
        yaxis_title="Eigenvalue",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


# Fixed-plot builders for main analysis (no dropdown)
_PLOT_SVD_XY = _plot_svd_xy_root_vs_test
_PLOT_SUBSPACE_OVERLAPS = _plot_subspace_overlaps
_PLOT_EIGENSPECTRA = _plot_eigenspectra


# -----------------------------------------------------------------------------
# Dash dashboard
# -----------------------------------------------------------------------------


def _make_config_panel(ids: dict[str, str]) -> list:
    """Build layout list for the config panel (inputs only; ids for callbacks)."""
    return [
        html.H3("Config", style=dict(marginBottom="0.5em")),
        html.Div(
            [
                html.Label("num_neurons"),
                dcc.Input(id=ids["num_neurons"], type="number", value=100, min=2),
                html.Label("shared_dimensions"),
                dcc.Input(id=ids["shared_dim"], type="number", value=10, min=1),
                html.Label("private_dim_1"),
                dcc.Input(id=ids["private_dim_1"], type="number", value=10, min=0),
                html.Label("private_dim_2"),
                dcc.Input(id=ids["private_dim_2"], type="number", value=10, min=0),
            ],
            style=dict(display="flex", flexWrap="wrap", gap="1em", alignItems="center"),
        ),
        html.Div(
            [
                html.Label("alpha_shared_1"),
                dcc.Input(id=ids["alpha_shared_1"], type="number", value=1.0, step="any"),
                html.Label("alpha_shared_2"),
                dcc.Input(id=ids["alpha_shared_2"], type="number", value=1.0, step="any"),
                html.Label("shuffle_shared"),
                dcc.Dropdown(
                    id=ids["shuffle_shared"],
                    options=[{"label": "True", "value": True}, {"label": "False", "value": False}],
                    value=True,
                ),
                html.Label("alpha_private_1"),
                dcc.Input(id=ids["alpha_private_1"], type="number", value=0.1, step="any"),
                html.Label("alpha_private_2"),
                dcc.Input(id=ids["alpha_private_2"], type="number", value=0.1, step="any"),
                html.Label("private_ratio"),
                dcc.Input(id=ids["private_ratio"], type="number", value=1.0, step="any"),
            ],
            style=dict(display="flex", flexWrap="wrap", gap="1em", alignItems="center", marginTop="0.5em"),
        ),
        html.Div(
            [
                html.Label("num_samples (S)"),
                dcc.Input(id=ids["num_samples"], type="number", value=10000, min=100),
                html.Label("rotation_angle (rad)"),
                dcc.Input(id=ids["rotation_angle"], type="number", value=0.785, step="any"),
                html.Label("noise_variance"),
                dcc.Input(id=ids["noise_variance"], type="number", value=0.1, step="any", min=0),
                dcc.Checklist(
                    id=ids["lock"],
                    options=[{"label": " Lock updates", "value": "lock"}],
                    value=[],
                    inline=True,
                ),
            ],
            style=dict(display="flex", flexWrap="wrap", gap="1em", alignItems="center", marginTop="0.5em"),
        ),
        html.Div(
            [
                html.Button("Run Optuna", id=ids["run_optuna"], n_clicks=0),
                html.Button("Load Optuna Best", id=ids["load_optuna_best"], n_clicks=0),
                html.Button("Reset", id=ids["reset"], n_clicks=0),
                html.Span(id=ids["optuna_status"], children="", style=dict(marginLeft="1em")),
            ],
            style=dict(display="flex", flexWrap="wrap", gap="0.5em", alignItems="center", marginTop="0.5em"),
        ),
    ]


def _parse_run_config_from_callback(
    num_neurons: int,
    shared_dim: int,
    private_dim_1: int,
    private_dim_2: int,
    alpha_shared_1: float,
    alpha_shared_2: float,
    shuffle_shared: bool,
    alpha_private_1: float,
    alpha_private_2: float,
    private_ratio: float,
    num_samples: int,
    rotation_angle: float,
    noise_variance: float,
) -> RunConfig:
    """Build RunConfig from Dash callback inputs."""
    return RunConfig(
        num_neurons=int(num_neurons),
        shared_dimensions=int(shared_dim),
        private_dim_1=int(private_dim_1),
        private_dim_2=int(private_dim_2),
        alpha_shared_1=float(alpha_shared_1),
        alpha_shared_2=float(alpha_shared_2),
        shuffle_shared=bool(shuffle_shared),
        alpha_private_1=float(alpha_private_1),
        alpha_private_2=float(alpha_private_2),
        private_ratio=float(private_ratio),
        num_samples=int(num_samples),
        rotation_angle=float(rotation_angle),
        noise_variance=float(noise_variance),
    )


def create_dashboard_app():
    """
    Create and return a Dash app for the shared-space config dashboard.

    Layout: top = config panel, bottom = plot selector + graph container.
    Callbacks recompute analysis on "Update plots" and update the active plot.

    Returns
    -------
    dash.Dash
        Configured app. Run with app.run(debug=True).
    """
    ids = {
        "num_neurons": "viz-num-neurons",
        "shared_dim": "viz-shared-dim",
        "private_dim_1": "viz-private-dim-1",
        "private_dim_2": "viz-private-dim-2",
        "alpha_shared_1": "viz-alpha-shared-1",
        "alpha_shared_2": "viz-alpha-shared-2",
        "shuffle_shared": "viz-shuffle-shared",
        "alpha_private_1": "viz-alpha-private-1",
        "alpha_private_2": "viz-alpha-private-2",
        "private_ratio": "viz-private-ratio",
        "num_samples": "viz-num-samples",
        "rotation_angle": "viz-rotation-angle",
        "noise_variance": "viz-noise-variance",
        "lock": "viz-lock",
        "run_optuna": "viz-run-optuna",
        "load_optuna_best": "viz-load-optuna-best",
        "reset": "viz-reset",
        "optuna_status": "viz-optuna-status",
        "optuna_store": "viz-optuna-store",
        "svd_xy_container": "viz-svd-xy",
        "subspace_overlaps_container": "viz-subspace-overlaps",
        "eigenspectra_container": "viz-eigenspectra",
        "optuna_parallel_container": "viz-optuna-parallel",
        "optuna_param_container": "viz-optuna-param",
        "random_reset": "viz-random-reset",
        "random_graph_container": "viz-random-graph-container",
    }

    app = Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Store(id=ids["optuna_store"], data=None),
            html.Div(
                _make_config_panel(ids),
                style=dict(
                    padding="1em",
                    backgroundColor="#f8f9fa",
                    borderRadius="8px",
                    marginBottom="1em",
                ),
            ),
            html.H3("SVD-XY", style=dict(marginBottom="0.25em", fontSize="1em")),
            dcc.Loading(
                id="viz-loading-svd",
                type="default",
                children=html.Div(id=ids["svd_xy_container"], children=[]),
            ),
            html.H3("Subspace overlaps (test)", style=dict(marginTop="1.5em", marginBottom="0.25em", fontSize="1em")),
            html.Div(id=ids["subspace_overlaps_container"], children=[]),
            html.H3("Eigenspectra (shared & private)", style=dict(marginTop="1.5em", marginBottom="0.25em", fontSize="1em")),
            html.Div(id=ids["eigenspectra_container"], children=[]),
            html.Hr(style=dict(margin="2em 0 1em 0")),
            html.H3("Optuna study", style=dict(marginBottom="0.5em")),
            html.Div(
                [
                    html.H4("Parallel coordinates", style=dict(marginBottom="0.25em", fontSize="1em")),
                    html.Div(id=ids["optuna_parallel_container"], children=[]),
                ],
                style=dict(marginBottom="1.5em"),
            ),
            html.Div(
                [
                    html.H4("Parameter importances", style=dict(marginBottom="0.25em", fontSize="1em")),
                    html.Div(id=ids["optuna_param_container"], children=[]),
                ],
                style=dict(marginBottom="1.5em"),
            ),
            html.Hr(style=dict(margin="2em 0 1em 0")),
            html.H3("Random baseline (no shared structure)", style=dict(marginBottom="0.5em")),
            html.Div(
                [
                    html.Button("Reset", id=ids["random_reset"], n_clicks=0),
                ],
                style=dict(marginBottom="0.5em"),
            ),
            dcc.Loading(
                id="viz-random-loading",
                type="default",
                children=html.Div(id=ids["random_graph_container"], children=[]),
            ),
        ],
        style=dict(
            width="100%",
            maxWidth="100%",
            margin="0 auto",
            padding="1em 2em",
            fontFamily="sans-serif",
            boxSizing="border-box",
        ),
    )

    @callback(
        [
            Output(ids["optuna_store"], "data"),
            Output(ids["optuna_status"], "children"),
        ],
        Input(ids["run_optuna"], "n_clicks"),
        [
            State(ids["num_neurons"], "value"),
            State(ids["shared_dim"], "value"),
            State(ids["private_dim_1"], "value"),
            State(ids["private_dim_2"], "value"),
        ],
        prevent_initial_call=True,
    )
    def run_optuna_callback(
        n_clicks,
        num_neurons,
        shared_dim,
        private_dim_1,
        private_dim_2,
    ):
        n_trials = 50
        defaults = RunConfig()
        num_neurons = num_neurons if num_neurons is not None else defaults.num_neurons
        shared_dim = shared_dim if shared_dim is not None else defaults.shared_dimensions
        private_dim_1 = private_dim_1 if private_dim_1 is not None else defaults.private_dim_1
        private_dim_2 = private_dim_2 if private_dim_2 is not None else defaults.private_dim_2
        try:
            _, figure_data = run_optuna_study(
                num_neurons=int(num_neurons),
                shared_dimensions=int(shared_dim),
                private_dim_1=int(private_dim_1),
                private_dim_2=int(private_dim_2),
                n_trials=n_trials,
            )
            return figure_data, f"Optuna done (best value from {n_trials} trials)"
        except Exception as e:
            return no_update, html.Span(f"Error: {e}", style=dict(color="red"))

    @callback(
        [
            Output(ids["alpha_shared_1"], "value"),
            Output(ids["alpha_shared_2"], "value"),
            Output(ids["shuffle_shared"], "value"),
            Output(ids["alpha_private_1"], "value"),
            Output(ids["alpha_private_2"], "value"),
            Output(ids["private_ratio"], "value"),
            Output(ids["num_samples"], "value"),
            Output(ids["rotation_angle"], "value"),
            Output(ids["noise_variance"], "value"),
        ],
        Input(ids["load_optuna_best"], "n_clicks"),
        State(ids["optuna_store"], "data"),
        prevent_initial_call=True,
    )
    def load_optuna_best_callback(n_clicks, store_data):
        if not n_clicks or not store_data or "best_params" not in store_data:
            return no_update
        bp = store_data["best_params"]
        defaults = RunConfig()
        return (
            bp.get("alpha_shared_1", defaults.alpha_shared_1),
            bp.get("alpha_shared_2", defaults.alpha_shared_2),
            bp.get("shuffle_shared", defaults.shuffle_shared),
            bp.get("alpha_private_1", defaults.alpha_private_1),
            bp.get("alpha_private_2", defaults.alpha_private_2),
            bp.get("private_ratio", defaults.private_ratio),
            bp.get("num_samples", defaults.num_samples),
            bp.get("rotation_angle", defaults.rotation_angle),
            bp.get("noise_variance", defaults.noise_variance),
        )

    @callback(
        [
            Output(ids["optuna_parallel_container"], "children"),
            Output(ids["optuna_param_container"], "children"),
        ],
        Input(ids["optuna_store"], "data"),
        prevent_initial_call=False,
    )
    def update_optuna_plots(optuna_store):
        placeholder = html.Div(
            "Run Optuna first.",
            style=dict(color="#666", fontStyle="italic", padding="2em", minHeight="200px"),
        )
        if not optuna_store:
            return placeholder, placeholder
        if not optuna_store.get("parallel_fig"):
            parallel_content = placeholder
        else:
            fig = go.Figure(optuna_store["parallel_fig"])
            parallel_content = dcc.Graph(
                figure=fig,
                config=dict(responsive=True),
                style=dict(width="100%", minHeight="400px"),
            )
        if not optuna_store.get("param_importance_fig"):
            param_content = placeholder
        else:
            fig = go.Figure(optuna_store["param_importance_fig"])
            param_content = dcc.Graph(
                figure=fig,
                config=dict(responsive=True),
                style=dict(width="100%", minHeight="400px"),
            )
        return parallel_content, param_content

    @callback(
        [
            Output(ids["svd_xy_container"], "children"),
            Output(ids["subspace_overlaps_container"], "children"),
            Output(ids["eigenspectra_container"], "children"),
        ],
        Input(ids["lock"], "value"),
        Input(ids["reset"], "n_clicks"),
        Input(ids["num_neurons"], "value"),
        Input(ids["shared_dim"], "value"),
        Input(ids["private_dim_1"], "value"),
        Input(ids["private_dim_2"], "value"),
        Input(ids["alpha_shared_1"], "value"),
        Input(ids["alpha_shared_2"], "value"),
        Input(ids["shuffle_shared"], "value"),
        Input(ids["alpha_private_1"], "value"),
        Input(ids["alpha_private_2"], "value"),
        Input(ids["private_ratio"], "value"),
        Input(ids["num_samples"], "value"),
        Input(ids["rotation_angle"], "value"),
        Input(ids["noise_variance"], "value"),
        prevent_initial_call=False,
    )
    def update_plots(
        lock,
        reset_clicks,
        num_neurons,
        shared_dim,
        private_dim_1,
        private_dim_2,
        alpha_shared_1,
        alpha_shared_2,
        shuffle_shared,
        alpha_private_1,
        alpha_private_2,
        private_ratio,
        num_samples,
        rotation_angle,
        noise_variance,
    ):
        locked = lock is not None and "lock" in lock
        if locked:
            return no_update, no_update, no_update

        # Regular analysis plots
        defaults = RunConfig()
        num_neurons = num_neurons if num_neurons is not None else defaults.num_neurons
        shared_dim = shared_dim if shared_dim is not None else defaults.shared_dimensions
        private_dim_1 = private_dim_1 if private_dim_1 is not None else defaults.private_dim_1
        private_dim_2 = private_dim_2 if private_dim_2 is not None else defaults.private_dim_2
        alpha_shared_1 = alpha_shared_1 if alpha_shared_1 is not None else defaults.alpha_shared_1
        alpha_shared_2 = alpha_shared_2 if alpha_shared_2 is not None else defaults.alpha_shared_2
        shuffle_shared = shuffle_shared if shuffle_shared is not None else defaults.shuffle_shared
        alpha_private_1 = alpha_private_1 if alpha_private_1 is not None else defaults.alpha_private_1
        alpha_private_2 = alpha_private_2 if alpha_private_2 is not None else defaults.alpha_private_2
        private_ratio = private_ratio if private_ratio is not None else defaults.private_ratio
        num_samples = num_samples if num_samples is not None else defaults.num_samples
        rotation_angle = rotation_angle if rotation_angle is not None else defaults.rotation_angle
        noise_variance = noise_variance if noise_variance is not None else defaults.noise_variance

        cfg = _parse_run_config_from_callback(
            num_neurons=num_neurons,
            shared_dim=shared_dim,
            private_dim_1=private_dim_1,
            private_dim_2=private_dim_2,
            alpha_shared_1=alpha_shared_1,
            alpha_shared_2=alpha_shared_2,
            shuffle_shared=shuffle_shared,
            alpha_private_1=alpha_private_1,
            alpha_private_2=alpha_private_2,
            private_ratio=private_ratio,
            num_samples=num_samples,
            rotation_angle=rotation_angle,
            noise_variance=noise_variance,
        )
        try:
            result = compute_analysis(cfg)
        except Exception as e:
            err = html.Div(f"Error: {e}", style=dict(color="red"))
            return err, err, err

        fig_svd = _PLOT_SVD_XY(result)
        fig_svd.update_layout(xaxis_type="log", yaxis_type="log", autosize=True)

        fig_overlaps = _PLOT_SUBSPACE_OVERLAPS(result)
        fig_overlaps.update_layout(autosize=True)

        fig_spectra = _PLOT_EIGENSPECTRA(result)
        fig_spectra.update_layout(autosize=True)

        graph_style = dict(width="100%", minHeight="400px")
        return (
            dcc.Graph(figure=fig_svd, config=dict(responsive=True), style=graph_style),
            dcc.Graph(figure=fig_overlaps, config=dict(responsive=True), style=graph_style),
            dcc.Graph(figure=fig_spectra, config=dict(responsive=True), style=graph_style),
        )

    @callback(
        Output(ids["random_graph_container"], "children"),
        Input(ids["random_reset"], "n_clicks"),
        Input(ids["num_neurons"], "value"),
        Input(ids["num_samples"], "value"),
        prevent_initial_call=False,
    )
    def update_random_panel(reset_clicks, num_neurons, num_samples):
        defaults = RunConfig()
        num_neurons = num_neurons if num_neurons is not None else defaults.num_neurons
        num_samples = num_samples if num_samples is not None else defaults.num_samples
        try:
            result = compute_analysis_random(
                num_neurons=int(num_neurons),
                num_samples=int(num_samples),
            )
        except Exception as e:
            return html.Div(f"Error: {e}", style=dict(color="red"))
        fig = _PLOT_SVD_XY(result)
        fig.update_layout(xaxis_type="log", yaxis_type="log", autosize=True)
        return dcc.Graph(
            figure=fig,
            config=dict(responsive=True),
            style=dict(width="100%", minHeight="400px"),
        )

    return app


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
    """
    Run the shared-space dashboard in the browser.

    Parameters
    ----------
    host : str, optional
        Bind address. Default 127.0.0.1.
    port : int, optional
        Port. Default 8050.
    debug : bool, optional
        Run in debug mode. Default True.
    """
    app = create_dashboard_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard()
