"""Score selection and the two trace shapes the figure-2 score panels share.

Two shapes recur across the performance, place-field-residual, and familiarity panels:

- one point (or dot) per model, with a faint line per subject behind the across-subject mean
  (:func:`draw_subject_traces`);
- one curve per model over a mouse's chronological session index, averaged across mice
  (:func:`draw_familiarity_series` and the ``mouse_session_curves`` machinery feeding it).

Both read scores through :func:`performance_scores`, which stacks one metric across models.
"""

import numpy as np
from syd import Viewer

from vrAnalysis.helpers.plotting import errorPlot
from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.registry import ModelName

# A RegressionConfig score is a whole-session number, so the familiarity x-axis is a mouse's own
# chronological session index -- there is no per-environment breakdown to align to.
SESSION_XLABEL = "Overall Session #"


def performance_scores(
    results: ResultsAggregator,
    model_names: list[ModelName],
    metric: str,
    selection: dict,
    avg_by_mouse: bool,
) -> np.ndarray:
    """Metric for every model in ``model_names``, shape (n_models, n_subjects).

    Subjects are mice when ``avg_by_mouse`` is True and sessions otherwise. Values are returned in
    their stored units, R^2 as a fraction.

    Parameters
    ----------
    results : ResultsAggregator
    model_names : list[ModelName]
        Models to stack along axis 0.
    metric : str
        Result key to pull out of each selection.
    selection : dict
        The remaining ``results.sel`` axes, from
        :func:`~dimensionality_manuscript.figure_scripts.panels.data_selection`.
    avg_by_mouse : bool
        Average sessions within a mouse before stacking.
    """
    scores = [results.sel(model_name=model_name, avg_by_mouse=avg_by_mouse, **selection)[metric] for model_name in model_names]
    return np.stack(scores, axis=0)


def draw_subject_traces(
    ax,
    xvals: np.ndarray,
    values: np.ndarray,
    colors,
    state,
    markersize: float | None = None,
) -> None:
    """Draw one faint line per subject, the across-subject mean, and colored mean dots.

    Reads ``mean_linewidth`` / ``subject_linewidth`` / ``subject_alpha`` / ``markersize`` from
    ``state`` (see :func:`~dimensionality_manuscript.figure_scripts.panels.add_trace_style_widgets`).
    ``colors`` supplies one color per x position. ``markersize`` overrides the state's value, for
    drawing the same traces smaller in an inset.
    """
    ax.plot(xvals, values, color=("k", state["subject_alpha"]), linewidth=state["subject_linewidth"])
    mean_values = np.nanmean(values, axis=1)
    ax.plot(xvals, mean_values, color="k", linewidth=state["mean_linewidth"])
    markersize = state["markersize"] if markersize is None else markersize
    for x, value, color in zip(xvals, mean_values, colors):
        ax.plot(x, value, color=color, marker="o", markersize=markersize, linestyle="none")


# ======================================================================================
# Familiarity curves (x = a mouse's own chronological session index)
# ======================================================================================


def add_familiarity_curve_widgets(
    viewer: Viewer,
    results: ResultsAggregator,
    *,
    style: str = "errorPlot",
    se: bool = True,
    min_mice: int = 2,
    linewidth: float = 1.5,
    subject_linewidth: float = 0.5,
    subject_alpha: float = 0.3,
    fill_alpha: float = 0.2,
) -> None:
    """Add the curve-rendering knobs consumed by :func:`draw_familiarity_series`."""
    viewer.add_selection("style", value=style, options=["errorPlot", "all"])
    viewer.add_boolean("se", value=se)
    viewer.add_integer("min_mice", value=min_mice, min=1, max=max(len(results.unique_mice), 1))
    viewer.add_float("linewidth", value=linewidth, min=0.25, max=5.0)
    viewer.add_float("subject_linewidth", value=subject_linewidth, min=0.0, max=3.0)
    viewer.add_float("subject_alpha", value=subject_alpha, min=0.0, max=1.0)
    viewer.add_float("fill_alpha", value=fill_alpha, min=0.0, max=1.0)


def _num_environments(session) -> int:
    """How many real environments a session ran, ignoring the invalid (negative) sentinel."""
    environments = np.asarray(session.environments)
    return int(np.unique(environments[environments >= 0]).size)


def chronological_mouse_sessions(results: ResultsAggregator, mouse: str, single_env_only: bool = False) -> np.ndarray:
    """Session indices for one mouse, sorted chronologically by date.

    Parameters
    ----------
    results : ResultsAggregator
        Provides ``sessions`` and ``mouse_names``.
    mouse : str
        Mouse to select.
    single_env_only : bool
        Keep only sessions that ran exactly one environment (see :func:`_num_environments`).

    Returns
    -------
    np.ndarray
        Indices into ``results.sessions``, in date order.
    """
    idx_mouse = np.where(results.mouse_names == mouse)[0]
    if single_env_only:
        idx_mouse = idx_mouse[[_num_environments(results.sessions[i]) == 1 for i in idx_mouse]]
    dates = np.array([results.sessions[i].date for i in idx_mouse])
    return idx_mouse[np.argsort(dates)]


def mouse_session_curves(results: ResultsAggregator, values: np.ndarray, single_env_only: bool) -> dict[str, np.ndarray]:
    """Per-mouse ``values`` in chronological session order, renumbered from 0.

    Filtered sessions are dropped rather than left as gaps, so index 0 is always the first
    *plotted* session of a mouse. With ``single_env_only`` that is the mouse's first session
    outright, since a second environment is only ever introduced later.

    Parameters
    ----------
    results : ResultsAggregator
        Provides ``sessions``, ``mouse_names`` and ``unique_mice``.
    values : np.ndarray
        One value per session, in ``results.sessions`` order, shape ``(n_sessions,)``.
    single_env_only : bool
        Keep only single-environment sessions.

    Returns
    -------
    dict[str, np.ndarray]
        ``{mouse: 1D array}``, ragged across mice. Mice left with no sessions are omitted.
    """
    curves: dict[str, np.ndarray] = {}
    for mouse in results.unique_mice:
        idx_sorted = chronological_mouse_sessions(results, mouse, single_env_only=single_env_only)
        if idx_sorted.size:
            curves[mouse] = values[idx_sorted]
    return curves


def _pad_stack(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Stack ragged per-mouse 1D curves into a NaN-padded ``(n_mice, max_len)`` array."""
    max_len = max((len(v) for v in curves.values()), default=0)
    stack = np.full((len(curves), max_len), np.nan)
    for i, values in enumerate(curves.values()):
        stack[i, : len(values)] = values
    return stack


def _supported_length(stack: np.ndarray, min_mice: int) -> int:
    """Columns up to the last one where at least ``min_mice`` mice have finite data."""
    support = np.sum(np.isfinite(stack), axis=0)
    valid = np.where(support >= min_mice)[0]
    return int(valid[-1]) + 1 if valid.size else 0


def draw_familiarity_series(ax, curves: dict[str, np.ndarray], color: str, label: str, state) -> tuple[int, np.ndarray]:
    """Draw one model's per-mouse session curves onto ``ax``.

    ``style == "all"`` draws every mouse thin plus a solid across-mouse mean; ``"errorPlot"``
    draws a mean +/- SE (or SD) band instead. Either way the mean is truncated where fewer than
    ``min_mice`` mice remain, since the tail of the x-axis is carried by whichever mouse ran
    longest. Style knobs come from :func:`add_familiarity_curve_widgets`.

    Returns
    -------
    tuple[int, np.ndarray]
        How many x positions carry a drawn artist (the mean's length, or the longest mouse when
        the individual traces are shown), and the values visible on the axis (for the caller's
        y autoscaling).
    """
    stack = _pad_stack(curves)
    length = _supported_length(stack, state["min_mice"])
    linewidth = state["linewidth"]
    if state["style"] == "all":
        for values in curves.values():
            ax.plot(np.arange(len(values)), values, color=(color, state["subject_alpha"]), linewidth=state["subject_linewidth"])
        if length:
            ax.plot(np.arange(length), np.nanmean(stack[:, :length], axis=0), color=color, linewidth=linewidth, label=label)
        return stack.shape[1], stack
    if not length:
        return 0, stack
    supported = stack[:, :length]
    se = state["se"]
    errorPlot(np.arange(length), supported, axis=0, se=se, ax=ax, color=color, linewidth=linewidth, alpha=state["fill_alpha"], label=label)
    # Only the band is drawn, not the individual mice, so the autoscale should track its edges
    # rather than single-mouse outliers well outside it.
    num_valid = np.sum(np.isfinite(supported), axis=0)
    spread = np.nanstd(supported, axis=0) / (np.sqrt(num_valid) if se else 1.0)
    mean = np.nanmean(supported, axis=0)
    return length, np.concatenate([mean - spread, mean + spread])
