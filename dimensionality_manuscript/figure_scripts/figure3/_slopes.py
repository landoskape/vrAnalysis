"""Per-env learning rates: collapsing each familiarity curve to a slope, and testing them.

Only the composite figure draws this panel, but it is a self-contained statistical model
(``slope ~ env + (1 | mouse)``) rather than drawing code, so it is kept apart from the panel
module that consumes it.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

from vrAnalysis.helpers.plotting import beeswarm, errorPlot, format_spines
from ...env_order import ENV_SLOT_COLORS

from ._ratios import COMPOSITE_SPINE_OFFSET

ENV_SLOPE_STYLES = ["beeswarm", "all", "errorPlot"]

_SLOPE_MODEL_FORMULAS = {"factor": "slope ~ C(env_index)", "linear": "slope ~ env_index", "null": "slope ~ 1"}


def env_slope_table(curves_by_env: dict[str, dict], min_sessions: int) -> pd.DataFrame:
    """Per-mouse learning rate in each env slot: the OLS slope of its Variance Ratio curve.

    One row per (mouse, env slot) that clears ``min_sessions`` finite sessions -- a slope from two
    points carries no information about a trend, so thin cells are dropped rather than fit. Unlike
    the curve panel, which truncates its x axis to where more than one mouse still has data, each
    mouse's slope uses every session it has in that env.

    Parameters
    ----------
    curves_by_env : dict
        ``familiarity_curves(..., "by_env")`` output: ``{env_label: {"svr": {mouse: 1D array}}}``.
    min_sessions : int
        Minimum number of finite sessions a mouse needs in an env slot to contribute a slope.

    Returns
    -------
    pandas.DataFrame
        Columns ``mouse``, ``env``, ``env_index``, ``slope``, ``n_sessions``.
    """
    rows = []
    for env_index, (env_label, data) in enumerate(curves_by_env.items()):
        for mouse, values in data["svr"].items():
            finite = np.isfinite(values)
            n_sessions = int(finite.sum())
            if n_sessions < min_sessions:
                continue
            session_index = np.arange(len(values))[finite]
            slope = float(np.polyfit(session_index, values[finite], 1)[0])
            rows.append(dict(mouse=mouse, env=env_label, env_index=env_index, slope=slope, n_sessions=n_sessions))
    return pd.DataFrame(rows, columns=["mouse", "env", "env_index", "slope", "n_sessions"])


def _holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of a family of p-values (order preserved)."""
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _fit_slope_models(table: pd.DataFrame) -> tuple[dict, bool]:
    """Fit the three ``slope ~ ... + (1 | mouse)`` models, dropping the random intercept if it collapses.

    When the between-mouse variance is estimated at zero -- which happens whenever the mouse-to-mouse
    spread is small relative to the residual spread -- the mixed model's Hessian is singular and
    statsmodels can't invert it for standard errors. At that boundary the mixed model *is* the OLS
    model, so the whole family is refit with OLS: the likelihoods stay comparable to each other, the
    LRT degrees of freedom are unchanged, and the contrasts become ordinary Wald tests. The returned
    flag says which happened, since it's worth knowing that the random intercept bought nothing.

    Returns
    -------
    tuple[dict, bool]
        ``({name: fitted result}, dropped_random_effect)``.
    """
    try:
        fits = {name: smf.mixedlm(f, table, groups=table["mouse"]).fit(reml=False, method="lbfgs") for name, f in _SLOPE_MODEL_FORMULAS.items()}
        return fits, False
    except np.linalg.LinAlgError:
        return {name: smf.ols(f, table).fit() for name, f in _SLOPE_MODEL_FORMULAS.items()}, True


def env_slope_stats(table: pd.DataFrame) -> dict | None:
    """Mixed-effects tests of whether the per-env learning slopes differ: ``slope ~ env + (1 | mouse)``.

    The random intercept is what makes this the right model rather than a one-way ANOVA: each mouse
    contributes a slope in several env slots, so its rows aren't independent.

    Three things are reported, because the omnibus alone doesn't answer the ordered question:

    - ``omnibus``: likelihood-ratio test of the env-as-factor model against the intercept-only null
      (``slope ~ 1 + (1 | mouse)``), on ``n_envs - 1`` df. Answers "do the slopes differ at all".
    - ``trend``: likelihood-ratio test of env as a *numeric* predictor against the same null, on
      1 df. This is the directional test for a monotonic env1 -> env2 -> env3 progression: it
      spends its single df on the ordering rather than spreading it over arbitrary differences,
      so it has more power than the omnibus for that specific hypothesis, and the sign of its
      coefficient (``trend_beta``) says which way the progression runs.
    - ``pairwise``: the three Wald contrasts between env slots from the factor model, with
      Holm-corrected p-values.

    Both LRTs are fit with ML (``reml=False``); REML likelihoods aren't comparable across models
    with different fixed effects.

    Returns
    -------
    dict or None
        ``None`` when the design is too thin to fit (fewer than two env slots or two mice with
        slopes); otherwise the tests above plus per-env ``n_mice``.
    """
    envs = sorted(table["env_index"].unique())
    if len(envs) < 2 or table["mouse"].nunique() < 2:
        return None

    fits, dropped_random_effect = _fit_slope_models(table)
    factor, linear, null = fits["factor"], fits["linear"], fits["null"]

    omnibus_lr = 2.0 * (factor.llf - null.llf)
    trend_lr = 2.0 * (linear.llf - null.llf)

    # Wald contrasts on the factor model's fixed effects. Treatment coding puts env ``envs[0]`` in
    # the intercept, so its "coefficient" is the zero vector and every pairwise difference is just
    # the difference of two such vectors.
    fe = factor.params if dropped_random_effect else factor.fe_params
    names = list(fe.index)
    cov = factor.cov_params().loc[names, names].to_numpy()

    def _level_vector(env: int) -> np.ndarray:
        vector = np.zeros(len(names))
        if env != envs[0]:
            vector[names.index(f"C(env_index)[T.{env}]")] = 1.0
        return vector

    pairwise = []
    for i, env_a in enumerate(envs):
        for env_b in envs[i + 1 :]:
            contrast = _level_vector(env_b) - _level_vector(env_a)
            difference = float(contrast @ fe.to_numpy())
            se = float(np.sqrt(contrast @ cov @ contrast))
            z = difference / se
            pairwise.append(dict(env_a=env_a, env_b=env_b, difference=difference, se=se, z=z, p=float(2.0 * norm.sf(abs(z)))))
    for entry, p_holm in zip(pairwise, _holm(np.array([entry["p"] for entry in pairwise]))):
        entry["p_holm"] = float(p_holm)

    return dict(
        omnibus_lr=float(omnibus_lr),
        omnibus_df=len(envs) - 1,
        omnibus_p=float(chi2.sf(omnibus_lr, len(envs) - 1)),
        trend_lr=float(trend_lr),
        trend_df=1,
        trend_p=float(chi2.sf(trend_lr, 1)),
        trend_beta=float((linear.params if dropped_random_effect else linear.fe_params)["env_index"]),
        pairwise=pairwise,
        n_mice={int(env): int((table["env_index"] == env).sum()) for env in envs},
        n_total=int(len(table)),
        dropped_random_effect=dropped_random_effect,
    )


def _format_p(p: float) -> str:
    """p-value for display: fixed decimals until they'd round to zero, then a bound."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def format_env_slope_stats(stats: dict | None, env_labels: list[str]) -> str:
    """Compact multi-line summary of :func:`env_slope_stats` for annotating the panel."""
    if stats is None:
        return "too few mice for stats"
    lines = [
        f"env: LR={stats['omnibus_lr']:.1f} (df {stats['omnibus_df']}), {_format_p(stats['omnibus_p'])}",
        f"trend: LR={stats['trend_lr']:.1f} (df 1), {_format_p(stats['trend_p'])}",
    ]
    for entry in stats["pairwise"]:
        label_a, label_b = env_labels[entry["env_a"]], env_labels[entry["env_b"]]
        lines.append(f"{label_a} vs {label_b}: {_format_p(entry['p_holm'])}")
    if stats["dropped_random_effect"]:
        lines.append("(mouse variance 0; OLS)")
    return "\n".join(lines)


def plot_env_slopes(
    ax,
    table: pd.DataFrame,
    env_labels: list[str],
    style: str,
    fontsize: float,
    stats_text: str | None = None,
) -> None:
    """Per-mouse Variance Ratio slope in each env slot.

    ``style`` picks the rendering: ``"beeswarm"`` swarms each env's per-mouse slopes around its
    tick with a mean line; ``"all"`` instead draws one faint line per mouse across the env slots
    (so a mouse's own progression is followed) plus the across-mouse mean; ``"errorPlot"`` reduces
    that to the mean +/- SE band. A dashed zero line marks "no change over sessions", which is the
    reference the slopes are read against.
    """
    if style not in ENV_SLOPE_STYLES:
        raise ValueError(f"Unknown style {style!r}. Options: {ENV_SLOPE_STYLES}")

    env_indices = np.arange(len(env_labels))
    # Reindexed onto every slot so a mouse missing one env leaves a NaN gap rather than shifting.
    per_mouse = (
        table.pivot(index="mouse", columns="env_index", values="slope").reindex(columns=env_indices)
        if len(table)
        else pd.DataFrame(np.empty((0, len(env_indices))), columns=env_indices)
    )
    values = per_mouse.to_numpy(dtype=float)

    ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--", zorder=0)

    if style == "beeswarm":
        for env_index in env_indices:
            slopes = values[:, env_index]
            slopes = slopes[np.isfinite(slopes)]
            if not slopes.size:
                continue
            color = ENV_SLOT_COLORS[env_index]
            ax.plot(
                env_index + 0.15 * beeswarm(slopes),
                slopes,
                color=color,
                linestyle="none",
                marker="o",
                markersize=3,
                alpha=0.4,
            )
            ax.plot(env_index + np.array([-0.25, 0.25]), np.full(2, np.nanmean(slopes)), color=color, linewidth=2.0)
    elif style == "all":
        for mouse_slopes in values:
            ax.plot(env_indices, mouse_slopes, color=("k", 0.3), linewidth=0.5, marker="o", markersize=2)
        ax.plot(env_indices, np.nanmean(values, axis=0), color="k", linewidth=2.0)
    elif values.size:
        errorPlot(env_indices, values, axis=0, se=True, ax=ax, color="k", linewidth=2.0, alpha=0.25)

    ax.set_xlabel("Environment", fontsize=fontsize)
    ax.set_ylabel("Slope (ratio / session)", fontsize=fontsize)
    ax.set_xlim(-0.5, len(env_labels) - 0.5)
    # Pinned rather than left on autoscale, so format_spines' fractional offsets stay put.
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
    format_spines(
        ax,
        x_pos=COMPOSITE_SPINE_OFFSET,
        y_pos=COMPOSITE_SPINE_OFFSET,
        spines_visible=["left", "bottom"],
        xbounds=[0, len(env_labels) - 1],
        ybounds=ylim,
        tick_fontsize=fontsize,
    )
    ax.set_xticks(env_indices, labels=env_labels, fontsize=fontsize)

    if stats_text is not None:
        # Upper left: the hypothesis under test is that slopes rise across env slots, so that
        # corner is the one the data vacates. It will collide if the effect ever runs the other way.
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=fontsize * 0.8,
            ha="left",
            va="top",
            linespacing=1.4,
        )
