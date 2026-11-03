"""Standalone saved-results report: run with vrAnalysis Python; no Viewer or refits."""

from __future__ import annotations

import html
import json
import pickle
import sqlite3
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BLUE, GREEN, ORANGE, RED, PURPLE = "#3569aa", "#248978", "#d18a30", "#be4d59", "#8869a8"
NEW = "internal_placefield_1d_structured_gain_residual"
EXT = "external_placefield_1d_structured_gain_residual"
OLD = "internal_placefield_1d_structured_gain"
MODELS = ("internal_placefield_1d", OLD, "internal_placefield_1d_structured_gain_sqrt", NEW, EXT, "rrr")
SUFFIX = "_spks=sigrebase_method=preferred_ap=std_rel=0.3_fa=0.1_seed=0_v2"


def mouse_means(frame, columns):
    """Equal environments within session, sessions within mouse; mice stay separate."""
    return frame.groupby(["mouse", "session"])[columns].mean().groupby("mouse").mean()


def read_results():
    with (ROOT / "paths.toml").open("rb") as handle:
        store = Path(tomllib.load(handle)["paths"]["storage"]) / "dimensionality-manuscript/cache/pipeline_v2"
    connection = sqlite3.connect((store / "results.db").as_uri() + "?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT result_uid,session_id,analysis_key,analysis_summary,computed_at,snapshot_path "
        "FROM results WHERE analysis_type='trial_placecell_regression' AND schema_version='v2' AND result_stored=1"
    ).fetchall()
    connection.close()
    stored, provenance = {}, []
    for uid, session, key, summary, computed, snapshot in rows:
        if not summary.endswith(SUFFIX):
            continue
        model = summary.removeprefix("trial_placecell_regression_")[: -len(SUFFIX)]
        if model not in MODELS:
            continue
        assert (model, session) not in stored, "Ambiguous configuration"
        stored[model, session] = pickle.loads((store / "blobs" / f"{uid}.pkl").read_bytes())
        provenance.append(
            dict(result_uid=uid, session=session, model=model, analysis_key=key, configuration=summary, computed_at=computed, snapshot_path=snapshot)
        )
    assert all(any(m == model for m, _ in stored) for model in MODELS), "Missing model"
    records, folds = [], []
    for session in sorted({s for m, s in stored if m == NEW}):
        assert all((model, session) in stored for model in MODELS), ("Missing comparison", session)
        new = stored[NEW, session]
        for slot, env in enumerate(new["env_slot_ids"]):
            if not np.isfinite(env) or not np.isfinite(new["r2"][slot]):
                continue
            base = dict(session=session, mouse=session.rsplit(".", 2)[0], slot=slot + 1, environment=int(env))
            record = dict(base)
            for model in MODELS:
                data = stored[model, session]
                assert data["env_slot_ids"][slot] == env and np.isfinite(data["r2"][slot])
                for key in ("source_roi_indices", "target_roi_indices"):
                    assert np.array_equal(new[key][slot], data[key][slot]), ("ROI mismatch", session, model)
                for key in ("cv_test_folds", "fixed_validation_fold"):
                    assert np.array_equal(new[key], data[key]), ("Fold mismatch", session, model)
                for a, b in zip(new["cv_trial_folds"][slot], data["cv_trial_folds"][slot]):
                    assert np.array_equal(a, b), ("Trial mismatch", session, model)
                record[model] = float(data["r2"][slot])
            for prefix, model in (("internal", NEW), ("external", EXT)):
                data = stored[model, session]
                for key, value in data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape == (3,) and np.issubdtype(value.dtype, np.number):
                        record[f"{prefix}_{key}"] = float(value[slot])
                for key in ("rank", "alpha", "rank_residual", "alpha_residual", "num_bins", "smooth_width"):
                    record[f"{prefix}_hp_{key}"] = data["selected_hyperparameters"][slot][key]
                scores = np.asarray(data["r2_folds"][slot])
                assert np.isfinite(scores).all()
                record[f"{prefix}_any_severe"] = bool(np.any(scores < -1))
                record[f"{prefix}_median_severe"] = bool(data["r2"][slot] < -1)
                record[f"{prefix}_worst_fold"] = float(scores.min())
                for index, fold in enumerate(data["cv_test_folds"]):
                    row = dict(base, branch=prefix, test_fold=int(fold))
                    for key in ("r2", "r2_gain", "r2_residual_component", "r2_gain_component", "delta_r2_gain", "delta_r2_residual"):
                        row[key] = float(data[key + "_folds"][slot, index])
                    folds.append(row)
                # A relative check at 1e98 would misleadingly accept lost increments.
                if not record[f"{prefix}_any_severe"]:
                    lhs = data["r2_template_pooled"][slot] + data["delta_r2_gain_pooled"][slot] + data["delta_r2_residual_pooled"][slot]
                    assert np.isclose(lhs, data["r2_pooled"][slot], rtol=0, atol=1e-12)
            records.append(record)
    data, fold_data = pd.DataFrame(records), pd.DataFrame(folds)
    assert not data.empty and not data.duplicated(["session", "slot"]).any()
    saved_sessions = sorted({s for m, s in stored if m == NEW})
    coverage = pd.DataFrame({"session": saved_sessions})
    coverage["n_scored_environments"] = coverage.session.map(data.groupby("session").size()).fillna(0).astype(int)
    coverage.to_csv(OUT / "session_coverage.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUT / "provenance.csv", index=False)
    data.to_csv(OUT / "session_environment_scores.csv", index=False)
    fold_data.to_csv(OUT / "fold_scores.csv", index=False)
    return store, data, fold_data


def panel(ax, letter, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=25)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#58616c", va="bottom")
    ax.text(-0.10, 1.13, letter, transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=9)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)
    print(f"Saved {name}", flush=True)


def figure_performance(data, folds):
    fig = plt.figure(figsize=(14.5, 10.8), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(grid[0, :])
    columns = [MODELS[0], OLD, MODELS[2], "internal_r2_gain", NEW, "rrr"]
    labels = ["Internal PF", "Existing trial gain", "Existing sqrt gain", "New G only", "New G + E", "Full peer RRR"]
    colors = ["#53616e", GREEN, PURPLE, ORANGE, RED, BLUE]
    rng = np.random.default_rng(20261103)
    low, high, counts = -0.08, 0.36, []
    for i, (column, color) in enumerate(zip(columns, colors)):
        y = data[column].to_numpy()
        visible = (y >= low) & (y <= high)
        ax.scatter(i + rng.uniform(-0.2, 0.2, visible.sum()), y[visible], s=10, alpha=0.25, color=color, linewidth=0)
        median = float(np.median(y))
        ax.plot([i - 0.28, i + 0.28], [median, median], color=color, lw=3)
        ax.text(i, high - 0.012, f"median {median:.3f}", ha="center", va="top", color=color, fontsize=10)
        below, above = int(np.sum(y < low)), int(np.sum(y > high))
        counts.append(dict(model=labels[i], below_window=below, above_window=above, median=median))
        if below:
            ax.scatter([i], [low + 0.006], marker="v", color=color, s=38)
            ax.text(i, low + 0.024, f"{below} below", ha="center", fontsize=9, color=color)
        if above:
            ax.scatter([i], [high - 0.003], marker="^", color=color, s=38)
            ax.text(i, high - 0.045, f"{above} above", ha="center", fontsize=9, color=color)
    ax.axhline(0, color="#9aa3ad", lw=0.8, ls="--")
    ax.set(xticks=np.arange(len(columns)), xticklabels=labels, ylim=(low, high), ylabel="Median test-fold R²")
    panel(
        ax,
        "A",
        "Performance comparison: typical scores and explicit tail counts",
        f"Same {len(data)} session–environment units in every column. Dots = units; bars = medians of ALL units, not mouse means.",
    )
    ax = fig.add_subplot(grid[1, 0])
    mice = sorted(data.mouse.unique())
    x = np.arange(len(mice))
    for offset, prefix, color in ((-0.19, "internal", RED), (0.19, "external", BLUE)):
        rates = data.groupby("mouse")[f"{prefix}_any_severe"].mean().reindex(mice)
        ax.bar(x + offset, rates * 100, width=0.36, color=color, alpha=0.85, label=prefix.capitalize())
    ax.set(xticks=x, xticklabels=[m.replace("CR_Hippocannula", "HC") for m in mice], ylabel="Units with ≥1 severe fold (%)", ylim=(0, 104))
    ax.tick_params(axis="x", rotation=60)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    panel(ax, "B", "Failures are distributed across mice", "Severe fold: R² < −1, i.e. SSE > 2 × target SST.")
    ax = fig.add_subplot(grid[1, 1])
    for prefix, color in (("internal", RED), ("external", BLUE)):
        part = folds[(folds.branch == prefix) & (folds.r2 < -1)]
        severity = np.sort(np.log10(1 - part.r2.to_numpy()))
        gain_severity = np.sort(np.log10(1 - part.r2_gain.to_numpy()))
        percentile = np.arange(1, len(part) + 1) / len(part) * 100
        ax.step(severity, percentile, where="post", color=color, lw=2, label=f"{prefix.capitalize()} G+E ({len(part)} folds)")
        ax.plot(gain_severity, percentile, color=color, lw=1.1, ls="--", alpha=0.8)
    ax.set(xlabel="log₁₀(SSE / SST) among severe combined-model folds", ylabel="Cumulative fraction of severe folds (%)", ylim=(0, 102))
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(
        0.03,
        0.94,
        "Solid: G+E    Dashed: G alone\nNearly overlapping curves: G dominates the failure",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )
    panel(ax, "C", "The negative tail is enormous, and follows G", "Every severe fold is included; no clipping or winsorization.")
    fig.suptitle("Trial gain + residual prediction: real-data stability audit", fontsize=17, fontweight="bold")
    save(fig, "01_performance_and_stability")
    return counts


def figure_structure(data):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.3), layout="constrained")
    ax = axes[0, 0]
    failed = data.internal_any_severe.to_numpy()
    for mask, color, label in ((~failed, GREEN, "No severe internal fold"), (failed, RED, "≥1 severe internal fold")):
        ax.scatter(
            data.loc[mask, "external_r2_residual_component"],
            data.loc[mask, "internal_r2_residual_component"],
            color=color,
            s=22,
            alpha=0.60,
            edgecolors="none",
            label=f"{label} (n={mask.sum()})",
        )
    bounds = (-0.025, 0.14)
    ax.plot(bounds, bounds, color="#89939e", ls="--", lw=1)
    ax.axhline(0, color="#b4bbc3", lw=0.7)
    ax.axvline(0, color="#b4bbc3", lw=0.7)
    ax.set(xlim=bounds, ylim=bounds, xlabel="External-position model: R² on E", ylabel="Internal-position model: R² on E")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    panel(ax, "A", "E carries held-out peer-predictable structure", "All 267 units; each axis scores its model's own measured E.")
    ax = axes[0, 1]
    mouse = mouse_means(data, ["external_r2_residual_component", "internal_r2_residual_component"])
    for _, row in mouse.iterrows():
        ax.plot([0, 1], row.to_numpy(), color="#b6bec8", lw=0.9, alpha=0.75)
        ax.scatter([0, 1], row.to_numpy(), color=[BLUE, GREEN], s=25, zorder=3)
    means = mouse.mean()
    ax.plot([0, 1], means.to_numpy(), color="#263441", lw=2.5, marker="D", ms=7, zorder=4)
    for i, value in enumerate(means):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(12, -3), textcoords="offset points", fontweight="bold", fontsize=11)
    ax.axhline(0, color="#a7afb8", lw=0.8, ls="--")
    ax.set(xticks=[0, 1], xticklabels=["External position", "Internal position"], xlim=(-0.3, 1.45), ylabel="R² predicting measured E")
    panel(ax, "B", "The signal survives mouse-level averaging", "14 mice; equal environments → sessions → mice. Diamond = mean.")
    ax = axes[1, 0]
    rng, diagnostic = np.random.default_rng(47), []
    for x, prefix, color in ((0, "internal", GREEN), (1, "external", BLUE)):
        selected = data.loc[~data[f"{prefix}_any_severe"]]
        values = mouse_means(selected, [f"{prefix}_delta_r2_residual_pooled"]).iloc[:, 0].to_numpy()
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(values)), values, color=color, s=36, alpha=0.8, zorder=3)
        mean = float(np.mean(values))
        ax.plot([x - 0.24, x + 0.24], [mean, mean], color=color, lw=3)
        ax.text(
            x, 0.96, f"{len(selected)}/{len(data)} units\n{len(values)} mice", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10
        )
        diagnostic.append(dict(branch=prefix, n_units=len(selected), n_mice=len(values), mean_mouse_residual_increment=mean))
    ax.axhline(0, color="#a7afb8", lw=0.8, ls="--")
    ax.set(xticks=[0, 1], xticklabels=["Internal position", "External position"], xlim=(-0.5, 1.5), ylabel="Additional original-activity R² from E")
    ax.margins(y=0.35)
    panel(ax, "C", "Adding E helps on the noncatastrophic subset", "Diagnostic only: every combined-model fold has R² ≥ −1.")
    ax = axes[1, 1]
    rank_summary = []
    for x, label, color in ((0, "Gain", ORANGE), (1, "Residual", GREEN)):
        effective = data["internal_structured_gain_effective_rank" if label == "Gain" else "internal_residual_effective_rank"]
        maximum = data["internal_structured_gain_max_trial_rank" if label == "Gain" else "internal_residual_max_rank"]
        ratio = effective / maximum
        ax.scatter(x + rng.uniform(-0.16, 0.16, len(ratio)), ratio, color=color, alpha=0.22, s=12, linewidth=0)
        ax.plot([x - 0.24, x + 0.24], [np.median(ratio)] * 2, lw=3, color=color)
        ax.text(
            x,
            0.06,
            f"Median rank: {np.median(effective):.0f}\nMedian bound: {np.median(maximum):.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
        )
        rank_summary.append(
            dict(
                branch=label,
                median_rank=float(np.median(effective)),
                median_rank_bound=float(np.median(maximum)),
                at_bound=int(np.isclose(effective, maximum).sum()),
            )
        )
    ax.axhline(1, lw=0.8, color="#a7afb8", ls="--")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Trial gain RRR", "Framewise residual RRR"],
        xlim=(-0.5, 1.5),
        ylim=(-0.04, 1.12),
        ylabel="Effective rank / available rank",
    )
    panel(ax, "D", "The branches use very different model capacity", "Internal model; gain bound ≈ trials, residual bound ≈ cells.")
    fig.suptitle("What can we currently say about the 'everything else' component?", fontsize=17, fontweight="bold")
    save(fig, "02_residual_structure")
    return mouse, diagnostic, rank_summary


def write_report(store, data, folds, counts, mouse, diagnostic, ranks):
    stability = []
    for prefix in ("internal", "external"):
        part = folds[folds.branch == prefix]
        stability.append(
            dict(
                branch=prefix,
                any_severe_units=int(data[f"{prefix}_any_severe"].sum()),
                median_severe_units=int(data[f"{prefix}_median_severe"].sum()),
                severe_folds=int((part.r2 < -1).sum()),
                total_folds=len(part),
                positive_residual_folds=int((part.r2_residual_component > 0).sum()),
                positive_residual_units=int((data[f"{prefix}_r2_residual_component"] > 0).sum()),
                median_residual_r2=float(data[f"{prefix}_r2_residual_component"].median()),
                mouse_mean_residual_r2=float(mouse[f"{prefix}_r2_residual_component"].mean()),
                worst_fold_r2=float(part.r2.min()),
            )
        )
    coverage = pd.read_csv(OUT / "session_coverage.csv")
    summary = dict(
        n_saved_sessions=len(coverage),
        n_unscored_sessions=int((coverage.n_scored_environments == 0).sum()),
        n_sessions=int(data.session.nunique()),
        n_mice=int(data.mouse.nunique()),
        n_units=len(data),
        store=str(store),
        schema="v2",
        configuration=SUFFIX,
        plot_window=counts,
        diagnostic_subset=diagnostic,
        rank_summary=ranks,
        stability=stability,
    )
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(stability).to_csv(OUT / "stability_summary.csv", index=False)
    mouse.to_csv(OUT / "mouse_residual_scores.csv")
    data.sort_values("internal_worst_fold").head(12)[
        [
            "session",
            "slot",
            "environment",
            "internal_worst_fold",
            "internal_r2",
            "internal_r2_gain",
            "internal_r2_residual_component",
            "internal_hp_alpha",
            "internal_hp_rank",
            "internal_hp_alpha_residual",
            "internal_hp_rank_residual",
        ]
    ].to_csv(OUT / "worst_internal_results.csv", index=False)
    internal, external = stability
    table = pd.DataFrame(stability)[
        [
            "branch",
            "any_severe_units",
            "median_severe_units",
            "severe_folds",
            "positive_residual_folds",
            "median_residual_r2",
            "mouse_mean_residual_r2",
        ]
    ]
    table.columns = [
        "Position",
        "Any severe fold: units / 267",
        "Severe median: units / 267",
        "Severe folds / 801",
        "E R² > 0: folds / 801",
        "Median E R²",
        "Equal-mouse mean E R²",
    ]
    caption1 = (
        "A shows stored median-fold scores, with one dot per session–environment unit: a descriptive distribution, not independent animals. "
        "The fixed display window is −0.08 to 0.36; out-of-window counts are explicit and medians use all units. "
        "B exposes failures hidden by median-fold scoring. C includes every severe combined-model fold on a log error-ratio scale. "
        "Dashed G-only curves use the same folds as the solid combined curves."
    )
    caption2 = (
        "A and B predict E on all units, including those with unstable gain prediction. Internal and external E targets differ because "
        "their template trajectories differ, and their component R² denominators differ. Component scores do not add to original-activity R². "
        "B averages environments within session, sessions within mouse, then mice equally. C is explicitly restricted to units whose three "
        "combined-model folds all have R² ≥ −1. It plots (SSE_G − SSE_G+E)/SST_original followed by the same mouse-level averaging. "
        "That subset is score-selected: this is a diagnostic, not an unbiased population estimate. D shows rank relative to different branch-specific bounds."
    )
    prose = f"""
<h2>What the saved results support</h2><ul>
<li><strong>There is predictable structure beyond a per-trial template rescaling.</strong>
Measured E remains after an explicit per-cell, per-trial template projection. Its component R² is positive in
{internal['positive_residual_units']}/267 internal units and {external['positive_residual_units']}/267 external units.
These are descriptive counts; folds and units share animals.</li>
<li><strong>The gain predictor is the current bottleneck.</strong> Failures affect {internal['any_severe_units']}/267 internal units
and {external['any_severe_units']}/267 external units. G-only and G+E errors track each other in the severe tail.
The worst internal fold has R² = {internal['worst_fold_r2']:.2e}.</li>
<li><strong>Median-fold scoring hides some failures.</strong> Only {internal['median_severe_units']} internal units have median R² &lt; −1,
although {internal['any_severe_units']} have at least one such fold. Pooled increments also lose precision when subtracting enormous,
nearly equal SSEs, so we do not interpret their numerical values in that tail.</li>
<li><strong>Residual prediction is not necessarily low dimensional.</strong> The median internal residual rank is {ranks[1]['median_rank']:.0f},
versus {ranks[0]['median_rank']:.0f} for gain. E has many more observations and a much higher available-rank bound.</li></ul>
<h2>Interpretation and next step</h2>
<p>Predictable E can include template mismatch and, for the internal model, position-decoding errors.
The result establishes structure beyond this chosen trial-gain template; it does not by itself identify a separate physiological process.</p>
<p>The failure pattern is consistent with ill-conditioned raw gain ratios when a cell's template energy is tiny.
The new decomposition uses unregularized gains to preserve exact orthogonality; the existing structured-gain model shrinks gain measurements toward 1.
That is a plausible explanation from the implementation and G-only error pattern, not a direct measurement of failing template energies.</p>
<p>A useful repair would preserve the explicit measured decomposition while stabilizing the gain features used by the learned predictor
(for example, with separate regularized source-gain features). The branches should then be reevaluated before claiming a clean additive result.
This report does not modify the model, delete results, or refit sessions.</p>
<h2>Data and reproducibility</h2>
<p>All six compared models match on session, environment identity, exact source/target ROI IDs, and trial-fold membership.
Every compared model has 149 saved sessions; 140 sessions have at least one finite scored environment (267 scored units in total).
The other 9 sessions contribute no scored environments and are listed in session_coverage.csv.
Exact retained prediction-frame masks are absent from cached summaries;
cross-model frame support cannot be independently audited here. Within the new model, G and E share a frame mask.</p>
<p>The main comparison uses the median of three held-out trial-fold scores. The validation quarter is fixed and hyperparameters are reused
across rotations. Place-cell selection uses full-session activity, as in the existing analysis. There is no noise-ceiling correction:
full peer RRR is a benchmark, not maximum explainable predictability.</p>
<p>Source store: <code>{html.escape(str(store / 'results.db'))}</code>, opened read-only.
Configurations and computation timestamps are recorded in provenance.csv.</p>
"""
    links = " · ".join(
        f'<a href="{name}">{label}</a>'
        for name, label in (
            ("01_performance_and_stability.svg", "Figure 1 SVG"),
            ("02_residual_structure.svg", "Figure 2 SVG"),
            ("session_environment_scores.csv", "Session–environment scores"),
            ("fold_scores.csv", "Fold scores"),
            ("mouse_residual_scores.csv", "Mouse summaries"),
            ("worst_internal_results.csv", "Worst results"),
            ("session_coverage.csv", "Session coverage"),
            ("provenance.csv", "Provenance"),
            ("manifest.json", "Audit manifest"),
            ("make_report.py", "Reproduction script"),
        )
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial gain + residual: saved-results report</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#253241;background:#f2f5f8;margin:0}}main{{max-width:1350px;margin:auto;padding:42px 36px}}
h1{{font-size:34px;line-height:1.2;margin:0 0 12px}}h2{{font-size:24px;margin:36px 0 12px}}.meta,figcaption{{color:#596776;font-size:14px}}
.warning{{background:#fff1ed;border-left:5px solid #be4d59;padding:18px 24px;border-radius:6px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}
.card{{background:white;border-radius:8px;padding:18px}}.big{{font-size:30px;font-weight:700;display:block}}figure{{margin:22px 0;background:white;padding:16px;border-radius:10px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:12px 10px;text-align:right;border-bottom:1px solid #dfe5ea}}th:first-child,td:first-child{{text-align:left}}
code{{font-size:13px}}a{{color:#3569aa}}li{{margin:8px 0}}@media(max-width:700px){{main{{padding:22px 14px}}.cards{{grid-template-columns:1fr}}table{{font-size:11px}}}}</style>
<main><h1>Trial gain + residual prediction</h1><p class="meta">Saved v2 results · 3 November 2026 · sigrebase / std / preferred / seed 0 · 140 scored sessions out of 149 saved, 14 mice, 267 matched session–environment units.</p>
<div class="warning"><strong>Archived unregularized v2 run.</strong> The gain-residual model has since been revised (v3) to reuse the standalone gain predictor.
The population figures below have not been recomputed. Two real-data equality checks are saved in <a href="regularized_validation.json">regularized_validation.json</a>.
<br><strong>The v2 combined model was not numerically reliable.</strong> The residual branch predicts held-out structure in E,
but catastrophic gain predictions prevent a clean population-level claim that G+E explains the gap to full peer RRR.
The synthetic implementation tests missed this real-data instability.</div>
<div class="cards"><div class="card"><span class="big">{internal['any_severe_units']} / 267</span>Internal-model units with at least one R² &lt; −1 fold.</div>
<div class="card"><span class="big">{internal['positive_residual_folds']} / 801</span>Internal-model folds with positive held-out E-component R².</div>
<div class="card"><span class="big">{internal['median_residual_r2']:.3f}</span>Median internal E-component R²; equal-mouse mean {internal['mouse_mean_residual_r2']:.3f}.</div></div>
<h2>1. Predictive performance and stability</h2><figure><img src="01_performance_and_stability.png" alt="Model comparisons, per-mouse failures, and full error tails"><figcaption>{caption1}</figcaption></figure>
<h2>2. The residual branch carries held-out predictability</h2><figure><img src="02_residual_structure.png" alt="Residual predictability, mouse comparisons, conditional benefit, and rank use"><figcaption>{caption2}</figcaption></figure>
<h2>Audit summary</h2>{table.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)}
{prose}<p>{links}</p></main></html>"""
    (OUT / "report.html").write_text(document, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334252",
            "text.color": "#253241",
            "axes.edgecolor": "#9ba6b2",
            "svg.fonttype": "none",
        }
    )
    store, data, folds = read_results()
    counts = figure_performance(data, folds)
    mouse, diagnostic, ranks = figure_structure(data)
    write_report(store, data, folds, counts, mouse, diagnostic, ranks)


if __name__ == "__main__":
    main()
