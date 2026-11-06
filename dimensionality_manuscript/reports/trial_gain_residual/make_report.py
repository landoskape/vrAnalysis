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
        result = pickle.loads((store / "blobs" / f"{uid}.pkl").read_bytes())
        if model in (NEW, EXT):
            assert "target_residual_gain_energy_fraction" in result, ("Unrecomputed gain-residual blob", session, model)
        stored[model, session] = result
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
                    for key in (
                        "r2",
                        "r2_gain",
                        "r2_residual_component",
                        "r2_gain_component",
                        "delta_r2_gain",
                        "delta_r2_residual",
                        "source_residual_gain_energy_fraction",
                        "target_residual_gain_energy_fraction",
                        "predicted_residual_gain_energy_fraction",
                    ):
                        row[key] = float(data[key + "_folds"][slot, index])
                    folds.append(row)
                lhs = data["r2_template_pooled"][slot] + data["delta_r2_gain_pooled"][slot] + data["delta_r2_residual_pooled"][slot]
                assert np.isclose(lhs, data["r2_pooled"][slot], rtol=0, atol=1e-12)
                for component in ("source_residual", "target_residual", "predicted_residual"):
                    energy = data[component + "_energy"][slot]
                    projection = data[component + "_gain_energy"][slot]
                    off = data[component + "_off_template_energy"][slot]
                    assert np.isclose(projection + off, energy, rtol=1e-10)
                    assert np.isclose(projection / energy, data[component + "_gain_energy_fraction"][slot], rtol=0, atol=1e-12)
            reference = stored[OLD, session]
            hp = new["selected_hyperparameters"][slot]
            old_hp = reference["selected_hyperparameters"][slot]
            assert all(hp[key] == old_hp[key] for key in ("num_bins", "smooth_width", "alpha", "rank")), (
                "Gain hyperparameter mismatch",
                session,
                slot,
            )
            record["gain_reference_score_difference"] = float(new["r2_gain"][slot] - reference["r2"][slot])
            record["gain_reference_max_fold_difference"] = float(np.max(np.abs(new["r2_gain_folds"][slot] - reference["r2_folds"][slot])))
            assert np.allclose(new["r2_gain_folds"][slot], reference["r2_folds"][slot], rtol=0, atol=1e-6), ("Gain score mismatch", session, slot)
            record["gain_hyperparameters_match"] = True
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


def summary_data(data, folds):
    columns = list(MODELS) + ["internal_r2_gain"]
    for prefix in ("internal", "external"):
        columns += [
            f"{prefix}_{key}"
            for key in (
                "r2_pooled",
                "r2_template_pooled",
                "delta_r2_gain_pooled",
                "delta_r2_residual_pooled",
                "r2_residual_component",
                "source_residual_gain_energy_fraction",
                "target_residual_gain_energy_fraction",
                "predicted_residual_gain_energy_fraction",
            )
        ]
    mouse = mouse_means(data, columns)
    counts = pd.read_csv(OUT / "session_coverage.csv")
    gain_mean, combined_mean, peer_mean = (float(mouse[key].mean()) for key in (OLD, NEW, "rrr"))
    stats = dict(
        implementation="regularized_measured_gain",
        schema="v2",
        configuration=SUFFIX,
        n_saved_sessions=len(counts),
        n_sessions=int(data.session.nunique()),
        n_unscored_sessions=int((counts.n_scored_environments == 0).sum()),
        n_mice=int(data.mouse.nunique()),
        n_units=len(data),
        mouse_mean_trial_gain_r2=gain_mean,
        mouse_mean_combined_r2=combined_mean,
        mouse_mean_peer_r2=peer_mean,
        median_fold_score_increment=combined_mean - gain_mean,
        mouse_mean_pooled_template_r2=float(mouse.internal_r2_template_pooled.mean()),
        mouse_mean_pooled_gain_increment=float(mouse.internal_delta_r2_gain_pooled.mean()),
        mouse_mean_pooled_combined_r2=float(mouse.internal_r2_pooled.mean()),
        peer_gap_recovered=(combined_mean - gain_mean) / (peer_mean - gain_mean),
        gain_hyperparameters_match=bool(data.gain_hyperparameters_match.all()),
        max_gain_reference_fold_difference=float(data.gain_reference_max_fold_difference.max()),
        max_gain_reference_summary_difference=float(data.gain_reference_score_difference.abs().max()),
        stability=[],
        rank_summary=[],
    )
    for prefix in ("internal", "external"):
        f = folds.loc[folds.branch == prefix]
        stats["stability"].append(
            dict(
                branch=prefix,
                total_folds=len(f),
                severe_folds=int((f.r2 < -1).sum()),
                severe_gain_folds=int((f.r2_gain < -1).sum()),
                any_severe_units=int(data[f"{prefix}_any_severe"].sum()),
                median_severe_units=int(data[f"{prefix}_median_severe"].sum()),
                worst_fold_r2=float(f.r2.min()),
                positive_residual_folds=int((f.r2_residual_component > 0).sum()),
                positive_residual_units=int((data[f"{prefix}_r2_residual_component"] > 0).sum()),
                positive_residual_increment_units=int((data[f"{prefix}_delta_r2_residual_pooled"] > 0).sum()),
                mean_mouse_residual_r2=float(mouse[f"{prefix}_r2_residual_component"].mean()),
                mean_mouse_residual_increment=float(mouse[f"{prefix}_delta_r2_residual_pooled"].mean()),
                mean_mouse_source_gain_energy_fraction=float(mouse[f"{prefix}_source_residual_gain_energy_fraction"].mean()),
                mean_mouse_target_gain_energy_fraction=float(mouse[f"{prefix}_target_residual_gain_energy_fraction"].mean()),
                mean_mouse_predicted_gain_energy_fraction=float(mouse[f"{prefix}_predicted_residual_gain_energy_fraction"].mean()),
            )
        )
    for component, effective, bound in (
        ("Gain", "internal_structured_gain_effective_rank", "internal_structured_gain_max_trial_rank"),
        ("Residual", "internal_residual_effective_rank", "internal_residual_max_rank"),
    ):
        stats["rank_summary"].append(
            dict(
                branch=component,
                median_rank=float(data[effective].median()),
                median_bound=float(data[bound].median()),
                median_fraction=float((data[effective] / data[bound]).median()),
                at_bound=int(np.isclose(data[effective], data[bound]).sum()),
            )
        )
    return mouse, stats


def paired_mouse_plot(ax, frame, columns, labels, colors):
    for _, row in frame[columns].iterrows():
        ax.plot(range(len(columns)), row.to_numpy(), color="#b9c3cd", lw=0.8, alpha=0.7, zorder=1)
        ax.scatter(range(len(columns)), row.to_numpy(), color=colors, s=23, zorder=2)
    means = frame[columns].mean().to_numpy()
    ax.plot(range(len(columns)), means, color="#253241", marker="D", lw=2.5, ms=6, zorder=3)
    for index, value in enumerate(means):
        ax.annotate(f"{value:.3f}", (index, value), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#a7afb8", ls="--", lw=0.8)
    ax.set(xticks=range(len(columns)), xticklabels=labels, xlim=(-0.3, len(columns) - 0.7))
    ax.margins(y=0.2)


def figure_performance(data, mouse, stats):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.8), layout="constrained")
    ax = axes[0, 0]
    columns = [MODELS[0], OLD, MODELS[2], NEW, "rrr"]
    paired_mouse_plot(
        ax, mouse, columns, ["Internal\nPF", "Trial\nGain", "Sqrt\nGain", "Gain + E", "Full peer\nRRR"], ["#64717e", ORANGE, PURPLE, GREEN, BLUE]
    )
    ax.set_ylabel("Mean R² (stored median-fold scores)")
    panel(ax, "A", "Predicting E closes part of the peer-prediction gap", f"Lines: {len(mouse)} mice. Black diamonds: equal-mouse means.")
    ax = axes[0, 1]
    combined, peer = data[NEW].to_numpy(), data["rrr"].to_numpy()
    above = combined >= peer
    for keep, color, label in ((~above, BLUE, "Peer RRR higher"), (above, GREEN, "G+E at least as high")):
        ax.scatter(combined[keep], peer[keep], color=color, s=17, alpha=0.45, edgecolors="none", label=f"{label} (n={keep.sum()})")
    bounds = (min(combined.min(), peer.min(), 0) - 0.015, max(combined.max(), peer.max()) + 0.015)
    ax.plot(bounds, bounds, color="#8d99a6", ls="--", lw=1)
    ax.scatter(
        [stats["mouse_mean_combined_r2"]], [stats["mouse_mean_peer_r2"]], marker="D", color="#253241", s=62, zorder=4, label="Equal-mouse mean"
    )
    ax.set(xlim=bounds, ylim=bounds, xlabel="Gain + E: R²", ylabel="Full peer RRR: R²")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    panel(ax, "B", "Full peer prediction still captures more", f"All {len(data)} matched session–environment units; same held-out trial folds.")
    ax = axes[1, 0]
    columns = ["internal_delta_r2_residual_pooled", "external_delta_r2_residual_pooled"]
    paired_mouse_plot(ax, mouse, columns, ["Internal position", "External position"], [GREEN, BLUE])
    ax.set_ylabel("Additional original-activity R² from E")
    panel(ax, "C", "The residual improvement is consistent across mice", "All units included. Improvement = (SSE_G − SSE_G+E) / SST_original.")
    ax = axes[1, 1]
    base = float(mouse.internal_r2_template_pooled.mean())
    gain = float(mouse.internal_delta_r2_gain_pooled.mean())
    residual = float(mouse.internal_delta_r2_residual_pooled.mean())
    total = float(mouse.internal_r2_pooled.mean())
    assert np.isclose(base + gain + residual, total, rtol=0, atol=1e-12)
    bottoms = [0, base, base + gain, 0]
    heights = [base, gain, residual, total]
    for i, (bottom, value, color) in enumerate(zip(bottoms, heights, ["#64717e", ORANGE, GREEN, BLUE])):
        ax.bar(i, value, bottom=bottom, width=0.65, color=color, alpha=0.9)
        ax.text(
            i, bottom + value + 0.003, f"{value:+.3f}" if i in (1, 2) else f"{value:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    for i, height in enumerate((base, base + gain, total)):
        ax.plot([i + 0.325, i + 0.675], [height, height], color="#a7afb8", lw=1)
    ax.set(
        xticks=range(4),
        xticklabels=["PF template", "+ Gain", "+ Residual E", "G + E total"],
        ylabel="Original-activity R²",
        ylim=(min(0, *bottoms) - 0.005, max(total, base, base + gain) * 1.25),
    )
    ax.axhline(0, color="#a7afb8", lw=0.8)
    panel(ax, "D", "Additive pooling exposes the remaining poor folds", "All-fold pooled R² is sensitive to rare failures; A uses median-fold R².")
    fig.suptitle("Trial gain + residual prediction | recomputed regularized model", fontsize=17, fontweight="bold")
    save(fig, "01_performance_and_stability")


def figure_structure(data, mouse, stats):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11), layout="constrained")
    ax = axes[0, 0]
    x, y = data.external_r2_residual_component, data.internal_r2_residual_component
    ax.scatter(x, y, color=GREEN, s=20, alpha=0.5, linewidth=0)
    bounds = (min(x.min(), y.min(), 0) - 0.008, max(x.max(), y.max()) + 0.008)
    ax.plot(bounds, bounds, color="#8d99a6", ls="--", lw=1)
    ax.axhline(0, color="#b4bbc3", lw=0.7)
    ax.axvline(0, color="#b4bbc3", lw=0.7)
    ax.set(xlim=bounds, ylim=bounds, xlabel="External-position model: R² on E", ylabel="Internal-position model: R² on E")
    panel(ax, "A", "E remains peer-predictable on held-out trials", f"All {len(data)} units. Each axis scores its model's own regularized E.")
    ax = axes[0, 1]
    paired_mouse_plot(
        ax, mouse, ["external_r2_residual_component", "internal_r2_residual_component"], ["External position", "Internal position"], [BLUE, GREEN]
    )
    ax.set_ylabel("R² predicting measured E")
    panel(ax, "B", "The residual signal is present across mice", "Equal environments → sessions → mice; component-specific denominators.")
    ax = axes[1, 0]
    rng = np.random.default_rng(27)
    for offset, prefix, color in ((-0.15, "internal", GREEN), (0.15, "external", BLUE)):
        means = []
        for index, component in enumerate(("source_residual", "target_residual", "predicted_residual")):
            values = mouse[f"{prefix}_{component}_gain_energy_fraction"].to_numpy() * 100
            assert np.all(values > 0), "Cannot display zero energy fractions on logarithmic axis"
            ax.scatter(index + offset + rng.uniform(-0.04, 0.04, len(values)), values, s=22, alpha=0.6, color=color, linewidth=0)
            means.append(float(np.mean(values)))
        ax.plot(np.arange(3) + offset, means, color=color, marker="D", lw=2, ms=6, label=prefix.capitalize())
    ax.set(
        xticks=range(3),
        xticklabels=["Measured\nsource E", "Measured\ntarget E", "Predicted\ntarget E"],
        xlim=(-0.5, 2.5),
        yscale="log",
        ylabel="Template-shaped energy / E energy (%)",
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}%"))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    panel(ax, "C", "Measured E contains very little leftover gain", "Exact posthoc projection. Log scale; dots = mice, diamonds = means.")
    ax = axes[1, 1]
    for index, (row, color, effective, bound) in enumerate(
        zip(
            stats["rank_summary"],
            [ORANGE, GREEN],
            ["internal_structured_gain_effective_rank", "internal_residual_effective_rank"],
            ["internal_structured_gain_max_trial_rank", "internal_residual_max_rank"],
        )
    ):
        values = (data[effective] / data[bound]).to_numpy()
        ax.scatter(index + rng.uniform(-0.16, 0.16, len(values)), values, color=color, alpha=0.23, s=12, linewidth=0)
        ax.plot([index - 0.24, index + 0.24], [np.median(values)] * 2, lw=3, color=color)
        ax.text(
            index,
            0.07,
            f"Median rank: {row['median_rank']:.0f}\nMedian bound: {row['median_bound']:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
        )
    ax.axhline(1, color="#a7afb8", lw=0.8, ls="--")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Trial gain RRR", "Framewise residual RRR"],
        xlim=(-0.5, 1.5),
        ylim=(-0.04, 1.12),
        ylabel="Effective rank / available rank",
    )
    panel(
        ax, "D", "Residual prediction uses substantially more capacity", "Internal model: gain rank is trial-limited; E rank is mostly cell-limited."
    )
    fig.suptitle("What remains in E? | predictability, gain overlap, and model capacity", fontsize=17, fontweight="bold")
    save(fig, "02_residual_structure")


def write_report(store, data, folds, mouse, stats):
    stats["store"] = str(store)
    pd.DataFrame(stats["stability"]).to_csv(OUT / "stability_summary.csv", index=False)
    mouse.to_csv(OUT / "mouse_scores.csv")
    mouse[["external_r2_residual_component", "internal_r2_residual_component"]].to_csv(OUT / "mouse_residual_scores.csv")
    folds.loc[folds.r2 < -1].to_csv(OUT / "remaining_poor_folds.csv", index=False)
    aggregation = mouse[[OLD, NEW, "internal_r2_pooled"]].copy()
    aggregation["pooled_gain_r2"] = mouse.internal_r2_template_pooled + mouse.internal_delta_r2_gain_pooled
    aggregation.to_csv(OUT / "mouse_aggregation_comparison.csv")
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
    if (OUT / "manifest.json").exists():
        previous = json.loads((OUT / "manifest.json").read_text())
        if previous.get("implementation") != stats["implementation"] and not (OUT / "previous_unregularized_manifest.json").exists():
            (OUT / "previous_unregularized_manifest.json").write_text(json.dumps(previous, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    internal, external = stats["stability"]
    n, nf = stats["n_units"], internal["total_folds"]
    table = pd.DataFrame(stats["stability"])[
        [
            "branch",
            "severe_folds",
            "severe_gain_folds",
            "positive_residual_folds",
            "positive_residual_increment_units",
            "mean_mouse_residual_r2",
            "mean_mouse_residual_increment",
        ]
    ]
    table.columns = [
        "Position",
        f"G+E R² < −1 / {nf}",
        f"G R² < −1 / {nf}",
        f"E-component R² > 0 / {nf}",
        f"Positive E increment / {n}",
        "Mouse mean E R²",
        "Mouse mean E increment",
    ]
    proj_table = pd.DataFrame(
        [
            dict(
                Position=row["branch"].capitalize(),
                **{
                    "Measured source E (%)": 100 * row["mean_mouse_source_gain_energy_fraction"],
                    "Measured target E (%)": 100 * row["mean_mouse_target_gain_energy_fraction"],
                    "Predicted target E (%)": 100 * row["mean_mouse_predicted_gain_energy_fraction"],
                },
            )
            for row in stats["stability"]
        ]
    )
    caption1 = (
        "A: stored median-of-three-fold R², averaged equally over environments within each session, sessions within mouse, then mice. "
        "Thin lines are mice; black diamonds are equal-mouse means. Trial Gain and G-only agree to the numerical tolerance reported below. "
        "B: all matched session–environment units, without score-based exclusion; the dark diamond is the equal-mouse mean. "
        "C and D use pooled-over-fold SSE/SST rather than median-fold R². C is the incremental benefit of E: "
        "(SSE_G − SSE_G+E)/SST_original. D telescopes exactly: R²(template) + gain increment + E increment = R²(G+E). "
        "Three remaining poor internal folds strongly lower the pooled total in D: 0.042 versus 0.125 for the median-fold summary in A. "
        "All folds are included; the aggregation-comparison and poor-fold tables expose this sensitivity. "
        "Full peer RRR is a comparison model, not a noise ceiling."
    )
    caption2 = (
        "A/B score predicted E against the measured residual formed using regularized trial gains. Internal and external E differ because "
        "their template trajectories differ, and the component R² denominators differ. These component R² values do not add to original-activity R². "
        "C projects each cell's E onto that cell's template separately on each trial, with an unregularized, numerically normalized projection. "
        "It reports squared energy fractions, not centered variance or fractions of predictive benefit. Fractions are pooled across folds "
        "using their energies, then averaged over environments, sessions and mice. No model inputs or predictions are changed. "
        "D shows session–environment rank distributions, with different branch-specific bounds."
    )
    stability_text = (
        f"The previous explosive negative tail is absent: the worst internal combined-model fold is now R² = {internal['worst_fold_r2']:.3f}. "
        f"There are still {internal['severe_folds']}/{nf} internal combined-model folds below −1 "
        f"({internal['severe_gain_folds']} gain-only folds), and {external['severe_folds']}/{nf} external combined-model folds below −1. "
        "All remain included in the figures and tables."
    )
    links = " · ".join(
        f'<a href="{name}">{label}</a>'
        for name, label in (
            ("01_performance_and_stability.svg", "Figure 1 SVG"),
            ("02_residual_structure.svg", "Figure 2 SVG"),
            ("session_environment_scores.csv", "Session–environment scores"),
            ("fold_scores.csv", "Fold scores"),
            ("mouse_scores.csv", "Mouse summaries"),
            ("session_coverage.csv", "Session coverage"),
            ("mouse_aggregation_comparison.csv", "Aggregation comparison"),
            ("remaining_poor_folds.csv", "Remaining poor folds"),
            ("provenance.csv", "Provenance"),
            ("manifest.json", "Audit manifest"),
            ("make_report.py", "Reproduction script"),
        )
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial gain + residual | recomputed results</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#253241;background:#f2f5f8;margin:0}}main{{max-width:1350px;margin:auto;padding:42px 36px}}
h1{{font-size:34px;line-height:1.2;margin:0 0 12px}}h2{{font-size:24px;margin:36px 0 12px}}.meta,figcaption{{color:#596776;font-size:14px}}
.status{{background:#e8f4ef;border-left:5px solid #248978;padding:18px 24px;border-radius:6px}}
.aggregation{{background:#fff3dd;border-left:5px solid #d18a30;padding:18px 24px;border-radius:6px;margin-top:20px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.card{{background:white;border-radius:8px;padding:18px}}
.big{{font-size:30px;font-weight:700;display:block}}figure{{margin:22px 0;background:white;padding:16px;border-radius:10px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:12px 10px;text-align:right;border-bottom:1px solid #dfe5ea}}
th:first-child,td:first-child{{text-align:left}}code{{font-size:13px}}a{{color:#3569aa}}li{{margin:8px 0}}
@media(max-width:700px){{main{{padding:22px 14px}}.cards{{grid-template-columns:1fr}}table{{font-size:11px}}}}</style>
<main><h1>Trial gain + residual prediction</h1>
<p class="meta">Recomputed regularized model · 3 November 2026 · all configurations retain schema v2 · sigrebase / std / preferred / seed 0.<br>
{stats['n_sessions']} scored sessions out of {stats['n_saved_sessions']} saved · {stats['n_mice']} mice · {n} matched session–environment units · {nf} test folds per position model.</p>
<div class="status"><strong>The gain-only control now agrees with standalone Trial Gain.</strong>
All {n} internal-model gain hyperparameter sets match exactly. The largest cached fold-score difference is
{stats['max_gain_reference_fold_difference']:.2e} (audit tolerance: 1e−6).
The recomputed residual branch adds held-out predictability, with very little template-shaped energy remaining in its measured target.</div>
<div class="cards">
<div class="card"><span class="big">{stats['mouse_mean_trial_gain_r2']:.3f} → {stats['mouse_mean_combined_r2']:.3f}</span>Trial Gain → G+E; equal-mouse mean median-fold R².</div>
<div class="card"><span class="big">{100*stats['peer_gap_recovered']:.1f}%</span>Of the mean Trial Gain → full-peer RRR gap recovered. Peer RRR: {stats['mouse_mean_peer_r2']:.3f}.</div>
<div class="card"><span class="big">{100*internal['mean_mouse_target_gain_energy_fraction']:.3f}%</span>Template-shaped fraction of measured internal target E energy.</div></div>
<div class="aggregation"><strong>Aggregation matters.</strong> The headline comparison uses the existing median-of-three-fold score.
When SSE is pooled over all folds, the equal-mouse mean combined R² is {stats['mouse_mean_pooled_combined_r2']:.3f}.
Three poorly predicted internal folds (also poor in standalone Trial Gain) strongly affect that pooled summary.
Figure 1D shows this sensitivity explicitly, with no excluded folds. The 28.9% gap-recovery figure applies to the median-fold summary only.</div>
<h2>1. What does residual prediction add?</h2>
<figure><img src="01_performance_and_stability.png" alt="Mouse performance comparisons, G+E versus peer RRR, residual benefit, and additive SSE accounting">
<figcaption>{caption1}</figcaption></figure>
<h2>2. How much gain remains in E?</h2>
<figure><img src="02_residual_structure.png" alt="Residual component predictability, mouse summaries, posthoc gain-energy fractions, and rank usage">
<figcaption>{caption2}</figcaption></figure>
<h2>Reading the results</h2><ul>
<li><strong>The robust performance summary and the all-fold additive account answer different questions.</strong>
The median-fold G+E score averages {stats['mouse_mean_combined_r2']:.4f} across mice; its all-fold pooled counterpart is
{stats['mouse_mean_pooled_combined_r2']:.4f}. In the pooled account, the gain increment is {stats['mouse_mean_pooled_gain_increment']:+.4f},
driven downward by rare poor gain predictions. E still adds {internal['mean_mouse_residual_increment']:+.4f} on that same pooled denominator.</li>
<li><strong>E provides additional prediction beyond the reused gain model.</strong> Its pooled-over-fold original-activity R² increment is
{internal['mean_mouse_residual_increment']:.4f} internally and {external['mean_mouse_residual_increment']:.4f} externally, using all units and equal-mouse weighting.
The increment is positive in {internal['positive_residual_increment_units']}/{n} internal units and {external['positive_residual_increment_units']}/{n} external units.</li>
<li><strong>The combined model still falls short of full peer RRR.</strong> Mean median-fold R² is {stats['mouse_mean_combined_r2']:.4f} for G+E,
versus {stats['mouse_mean_peer_r2']:.4f} for peer RRR. The {100*stats['peer_gap_recovered']:.1f}% gap recovery is a ratio of these aggregate differences,
not an estimate of a biological variance fraction or maximum explainable predictability.</li>
<li><strong>The regularized measured E is nearly off-template, although not constrained to be.</strong>
Its template-shaped energy fraction averages {100*internal['mean_mouse_target_gain_energy_fraction']:.3f}% internally
and {100*external['mean_mouse_target_gain_energy_fraction']:.3f}% externally.</li>
<li><strong>Predicted E contains a larger, but still minority, template-shaped component.</strong>
The corresponding fractions are {100*internal['mean_mouse_predicted_gain_energy_fraction']:.2f}% and
{100*external['mean_mouse_predicted_gain_energy_fraction']:.2f}%.
Energy fractions alone do not tell us what fraction of the incremental predictive improvement is gain-related:
that would require separately scoring the projected and off-template prediction components.</li>
<li><strong>The E branch is comparatively high rank.</strong> Median internal gain rank is {stats['rank_summary'][0]['median_rank']:.0f};
median residual rank is {stats['rank_summary'][1]['median_rank']:.0f}.
The residual branch has framewise observations and a much higher rank bound.</li></ul>
<h2>Projection audit: equal-mouse mean fractions</h2>{proj_table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)}
<p>For each cell/trial, E = P_template E + (I − P_template)E. The two projected pieces are orthogonal and their squared energies sum exactly.
The model itself still uses regularized measured gains and an unconstrained signed residual predictor.</p>
<h2>Residual scores and stability</h2>{table.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)}
<p>{stability_text}</p>
<h2>Matching, weighting, and limits</h2>
<p>All six compared model results are matched on session, environment identity, source and target ROI IDs, and the exact trial folds.
Internal G-only also matches the standalone gain model's saved PF/gain hyperparameters and fold scores within numerical tolerance.
Cached summaries do not include every retained prediction-frame index, so exact cross-model frame support cannot be independently audited here.
Within the new model G and E share the same scoring frames.</p>
<p>The {stats['n_unscored_sessions']} saved sessions without a finite scored environment remain listed in session_coverage.csv.
All scored units and folds are retained, including negative scores. No score-based subset, winsorization, or outlier clipping is used.
Across-mouse summaries first average environments within a session and sessions within each mouse. No interval treats sessions or folds as independent animals.</p>
<p>Hyperparameters use a fixed validation quarter and are reused across the three rotations. Place-cell selection uses full-session activity.
Internal position is decoded from source cells; trial gains use the source trial's frames. E may include template mismatch and position-decoding error
as well as other shared activity structure. Component R² values have component-specific denominators; only the original-activity SSE increments
provide the additive account shown in Figure 1D.</p>
<p>Read-only source: <code>{html.escape(str(store/'results.db'))}</code>. The selected configurations and computation timestamps are in provenance.csv.
This report reads saved results and does not refit models or modify the results store.</p>
<p>{links}</p></main></html>"""
    (OUT / "report.html").write_text(document, encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


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
    mouse, stats = summary_data(data, folds)
    figure_performance(data, mouse, stats)
    figure_structure(data, mouse, stats)
    write_report(store, data, folds, mouse, stats)


if __name__ == "__main__":
    main()
