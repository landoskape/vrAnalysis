"""Audit global ROI recurrence and session-specific dropout using Tracker labels.

Reads existing tracking and analysis data. Writes diagnostic artifacts only.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrAnalysis.tracking import Tracker
from inspect_atl012_reliability import ReadOnlyStore
from dimensionality_manuscript import PFPredQualityConfig, ResultsAggregator


def recurrence(labels, presence):
    """Number of other sessions per ROI, with unmatched -1 contributing zero."""
    counts = np.zeros(len(labels), dtype=int)
    valid = labels >= 0
    counts[valid] = presence[labels[valid]].sum(axis=1) - 1
    return counts


def run():
    output = Path(__file__).parent / "atl012_tracking_inspection"
    output.mkdir(exist_ok=True)
    tracker = Tracker("ATL012")
    order = sorted(range(tracker.num_sessions), key=lambda i: (str(tracker.sessions[i].date), str(tracker.sessions[i].session_id)))
    sessions = [tracker.sessions[i] for i in order]
    labels = [np.asarray(tracker.labels[i]) for i in order]
    n = len(sessions)
    for s, lab in zip(sessions, labels):
        assert len(lab) == s.values.numROIs
        ids, counts = np.unique(lab[lab >= 0], return_counts=True)
        assert np.all(counts == 1), "Ambiguous duplicate cluster IDs in a session"
    n_clusters = max(int(lab.max()) for lab in labels) + 1
    presence = np.zeros((n_clusters, n), dtype=bool)
    for i, lab in enumerate(labels):
        presence[lab[lab >= 0], i] = True
    masks, classes, planes = [], [], []
    for i, s in enumerate(sessions):
        s.params.spks_type = PFPredQualityConfig().spks_type
        mask = s.idx_rois.copy()
        masks.append(mask)
        classes.append(np.asarray(s.roicat_classifier["class_predictions"]))
        planes.append(s.get_plane_idx())
        print("FILTERED", i + 1, str(s.date), int(mask.sum()), flush=True)
        s.clear_cache()

    # Raw target presence deliberately ignores its cell-class/activity filters.
    # This separates a missing tracking match from a match rejected by analysis.
    selected_presence = np.zeros_like(presence)
    for i, (lab, mask) in enumerate(zip(labels, masks)):
        ids = lab[mask & (lab >= 0)]
        selected_presence[ids, i] = True
    rows, hist_rows, pair_rows, bridge_rows = [], [], [], []
    for i, s in enumerate(sessions):
        lab = labels[i]
        other = recurrence(lab, presence)
        selected_other = np.zeros(len(lab), dtype=int)
        valid = lab >= 0
        selected_other[valid] = selected_presence[lab[valid]].sum(axis=1) - masks[i][valid]
        group_masks = {"all_raw": np.ones(len(lab), dtype=bool), "analysis_selected": masks[i],
                       "somatic_selected": masks[i] & (classes[i] == 0),
                       "dendritic_selected": masks[i] & (classes[i] == 3)}
        group_masks.update({f"plane_{p}_selected": masks[i] & (planes[i] == p) for p in np.unique(planes[i])})
        for name, mask in group_masks.items():
            values = other[mask]
            row = dict(date=str(s.date), session_id=str(s.session_id), tracking_session=i + 1,
                       subset=name, n_rois=int(mask.sum()), assigned_fraction=float(np.mean(lab[mask] >= 0)),
                       found_any_other_fraction=float(np.mean(values >= 1)),
                       found_ge4_others_fraction=float(np.mean(values >= 4)),
                       found_ge9_others_fraction=float(np.mean(values >= 9)),
                       mean_other_sessions=float(np.mean(values)), median_other_sessions=float(np.median(values)),
                       mean_pairwise_recovery_fraction=float(np.mean(values) / (n - 1)),
                       found_any_other_selected_fraction=float(np.mean(selected_other[mask] >= 1)))
            rows.append(row)
            for count in range(n):
                hist_rows.append(dict(date=str(s.date), subset=name, other_sessions=count, n_rois=int(np.sum(values == count))))
            for j, target in enumerate(sessions):
                if i == j:
                    continue
                found = np.zeros(len(lab), dtype=bool)
                found_selected = np.zeros(len(lab), dtype=bool)
                found[valid] = presence[lab[valid], j]
                found_selected[valid] = selected_presence[lab[valid], j]
                pair_rows.append(dict(source_date=str(s.date), target_date=str(target.date), subset=name,
                                      source_n=int(mask.sum()), shared_raw_target=int(np.sum(mask & found)),
                                      source_fraction_found=float(np.mean(found[mask])),
                                      shared_selected_target=int(np.sum(mask & found_selected)),
                                      source_fraction_found_selected=float(np.mean(found_selected[mask]))))

        if 0 < i < n - 1:
            # Select the cohort from flanking days without using the focal day.
            left_ids = set(labels[i - 1][labels[i - 1] >= 0])
            right_ids = set(labels[i + 1][labels[i + 1] >= 0])
            raw_shared = sorted(left_ids & right_ids)
            source_sets = {"all_raw": raw_shared}
            left_sel = set(labels[i - 1][masks[i - 1] & (labels[i - 1] >= 0)])
            right_sel = set(labels[i + 1][masks[i + 1] & (labels[i + 1] >= 0)])
            source_sets["analysis_selected_both_flanks"] = sorted(left_sel & right_sel)
            for kind, class_id in (("somatic", 0), ("dendritic", 3)):
                left = set(labels[i - 1][masks[i - 1] & (classes[i - 1] == class_id) & (labels[i - 1] >= 0)])
                right = set(labels[i + 1][masks[i + 1] & (classes[i + 1] == class_id) & (labels[i + 1] >= 0)])
                source_sets[f"{kind}_selected_both_flanks"] = sorted(left & right)
            for plane in np.unique(planes[i - 1]):
                left = set(labels[i - 1][masks[i - 1] & (planes[i - 1] == plane) & (labels[i - 1] >= 0)])
                right = set(labels[i + 1][masks[i + 1] & (planes[i + 1] == plane) & (labels[i + 1] >= 0)])
                source_sets[f"plane_{plane}_selected_both_flanks"] = sorted(left & right)
            for name, source in source_sets.items():
                source = np.asarray(source, dtype=int)
                if not len(source):
                    continue
                bridge_rows.append(dict(date=str(s.date), previous_date=str(sessions[i - 1].date),
                                        next_date=str(sessions[i + 1].date), cohort=name, n_reference=len(source),
                                        recovered_raw=int(presence[source, i].sum()),
                                        recovered_fraction=float(presence[source, i].mean()),
                                        recovered_selected=int(selected_presence[source, i].sum()),
                                        recovered_selected_fraction=float(selected_presence[source, i].mean())))

    summary, pairs, bridges = pd.DataFrame(rows), pd.DataFrame(pair_rows), pd.DataFrame(bridge_rows)
    summary.to_csv(output / "session_recurrence.csv", index=False)
    pairs.to_csv(output / "pairwise_recovery.csv", index=False)
    bridges.to_csv(output / "flanking_cohort_recovery.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(output / "recurrence_distribution.csv", index=False)

    # Explicit use of Tracker's shared-index API validates our global-label counting.
    late = [i for i,s in enumerate(sessions) if str(s.date) in ("2023-02-28", "2023-03-01", "2023-03-03")]
    assert len(late) == 3
    for a, b in zip(late[:-1], late[1:]):
        tracked, extras = tracker.get_tracked_idx(idx_ses=[order[a], order[b]], use_session_filters=False)
        expected = set(labels[a][labels[a] >= 0]) & set(labels[b][labels[b] >= 0])
        assert set(extras["cluster_ids"]) == expected
        assert len(tracked[0]) == len(expected)

    # Compare the same ROIs across all three late sessions, with the quality cohort
    # selected from the two flanks only to avoid conditioning on the dip session.
    a, dip, b = late
    idx_tracked, extras = tracker.get_tracked_idx(idx_ses=[order[i] for i in late], use_session_filters=False)
    all_selected = np.all(np.stack([masks[i][idx] for i,idx in zip(late,idx_tracked)]),axis=0)
    idx_tracked = idx_tracked[:,all_selected]
    result_arrays = ResultsAggregator(PFPredQualityConfig, ReadOnlyStore(), [sessions[i] for i in late],
                                     keys=["reliability_slot", "fraction_active_slot"]).sel(squeeze_ones=False)
    local_idx = []
    for i, idx in zip(late, idx_tracked):
        lookup = np.full(len(masks[i]), -1, dtype=int)
        lookup[masks[i]] = np.arange(int(masks[i].sum()))
        local_idx.append(lookup[idx])
    rel = np.stack([result_arrays["reliability_slot"][j][:,idx] for j,idx in enumerate(local_idx)])
    fa = np.stack([result_arrays["fraction_active_slot"][j][:,idx] for j,idx in enumerate(local_idx)])
    fixed_rows = []
    for slot,env in enumerate((1,3,4)):
        flank_quality = (rel[0,slot] > .3) & (fa[0,slot] > .1) & (rel[2,slot] > .3) & (fa[2,slot] > .1)
        both_somatic = (classes[a][idx_tracked[0]] == 0) & (classes[b][idx_tracked[2]] == 0)
        for subset, mask in (("all_three_selected", np.ones(rel.shape[-1],dtype=bool)),
                             ("somatic_all_three_selected", both_somatic),
                             ("quality_in_both_flanks",flank_quality),
                             ("somatic_quality_in_both_flanks",flank_quality & both_somatic)):
            for j,i in enumerate(late):
                values=rel[j,slot,mask]
                fixed_rows.append(dict(date=str(sessions[i].date),environment=env,subset=subset,n_rois=int(mask.sum()),
                                       finite_rois=int(np.isfinite(values).sum()),mean_reliability=float(np.nanmean(values)),
                                       median_reliability=float(np.nanmedian(values))))
    pd.DataFrame(fixed_rows).to_csv(output / "fixed_cohort_reliability.csv",index=False)

    # Map recurrence on the affected day's FOV. An unmatched cluster is not proof
    # that a cell is absent: segmentation and tracking can also miss the match.
    fig_spatial, spatial_axes = plt.subplots(1,5,figsize=(15,4),layout="constrained")
    dip_session = sessions[dip]
    dip_other = recurrence(labels[dip], presence)
    start = 0
    for plane, ax in enumerate(spatial_axes):
        count = dip_session.values.roiPerPlane[plane]
        stats = np.load(dip_session.s2p_path / f"plane{plane}" / "stat.npy",allow_pickle=True)
        image = np.load(dip_session.s2p_path / f"plane{plane}" / "ops.npy",allow_pickle=True).item()["meanImg"]
        ax.imshow(image,cmap="gray",vmin=np.percentile(image,2),vmax=np.percentile(image,99.5))
        mask = masks[dip][start:start+count]
        matches = dip_other[start:start+count] >= 1
        positions = np.array([stat["med"] for stat in stats])
        for found,color,label in ((True,"#00c9b0","Found elsewhere"),(False,"#ff5685","No match elsewhere")):
            keep = mask & (matches == found)
            ax.scatter(positions[keep,1],positions[keep,0],s=4,c=color,alpha=.55,label=label)
        ax.set_title(f"Plane {plane}\n{100*np.mean(matches[mask]):.1f}% found elsewhere",fontsize=10)
        ax.axis("off")
        start += count
    handles, legend_labels = spatial_axes[-1].get_legend_handles_labels()
    fig_spatial.legend(handles, legend_labels, loc="outside lower center",ncol=2,fontsize=9,frameon=False)
    fig_spatial.savefig(output / "march1_roi_recurrence_map.png",dpi=150)
    plt.close(fig_spatial)

    dates = [str(s.date) for s in sessions]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for subset,label in (("all_raw","All raw ROIs"),("analysis_selected","Analysis selected"),
                          ("somatic_selected","Somatic selected"),("dendritic_selected","Dendritic selected")):
        table=summary[summary.subset==subset].set_index("date").loc[dates]
        axes[0,0].plot(range(n),100*table.found_any_other_fraction,marker=".",label=label)
        axes[0,1].plot(range(n),table.mean_other_sessions,marker=".",label=label)
    axes[0,0].set_ylabel("ROIs found in ≥1 other session (%)")
    axes[0,1].set_ylabel("Mean number of other sessions per ROI")
    for ax in axes[0]:
        ax.axvline(dip,color="red",ls=":",label="Mar 1 dip")
        ax.set_xticks(range(n), [d[5:] for d in dates], rotation=90, fontsize=7)
        ax.legend(fontsize=8,frameon=False)
    matrix=pairs[pairs.subset=="analysis_selected"].pivot(index="source_date",columns="target_date",values="source_fraction_found").reindex(index=dates,columns=dates)
    im=axes[1,0].imshow(100*matrix,vmin=0,vmax=100,cmap="viridis")
    axes[1,0].set_xticks(range(n),[d[5:] for d in dates],rotation=90,fontsize=7)
    axes[1,0].set_yticks(range(n),[d[5:] for d in dates],fontsize=7)
    axes[1,0].set_xlabel("Target session (any raw match)")
    axes[1,0].set_ylabel("Source session (selected ROIs)")
    fig.colorbar(im,ax=axes[1,0],label="Source ROIs recovered (%)",shrink=.8)
    table=bridges[bridges.cohort=="analysis_selected_both_flanks"]
    axes[1,1].plot(range(1,n-1),100*table.recovered_fraction,marker=".",label="Any tracking match")
    axes[1,1].plot(range(1,n-1),100*table.recovered_selected_fraction,marker=".",label="Match also passes analysis filters")
    axes[1,1].axvline(dip,color="red",ls=":")
    axes[1,1].set_xticks(range(1,n-1),[d[5:] for d in dates[1:-1]],rotation=90,fontsize=7)
    axes[1,1].set_ylabel("Cohort shared by flanking sessions recovered (%)")
    axes[1,1].legend(fontsize=8,frameon=False)
    fig.savefig(output / "tracking_summary.png",dpi=150)
    plt.close(fig)
    print("LATE RECURRENCE\n",summary[(summary.date.isin([dates[i] for i in late])) & summary.subset.isin(["all_raw","analysis_selected","somatic_selected","dendritic_selected"])].to_string(index=False),flush=True)
    print("LATE PAIRS\n",pairs[pairs.source_date.isin([dates[i] for i in late]) & pairs.target_date.isin([dates[i] for i in late]) & pairs.subset.eq("analysis_selected")].to_string(index=False),flush=True)
    print("BRIDGES\n",bridges[bridges.cohort.eq("analysis_selected_both_flanks")].to_string(index=False),flush=True)
    print("FIXED COHORT\n",pd.DataFrame(fixed_rows).to_string(index=False),flush=True)
    print("SAVED",output,flush=True)


if __name__ == "__main__":
    run()
