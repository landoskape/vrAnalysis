"""Compare existing images and ROI morphology without modifying session data."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors.spkmaps import SpkmapProcessor, SpkmapParams

output = Path(__file__).parent / "atl012_reliability_inspection"
dates = ["2023-02-28", "2023-03-01", "2023-03-03"]
fig, axes = plt.subplots(5, 3, figsize=(9, 14), layout="constrained")
rows, occupancy_rows = [], []
reference = {}
for col, date in enumerate(dates):
    session = B2Session.create("ATL012", date, "701", params={"spks_type": "sigrebase"})
    selected = session.idx_rois
    plane_ids = session.get_plane_idx()
    offsets = np.r_[0, np.cumsum(session.values.roiPerPlane)]
    print(date, "classifier keys", list(session.roicat_classifier), flush=True)
    for plane in range(5):
        ops = np.load(session.s2p_path / f"plane{plane}" / "ops.npy", allow_pickle=True).item()
        image = np.asarray(ops["meanImg"], dtype=float)
        mask = selected[offsets[plane]:offsets[plane + 1]]
        stats = np.load(session.s2p_path / f"plane{plane}" / "stat.npy", allow_pickle=True)
        entry = dict(date=date, plane=plane, raw_rois=len(stats), selected_rois=int(mask.sum()),
                     mean_intensity=float(np.mean(image)), contrast=float(np.std(image)),
                     median_selected_npix=float(np.median([s["npix"] for s in stats[mask]])))
        if plane not in reference:
            reference[plane] = image
        ref = reference[plane]
        if ref.shape == image.shape:
            a, b = image - image.mean(), ref - ref.mean()
            corr = fftconvolve(a, b[::-1, ::-1], mode="same")
            cy, cx = np.array(corr.shape) // 2
            window = corr[cy-30:cy+31, cx-30:cx+31]
            dy, dx = np.array(np.unravel_index(np.argmax(window), window.shape)) - 30
            ay = slice(max(0, dy), min(image.shape[0], image.shape[0] + dy))
            ax = slice(max(0, dx), min(image.shape[1], image.shape[1] + dx))
            by = slice(max(0, -dy), min(image.shape[0], image.shape[0] - dy))
            bx = slice(max(0, -dx), min(image.shape[1], image.shape[1] - dx))
            entry.update(shift_y=int(dy), shift_x=int(dx), aligned_image_correlation=float(np.corrcoef(image[ay, ax].ravel(), ref[by, bx].ravel())[0,1]))
        labels = session.roicat_classifier["class_predictions"][offsets[plane]:offsets[plane+1]]
        entry["class_counts"] = json.dumps({str(k): int(v) for k,v in zip(*np.unique(labels, return_counts=True))})
        rows.append(entry)
        axes[plane,col].imshow(image, cmap="gray", vmin=np.percentile(image, 2), vmax=np.percentile(image, 99.5))
        axes[plane,col].set_title(f"{date} | plane {plane}\n{int(mask.sum())} selected ROIs", fontsize=9)
        axes[plane,col].axis("off")
        print(entry, flush=True)
    # Occupancy has no dependence on spike type. Read only the matching cached
    # occupancy arrays, without loading the large neural maps or writing caches.
    smp = SpkmapProcessor(session, SpkmapParams())
    cache = session.data_path / "spkmaps" / "env_maps_significant"
    h = smp._params_hash("env_maps")
    for env in (1,3,4):
        path = cache / f"data_occmap_{env}_{h}.npy"
        if path.exists():
            occ = np.load(path)
            occupancy_rows.append(dict(date=date, environment=env, retained_trials=occ.shape[0],
                                       position_bins=occ.shape[1], nan_fraction=float(np.isnan(occ).mean()),
                                       median_trial_occupancy_seconds=float(np.nanmedian(np.nansum(occ,axis=1)))))
    session.clear_cache()
pd.DataFrame(rows).to_csv(output / "image_roi_comparison.csv", index=False)
pd.DataFrame(occupancy_rows).to_csv(output / "retained_trials.csv", index=False)
fig.savefig(output / "mean_images.png", dpi=120)
print("OCCUPANCY", occupancy_rows, flush=True)
summary = pd.read_csv(output / "plane_reliability.csv")
for (date, env), group in summary.groupby(["date", "environment"]):
    group = group[group.plane != 1]
    print("EXCLUDING PLANE 1", date, env, float(np.average(group.reliability_quality_mean, weights=group.quality_cells)), flush=True)
