# ATL012 reliability dip: 2023-03-01 / 701

The plotted dip is chronological session 22 of 23. Cached per-ROI results reproduce
the viewer using reliability > 0.3 AND fraction active > 0.1, with an across-ROI
mean, and the default pfpred_quality spike type (sigrebase).

## Comparison with adjacent sessions

| Measurement | Feb 28 | Mar 1 (dip) | Mar 3 |
|---|---:|---:|---:|
| Imaging frames | 12,361 | 14,149 | 10,851 |
| Imaging duration, minutes | 34.33 | 39.30 | 30.14 |
| Total trials | 214 | 268 | 205 |
| Trials in environments 1 / 3 / 4 | 50 / 42 / 122 | 58 / 53 / 157 | 48 / 40 / 117 |
| Selected ROIs | 2,431 | 3,720 | 3,069 |
| Selected somatic ROIs, classifier c | 1,784 | 1,974 | 1,906 |
| Selected dendritic ROIs, classifier d | 647 | 1,746 | 1,163 |
| Quality ROIs in environments 1 / 3 / 4 | 454 / 472 / 346 | 789 / 767 / 415 | 557 / 598 / 402 |
| Quality mean reliability, environment 1 | .672 | .620 | .665 |
| Quality mean reliability, environment 3 | .680 | .620 | .693 |
| Quality mean reliability, environment 4 | .638 | .517 | .596 |

Quality ROIs are the viewer's operational selection, not an independently validated
place-cell count. ROI identities are not matched across days in these comparisons.

## Main anomaly: cell population composition

Plane 1 accounts for 929 of the 1,289 extra selected ROIs relative to Feb 28.
Its selected count is 343 -> 1,272 -> 552. Raw Suite2p ROI counts in that plane
are 1,131 -> 3,932 -> 2,176. The median selected mask size there remains about
12 pixels, consistent with the many dendritic detections.

The pooled mean weights every selected quality ROI equally. The larger contribution
from lower-reliability dendritic ROIs therefore lowers the average. This is not the
whole explanation: quality-selected somatic means also drop:

| Somatic ROI mean reliability | Feb 28 | Mar 1 | Mar 3 |
|---|---:|---:|---:|
| Environment 1 | .701 | .680 | .726 |
| Environment 3 | .727 | .693 | .762 |
| Environment 4 | .682 | .586 | .656 |

Excluding plane 1 also leaves a dip: environment 4 becomes .655 -> .558 -> .612.
All five planes have lower quality-selected means on Mar 1 than Feb 28.

## Acquisition and behavior checks

- No shortage of trials, frames, or selected quality ROIs.
- Frame intervals remain regular: median approximately .167 s and maximum .167 s
  in all three sessions. Plane frame counts differ by at most one on Mar 1.
- Behavior-aligned frame fractions are 84.0%, 82.3%, and 81.0%; Mar 1 is not an
  isolated alignment failure by this measure.
- Median running speeds by environment rise from approximately 23-25 cm/s on
  Feb 28 to 28-31 cm/s on Mar 1; Mar 3 is similarly fast.
- Existing occupancy caches with matching default mapping parameters retain
  44 / 34 / 111 trials on Feb 28, 52 / 51 / 154 on Mar 1, and 25 / 31 / 106
  on Mar 3. These were read from significant-spike caches: occupancy is independent
  of spike type. Position-bin missingness stays around 1.6-1.9%.
- Suite2p rigid x-offset 95th absolute deviation increases from 2-3 to 4-5 pixels;
  y-offset deviations remain 2-3 pixels. Bad-frame fraction is zero in four planes
  and approximately .007% in plane 0 on Mar 1. Registration correlation medians
  do not show a large isolated collapse.
- Mean images show the same broad field of view without a gross registration
  failure. Mar 1 images are brighter than Feb 28; Mar 3 is similarly bright.

## Interpretation and limits

The evidence supports a substantial change in the sampled ROI population, especially
dendritic detections in plane 1, plus a within-class reliability reduction that is
strongest in environment 4. It does not establish whether the remaining reduction
reflects neural variability, residual motion, or segmentation/extraction changes.
No definitive processing failure was identified. Mean images and motion summaries
do not substitute for inspecting the raw movie or matched-cell activity.

A soma-only comparison would reduce the composition effect, but it would not remove
the third-environment dip. No session data, cached results, or viewer settings were
changed during this inspection. Only diagnostic artifacts were written.

## Artifacts

- session_environment_summary.csv: late five sessions, counts, alignment, speed, reliability
- plane_reliability.csv: reliability and quality counts by plane
- cell_class_summary.csv: somatic versus dendritic counts and reliability
- motion_summary.csv: Suite2p registration summaries
- image_roi_comparison.csv: image intensity, alignment, morphology and classification counts
- retained_trials.csv: cached occupancy coverage
- mean_images.png: adjacent-session mean-image comparison

Diagnostic scripts: ../inspect_atl012_reliability.py and ../inspect_atl012_images.py.
