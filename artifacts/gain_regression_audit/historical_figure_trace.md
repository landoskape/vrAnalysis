# Original gain figure traced to PF=train

The original files are D:/localData/dimensionality-manuscript/figures/figure2/placefield_gain_example.png and .svg. The SVG embeds 2026-08-11T23:00:34.834406. This resolves the Explorer date as 11 August, not 8 November.

The next-morning code snapshot snapshot_20260812T075603Z.zip contains the notebook cell generating this filename. It selects sqrt gain, mouse ATL027, session ATL027/2023-08-04/702, environment 1, rastermap gain sorting, and pooled performance. It omits placefield_split; both viewer and analysis defaults in that snapshot are train. The config is schema v2 and uses only the Gaussian estimator, predating the lag sweep and least-squares estimator.

## Numeric confirmation

The 14 data and 14 shuffle marker values were recovered directly from the SVG vector coordinates. They match the archived v3 Gaussian/PF=train/sqrt native-score results, in mouse order, with maximum absolute error 4.57e-09, consistent with rounded SVG coordinates. v3 added an estimator option while preserving the Gaussian computation. This numeric match independently confirms the settings; the conclusion does not rely solely on notebook defaults or Git commit dates.

| Archived setup | Mean data R2 | Mean random-roll R2 |
|---|---:|---:|
| PF=train, Gaussian, sqrt fit/sqrt score | 0.134335004 | 0.030105325 |
| PF=all, Gaussian, sqrt fit/sqrt score | 0.081579082 | 0.017155833 |

The selected red point is 0.11503527504857947; the original SVG red point is 0.115035278. The latest v6 all-trial Gaussian sqrt/sqrt mean is 0.082844, close to the archived all-trial value, rather than a scoring-schema collapse.

## Important historical viewer distinction

In this version, the raster always built its direct-session template and screen from all trials. The viewer documentation explicitly says placefield_split applies only to stored regression results. Consequently the left raster was based on all-trial fields while the right performance panel selected PF=train results. The image alone could therefore suggest an all-trial setup even though its plotted R2 was from PF=train.

## Why the PF option changes the result

It changes more than how the same prediction is scored. In the historical config, PF=train builds each place-field template and Gaussian gain denominator using training trials. It also computes reliability and fraction-active screening on training trials. PF=all builds the templates/denominators and screen on all retained trials. Different neuron sets then produce different source/target splits. The gain targets, weighting under pooled R2, fitted model, and possibly populations all differ. Thus larger PF=train R2 is not automatically a better prediction of an identical target variable, and all-trial template estimation need not increase this R2.

For 265 paired usable session/environment observations, median target-neuron count is 715.0 for PF=train and 732.0 for PF=all. Median target-ROI Jaccard overlap is 0.282. The median paired sqrt-target variance ratio (train/all) is 0.905; the median paired MSE ratio is 0.843. These confirm that changing the PF option changes both the evaluated target distribution and the residual error; it is not merely a plot-label switch. In fact the median variance ratio is below one: the higher PF=train R2 is not explained by a simply inflated target denominator. The paired MSE ratio is lower still. These medians summarize distinct ratios and should not be substituted into an exact decomposition of the mean R2.

The exact causal contribution of template normalization versus neuron screening versus refitting is not isolated by these saved results. That would require a crossed comparison with fixed neurons/source-target split/trial folds and separately varied template estimation and screen. No existing results or configuration defaults were changed by this investigation.

## Historical source references

- Git 4845046 (11 August 2026, 20:34 BST): Gaussian gain config v2 with roll nulls, train/all and raw/sqrt options.
- Git 190da69 (12 August 2026, 09:36 BST): committed PlacefieldGainViewer, with PF=train as the default. The figure was generated before that commit from the working code, so the saved next-morning snapshot and exact numeric match are stronger evidence than assuming commit time is execution time.
- Git 4df9097 (7 October 2026): v3 adds the least-squares estimator while preserving Gaussian behavior.
- Next-morning snapshot also preserves the exact notebook figure-generation call.

CSV files historical_figure_match.csv and historical_pf_train_vs_all.csv retain the recovered marker values and paired diagnostics.
