# ATL012 March 1: exclude the final 70 trials

Session: ATL012 / 2023-03-01 / 701. Retain chronological trials 1-198 of 268;
exclude trials 199-268, starting about 29.87 minutes after behavior began.
Original session data, existing processor caches, and the result database were
not changed. This is a diagnostic recomputation only.

## Results matching the viewer's quality selection

Quality means spatial reliability > .3 AND fraction active > .1. Both measurements
are recomputed after exclusion. Analysis uses the same original 3,720 filtered ROIs
and sigrebase activity, with the existing spatial-map and weighted leave-one-out
reliability implementations.

| Environment / experience order | Full mean | First 198 mean, quality reselected | First 198 mean, original quality cohort fixed | Quality count, full -> first 198 |
|---|---:|---:|---:|---:|
| 1 / first | .620147 | .647699 | .660094 | 789 -> 922 |
| 3 / second | .620490 | .662437 | .676555 | 767 -> 893 |
| 4 / third | .516819 | .587880 | .584186 | 415 -> 486 |

Reselected quality means improve by .027552, .041947, and .071061. Holding the
original full-session quality cohort fixed also improves all three means, so
the increase is not solely an effect of changing which cells pass thresholds.

The third environment's trimmed mean (.588) is much closer to the following
session's .596, though still below the preceding session's .638. The other two
trimmed means (.648 and .662) also move toward the neighboring-session values.

## Somatic versus dendritic ROIs

| Environment | Full somatic mean | Trimmed somatic mean | Full dendritic mean | Trimmed dendritic mean |
|---|---:|---:|---:|---:|
| 1 | .680444 | .710899 | .587526 | .611968 |
| 3 | .692737 | .744368 | .571482 | .609875 |
| 4 | .586443 | .662025 | .446180 | .510619 |

These rows reselect quality within each condition. In environment 4, the trimmed
somatic mean (.662) is comparable to Feb 28 (.682) and Mar 3 (.656).

## Trial and spatial coverage

| Environment | Full raw trials | Removed | Retained raw trials | Full eligible trials | Trimmed eligible trials |
|---|---:|---:|---:|---:|---:|
| 1 | 58 | 12 | 46 | 52 | 41 |
| 3 | 53 | 13 | 40 | 51 | 38 |
| 4 | 157 | 45 | 112 | 154 | 110 |

Reliability uses full-trial eligibility from SpkmapProcessor. The same 194 spatial
bins survive the cross-environment missing-position check in both conditions.
Fraction-active uses the separate imaging-frame trial maps, as in PFPredQualityConfig.

## Method and validation

The diagnostic processes ROIs in chunks to limit memory use. It calls the existing
SpkmapProcessor.get_processed_maps and helpers.reliability_loo, explicitly disabling
all processor cache reads/writes. Truncation is applied to chronological trial IDs
before the reliability calculation, rather than selecting a different number of
trials independently in each environment. The initial chunk's full-session result
also agrees with SpkmapProcessor.get_reliability to approximately 6.2e-8.

Per-ROI activity normalization is computed over the full trace; reliability and
participation fraction-active are invariant to this positive scalar scaling, and
the underlying trial maps are unchanged by dropping other trials. Session ROI
filters are held fixed, preserving ROI identity for paired comparisons.

The full-session recomputation reproduces the stored quality membership exactly.
For every activity-eligible ROI/environment entry (stored fraction active > .1),
the maximum reliability difference is 7.14e-8. Recomputed fraction-active agrees
over all entries within 2.39e-7. The full quality means match the plotted values.

An unresolved numerical discrepancy exists outside the activity-eligible set:
763 low-activity ROI/environment entries differ from stored reliability by more
than 2e-5, with maximum difference .726. None pass the activity criterion, and
none change the quality-selected baseline. The reason for those differences was
not established here. Unfiltered all-ROI means are included in the raw summary
CSV for audit, but are not used for the headline comparison.

## Interpretation

The final 70 trials materially reduce the session's quality-selected spatial
reliability, including within a fixed ROI cohort and within somatic ROIs. Excluding
them substantially reduces the original dip. This calculation does not identify
why the late trials are less consistent or establish that exclusion is appropriate
for the final analysis; a change of activity pattern across time could also lower
full-session reliability.

## Outputs and reproduction

- reliability_summary.csv: all, somatic, dendritic, and per-plane summaries; fixed and reselected cohorts
- per_roi_metrics.csv: full and trimmed reliability, fraction active, and quality membership
- trial_coverage.csv: raw and eligible trial counts
- recomputed_metrics.npz: compact per-ROI result arrays
- baseline_validation.json: numerical checks against stored full-session results
- trim70_comparison.png: comparison of the two quality-cohort definitions

Run `artifacts/inspect_atl012_trim70.py` in the vrAnalysis Conda environment.
`--reuse-computed` rebuilds validation and summaries from the diagnostic NPZ instead
of recomputing neural maps.
