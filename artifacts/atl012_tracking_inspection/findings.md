# ATL012 global tracking and March 1 dropout

## Scope and definitions

Used `vrAnalysis.tracking.Tracker("ATL012")` and its existing ROICaT global cluster
labels, without rerunning tracking or changing session data. Tracking covers 22
sessions, Jan 23 through Mar 3, 2023; the first plotted imaging session, Jan 20,
has no tracking labels and is excluded.

For each source-session ROI, count the OTHER sessions containing its nonnegative
global tracking label. Label -1 contributes zero matches. All source ROIs remain
in the denominator, including unmatched ROIs. No extra silhouette threshold is
applied. Each session's label array matches its raw ROI count and contains no
duplicate nonnegative labels. Pair counts were checked against
`Tracker.get_tracked_idx(..., use_session_filters=False)`.

Analysis-selected ROIs use the viewer's session filters with sigrebase, including
classifier c/d, non-silent and non-redundant filters. This does NOT additionally
require the environment-specific spatial-reliability/fraction-active thresholds.
For primary recovery estimates, a match in the target may be any raw ROI; target
analysis filters are reported separately to distinguish matching from filtering.

## Recurrence fractions

| Fraction found in at least one other tracked session | Feb 28 | Mar 1 | Mar 3 |
|---|---:|---:|---:|
| All raw ROIs | 85.6% | 65.8% | 74.5% |
| Analysis-selected ROIs | 91.0% | 69.7% | 82.4% |
| Selected somatic ROIs | 96.1% | 91.7% | 94.2% |
| Selected dendritic ROIs | 77.0% | 44.7% | 63.1% |

Mean number of OTHER sessions containing an analysis-selected ROI is
11.26 -> 7.54 -> 9.13, out of 21 possible. Equivalently, the chance of finding
a randomly chosen selected source ROI in a uniformly chosen other session is
53.6% -> 35.9% -> 43.5%. These are descriptive fractions, not independent
Bernoulli observations or calibrated biological detection probabilities.

March 1 has 1,128 selected ROIs with no match in any other session; 965 (85.5%)
are classified dendrites. Plane 1 alone contains 763 of those unmatched ROIs.
The March 1 selected recurrence fraction is third-lowest of the 22 sessions;
dendritic recurrence is second-lowest. Feb 22 also shows pronounced tracking
churn, so March 1 is not the only session with this pattern.

## Direction matters: added ROIs versus missing persistent cells

73.4% of Feb 28's selected ROIs are found on Mar 1, including 83.4% of its somata.
Conversely, only 46.9% of Mar 1's larger selected ROI population is found on Feb 28.
The asymmetry is compatible with many added detections rather than only loss of
the previous field. Selected pairwise counts can be asymmetric because the source
is filtered while the target accepts any raw tracking match.

Define a persistent reference independently of March 1: cluster IDs selected in
BOTH Feb 28 and Mar 3. Of 1,193 reference ROIs, March 1 has a tracking match for
1,049 (87.9%). Of the same reference, 888 (74.4%) also pass March 1's analysis
filters. Thus 144 have no tracking match and another 161 have a match but fail
the analysis filters.

| Reference subgroup, selected in both flanks | Reference ROIs | Found on Mar 1 |
|---|---:|---:|
| All | 1,193 | 87.9% |
| Somatic in both flanks | 996 | 92.3% |
| Dendritic in both flanks | 173 | 63.0% |
| Plane 0 | 256 | 92.2% |
| Plane 1 | 117 | 70.9% |
| Plane 2 | 146 | 68.5% |
| Plane 3 | 234 | 89.7% |
| Plane 4 | 440 | 95.5% |

The soma/dendrite reference subgroups do not exhaust the full reference because
some tracked ROIs change classifier category between the two flanks.

The same flanking-cohort calculation across other interior sessions has median
recovery 85.5%; March 1's 87.9% is not unusually low overall. These controls have
different date gaps and cohort sizes and are descriptive, not a matched experiment.

## Spatial distribution and interpretation

For the ROIs detected on March 1, fractions found anywhere else are 89.7%, 40.0%,
65.0%, 85.6%, and 93.7% in planes 0 through 4. The spatial plot shows many unmatched
detections throughout plane 1 and part of plane 2, while the other planes retain
a substantial recurring population. This does not resemble wholesale replacement
of the field across every plane.

The combined evidence supports a large influx of poorly recurring dendritic
detections, especially in plane 1, plus localized loss of some persistent matches
in planes 1-2. A modest lateral or axial FoV change is plausible. Existing tracking
cannot establish whether any unmatched ROI reflects a physically missing structure,
changed fluorescence/segmentation, or a tracking failure. It cannot by itself prove
the acquisition moved in z.

## Reliability of a fixed matched cohort

888 ROIs have a global match and pass session filters in all three late sessions.
798 are somatic in both flanks. Without selecting on reliability, the same 798
somatic ROIs have mean spatial reliabilities:

| Environment | Feb 28 | Mar 1 | Mar 3 |
|---|---:|---:|---:|
| 1 | .099 | .065 | .290 |
| 3 | .187 | .087 | .346 |
| 4 | .304 | .277 | .327 |

These levels differ from the viewer because the viewer additionally selects quality
ROIs separately in each session. This fixed-cohort comparison shows lower mean
reliability on Mar 1 even among matched somata; replacement of the ROI population
does not explain the entire difference.

The saved table also includes cohorts required to pass quality thresholds in both
flanks. Those endpoint-conditioned comparisons favor high endpoint values and
can show regression to the mean in the middle session; they are not unbiased
estimates of a March 1-specific physiological effect.

## Outputs

- session_recurrence.csv: every session, source subset, and global recurrence statistics
- recurrence_distribution.csv: number of ROIs found in exactly k other sessions
- pairwise_recovery.csv: directional source-normalized recovery for every ordered pair
- flanking_cohort_recovery.csv: persistent-reference recovery, including class and plane
- fixed_cohort_reliability.csv: reliability of identical matched ROI cohorts
- tracking_summary.png: recurrence curves, pairwise matrix, flanking-cohort recovery
- march1_roi_recurrence_map.png: ROI centroid overlay on each plane's mean image

Reproduce with `artifacts/inspect_atl012_tracking.py` in the vrAnalysis Conda
environment. Only diagnostic files under artifacts are written.
