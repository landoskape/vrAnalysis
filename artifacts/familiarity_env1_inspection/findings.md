# First-environment drop at x=6

Inspected stored RegressionConfig v5 results (sigrebase, std), using external_placefield_1d
as baseline and internal_placefield_1d_structured_additive as reference. The results database
was opened read-only. No models were recomputed and no observations were excluded from saved results.
The x axis is zero-based: x=6 is the seventh finite first-environment session for each mouse.

| x | Mice | PF mean R2 | PF median R2 | Mean PF/Residual | Median PF/Residual |
|---|---:|---:|---:|---:|---:|
| 5 | 12 | 0.046423 | 0.026761 | 0.348349 | 0.378373 |
| 6 | 8 | 0.010032 | 0.008208 | 0.085843 | 0.191969 |
| 7 | 6 | 0.047750 | 0.032767 | 0.434223 | 0.375963 |

## Changing cohort

ATL045, ATL057, CR_Hippocannula6 and CR_Hippocannula7 have no further first-environment
observations in the selected session set after x=5. Their selected session histories end there;
the lower mouse count at x=6 is not caused by a newly missing cached score in a later selected session.
ATL045 and the two CR mice have above-average PF scores at x=5.

Restricting x=5 to the eight mice contributing at x=6 lowers its PF mean from 0.046423
to 0.025728. Those same mice then fall to 0.010032 at x=6. This decomposes the absolute
drop into 0.020695 from cohort composition (57%) and 0.015696 within mice (43%).
For PF/Residual, the analogous decomposition is 0.348349 -> 0.280703 -> 0.085843:
26% composition, 74% within mice. These are descriptive decompositions, not causal estimates.

## Most influential sessions

| Mouse | Session at x=6 | PF R2 x=5 -> x=6 | PF/Residual x=5 -> x=6 | Held-out env-1 frames at x=6 |
|---|---|---|---|---:|
| ATL049 | 2024-05-02 / 701 | 0.073123 -> -0.006598 | 0.585740 -> -0.300835 | 395 |
| ATL012 | 2023-01-30 / 702 | 0.021276 -> -0.005045 | 0.446514 -> -0.195529 | 74 |
| ATL060 | 2024-07-19 / 701 | -0.006158 -> -0.004287 | -0.208890 -> -0.113210 | 206 |

ATL049 and ATL012 have the largest decreases. Both models' scores decline in those sessions,
but PF becomes negative while Residual stays positive (0.021931 and 0.025800 respectively).
ATL012's session is its first selected session with a second environment and has only 74
held-out frames in the first environment, versus 596 at x=5. This is context for possible
sampling variability, not evidence of a specific cause. ATL049's 395 frames are similar to
its neighboring sessions; a very small frame count does not explain that session's dip.

ATL060 lowers the level at x=6 but improves from x=5, so it does not drive the sudden decline.
Across the eight paired mice, PF falls in five and rises in three. ATL049 and ATL012 account
for about 84% of the net paired PF decrease and 98% of the net paired ratio decrease;
other mice include offsetting improvements.

Leaving either mouse out still leaves a dip. Excluding both diagnostically raises the x=6
ratio mean from 0.085843 to 0.197185, compared with 0.314793 at x=5 and 0.418258 at x=7
under the same two-mouse exclusion. This is not a recommendation to remove these observations.

## Mean versus median

Median aggregation softens the ratio dip (0.378373 -> 0.191969 -> 0.375963), but preserves
the absolute-score dip (0.026761 -> 0.008208 -> 0.032767). It is therefore not solely a
single-mouse mean artifact. Cohort composition still changes under median aggregation.

The new viewer statistic selector controls baseline, reference and comparison centers.
Bands retain across-mouse SD or SD/sqrt(n), centered on the chosen statistic; they are not
confidence intervals for the median. All 38 focused per-environment viewer tests pass.

Files: per_mouse_session_scores.csv, paired_changes_5_to_6.csv, leave_one_mouse_out.csv,
summary.csv, mean_curves.png, median_curves.png. inspect_results.py reproduces the stored-score
inspection; paired_changes_5_to_6.csv records the additional paired comparison.
