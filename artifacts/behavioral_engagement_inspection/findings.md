# Exploratory within-session engagement and spatial-code analysis

## Conclusion

ATL012 on 2023-03-01 is a strong candidate for a late-session state change:
anticipatory reward-related slowing declines, held-out agreement with an earlier
place-field template collapses late in the session, and identical somatic ROIs
have lower spatial reliability afterward. This survives equal trial counts,
trial-mean speed matching, exclusion of the reward region, and an alternative
Oasis activity extraction.

The broader analysis does **not** yet justify a validated disengagement detector
or automatic trimming. Behavior-screen positives have modest neural effects in
other mice, several sham controls also deteriorate, and speed-matched group
averages do not separate positives from controls. Imaging and signal-stability
changes remain potential confounds in the motivating session.

## Relation to the proposed paper

[Pettit, Yuan & Harvey (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9076532/)
defined engagement from licking independently of neural outcomes. They compared
spatial coding using matched trial counts and controlled for running speed;
poor coding within the disengaged period helped distinguish degradation from a
stable alternate map. Here the running-based screen is an adaptation, not a
replication of their behavioral classifier or proof of the same internal state.

## Behavior discovery and independent screening

Four discovery sessions were selected before inspecting the broader neural
outcomes: ATL012 on Feb 27, Feb 28, Mar 1, and Mar 3, 2023.

All ATL sessions have zero registered lick events in these data. The motivating
session's raw lick-detector channel is also flat. The absence of recorded licking
is treated as unavailable measurement, not evidence that the animal stopped
licking. CR sessions have some lick events, but the primary screen is consistently
based on running. Rewards are automatically delivered in these sessions: the
active-licking and active-stopping flags are false, so reward receipt itself is
not a task-success measure comparable to a lick-contingent task.

The feature is a pre-reward slowing index:

`(control speed - approach speed) / (control speed + approach speed)`.

Approach speed is measured over the 10% of track immediately preceding the true
reward-zone onset. The equal-width control window spans 22.5%-12.5% of track
before that onset. This uses actual per-trial reward metadata and includes
stationary behavior. Measuring before reward availability reduces consumption
confounding. It does not make running an exclusive measure of engagement.

The rule was frozen in `behavior_rule_v1.json` after discovery and before broad
neural comparisons. It fits one downward change point with environment-specific
intercepts, leaving at least 20 trials and 20% of the session on each side. It
requires a fitted drop of at least .08, mean pre-split slowing of at least .10,
and post-split slowing at most half the pre-split value. Searching over cutoffs
is included in a 1,999-repeat block-permutation test (10-trial blocks). Session
p-values are adjusted using Benjamini-Hochberg FDR .05. These exploratory
permutation results depend on the residual/block-stationarity assumptions.

140 available imaging sessions from 14 mice were extracted successfully. 126 had
enough data for a fitted screen. Six sessions from three mice passed both effect
and FDR criteria:

| Session | First trial after split | Minutes | Slowing before -> after | FDR q |
|---|---:|---:|---:|---:|
| ATL012 2023-02-06 / 701 | 26 | 5.22 | .112 -> -.012 | .0158 |
| ATL012 2023-02-09 / 701 | 61 | 11.01 | .125 -> .015 | .0158 |
| ATL012 2023-02-15 / 701 | 94 | 22.22 | .167 -> .060 | .0189 |
| ATL012 2023-03-01 / 701 | 167 | 25.44 | .191 -> .067 | .0158 |
| ATL045 2024-02-01 / 701 | 173 | 35.76 | .134 -> -.051 | .0158 |
| ATL049 2024-06-05 / 701 | 81 | 17.81 | .130 -> .015 | .0158 |

The first two examples occur early and should not be called end-of-session
disengagement without further evidence. Five of the six positives were outside
the four discovery sessions, but three of those five are additional ATL012
sessions; independent-mouse validation is limited.

Some trends are gradual. In ATL012 Mar 1 a linear time trend fits the slowing
index slightly better than a single step, so trial 167 is an operational split,
not evidence of an abrupt physiological state transition.

## Neural comparisons and controls

The planned neural manifest contains six positives, four explicitly exploratory
borderline sessions, and eight sham-control sessions chosen from behavioral
information before their neural outcomes were examined. Controls use the same
fractional session time as a reference positive/borderline session. They are
imperfect controls: training history, session duration, and environment support
are not exactly matched.

16 of the 18 planned sessions yield neural comparisons: six positives, three
borderline sessions, and seven controls, across six mice. The two ATL022 examples
have too few trials before their boundaries for disjoint calibration/evaluation
and are explicitly excluded rather than analyzed with overlapping data.

Within each session/environment:

- Original filtered ROI identities are held fixed across both periods.
- Earlier pre-split trials form an independent calibration set. Cells are selected
  using calibration reliability > .3 and participation fraction-active > .1;
  primary results use somatic classifier category c. The calibration fraction-active
  is measured on the full-trial neural maps for this diagnostic, rather than the
  pipeline's separate imaging-frame maps.
- Evaluation takes the nearest equal number of eligible trials on each side,
  up to 20 per period, with at least eight per period and five calibration trials.
- Calibration trials do not overlap either evaluation period. Evaluation trials
  are never used to select the primary quality cohort.
- Weighted leave-one-out spatial reliability is recomputed on both evaluation
  periods using the same cells, trial counts, and valid spatial bins.
- The calibration PF template is compared independently with the average PF in
  each evaluation period. Within-period odd/even split-half template correlations
  distinguish a reproducible alternate map from reduced spatial consistency.
- Sensitivity analyses match trial-mean running speed using one-to-one assignment
  with an absolute log-speed-ratio caliper of .25. This is **not** full position-wise
  speed matching as in the paper, and it can change the calibration/evaluation
  trial sets. A second analysis excludes the spatial interval from reward onset
  minus 10 cm to onset plus 30 cm.

## ATL012 Mar 1: same-cell results

The behavior-only cutoff keeps trials 1-166 and treats trials 167-268 as later.

| Environment | Somatic cells | Trials per period | Reliability before | After | Calibration-template r before | After |
|---|---:|---:|---:|---:|---:|---:|
| 1, first | 322 | 18 | .517 | .244 | .729 | .370 |
| 3, second | 343 | 18 | .548 | .245 | .720 | .365 |
| 4, third | 255 | 20 | .378 | .263 | .711 | .562 |

These are independently selected, matched-trial comparisons; their absolute
values differ from the original viewer's full-session, quality-reselected means.
Cells may participate in multiple environment comparisons, so the environment
cell counts are not independent sample sizes.

Within-period split-half correlations drop .694 -> .428, .704 -> .417, and
.484 -> .375. Reliability excluding the reward region declines by .263, .295,
and .117. These support reduced spatial consistency beyond a simple stable
alternate-map interpretation or a change confined to the reward zone.

The later code is not completely absent, and remapping may coexist with reduced
reliability. No place-cell significance shuffle test or population decoder was
implemented in this exploratory analysis.

Trial-mean speed-matched same-cell reliability also decreases in all three
environments. Oasis deconvolution in exactly the same sigrebase-selected somata
gives .397 -> .155, .397 -> .133, and .293 -> .194, respectively. Thus the finding
is not specific to significant-activity extraction. These signals come from the
same imaging data and do not constitute independent biological replication.

### Timing and trim comparisons

An independently calculated, descriptive change point in held-out PF-template
agreement falls after trial 195, at 29.44 minutes. It was **not** used to select
the primary behavioral boundary or its neural comparisons. The behavioral
decline is gradual and leads this sharp neural deterioration by about four
minutes. The earlier ad hoc exclusion of trials 199-268 begins at 29.87 minutes,
close to this neural timing, but that agreement is descriptive and data-informed.

For the original full-session quality cohort, recomputing reliability after
trimming at the behavioral boundary gives:

| Environment | Full-session mean | First 166 trials, same original quality cells |
|---|---:|---:|
| 1 | .620147 | .631658 |
| 3 | .620490 | .657342 |
| 4 | .516819 | .591245 |

The previous first-198-trial calculation is retained separately under
`artifacts/atl012_trim70_inspection`. Its fixed-cohort means are .660, .677, .584;
its newly reselected quality means are .648, .662, .588.

## Generalization is limited

Average over environments within each session, then give sessions equal weight:

| Group | Sessions / mice | Same-cell reliability before -> after | Mean change |
|---|---:|---:|---:|
| Behavior-screen positive | 6 / 3 | .472 -> .382 | -.090 |
| Borderline, exploratory | 3 / 3 | .470 -> .360 | -.110 |
| Sham control | 7 / 6 | .471 -> .420 | -.050 |

All six positives decline, but five of seven controls also decline. Giving mice
equal weight instead produces changes of -.061 in positives and -.043 in controls.
With trial-speed matching, the session-average changes are -.093 for positives
and -.091 for controls. Therefore the screen has not demonstrated useful
specificity for neural disengagement in this selected sample.

The other-mouse positive examples have smaller count-matched effects:
ATL045 Feb 1 is .423 -> .403 and ATL049 June 5 is .553 -> .507. ATL012 Mar 1's
.481 -> .251 is the standout, but was already the motivating example.

Trimming does not consistently improve the full-session quality mean across
positives. It can reduce it because fewer trials reduce estimator support, and
early cutoffs may remove useful later trials. Those whole-versus-trim comparisons
have unequal trial counts and are descriptive; the primary before/after comparison
explicitly matches trial counts.

## Imaging and activity caveats

In ATL012 Mar 1, mean activity of the independently selected quality somata falls
from approximately .407 to .186 in standardized significant-activity map units.
The Oasis comparison also shows lower activity. This is not an activity-matched
test, and selection of an early quality cohort can itself accentuate activity
loss with time.

At the behavioral cutoff the mean registration correlation across planes changes
.0599 -> .0518. Mean rigid x offsets shift by roughly 3.3-3.9 pixels, with very
few bad frames flagged. Plane 4 retains approximately stable registration
correlation but still shows a large neural decline in the first two environments.
That argues against explaining the entire effect by one failing plane, while
leaving residual/axial motion as a confound.

Raw ROI fluorescence changes are heterogeneous: in the selected somata, plane 2's
post/pre mean ratio is .79, plane 3's .88, plane 1's 1.00, and plane 4's 1.07.
Neuropil and corrected fluorescence also change. Registered movie binaries are
not present in the local plane directories, so no time-resolved FOV image
stability test was performed. These summaries cannot rule out focus or extraction
changes, nor establish whether neural, behavioral, and motion changes share a cause.

## Pupil data: useful next validation channel

The motivating session has eye frame-time metadata locally, but no processed
pupil trace was found at the session root. No pupil analysis was performed here.
A practical addition is a read-only per-eye-frame table with timestamp, pupil
area or diameter, center position, tracking confidence, and blink/invalid flags.
Align it using the existing eye frame-time metadata, summarize within trial, and
control for environment/position and running speed before comparing it with
behavior-flagged periods. Pupil size alone should not define disengagement by
an assumed direction or threshold.

For now I would call the running measure a **loss-of-reward-approach-modulation
flag**, preserve its continuous score, and allow an indeterminate state. I would
not automatically delete trials. The useful next validation is an independent
pupil/video check of ATL012's final approximately ten minutes and one control,
alongside time-resolved imaging stability if the source movies are available.

## Validation and artifacts

The analysis writes only under artifacts. Original data, result-store entries,
processor caches, and viewer defaults were unchanged. All neural map cache reads
and writes are explicitly disabled for the recomputation. Full-session quality
means are checked against the stored source results; numerical per-cell differences
are retained in `analysis_validation.csv`. Exact disjoint evaluation trial IDs are
saved as `validated_trial_indices.json` in each usable neural-session directory.
All 16 usable sessions and 31 environment comparisons passed those checks. The
largest full-quality-cohort mean difference from stored results was 9.80e-7;
the largest individual-cell difference was .00154 and is recorded rather than
hidden. Calibration/evaluation trial disjointness and matched counts also passed.

Main outputs:

- `atl012_case_summary.png`: compact motivating example
- `behavior_neural_examples.png`: four example session panels
- `neural_summary.png`: all analyzed session comparisons and controls
- `behavior_examples.png`: behavior-screen positive examples
- `behavior_screen.csv`: all 140 sessions and screening decisions
- `neural_manifest.csv`: preselected roles and boundaries for all 18 planned sessions
- `neural_environment_summary.csv`, `neural_session_summary.csv`, `neural_group_summary.csv`
- `trim_summary.csv`, `neural_trial_coverage.csv`, `analysis_validation.csv`
- `alternative_signal_comparison.csv`, `fluorescence_stability.csv`, `neural_motion_summary.csv`
- `neural/<session>/per_roi_comparison.csv`: same-cell metrics and selection information
- `neural/<session>/trial_template_overlap.csv`: trial-level held-out template agreement

Reproduction scripts, run in the vrAnalysis Conda environment:

1. `artifacts/inspect_behavioral_engagement.py` for the discovery sessions.
2. `artifacts/detect_behavioral_engagement.py` to freeze/apply the initial rule.
3. `artifacts/inspect_behavioral_engagement.py --all` and `artifacts/detect_behavioral_engagement.py --all`.
4. `artifacts/compare_engagement_neural.py --all` (resumes completed sessions).
5. `artifacts/check_engagement_signal_robustness.py`.
6. `artifacts/validate_engagement_analysis.py`.
7. `artifacts/summarize_behavioral_engagement.py` and `artifacts/plot_engagement_case.py`.
