# Deep dive: the PF=train shift-curve offset

All analyses use v6 Gaussian-estimator sqrt models scored on sqrt gain, with the same 240 session/environment observations, 135 sessions, and 14 mice. Summaries average environments within sessions, sessions within mice, then mice equally. The investigation uses saved trial-gain matrices and reconstructed predictions; it does not reprocess the original spike/activity files. No stored results or analysis definitions were changed.

## Main conclusion

The approximately 0.05 vertical difference between PF=train and PF=all is almost entirely a difference in predicting neuron-average gain levels under pooled R2. It is not approximately 0.05 more prediction of trial-to-trial fluctuations. Both models have a comparable remaining off-zero temporal contribution; a simple time trend captures much of that aggregate contribution. The zero-lag peak remains substantially larger.

## Exact prediction decomposition

For test arrays Y and P (trials by neurons), write each as its per-neuron test mean plus its trial-centered residual. Orthogonality gives SSE = N_trials * sum((mean_Y - mean_P)^2) + sum((centered_Y - centered_P)^2). With pooled SST around the single test grand mean, define the mean-level contribution as 1 - SSE(Y, broadcast(mean_P))/SST, and the trial-varying contribution as (within-neuron SST - centered SSE)/SST. Their sum is exactly the stored pooled R2. This is a descriptive decomposition, not fitting a predictor to test labels. The per-neuron test means are only used to partition error.

960 models (240 observations x two PF settings x lags 0/+3) were reconstructed at their saved alpha and rank. A thin-SVD implementation was used for computational efficiency; reconstructed scores matched saved scores within numerical tolerance.

| PF | Lag | Total R2 | Neuron-mean contribution | Trial-varying contribution |
|---|---:|---:|---:|---:|
| all | 0 | 0.082844 | 0.001789 | 0.081054 |
| all | 3 | 0.019478 | -0.011136 | 0.030614 |
| train | 0 | 0.129876 | 0.050203 | 0.079673 |
| train | 3 | 0.068418 | 0.037527 | 0.030890 |

At +3, train minus all total R2 is 0.048939; the mean-level contribution changes by 0.048663 and the trial-varying contribution by only 0.000276. At zero lag the trial-varying contributions are also similar (0.079673 train vs 0.081054 all).

## Why the mean-level term changes

- The scored target is a derived gain, and its normalization and neuron selection differ between PF settings. In PF=train, templates and reliability/fraction-active screening are estimated on training observations. This conditions the target distribution on the training fold. PF=all includes held-out observations in that definition. Neither is the identical target/population with merely a relabeled score.
- Across the matched cohort, the mean sqrt target is 0.4960 in training versus 0.4417 in test for PF=train. For PF=all it is 0.4920 versus 0.4939. The training-defined normalization/screen therefore produces a clear training-to-held-out distribution difference. This observation does not by itself isolate denominator estimation from selection effects.
- A predictor containing only each target neuron's training mean already scores +0.0300 under PF=train, versus -0.0133 under PF=all, without using source activity. Evaluating RRR against this training-mean MSE baseline makes the PF curves much closer: at +3 the error reductions are 0.0395 and 0.0326, respectively. This does not establish that the residual is all temporal prediction: RRR can also improve mean-level calibration over a noisy training mean.
- The pooled metric includes stable differences between neurons, not only within-neuron trial variation. The fraction of held-out SST due to between-neuron test means is 0.1697 for train versus 0.1255 for all (an oracle descriptive statistic). Thus per-neuron mean estimation can contribute materially even at unrelated lags.
- Predictions are systematically lower than the raw training target means. At +3, average predictions are 0.3987 (train) and 0.4054 (all); true test means are 0.4417 and 0.4939. This is consistent with stronger relative underprediction under PF=all. The existing model regularizes its appended intercept and uses nonnegative uncentered fitting; this is one calibration mechanism, not a proven isolated cause.
- In two examples, rescaling each fixed neuron's entire raw gain series to mean one (keeping the neuron groups and alpha/rank fixed) reduces PF=train lag +3 R2 from 0.0500 to 0.0326 and from 0.0580 to 0.0468. This supports a normalization contribution even with fixed neurons. This all-trial rescaling is a diagnostic using held-out values, not a proposed training-only evaluation. It also changes loss weighting/regularization units, so it is not a clean causal decomposition of normalization versus screening.

## Slow temporal structure

A time-only model was fit from each target neuron's training sqrt gains against standardized retained-trial index. The intercept is its training mean; one shared ridge penalty for per-neuron slopes was chosen by validation MSE from a fixed log-spaced grid. Test values were not used for fitting or penalty selection. This model has no source-neuron inputs.

| PF | Time-only total R2 | Time-only trial-varying contribution |
|---|---:|---:|
| all | 0.017343 | 0.026907 |
| train | 0.061431 | 0.027439 |

The corresponding RRR lag +3 trial-varying contributions are 0.0306 and 0.0309. Time-only prediction gives a similar aggregate contribution (0.0269 and 0.0274). This is consistent with slow changes in population gain carrying much of the off-zero temporal prediction; it is not proof that every bit of lagged prediction is drift or that the two predictions explain identical residuals. Retained-trial index is not elapsed physical time.
The examples ATL027/2023-08-04/702 environment 1 and ATL049/2024-05-01/702 environment 4 show a conspicuous decline in mean source gain over retained trials. Their time-only PF=train R2 values are 0.0706 and 0.0541, compared with RRR lag +3 values 0.0500 and 0.0580. These slow trends exist in the stored gain series; their behavioral or physiological cause was not identified.

## Permutation-control checks

The stored control independently rolls every source and target neuron on the full series, then splits trials and refits with real zero-lag alpha/rank. It changes both held-out target SST and how template-conditioned training/test distributions are mixed. The mean-only predictor itself drops from 0.0300 on real PF=train data to about -0.0064 after full-row target rolling (20 deterministic diagnostic replicates per observation). Therefore a real-minus-full-roll gap is not automatically extra trial-specific prediction.
For each example I additionally performed 20 source-only full-row rolls at lag +3, leaving target values, target folds, and target denominator fixed, and using that lag's own saved alpha and rank.

| Session | PF | Real lag +3 R2 | Fixed-target source-roll mean | Roll SD |
|---|---|---:|---:|---:|
| ATL027.2023-08-04.702 | all | -0.04938 | -0.04199 | 0.00275 |
| ATL027.2023-08-04.702 | train | 0.04996 | 0.05095 | 0.00277 |
| ATL049.2024-05-01.702 | all | 0.02439 | 0.03094 | 0.00230 |
| ATL049.2024-05-01.702 | train | 0.05803 | 0.06317 | 0.00194 |

PF=train lag +3 does not exceed this fixed-target source-roll mean in either example. This is a descriptive diagnostic from two examples and 20 rolls, not a calibrated cohort-wide significance test. Source-only full-row rolls still mix source fold-dependent distributions. I also tested source-only and both-population independent shuffles within each fold; these preserve fold-specific marginal distributions but erase temporal alignment, and give different floors. That dependence on the null definition is itself informative. All these controls use fixed real-fit hyperparameters, not fresh validation optimization for every replicate.

## Code/reproducibility checks

Trial folds are randomly shuffled within environment; the observed temporal decline is not simply a deliberately early-train/late-test split. Nonnegative predictions and scale transformations behaved as intended. No new pooled-denominator or indexing bug was found that accounts for the offset.
One smaller reproducibility issue emerged: for ATL023/2023-04-28/702 environment slot 0, separately saved PF=train lag 0 and lag +3 gain matrices differ beyond floating-point roundoff, despite identical retained neuron identities. The maximum sqrt-gain difference is 0.0929. Repeating the lag +3 fit on the lag-zero gain matrix changes its R2 from 0.1302244 to 0.1302015 (about 2.3e-5 for that observation). Other observed differences were around 1e-7 or less. The root cause of this preprocessing discrepancy was not traced; it is far too small in effect to explain the cohort offset. Reusing one immutable gain/template computation across lag jobs would enforce the intended invariant.

## Recommended interpretation and next changes

1. Report original pooled R2 together with prediction beyond a training-neuron-mean baseline. Do not equate pooled R2 with exclusively temporal shared-gain prediction.
2. Include a time-only baseline and the exact diagnostic mean/trial decomposition. The narrow zero-lag component is more distinctive than the broad offset.
3. If the scientific question is trial-specific cross-neuron coupling, add a fixed-target source shuffle preserving chosen fold/time structure, and save per-replicate scores. A time-preserving versus time-destroying null asks different questions. For formal significance, mirror hyperparameter selection and the gain-estimation/screening dependencies in the null procedure.
4. Consider a crossed template/screen comparison with identical neurons and an explicitly calibrated intercept. This is needed before attributing the mean-level improvement solely to PF normalization.
No schema changes or numerical modifications were made during this investigation.

## Reproduction and files

- ../investigate_gain_offset.py: cohort mean baselines, lag scores, and rolled-mean controls.
- ../gain_offset_examples.py: example predictions, fixed-neuron renormalization, and alternative shuffles.
- ../decompose_gain_predictions.py: reconstruct and decompose cohort predictions at lags 0/+3.
- ../gain_offset_time_baseline.py: validation-tuned time-only baseline.
- cohort.csv, prediction_decomposition.csv, time_baseline.csv: observation-level outputs.
- fixed_target_lag3_controls.csv, example_results.csv, example_rolls.json: example controls and replicate scores.
- decomposition.png, baseline_adjusted_curves.png, examples.png: diagnostic figures.
- numerical_anomaly.json and gain_lag_invariance.json: smaller preprocessing reproducibility discrepancy.
