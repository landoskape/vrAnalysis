# Gain regression v6 audit - 30 October 2026

Complete stored v6 run: 4,172 session/configuration results (149 sessions x 28 configurations). Both estimators have 240 matched session/environment observations from 135 sessions and 14 mice. Minimum full-environment trial count is 10; unfit slots remain excluded. Environment scores are averaged within session, then sessions within mouse, then mice are equally weighted. Raw and sqrt fits and both scoring scales use matched observations within each estimator.

## Performance

Each row compares a fit and its own controls on one explicit scoring scale. Adjacent-lag control is the arithmetic mean of separately fitted lag -1 and +1 test R2 values. Random-roll control is the stored 10-roll average at lag zero. These are descriptive averages, not p-values.

| Estimator | Scoring scale | Fit | Same trial | Mean lag -1/+1 | Random roll |
|---|---|---|---:|---:|---:|
| gaussian | raw | raw | 0.00386 | -0.03074 | -0.03469 |
| gaussian | raw | sqrt | -0.05057 | -0.09052 | -0.08167 |
| gaussian | sqrt | raw | -0.12698 | -0.14284 | -0.20328 |
| gaussian | sqrt | sqrt | 0.08284 | 0.02774 | 0.01856 |
| least_squares | raw | raw | 0.00501 | -0.03039 | -0.03495 |
| least_squares | raw | sqrt | -0.05632 | -0.09656 | -0.08857 |
| least_squares | sqrt | raw | -0.12092 | -0.13745 | -0.19448 |
| least_squares | sqrt | sqrt | 0.07531 | 0.02361 | 0.01344 |

The apparent discrepancy is a scale effect: raw fits on raw gain average roughly 0.004-0.005, whereas sqrt fits on sqrt gain average 0.075-0.083, with the best mouse near 0.20. The same-trial native-scale fit exceeds both controls in all 14 mice for both estimators. Raw-fit advantages over lag/roll controls are about 0.035/0.040 for least squares; sqrt-fit advantages on sqrt scale are about 0.052/0.062.

## Historical check

v5 and v6 native-space averages match to floating-point precision. The v3 all-trial-placefield results were already similar: Gaussian raw/sqrt 0.00289/0.08158; least squares raw/sqrt 0.00407/0.07443. v3 has 140 usable sessions versus 135 now, with median test-fold trial counts of 10 versus 9 after lag-boundary trimming. This historical comparison is descriptive because the cohort/trial sets differ. It does not show a new performance collapse.

## Independent checks

- For every finite v6 test score across both estimators, both score scales, both fits, and lags -1/0/+1, recovered pooled target variance as MSE/(1-R2). Across 24 configuration comparisons, the largest relative discrepancy was 6.95e-16. This checks the score denominator even when predictions and fitted hyperparameters differ.
- Refit four representative saved models (two estimators x two transforms, one environment in ATL012.2023-02-22.701) at stored alpha/rank. Independently computed validation/test SSE, SST, MSE, and R2 using NumPy for both scoring scales. All matched saved values within numerical tolerance.
- Reproduced all ten deterministic random rolls for each representative fit; averaged null R2 matched the stored null scores.
- Verified matching retained ROI indices, source/target neuron rows, raw versus squared sqrt gain matrices, and target-fold indices across sampled raw/sqrt and lag fits.
- Added independent unit tests with heterogeneous neuron means to check pooled target centering without using the implementation metric as the expected answer. Added a test distinguishing lag-invariant targets from full-row-roll target distributions. 91 focused tests passed in the vrAnalysis environment with pytest plugin autoload disabled.

## Code and fairness review

1. **Model/scale comparison is fair within an explicit score scale.** Predictions are rectified before square/sqrt conversion; true targets get the same conversion. Target variance is pooled around the single held-out grand mean, not averaged per-neuron R2. This is an intentional metric definition. Do not compare R2 values across different scoring scales as the same variance fraction.
2. **The gain-template denominator is separate from the R2 denominator.** Both transforms use the same all-trial templates and the same positive gain denominator before transformation. Gaussian gain divides a bump-weighted trial response by the bump-weighted fitted curve, including the fitted baseline; least squares divides template dot-products by template energy. A Gaussian fit need not make each neuron's all-trial empirical mean gain exactly one. The two estimators nevertheless give similar results here.
3. **All-trial gain templates and screening include held-out trials, intentionally.** Regression weights are held out, but the definition of the gain and the neuron screen are not learned from training trials alone. Keep this interpretation explicit; it is the requested analysis convention, not an unseen-test-template evaluation.
4. **Lag controls have identical target denominators.** All shifts use the same interior targets and the same neuron split; source inputs shift in original retained environment order without wraparound. Each lag separately selects alpha/rank. They measure temporal prediction, not a randomization p-value. Source inputs can come from a different target-assigned fold at nonzero lag; this is the documented lag-prediction design.
5. **Full-row random rolls do not hold test targets fixed.** They preserve each neuron's full-series distribution but change the values assigned to target test columns. Each roll correctly computes its own denominator; the saved value averages R2 ratios, rather than dividing average SSE by the real fit's SST. In the audited Gaussian example, raw-scale null SST ranged from 0.944 to 1.459 times the real SST; sqrt-scale null SST ranged from 0.924 to 1.145 times real SST. This is a genuine limitation for an exactly fixed-target comparison, not evidence that the metric code is wrong.
6. **Null tuning differs from real and lag tuning.** The real alpha/rank are selected on validation data; each roll uses those fixed choices. That asks how the selected model behaves when shared structure is removed. It is not the performance of a separately optimized null model. Only 10-roll averages are saved, so null uncertainty and calibrated permutation significance cannot be recovered from the stored averages.
7. **Each fit optimizes its own fitting-space loss.** A raw fit evaluated on sqrt scale has not been tuned to maximize sqrt-scale performance, and conversely. Square-root loss reduces tail weighting; squaring a conditional-mean sqrt prediction generally underpredicts the raw conditional mean (squaring does not commute with averaging). Cross-scale failures are therefore plausible even with correct transformations.
8. **Other modeling details:** ridge regularization also penalizes the appended intercept; rank includes the mean component; alpha is selected at maximal rank before rank selection, rather than jointly optimizing alpha for every rank. These are existing design choices, not newly introduced scoring defects. Small test folds (median nine trials) and heavy gain tails make absolute test R2 noisy.

## Changes and next scientific decision

No numerical scoring or stored-result changes were warranted by these checks. Corrected a stale template-training comment and documented roll-denominator semantics; no schema bump. For a stricter fixed-target null, roll only source neurons while leaving targets and fold assignments intact, use the same target denominator, and retain per-roll scores. If the question is significance of an optimized pipeline, apply the same hyperparameter-selection procedure to each null replicate. That is a separate scientific definition requiring a new schema, not a post-hoc denominator replacement.

## Files

- comparison.png: four-panel comparison (both fits and controls on both scoring scales, for each estimator).
- summary.csv and mouse_scores.csv: means and individual mouse values.
- control_advantages.csv: descriptive paired advantages and numbers of mice above each control.
- history.csv: native-scale comparison across stored schema versions.
- target_variance_checks.csv: all target-variance consistency checks.
- audit.json: representative independent reconstruction checks and null-SST ranges.
- ../gain_regression_audit.py: reproducible read-only analysis of stored results.
