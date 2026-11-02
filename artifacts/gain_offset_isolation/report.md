# Isolating the PF=train offset from the full-row roll control

## What was changed independently

Two real sessions were analyzed: ATL027/2023-08-04/702 environment 1 and ATL049/2024-05-01/702 environment 4. Raw session activity was reprocessed to trial place-field maps using the existing loader. Training folds and the lag +3 target columns were kept fixed.

For each session I computed both train-only and all-trial reliability/fraction-active screens. I fitted Gaussian fields under both reference conventions on the union of screened neurons. A fixed hash-based source/target assignment was made per original ROI, independent of either screen. Fits that failed under either reference were excluded from all conditions. This creates four conditions: all/all, all/train, train/all, train/train (screen/reference). Within a screen comparison, reference changes use exactly the same neurons and source/target assignments. Within a reference comparison, changing the screen changes neuron membership but never reassigns shared neurons between source and target.

Each condition independently optimizes alpha/rank by the existing validation procedure on sqrt gain. I evaluated lag +3 real R2 and 20 full-row-roll controls using that condition's selected hyperparameters. This is a controlled diagnostic, not a reproduction of the original stored scores: source/target assignments have deliberately been made stable across screens. Reference changes include the Gaussian field fit, spatial weights, and denominator; they do not isolate scalar denominator changes from Gaussian shape/weight changes.

## Real data results

| Session | Screen | Reference | Real R2 | Roll mean R2 | Real minus roll |
|---|---|---|---:|---:|---:|
| ATL027.2023-08-04.702 | all | all | -0.02233 | -0.01100 | -0.01132 |
| ATL027.2023-08-04.702 | all | train | 0.00542 | -0.00107 | 0.00649 |
| ATL027.2023-08-04.702 | train | all | 0.01190 | -0.00543 | 0.01732 |
| ATL027.2023-08-04.702 | train | train | 0.05053 | 0.01559 | 0.03494 |
| ATL049.2024-05-01.702 | all | all | 0.02706 | 0.03262 | -0.00556 |
| ATL049.2024-05-01.702 | all | train | 0.04369 | 0.03557 | 0.00812 |
| ATL049.2024-05-01.702 | train | all | 0.04489 | 0.03436 | 0.01053 |
| ATL049.2024-05-01.702 | train | train | 0.06024 | 0.04356 | 0.01669 |

Both training-only reference estimation and training-only screening increase the real-versus-roll gap in these two examples. Holding the screen all-trial, the reference change increases the gap by about 0.0178 and 0.0137. Holding the reference all-trial, changing the screen increases the gap by about 0.0287 and 0.0161. Effects are not assumed additive or generalizable to every session. The analogous contributions to mean-level and trial-varying R2 are in crossed_examples.csv; changes are predominantly mean-level.

## Why the reference alone can make a gap without shared activity

I simulated 400 neurons over 60 trials, with 40 random training trials and 20 test trials. Each neuron has independent gamma-distributed responses, with different fixed gamma shapes across neurons. There are no shared trial fluctuations, no source-neuron inputs to the predictor, and no temporal trend. The gain is the scalar-response analogue of least-squares field gain: response divided by that neuron's training or all-trial mean, followed by sqrt. The only predictor is each neuron's mean sqrt gain on training trials. Results below average 500 simulations.

| Reference | Real data | Roll old gains | Roll responses, then re-estimate reference |
|---|---:|---:|---:|
| All | 0.0124 | 0.0127 | 0.0127 |
| Train | 0.0307 | 0.0126 | 0.0298 |

This does not model all the real data or estimate how much of its offset is artificial. It establishes a mechanism: the real-minus-roll advantage can occur without shared activity or a source predictor, and repeating normalization after randomization largely removes that advantage.

## Intuition: the training reference remembers which trials trained it

For scalar gain g_it = response_it / mean_training(response_i), the mean raw gain over the original reference-training trials is exactly one for every neuron. In sqrt units, this means the training second moment (mean squared sqrt gain) is exactly one. The training mean sqrt gain is not fixed: it reflects each neuron's response sparsity/shape, but is estimated under this self-normalization constraint. All-trial normalization instead gives that constraint over all trials. Gaussian field gain is not exactly this scalar identity, because its fitted curve need not equal the empirical template, but it shares the training-reference dependency.

Now roll the already normalized gain series. The reference still comes from the original training trials, but the rows now put different trial responses in the training columns. Their training raw mean is no longer constrained to one. The mean-only predictor is therefore trained under a different finite-sample normalization relationship. Its estimate of the held-out neuron means can deteriorate even though cross-neuron activity was never involved. In PF=all, rolling each complete row leaves the all-trial scalar reference invariant, so this particular mismatch does not arise.

Another way to express it: the real pipeline says "estimate the training reference, then learn the training-to-test relationship." The current null says "keep that estimated reference, but move different observations into the training set." That null does more than remove shared trial structure. Re-estimating the reference after rolling restores the intended training-reference relationship in the simulation.

## What screening adds

The train-only screen also remembers which observations were training observations: selected cells passed reliability/fraction-active criteria on those particular trials. Full-row rolling keeps the selected cells but moves observations between folds, so the null training columns are no longer the columns on which they were selected. That changes the conditional distribution being evaluated. The crossed results show that this screening choice adds to the real-versus-roll gap even with an all-trial reference and stable neuron assignments. This is not proof that the complete screening effect is spurious or that it always has this sign; the cells themselves also differ between screens.

## Relation to the earlier mean/trial decomposition

The earlier cohort results showed nearly identical lag +3 trial-varying contributions (0.0309 train versus 0.0306 all), despite different total scores. The new experiments explain why retaining a mean-level offset above the old roll line is not enough to infer extra cross-neuron predictive structure. The pooled R2 can genuinely credit stable differences between neuron means, while the old randomization makes those means harder to estimate because it changes the reference/screen conditioning. Those are separate statements: pooled R2 is not numerically wrong, and the roll gap does not uniquely measure shared-trial prediction.

## Implications for controls

For a null for the complete preprocessing-and-fitting procedure, randomization should occur before the train-dependent reference/screen computations and those steps should be repeated consistently, with careful attention to how neuron selection changes the evaluated target population. Alternatively, use independent reference data so the regression train/test observations do not also define the gain reference.

If the question is whether source activity adds trial-specific information for a fixed set of target neurons and a fixed gain definition, a fixed-target source randomization preserving the intended training/evaluation conditioning is a cleaner question. Source indices under nonzero lag require care: shuffling target-fold columns is not automatically shuffling the actual source-training observations. Within-fold shuffles also destroy temporal correlations and ask a different question from full-series circular rolls.

The crossed script includes an exploratory within-target-fold shuffle column. Because the analysis is shifted, that operation does not preserve the actual source-fold marginals; it is intentionally not used in the conclusions above.

No pipeline source, schema, or saved results were modified. The real-data experiment concerns two examples, not a cohort-wide attribution.

## Files

- crossed_examples.csv: all real/reference/screen conditions and mean/trial decompositions.
- independent_simulation.csv: all 500 simulation replicates.
- independent_simulation.png: reference-conditioning demonstration.
- ../isolate_gain_reference_screen.py: real-data experiment.
- ../gain_reference_simulation.py: independent-neuron simulation.
