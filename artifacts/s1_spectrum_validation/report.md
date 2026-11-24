# S1-style spectrum check

Session: `ATL057.2024-07-08.701`, chosen with seed 731;
682 successfully fitted neurons, 100 spatial bins,
best-sampled environment 1.

## Construction

- Known signal: the stored unregularized generalized/Tilbury tuning curves.
- Signal and empirical responses converted to full-recording frame-SD units.
- Six disjoint trial-group means from the same neurons/environment/positions.
- Empirical residual E_i = measured repeat i - mean of the six measured repeats.
- Surrogate repeat i = known signal + E_i, following the paper's Figure S1 recipe.
- Evaluate all 60 choices of reference repeat and unordered disjoint scoring pair.
  This uses production estimator functions but more fold assignments than the
  default four-fold StimSpace / three-fold CVPCA pipelines. It does not reproduce
  an existing stored session spectrum.
- Basis smoothing: 5 cm, using the normalized placefield smoother for StimSpace
  and CVPCAConfig.fit_fixed_model for reg_covariances_fixed. Scoring inputs are
  unsmoothed. No sum normalization, log smoothing, or rank re-sorting is applied.

## Ground truth and scaling

ss_cvpca and reg_covariances_fixed center across neurons and use SVD^2/(N-1).
ss_cv centers each neuron across positions and uses SVD^2/(P-1).
The dimilibi PCA class uses SVD^2/N or SVD^2/P respectively. Reference PCA is
converted to sample-covariance units before the comparison.
For this session the corrections are N/(N-1) = 1.00146843
and P/(P-1) = 1.01010101.
Centering axes differ, so there is no universal scalar conversion between the
spatial and neural covariance spectra. The two ground-truth totals below differ
for that reason as well as the different denominators.

## Measurements

All ratios use each method's matching known-signal PCA target.
The final column sums ranks 10 through 50 inclusive, retaining negative estimates.

| Estimator | Target total | Clean/no smoothing total ratio | Literal S1 total ratio | Independent sign control total ratio | Independent sign control ranks 10-50 ratio |
|---|---:|---:|---:|---:|---:|
| ss_cvpca | 6.90043 | 1.000000 | 0.800 | 1.000 | 1.095 |
| reg_covariances_fixed | 6.90043 | 1.000000 | 0.800 | 1.000 | 1.113 |
| ss_cv | 43.23919 | 1.000000 | 0.810 | 0.965 | 0.765 |

The clean, unsmoothed spectra agree elementwise with their targets to relative
error below 1e-8 (actual maxima are recorded in metrics.json). This rules out a
large multiplicative normalization error for these inputs.

## Why the literal S1 result is low

Residuals estimated by subtracting a finite repeat mean are dependent: they sum
to zero. Their average cross-repeat covariance is negative. In the spatial
covariance methods the exact resulting total-variance bias is minus the mean
residual covariance trace divided by R-1, here -1.38254.
This accounts for their observed roughly 20% deficit; it is not a scaling factor
inside cvPCA.

The independent sign control gives each input's empirical residual an independent
random +/- sign, preserving its spatial/cross-neuron covariance while making
cross-repeat noise covariance zero. The plotted result is the exact average over
scoring signs (computed analytically), averaged over training signs and repeat
assignments. It is not one additional noisy realization.

ss_cvpca and reg_covariances_fixed then recover the total signal variance exactly,
though estimated bases redistribute variance between ranks and distort the very
small tail. ss_cv still loses about 3.5% of the total and about 24% in ranks 10-50.
Its learned left/right cross-SVD bases differ from a PCA basis; its projection
input is also reused in basis fitting and scoring. This check establishes an
estimator/basis effect, not a unique causal attribution or a universal correction.

## Reproduction and scope

Run `python -u -m artifacts.s1_spectrum_validation` in the vrAnalysis Conda
environment. No stored fits or production analysis code are changed.

Outputs: s1_comparison.png, metrics.json, spectra.npz, prepared.npz.
This is a one-session diagnostic, not a validation across all sessions.
