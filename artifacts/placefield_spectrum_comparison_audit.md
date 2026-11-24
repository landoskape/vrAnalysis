# Placefield spectrum comparison audit

Audited the first stored fit session used by the notebook's default example:
`ATL012.2023-01-20.702`, 397 neurons successfully fitted under all three models,
100 spatial positions, best-sampled environment 1. Used the `even` population
registry, full-recording `std` activity units, neuron-centered covariance, and
the N-1 denominator. No spectrum smoothing or sum normalization was applied
to the numerical comparisons below.

| Quantity | Sum of spectrum |
| --- | ---: |
| Stored pooled-environment ss_cvpca | 11.56065 |
| Same fitted neurons/environment: train/validation/test cvPCA | 9.66579 |
| Shrinkage model, its own eigenbasis | 25.84061 |
| Shrinkage model, projected into the data training basis | 25.84061 |
| Ordinary training-data covariance | 35.94805 |
| Ordinary validation-data covariance | 47.30226 |
| Ordinary test-data covariance | 42.14272 |

The matched cvPCA uses the fit's combined training split for its basis and its
validation and test splits for scoring. It therefore deliberately differs from
the stored four-fold estimator while matching environment, neuron set, units,
and covariance convention. It is not a numerical reproduction of stored ss_cvpca.

Recomputed unregularized generalized-model test R2 agrees with saved test R2
to a maximum absolute difference of 5.7e-14. Its median is -0.31605.
The shrinkage model's recomputed median test R2 is -0.28394.

These measurements do not support another uniform amplitude correction.
The data trace is a cross-covariance of independent observations; the model
trace is an autocovariance of one fitted prediction. Fitted structure that
does not reproduce in held-out data can contribute to the latter without
contributing to the former. The negative median held-out R2 is consistent
with this explanation. The audit does not establish a unique biological or
fitting cause for the low held-out performance.

For a direct descriptive spectrum comparison, compare models with ordinary
data covariance on the same neurons/environment. For a reproducible-variance
comparison, fit models independently on disjoint data and cross-score their
predictions using the same convention as the data estimator. Reusing one
prediction as both repeats is still autocovariance. Scaling either curve to
force overlap would hide the observed difference.

Reproduce with the vrAnalysis interpreter:

```powershell
python -u -m artifacts.audit_placefield_spectrum_comparison
```

The script reads existing stored fits and activity; it does not refit models.
