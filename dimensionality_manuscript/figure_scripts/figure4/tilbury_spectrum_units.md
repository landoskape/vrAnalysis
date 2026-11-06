# Tilbury v11 spectrum comparisons

TilburyFitConfig now defaults to `activity_parameters_name="std"` and
`activity_scaling_scope="full_recording"`. Each neuron's activity is divided
by its sample SD across the full recording before selecting split frames.
No mean subtraction is applied to activity. This matches the scaling scope
used by StimSpaceSpectraConfig. `retained_vr` remains an explicit option.
The parameter grid contains `default`, `raw`, and `std`.

The schema is v11. New preprocessing/config fields change result keys; rerun
the desired v11 variations. Existing stored results are not invalidated or
overwritten by these code changes. Do not combine old v10 retained-frame units
with new v11 full-recording units by relabelling them as the same result.

## Stored outputs

| Keys | Meaning |
| --- | --- |
| `eig_control`, `eig_tilbury`, `eig_shrinkage`, `eig_better` | Historical ordinary neural PCA: center each neuron over positions, SVD squared divided by P. |
| `eig_raw_train`, `eig_raw_test` | Historical ordinary PCA of measured train/test curves, with the same convention. `raw` means unfitted, not an activity preset. |
| `eig_spatial_control`, `eig_spatial_tilbury`, `eig_spatial_shrinkage`, `eig_spatial_better` | Ordinary PCA of fitted curves in the ss_cvpca convention: center each position over neurons, SVD squared divided by N-1. |
| `eig_spatial_raw_train`, `eig_spatial_raw_test` | Ordinary spatial PCA of the measured train/test curves. |
| `ss_cvpca_matched` | Four-fold data cvPCA on the same successful neurons, environment, spatial support, and activity units as the spatial model outputs. |
| `spectrum_roi_indices` | Original session ROI indices for all new spatial/data spectra. |
| `spectrum_dist_centers` | Common spatial support of all new spatial/data spectra. |
| `spectrum_metadata` | Activity preset and scaling scope, frame count, signal type, environment, covariance axes and denominators, and fold/smoothing definitions. |

The native spectra retain their original fitting position support. The new
spatial/data spectra use its intersection with bins observed in each of the
four folds. Both supports and their separate denominators are recorded.
All spectra use the same neurons successful under all three fitted models.

For the matched data estimator, the four original population time groups are
used, with the same `(0,1,2)`, `(0,1,3)`, `(0,2,3)`, `(1,2,3)` assignments as
StimSpace. The first fold defines a smoothed spatial basis; the other two
are unsmoothed scoring folds. `spectrum_smooth_width` controls basis smoothing
and defaults to 5 cm. Fitted model PCA does not use that smoothing.

## Viewer

```python
results_fit = ResultsAggregator(TilburyFitConfig(), store, sessions)
viewer = PlacefieldSpectrumExampleViewer(
    results_fit,
    source_key="matched_ss_cvpca",  # default for v11
    activity_parameters_name="std",
)
viewer.show()
```

No registry or raw activity access is needed for this stored comparison.
`viewer.comparison_metadata` exposes the selected session's stored conventions.

Other sources:

- `matched_pca_train` / `matched_pca_test`: ordinary measured spatial PCA,
  useful for checking the effect of noise without changing units or support.
- `ss_cvpca` (with `results_spectra`): the external pooled StimSpace result,
  labelled "Pooled data". The model conversion path remains available for
  this comparison; environment/neuron populations still differ.
- Existing CVPCA sources remain available when `results_cvpca` is supplied.

Matched sources load stored spatial model spectra directly, without the legacy
P/(P-1) correction or activity conversion. Sum normalization and `force_scale`
remain explicit display options, both off by default.

## Validation

Focused tests cover full-recording versus retained-frame scaling, spatial
PCA/cvPCA normalization, missing-bin intersections, original ROI identity,
metadata, and direct viewer loading. `artifacts/check_tilbury_v11.py` performs
a two-optimizer-step integration check on a real session, round-trips through
a temporary ResultsStore, renders the viewer, and checks the S1 reference.
It does not save scientific fits to the real results store.
