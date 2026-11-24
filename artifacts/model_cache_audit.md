# Trial model cache audit — 2026-11-05

Scope: plain PF, global gain, raw trial gain, square-root trial gain, gain + residual, their trial registries, model-specific joblib caches, and result-store separation. No production cache policy or analysis schema was changed during this audit.

## Findings

1. **Cache construction and reuse work in the tested dependency paths.** The isolated audit exercised the real public cache load/save/check/clear methods and real inheritance methods, with inexpensive optimizer stand-ins. A completely cold PF → G → G+E tree builds all dependencies; a second model instance loads the files without optimizing again. Both internal and external branches pass. Square-root G shares PF optimization but has its own gain cache. Score cache hits avoid refitting. Preferred-method aliases resolve to the same entries as their concrete methods.
2. **The PF selection policy is history-dependent.** Standalone PF uses preferred = Optuna. Trial gain inheritance explicitly asks PF for best. Best selects the lowest validation score among files currently present; it runs the supported methods only if no method cache exists. Thus identical data/code with different cache coverage can select different spatial settings. Adding a better parent cache does not invalidate existing descendants.
3. **This difference occurs in the saved results.** Raw internal G and square-root G spatial settings differ from standalone internal PF in 61/267 environments. All 267 raw G spatial settings match the actual best cached PF candidate. The best candidates are Optuna in 204 environments and grid in 63; two grid choices happen to match the standalone spatial settings. This is the implemented policy, not evidence of stale cache loading. It matters if the intended paper comparison requires exactly the same PF template for standalone PF and G.
4. **G+E inherits the intended G settings.** All 73 completed internal G+E environments available at the audit snapshot match standalone G in num_bins, smooth_width, alpha, and rank. Both internal and external regenerated G+E results also match their own exact cache entries (73 each). The repository tests check exact G prediction equivalence as well.
5. **Invalidation is explicit, not transitive.** Clearing G+E preserves PF, raw G, and square-root G. force_remake on hyperparameter selection rebuilds only the requested model; it does not force dependency reoptimization. Updating PF or G requires explicitly invalidating descendants if they must adopt the new settings. Results-store blobs are separate from model caches.
6. **Registered default models have distinct identities for the tested axes.** Session, spike type, environment, selected ROIs, seed, scaling, train/validation splits, optimization method, internal/external flavor, raw/sqrt/residual flavor, and CV rotation distinguish tested cache keys. Score keys additionally distinguish test split. Gain-residual revision keys include _regularized_v2. Optimization happens on the original train/validation partition; the rotated test folds refit with those fixed settings.
7. **Custom constructor settings are not fully keyed.** Changing gain_regularization or min_frames_per_unit directly does not change the structured-gain cache key. The registered factory models use fixed defaults, so this was not found to affect the current trial grid. Such custom runs require separate cache handling or a future key extension.
8. **Missing caches need not imply missing results.** sync_from_myriad transfers database rows and result blobs by default; model caches require --include-model-caches, and population caches require --include-population-cache. No evidence of erroneous deletion of PF/G caches was found. Historical sync completeness cannot be established from the local files alone.

## Actual file audit

Before the PF+E rerun, exact cache coverage for the 267 scored session–environment combinations was:

| Cache | Present |
|---|---:|
| Internal PF | 267 |
| Internal raw G | 267 |
| Internal square-root G | 267 |
| External PF | 246 |
| External raw G | 0 |
| Internal/external G+E | 0 |

The 21 missing external PF entries span 10 sessions. Missing external raw G caches explain why every external branch needs gain optimization; only the missing PF dependencies trigger additional PF grid/Optuna optimization.

A separate comparison of saved results to their exact preferred-method cache entries found all existing internal model entries matching. Missing external-global-gain and RRR entries were also observed (68 and 41 respectively), consistent with incomplete local model-cache coverage rather than an external-PF-only problem.

Of four external PF entries rebuilt by the snapshot, two reproduce old saved settings to floating-point precision (roughly 1e-15). Two select materially different hyperparameters from the old saved result. Rebuilding a missing optimization cache therefore must not be treated as exact replay of saved analyses. Existing standalone PF result blobs were not replaced.

## Validation

- 63 focused repository tests passed: gain-residual, square-root gain, structured gain, trial registries/fold reuse, and scoped cache invalidation.
- 15 isolated cache audit checks passed. See audit_model_cache_dependencies.py and audit_model_cache_dependencies.log.
- The isolated audit writes only to a temporary directory and removes it on completion. It does not modify production caches.
- The current batch had 44 saved sessions per gain-residual configuration and zero recorded errors at the read-only audit snapshot. Its tool-bound process had stopped; a detached resume was launched for missing jobs only, with no reinvalidation. See resume_trial_residual.log, resume_trial_residual.stderr.log, and eventual resume_trial_residual_status.json.

## Submission implication

The core cache plumbing passed these checks. Incomplete cache transfer is a plausible explanation for local gaps, not a proven historical cause. The two policy considerations to resolve explicitly are (a) whether plain PF and trial G must use identical spatial hyperparameters, given preferred versus best, and (b) freezing the saved selected hyperparameters for exact reproduction instead of relying on the contents of a mutable cache directory.

No automatic rerun of other models, change to optimization policy, or schema-version change was made for this audit.

