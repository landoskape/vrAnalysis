# Shared Variance Spectrum Methods

Quick reference. All methods produce a mode spectrum; ratios (candidate / reference) give SVR-like scalars.

---

## Population (oracle) — `simulations/shared_variance.py`

| Name | Description | Location | Equation |
|------|-------------|----------|----------|
| **kappa_modes** | Mode-wise geometric mean of two covs | `:334` | `σᵢ(Σ_A^{½} Σ_B Σ_A^{½})` |
| **energy_modes** | Like kappa but no sqrt; sum = tr(AB) | `:340` | `λᵢ(Σ_A^{½} Σ_B Σ_A^{½})` |
| **stimulus_space_kappa_modes** | kappa using pre-cov G (A = G Gᵀ) | `:355` | `σᵢ(Gᵀ Σ_B G)` |
| **stimulus_space_energy_modes** | energy using pre-cov G | `:360` | `λᵢ(Gᵀ Σ_B G)` |
| **population stimstim** | Oracle stim×stim energy; directions from Gᵀ Σ_stim G | `_population_stimstim_comparison :628` | `uᵢᵀ (Gᵀ Σ_stim G) uᵢ`, dirs from same |

---

## Empirical / cross-validated — `simulations/shared_variance.py`

| Name | Description | Location | Equation / protocol |
|------|-------------|----------|---------------------|
| **empirical kappa** | Biased kappa from a single train/test draw pair | `_stim_full_empirical_result :1091` | `σᵢ(G_train^T Σ_full_test G_train)` |
| **cvSER** | 3-fold balanced CV energy ratio; CKA-adjacent | `_stim_full_cvser_result :695` | dirs: `G₀ᵀ F_test G₀`; score: `uᵢᵀ G₁ᵀ F_test G₂ uᵢ`; ref: `energy_modes(F_train, F_test)` |
| **cv_kappa** | 4-draw CV kappa for stim-full | `_stim_full_cv_kappa_result :719` | `fit: SVD(root_stim_train @ root_full_train); score: Uᵀ root_stim_test @ root_full_test Vᵀ` |
| **cv_stimstim** | CV stim×stim energy | `_stim_full_cv_stimstim_result :755` | dirs: `G₀ᵀ cov₃ G₀`; score: `uᵢᵀ G₁ᵀ cov₃ G₂ uᵢ`; ref: `stim_energy(G_t, cov₃)` |
| **cv_variance_scale** | CV variance-scale stim-full; average over all C(4,3) combos | `_stim_full_cv_variance_scale_result :799` | `svd(stim_i^T data_k)`, score `stim_j^T data_k`, norm `√((S-1)(T-1))` |
| **cv_kappa (context)** | 6-draw CV kappa for context-pair pipeline | `_context_cv_kappa_result :913` | same SVD fit/score structure as stim-full cv_kappa |
| **round-the-house (sym)** | Cross-neuron-split round-the-house; symmetrized, clipped ≥0 | `_stim_full_roundhouse_result :1005` / `_context_roundhouse_result :1051` | `λᵢ(G₀₀ G₁₀ᵀ G₁₁ G₀₁ᵀ)` sym |
| **round-the-house (asym)** | Same kernel, not symmetrized; eigenvalues can be negative/complex | same | `λᵢ(G₀₀ G₁₀ᵀ G₁₁ G₀₁ᵀ)` raw |
| **MTFA kappa** | Kappa between MTFA-shrunk (private-stripped) covariances | `_mtfa_kappa_comparison :948` | `kappa(shared(Σ_A), shared(Σ_B))` where shared = max-trace PSD residual |
| **cv_kappa_modes** (utility) | Singular values of root_A @ root_B from independent draws | `:365` | `σᵢ(Σ_A^{½} Σ_B^{½})`, independent draws |
| **rCVPCA** | Regularized cvPCA in position space vs neuron space; ratio = position SVR | `_stim_full_rcvpca_result :861` | `CVPCA(on_stimuli=True).fit(r0_smooth).score(r2, r3)` vs neuron-space analog |

---

## Real-data subspace models — `subspace_analysis/stimspace.py` (`StimSpaceSubspace`)

All require a `fit()` (train) / `score()` (test) call pair.

| Name | Description | Location |
|------|-------------|----------|
| **variance_activity** | Sqrt evals of inner-block kappa; PCA train full-data dirs × test full-cov | `score :468` |
| **variance_placefields** | Same but PCA dirs come from train place fields | `score :469` |
| **variance_placefield_placefield** | PF PCA dirs × test PF cov (stim-stim analog) | `score :470` |
| **cv_variance_squared_placefields** | CV stim-full energy: dirs `u_pf_full` from `G_train^T cov_data_test G_train`; score `uᵀ (G_cv1^T cov_test G_cv2) u` | `score :452` |
| **cv_variance_squared_placefield_placefield** | Same CV structure but against test PF cov (stim-stim) | `score :453` |
| **cv_variance_scale_placefields** | Mirror of `cv_variance_scale` simulation method; C(4,3) combos of smoothed train PF × raw test PF × raw data | `compute_cv_variance_scale :480` |
| **cv_variance_scale_placefields_raw_test** | Same but test PFs are unsmoothed | `compute_cv_variance_scale :558` |
| **variance_scale_placefields_raw_test** | Non-double-CV version (train = test PF fold); biased baseline | `compute_cv_variance_scale :559` |

---

## Real-data subspace models — `subspace_analysis/subspaces.py`

| Class | Name | Description | Location |
|-------|------|-------------|----------|
| `PCASubspace` | **pca** | `var(test^T u_i)` in train PCA directions | `score :105` |
| `SVCASubspace` | **svca** | SVCA on source/target neuron halves; singular values = shared variance | `score :252` |
| `CovCovSubspace` | **covcov** | Inner-block kappa: `√λᵢ(Σ_train^{½} Σ_test Σ_train^{½})` | `score :358` |
| `CovCovCrossvalidatedSubspace` | **covcov_cv** | CV kappa via SVD of `root_cov_train0 @ root_cov_train1` scored on test roots | `score :567` |

---

## Ad-hoc notebook prototypes — `explore_dimensionality.ipynb`

These are exploratory predecessors; canonical versions live in `shared_variance.py`.

| Name | Description | Cell lines | Notes |
|------|-------------|------------|-------|
| **cvsvd_stimfull** | Prototype of `cv_variance_scale`; same SVD-based stim×full CV | `:269`, `:477`, `:905` | Duplicated across 3 notebook sections |
| **inner_block kappa** (stst / stim / full) | PCA-root inner-block kappa in 3 flavors: stim-stim, stim-full, full-full | `:294–299`, `:501–503`, `:929–931` | Same math as `CovCovSubspace.score` |
| **FA kappa** | sklearn `FactorAnalysis` shared-cov estimate, then kappa | `:506–522` | Abandoned in favor of MTFA |
| **MTFA kappa (notebook)** | Notebook-local MTFA + kappa; precursor to `_mtfa_kappa_comparison` | `:527–545` | Uses SCS solver, simpler than production version |
