# Shared Variance Math Overview

This document gives a detailed, equation-level walkthrough of every analytical
object in the shared-variance system — generative model, covariance algebra,
all four metrics (SVR, cvSER, StimStim, CV-Kappa), the SubspaceGeometry
diagnostics, and the two companion systems (CVPCA, StimSpace). It also explains
how the simulation atlas cases are wired up and what each metric actually
measures. Written as a reference for `explore_dimensionality.ipynb`.

---

## 1. Generative Model (`StimFullGenerator`)

All `stim_full.*` atlas cases are built on the same latent model:

$$x_t = g(s_t) + h(n_t) + \varepsilon_t, \quad x_t \in \mathbb{R}^N$$

### 1.1 Stimulus component $g(s_t)$

**Stimulus space** $\mathbf{U}_s \in \mathbb{R}^{N \times D}$ — an orthonormal matrix
whose $D$ columns are the neural directions that carry stimulus information.
Generated as a random orthonormal frame via QR decomposition.

**Stimulus spectrum** $\boldsymbol{\lambda}_s \in \mathbb{R}^D$ with entries

$$[\boldsymbol{\lambda}_s]_d = d^{-\alpha_s}, \quad d = 1, \ldots, D$$

The power-law exponent $\alpha_s$ (`alpha_stim`) controls how steeply variance
falls across stimulus dimensions.

**Stimulus latents** $\mathbf{L} \in \mathbb{R}^{D \times S}$ — a tight frame.
When $D = S$ this is the identity. When $D < S$ it is a random orthonormal matrix
scaled by $\sqrt{S}$ so that $\mathbf{L}\mathbf{L}^T = S \mathbf{I}_D$. This
scaling ensures that the outer product of the $S$ stimulus-response vectors
exactly reproduces the intended covariance (see below).

Each stimulus $s \in \{1, \ldots, S\}$ has a fixed mean response:

$$g(s) = \mathbf{U}_s \,\mathrm{diag}(\sqrt{\boldsymbol{\lambda}_s})\, \mathbf{l}_s$$

where $\mathbf{l}_s$ is the $s$-th column of $\mathbf{L}$. The full
$(N \times S)$ matrix of stimulus responses is therefore

$$\mathbf{R} = \mathbf{U}_s \,\mathrm{diag}(\sqrt{\boldsymbol{\lambda}_s})\, \mathbf{L}$$

Because $\mathbf{L}\mathbf{L}^T = S\mathbf{I}_D$, the *row-wise* covariance of
$\mathbf{R}$ is

$$\mathrm{cov}_s(\mathbf{R}) = \frac{1}{S-1} \mathbf{R}_c \mathbf{R}_c^T
\approx \frac{S}{S-1} \mathbf{U}_s \,\mathrm{diag}(\boldsymbol{\lambda}_s)\, \mathbf{U}_s^T
\longrightarrow \Sigma_{\mathrm{stim}} := \mathbf{U}_s \,\mathrm{diag}(\boldsymbol{\lambda}_s)\, \mathbf{U}_s^T$$

**Important:** `true_covariance()` explicitly computes `np.cov(stim_responses)` — it does not use the formula above. The tight-frame construction ensures these agree up to a $(S-1)/S$ Bessel correction that vanishes as $S$ grows. The net effect is that $\Sigma_\mathrm{stim}$ is rank $D$ and has a power-law spectrum with exponent $\alpha_s$.

### 1.2 Nuisance component $h(n_t)$

$$h(n_t) = \nu \cdot \mathbf{U}_n \,\mathrm{diag}(\sqrt{\boldsymbol{\lambda}_n})\, z_t, \quad z_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_H)$$

where $\nu$ is `nuisance_scale` and $\boldsymbol{\lambda}_n$ is a power-law
spectrum with exponent $\alpha_n$.

**Nuisance alignment** — four options for how $\mathbf{U}_n$ relates to $\mathbf{U}_s$:

| `nuisance_alignment` | Construction |
|---|---|
| `"orthogonal"` | $\mathbf{U}_n$ drawn from the orthogonal complement of $\mathbf{U}_s$ |
| `"random"` | $\mathbf{U}_n$ drawn uniformly from Stiefel manifold (independent of $\mathbf{U}_s$) |
| `"aligned"` | $\mathbf{U}_n$ copies the first $\min(D,H)$ columns of $\mathbf{U}_s$ |
| `"angle"` | first $\min(D,H)$ columns of $\mathbf{U}_s$ are rotated by `nuisance_angle` radians |

The true nuisance covariance is

$$\Sigma_\mathrm{nuisance} = \nu^2 \,\mathbf{U}_n \,\mathrm{diag}(\boldsymbol{\lambda}_n)\, \mathbf{U}_n^T$$

### 1.3 Private noise $\varepsilon_t$

Each neuron $n$ has its own noise variance drawn once at construction:

$$\sigma_n^2 \sim \mathrm{Exp}(\texttt{noise\_scale}), \quad
[\Sigma_\varepsilon]_{nn} = \sigma_n^2$$

The resulting diagonal matrix is $\Sigma_\varepsilon = \mathrm{diag}(\boldsymbol{\sigma}^2)$.
With `noise_scale = 0` all entries are exactly zero (no per-neuron noise).

### 1.4 Full covariance decomposition

$$\Sigma_\mathrm{full} = \Sigma_\mathrm{stim} + \Sigma_\mathrm{nuisance} + \Sigma_\varepsilon$$

Because all three are PSD and $\Sigma_\mathrm{stim} \preceq \Sigma_\mathrm{full}$
(Loewner order), the SVR is guaranteed to lie in $[0, 1]$ when $A = \Sigma_\mathrm{stim}$
and $B = \Sigma_\mathrm{full}$.

### 1.5 Data generation

At each call to `generate(num_samples)`:

1. Draw $T$ stimulus indices $s_t \sim \mathrm{Uniform}\{1,\ldots,S\}$.
2. Look up $g(s_t) = \mathbf{R}[:, s_t]$ (column indexing into the fixed response matrix).
3. Draw nuisance loadings $z_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_H)$ and form $h_t$.
4. Draw per-neuron noise $\varepsilon_t$.
5. Return $x_t = g(s_t) + h_t + \varepsilon_t$.

The `rotation_angle` parameter rotates both $\mathbf{U}_s$ and $\mathbf{U}_n$ before
generating responses, simulating a test-set coordinate shift (session-to-session drift).

---

## 2. Core Matrix Operators (`operators.py`)

All kappa computations reduce to three operations.

### 2.1 Matrix square root `sqrtm_spd`

For a symmetric PSD matrix $A = V \Lambda V^T$:

$$A^{1/2} = V \,\mathrm{diag}(\sqrt{\max(\lambda_i, 0)})\, V^T$$

Implemented via `np.linalg.eigh` (stable for symmetric matrices).
Negative eigenvalues from numerical noise are clamped to zero.

### 2.2 Inverse matrix square root `invsqrtm_spd`

$$A^{-1/2} = V \,\mathrm{diag}(1/\sqrt{\max(\lambda_i, \epsilon)})\, V^T$$

Used in `geometric_mean_spd` but not in the main kappa pipeline.

### 2.3 Root sandwich `root_sandwich`

$$\mathrm{rootsandwich}(A, B) = A^{1/2} B A^{1/2}$$

The eigenvalues of this expression equal the eigenvalues of $A B$ when both are
PSD (they are related by a similarity transform via $A^{1/2}$).

---

## 3. The $\kappa$ Operator and Stimulus-Space Kernel

### 3.1 Definition

For two PSD matrices $A, B \in \mathbb{R}^{N \times N}$:

$$\kappa(A, B) := \|A^{1/2} B^{1/2}\|_* = \sum_i \sigma_i(A^{1/2} B^{1/2})$$

where $\|\cdot\|_*$ is the nuclear norm (sum of singular values). This is
equivalently

$$\kappa(A, B) = \sum_i \sqrt{\lambda_i(A^{1/2} B A^{1/2})} = \mathrm{tr}\sqrt{A^{1/2} B A^{1/2}}$$

The mode-wise values $\kappa_i(A, B) = \sqrt{\lambda_i(A^{1/2} B A^{1/2})}$ are
the **kappa modes** and form the sequence stored in `ModeComparison.candidate_modes`.

#### Implementation: `kappa_modes(A, B)`

```python
Aroot = sqrtm_spd(symmetrize(A))
return sqrt(sorted_eigenvalues(Aroot @ symmetrize(B) @ Aroot))
```

The symmetrization (`0.5*(M + M.T)`) guards against tiny numerical
asymmetries that can produce complex eigenvalues.

#### Geometric interpretation

Let $\mathcal{E}_A = \{A^{1/2} u : \|u\|=1\}$ be the ellipsoid defined by $A$,
and $\mathcal{E}_B$ similarly. Then:

$$\kappa_1(A,B) = \max_{\|u\|=\|v\|=1} \langle A^{1/2} u,\, B^{1/2} v \rangle$$

and each subsequent mode solves the same problem under orthogonality constraints
on the singular vectors. The modes find pairs of directions — one in $A$'s
covariance structure, one in $B$'s — that are maximally correlated.

### 3.2 Energy modes (squared-amplitude scale)

The **energy modes** skip the square root:

$$\omega_i(A, B) = \lambda_i(A^{1/2} B A^{1/2})$$

Their sum is $\mathrm{tr}(A B)$, the Frobenius inner product. When $A = B$:
$\omega_i(A, A) = \lambda_i^2(A)$, so energy modes over-weight the high-variance
modes relative to kappa modes (which give $\kappa_i(A,A) = \lambda_i(A)$).

#### Implementation: `energy_modes(A, B)`

```python
Aroot = sqrtm_spd(symmetrize(A))
return maximum(sorted_eigenvalues(Aroot @ symmetrize(B) @ Aroot), 0.0)
```

### 3.3 Stimulus-space kernel $K_B(A)$

Let $G_A \in \mathbb{R}^{N \times S}$ be the **pre-covariance** matrix, i.e.
the centered, scaled data matrix satisfying $G_A G_A^T = A$. Define:

$$K_B(A) := G_A^T B G_A \in \mathbb{R}^{S \times S}$$

**Key identity:** $K_B(A)$ has the same nonzero eigenvalues as $A^{1/2} B A^{1/2}$.

*Proof:* Let $X = B^{1/2} G_A$. Then $X^T X = G_A^T B G_A = K_B(A)$ and
$X X^T = B^{1/2} A B^{1/2}$. Since $X^T X$ and $X X^T$ share nonzero eigenvalues,
and $B^{1/2} A B^{1/2}$ and $A^{1/2} B A^{1/2}$ also share eigenvalues (symmetric
similarity), we have $\lambda_i(K_B(A)) = \lambda_i(A^{1/2} B A^{1/2})$.

Therefore $\kappa(A, B) = \mathrm{tr}\sqrt{K_B(A)}$.

#### Why this matters for cross-validation

$A^{1/2}$ requires $A$ to be PSD, but the empirical estimate
$\tilde{\Sigma}_\mathrm{stim}$ — computed from stimulus-averaged responses — can
be replaced by its pre-covariance $G$ without forming the full PSD matrix. This
allows independent folds to provide $G_1$ and $G_2$ as two views of the same
underlying space, yielding an **unbiased** bilinear estimator.

#### Pre-covariance construction: `_precov(data)`

For a data matrix $X \in \mathbb{R}^{N \times T}$ (neurons × samples):

$$G = \frac{X - \bar{X}}{\sqrt{T-1}}, \quad \text{so that} \quad G G^T = \mathrm{cov}(X)$$

Applied to the $(N \times S)$ matrix of per-stimulus mean responses, we get
$G_\mathrm{stim}$ satisfying $G_\mathrm{stim} G_\mathrm{stim}^T = \tilde{\Sigma}_\mathrm{stim}$.

#### Implementation: `stimulus_space_kappa_modes(G_A, B)`

```python
return sqrt(sorted_eigenvalues(G_A.T @ symmetrize(B) @ G_A))
```

This is numerically cheaper than forming the full $N \times N$ matrix
$A^{1/2} B A^{1/2}$ when $S \ll N$.

---

## 4. Population Block: What the "True" Spectra Show

`_stim_full_population_block(gen)` operates entirely on **population-level**
(infinite-sample) covariances returned by `gen.true_covariance()`.

### 4.1 Population stim-full kappa modes

$$\kappa_i(\Sigma_\mathrm{stim}, \Sigma_\mathrm{full}) = \sqrt{\lambda_i(G_\mathrm{stim}^T \Sigma_\mathrm{full}\, G_\mathrm{stim})}$$

where $G_\mathrm{stim} = \mathrm{precov}(\mathbf{R})$ is the pre-covariance of
the true stimulus responses ($S$ columns, no noise). This is compared against

$$\kappa_i(\Sigma_\mathrm{full}, \Sigma_\mathrm{full}) = \lambda_i(\Sigma_\mathrm{full})$$

because when $A = B$ the sandwich $A^{1/2} A A^{1/2} = A^2$ and
$\sqrt{\lambda_i(A^2)} = \lambda_i(A)$.

**Verified identity** (`extras["stimulus_space_modes_match_covariance_modes"]`):
The population stimulus-space kappa modes equal the kappa modes computed from
$\Sigma_\mathrm{stim}$ directly via `kappa_modes(sigma_stim, sigma_full)`. This
confirms that the stimulus-space kernel is exact in the noiseless population limit.

### 4.2 Population energy modes

$$\omega_i(\Sigma_\mathrm{stim}, \Sigma_\mathrm{full}) = \lambda_i(G_\mathrm{stim}^T \Sigma_\mathrm{full}\, G_\mathrm{stim})$$

These are the eigenvalues *without* taking the square root.

### 4.3 Population StimStim (`stimstim` field)

This is the "oracle" analog of the CV stim-stim measurement.

**Kernel:** $K_\mathrm{ss} = G_\mathrm{stim}^T \Sigma_\mathrm{stim}\, G_\mathrm{stim}$

**Directions:** $\{u_i\}$ are the eigenvectors of $K_\mathrm{ss}$.

**Candidate energy modes:**
$e_i = u_i^T K_\mathrm{ss} u_i = \lambda_i(K_\mathrm{ss})$

**Reference energy modes:** `stimulus_space_energy_modes(G_stim, Sigma_stim)` which
computes $\lambda_i(G_\mathrm{stim}^T \Sigma_\mathrm{stim}\, G_\mathrm{stim})$.

Since both candidate and reference use the *same* kernel, the candidate modes
equal the reference modes, so the ratio is exactly 1 in the noiseless population.
The stim-stim measurement is therefore most informative in the empirical/CV form,
where noise splits the two estimates.

---

## 5. Empirical Measurement (Finite Samples)

When `num_samples` is provided, `_stim_full_empirical_result` draws finite data
and computes the following.

### 5.1 Data splits

One call to `gen.generate(num_samples)` yields **train** data; a second
independent call (optionally with `rotation_angle`) yields **test** data.

$$\tilde{\Sigma}_\mathrm{full}^\mathrm{train} = \mathrm{cov}(X_\mathrm{train}), \quad
\tilde{\Sigma}_\mathrm{full}^\mathrm{test} = \mathrm{cov}(X_\mathrm{test})$$

The train data is also used to compute per-stimulus means:

$$\bar{x}_s^\mathrm{train} = \frac{1}{|\mathcal{F}_s|} \sum_{t: s_t=s} x_t$$

### 5.2 Empirical SVR (the `kappa` field in `EmpiricalBlock`)

$$\text{SVR} = \frac{\kappa(\tilde{\Sigma}_\mathrm{stim}^\mathrm{train},\, \tilde{\Sigma}_\mathrm{full}^\mathrm{test})}{\kappa(\tilde{\Sigma}_\mathrm{full}^\mathrm{train},\, \tilde{\Sigma}_\mathrm{full}^\mathrm{test})}$$

**Numerator:** `stimulus_space_kappa_modes(G_train, Sigma_full_test)`,
where $G_\mathrm{train} = \mathrm{precov}(\bar{X}^\mathrm{train})$.

**Denominator:** `kappa_modes(Sigma_full_train, Sigma_full_test)`.

**Bias note:** The empirical stim mean is:
$$\mathbb{E}[\bar{x}_s] = g(s) + \frac{1}{m_s}\sum_{t: s_t=s} h(n_t) + \frac{1}{m_s}\sum_{t: s_t=s} \varepsilon_t$$

So the expected value of the empirical stim covariance is:
$$\mathbb{E}[\tilde{\Sigma}_\mathrm{stim}] \approx \Sigma_\mathrm{stim} + \frac{1}{m_s}\Sigma_\mathrm{nuisance} + \frac{1}{m_s}\Sigma_\varepsilon$$

This is always $\succeq \Sigma_\mathrm{stim}$, so the empirical SVR is an
upward-biased estimator of the population SVR. More trials per stimulus
($m_s$ larger) reduce this bias. The $1/m_s$ attenuation also means nuisance
structures that dominate the full covariance are suppressed in the numerator.

**Upper bound:** Because $\tilde{\Sigma}_\mathrm{stim} \preceq \tilde{\Sigma}_\mathrm{full}$
(law of total covariance: $\mathrm{cov}(X) = \mathrm{cov}(\bar{X}_s) +
\mathbb{E}_s[\mathrm{cov}(X|s)]$), the empirical SVR is also bounded above by 1
when train and test covariances come from the same distribution.

---

## 6. Cross-Validated Shared Energy Ratio (cvSER)

`_stim_full_cvser_result` implements the 3-fold stimulus-balanced CV.

### 6.1 Stimulus-balanced folds

The train data is partitioned into 3 folds such that each fold contains
approximately the same number of trials per stimulus. Within each stimulus,
trials are randomly permuted and split.

### 6.2 Per-fold pre-covariances

For each fold $k \in \{0, 1, 2\}$, compute the per-stimulus mean responses
restricted to fold $k$, then form the pre-covariance:

$$G^{(k)} = \mathrm{precov}(\bar{X}^{(k)}), \quad
G^{(k)} \in \mathbb{R}^{N \times S}$$

### 6.3 Direction learning (fold 0)

The direction kernel uses fold 0's stimulus pre-covariance projected through
the **test** full covariance:

$$K_\mathrm{dir} = (G^{(0)})^T \,\tilde{\Sigma}_\mathrm{full}^\mathrm{test}\, G^{(0)}$$

Eigenvectors $\{u_i\}$ of $K_\mathrm{dir}$ are the **energy directions** (largest
eigenvalue first). These identify the modes of stimulus structure that best predict
the test full covariance.

**Why test covariance for directions?** Using an independent test covariance for
direction learning prevents circular over-fitting. The directions can then be
scored on the remaining independent folds 1 and 2.

### 6.4 Cross-validated scoring (folds 1 and 2)

The asymmetric cross-covariance kernel:

$$K_{12} = (G^{(1)})^T \,\tilde{\Sigma}_\mathrm{full}^\mathrm{test}\, G^{(2)}$$

Candidate (cvSER numerator per mode):

$$w_i^\mathrm{cv} = u_i^T K_{12} u_i = (G^{(1)} u_i)^T \tilde{\Sigma}_\mathrm{full}^\mathrm{test} (G^{(2)} u_i)$$

This is an **unbiased** estimator of $\lambda_i(G^T \Sigma_\mathrm{stim} G \cdot G^T \Sigma_\mathrm{full} G)$
in the population limit, because $G^{(1)}$ and $G^{(2)}$ are independent — their
noise does not correlate. Values can be negative (when stimulus noise in folds 1 and
2 is anti-correlated by chance), which is a feature rather than a bug: it provides an
honest noise floor.

### 6.5 Reference energy (normalization)

$$\mathrm{reference}_i = [\mathrm{energy\_modes}(\tilde{\Sigma}_\mathrm{full}^\mathrm{train}, \tilde{\Sigma}_\mathrm{full}^\mathrm{test})]_i$$

The sum of reference modes equals $\mathrm{tr}(\tilde{\Sigma}_\mathrm{full}^\mathrm{train} \tilde{\Sigma}_\mathrm{full}^\mathrm{test})$,
an unbiased estimator of $\mathrm{tr}(\Sigma_\mathrm{full}^2)$.

### 6.6 cvSER ratio

$$\text{cvSER} = \frac{\sum_i w_i^\mathrm{cv}}{\sum_i [\mathrm{energy\_modes}(\tilde{\Sigma}_\mathrm{full}^\mathrm{train}, \tilde{\Sigma}_\mathrm{full}^\mathrm{test})]_i}$$

This is stored in `empirical.cv_energy`.

#### Scale difference from SVR

| Metric | Numerator | Weight on mode $i$ |
|---|---|---|
| SVR | $\sum_i \kappa_i$ | $\sqrt{\lambda_i(\Sigma_\mathrm{stim})} \cdot \sqrt{\lambda_i(\Sigma_\mathrm{full})}$ |
| cvSER | $\sum_i \omega_i$ | $\lambda_i(\Sigma_\mathrm{stim}) \cdot \lambda_i(\Sigma_\mathrm{full})$ |

cvSER is biased toward the highest-variance modes. When a single large mode
dominates, cvSER and SVR can diverge dramatically.

---

## 7. CV Kappa (`cv_kappa` field)

`_stim_full_cv_kappa_result` implements the cross-validated kappa estimator that
removes the finite-sample upward bias in the numerator of SVR.

### 7.1 Four independent data draws

| Draw | Used for |
|---|---|
| `data_train0` | stimulus covariance root (candidate, train) |
| `data_train1` | full covariance root (reference, train) |
| `data_test0` | stimulus covariance root (candidate, test) |
| `data_test1` | full covariance root (reference, test) |

### 7.2 Matrix square roots

$$R_{\mathrm{stim}}^\mathrm{train} = \sqrt{\mathrm{cov}(\bar{X}_0^\mathrm{train})}, \quad
R_{\mathrm{stim}}^\mathrm{test} = \sqrt{\mathrm{cov}(\bar{X}_0^\mathrm{test})}$$

$$R_{\mathrm{full}}^\mathrm{train} = \sqrt{\tilde{\Sigma}_\mathrm{full}^\mathrm{train_1}}, \quad
R_{\mathrm{full}}^\mathrm{test} = \sqrt{\tilde{\Sigma}_\mathrm{full}^\mathrm{test_1}}$$

### 7.3 Fit-score structure: `_cv_kappa_fit_score`

**Train:** Compute the SVD of $R_A^\mathrm{train} R_B^\mathrm{train} = U S V^T$.

**Score:** Project the test cross-product onto the train singular vectors:
$$\hat{s}_i = u_i^T (R_A^\mathrm{test} R_B^\mathrm{test}) v_i$$

This mirrors SVCA: fit shared directions on train data, evaluate on independent
test data.

**Candidate CV-kappa:** `_cv_kappa_fit_score(R_stim_train, R_full_train1, R_stim_test, R_full_test1)`

**Reference CV-kappa:** `_cv_kappa_fit_score(R_full_train0, R_full_train1, R_full_test0, R_full_test1)`

### 7.4 Why this removes bias

In the non-CV form, both roots in $R_\mathrm{stim} R_\mathrm{full}$ come from
data that includes finite-sample stimulus noise. The resulting singular values are
inflated relative to the population. By using **four independent draws** (two for
directions, two for scoring), the cross-validated scores have expectation equal to
the population singular values, at the cost of higher variance and requiring 4×
the data.

---

## 8. CV StimStim (`cv_stimstim` field)

`_stim_full_cv_stimstim_result` measures the self-consistency of the stimulus
representation — how well one estimate of $\Sigma_\mathrm{stim}$ predicts another.

### 8.1 Five independent draws

| Draw | Used for |
|---|---|
| `data_0` | direction kernel (fold 0 stimulus means) |
| `data_1` | CV score numerator (fold 1 stimulus means) |
| `data_2` | CV score numerator (fold 2 stimulus means) |
| `data_3` | reference covariance (for directions and scoring) |
| `data_t` | reference normalization (fresh independent draw) |

### 8.2 Direction kernel

Compute $\mathrm{cov}_3 = \mathrm{cov}(\bar{X}^{(3)})$, the covariance of
per-stimulus means from draw 3. Then:

$$K_\mathrm{dir} = G_0^T \,\mathrm{cov}_3\, G_0$$

where $G_0 = \mathrm{precov}(\bar{X}^{(0)})$. Eigenvectors $\{u_i\}$ define the
stimulus-space directions.

### 8.3 CV bilinear score

$$w_i^\mathrm{stim} = u_i^T \,(G_1^T \,\mathrm{cov}_3\, G_2)\, u_i = (G_1 u_i)^T \,\mathrm{cov}_3\, (G_2 u_i)$$

This estimates how much variance the direction $u_i$ carries in $\Sigma_\mathrm{stim}$,
using three independent stimulus estimates to separate directions from amplitudes.

### 8.4 Reference normalization

$$\mathrm{ref}_i = [\mathrm{stimulus\_space\_energy\_modes}(G_t, \mathrm{cov}_3)]_i = \lambda_i(G_t^T \,\mathrm{cov}_3\, G_t)$$

This is a symmetric stimulus-space energy measurement from a fresh draw. The
reference sum $\sum_i \mathrm{ref}_i = \mathrm{tr}(G_t^T \,\mathrm{cov}_3\, G_t) \approx \mathrm{tr}(\Sigma_\mathrm{stim}^2)$.

**Key difference from cvSER:** cvSER normalizes against full-covariance energy;
cvStimStim normalizes against *stimulus* covariance energy. The ratio therefore
measures self-consistency of $\Sigma_\mathrm{stim}$, not its fraction of
$\Sigma_\mathrm{full}$.

### 8.5 `as_variance_scale()` conversion

The stim-stim modes are on the energy (squared) scale:
$w_i \approx \lambda_i(\Sigma_\mathrm{stim})^2$. To compare against the true
eigenspectrum $\lambda_i(\Sigma_\mathrm{stim})$, convert to variance scale:

$$\tilde{w}_i = \sqrt{\max(w_i, 0)}$$

This is what `result.empirical.cv_stimstim.as_variance_scale()` returns.
A well-estimated stim-stim spectrum should track the true $\Sigma_\mathrm{stim}$
eigenvalues shown in `result.population.geometry.candidate_spectrum`.

---

## 9. `ModeComparison` and Ratio Calculations

`ModeComparison` is a frozen dataclass holding:
- `candidate_modes`: the per-mode values for the candidate (stim or context A)
- `reference_modes`: the per-mode values for the reference (full or context B)
- `ratio`: $\sum_i \mathrm{candidate}_i / \sum_i \mathrm{reference}_i$
- `cumulative_ratio`: $\mathrm{cumsum}(\mathrm{candidate}) / \mathrm{cumsum}(\mathrm{reference})$
- `metric`: `"kappa"` or `"energy"` (records the scale of the stored modes)

The `cumulative_ratio` at mode $k$ answers: "what fraction of the total shared
signal is explained by the top $k$ modes?" A curve that flattens quickly implies
low-dimensional alignment.

---

## 10. `SubspaceGeometry` Diagnostics

`_build_geometry(candidate_cov, reference_cov)` computes the following using
eigendecompositions of both covariances.

Let $\Sigma_A = V_A \Lambda_A V_A^T$ and $\Sigma_B = V_B \Lambda_B V_B^T$.

### 10.1 Cross-subspace overlap

**Reference-on-candidate** — how much variance of $\Sigma_B$ falls on each
*candidate* eigenvector $v_i^A$:

$$[\mathrm{ref\_on\_cand}]_i = \sum_j \langle v_i^A, v_j^B \rangle^2 \lambda_j^B
= [(V_A^T V_B)^{\circ 2} \boldsymbol{\lambda}_B]_i$$

This is stored in `reference_on_candidate_overlap` and plotted in the notebook
as "reference overlap" against the candidate spectrum. If it tracks the candidate
spectrum closely, the reference covariance is concentrating its variance along
the same directions as the candidate.

**Candidate-on-reference** — symmetric version: how much $\Sigma_A$ variance
lies on each reference eigenvector.

### 10.2 CKA (matrix form)

$$\mathrm{CKA}(\Sigma_A, \Sigma_B) = \frac{\mathrm{tr}(\Sigma_A \Sigma_B)}{\sqrt{\mathrm{tr}(\Sigma_A^2) \mathrm{tr}(\Sigma_B^2)}}$$

This is the cosine similarity of the two matrices under the Frobenius inner
product $\langle A, B \rangle_F = \mathrm{tr}(A^T B)$. It is always in $[-1,1]$
and equals 1 only when $\Sigma_A = c \Sigma_B$ for some scalar $c > 0$.

**CKA vs SVR:** CKA normalizes by the total variance in each matrix separately
(geometric mean), whereas SVR normalizes only by the reproducibility of the
reference matrix. CKA is scale-invariant; SVR is not. Two matrices that share
geometry but differ greatly in scale will have CKA $\approx 1$ but SVR $<1$.

### 10.3 Traces

`trace_candidate`, `trace_reference`: total variance of each matrix. When
`trace_nuisance` and `trace_eps` are available (stim-full pipeline), they
give the variance budget breakdown.

---

## 11. Context-Pair Pipeline (`context.*`, `shared_space.*`)

### 11.1 Direct covariance-to-covariance comparison

For context and shared-space cases there is no "stimulus space" — both candidate
and reference are full-rank covariance matrices. The pipeline applies kappa
directly:

**Population block:**
$$\kappa_i(\Sigma_A, \Sigma_B) = \sqrt{\lambda_i(\Sigma_A^{1/2} \Sigma_B \Sigma_A^{1/2})}$$

**Reference modes:**
$$\kappa_i(\Sigma_B, \Sigma_B) = \lambda_i(\Sigma_B)$$

SVR is unbounded from above here (the Loewner nesting argument does not apply),
as illustrated by `context.spectrum_mismatch` which produces SVR $\approx 1.94$.

### 11.2 `CovariancePairGenerator` geometry options

| `geometry` | Construction of $\Sigma_B$ |
|---|---|
| `"same"` | Shares eigenvectors with $\Sigma_A$; possibly fewer |
| `"orthogonal"` | $\Sigma_B$ basis drawn from the orthogonal complement of $\Sigma_A$ basis |
| `"random"` | $\Sigma_B$ basis drawn independently (random overlap by chance) |
| `"angle"` | First $\min(r_A, r_B)$ axes of $\Sigma_A$ basis rotated by `angle` |
| `"partial"` | First `shared_rank` axes shared; remaining axes orthogonal to $\Sigma_A$ |

Both covariances have power-law spectra with exponents $\alpha_A$, $\alpha_B$ and
can be independently scaled. Samples are generated as
$x = \mathbf{U} \,\mathrm{diag}(\sqrt{\boldsymbol{\lambda}})\, z$, $z \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### 11.3 `SharedSpaceGenerator` latent model

Candidate and reference share a common $D_\mathrm{shared}$-dimensional subspace
but have independent private subspaces:

$$\Sigma_A = \mathbf{U}_\mathrm{sh} \,\mathrm{diag}(\boldsymbol{\lambda}_\mathrm{sh,1})\, \mathbf{U}_\mathrm{sh}^T
+ r^2 \,\mathbf{U}_{p1} \,\mathrm{diag}(\boldsymbol{\lambda}_{p1})\, \mathbf{U}_{p1}^T$$

$$\Sigma_B = \mathbf{U}_\mathrm{sh} \,\mathrm{diag}(\boldsymbol{\lambda}_\mathrm{sh,2})\, \mathbf{U}_\mathrm{sh}^T
+ r^2 \,\mathbf{U}_{p2} \,\mathrm{diag}(\boldsymbol{\lambda}_{p2})\, \mathbf{U}_{p2}^T$$

where $r$ is `private_ratio`, the private subspaces are orthogonal to the shared
space and to each other, and both spectra are power-law. The `shuffle_shared`
option permutes the second condition's shared eigenvalues so the two conditions
have the same shared subspace but assign variance in a different order.

**Empirical CV-kappa** for context pairs uses the same `_cv_kappa_fit_score`
machinery as the stim-full pipeline, but here all roots come from covariance of
raw activity rather than stimulus means.

---

## 12. CVPCA (`CVPCAConfig`)

CVPCA is a separate system that measures the dimensionality of **place field
covariance** via 3-fold cross-validation, operating entirely in the spatial-bin
domain rather than on raw trial activity.

### 12.1 Place field construction

Trials are split into three folds. For each fold $k$ and spatial bin $b$, the
place field is the average firing rate across all trials in that fold when the
animal is in bin $b$:

$$[P^{(k)}]_{n,b} = \frac{1}{|\mathcal{T}_{kb}|} \sum_{t \in \mathcal{T}_{kb}} x_{nt}$$

This yields a matrix $P^{(k)} \in \mathbb{R}^{N \times B}$ (neurons × bins) for
each fold. Entries where the animal never visited bin $b$ in fold $k$ are NaN.

### 12.2 Normalization options

When `normalize=True`, each neuron is divided by its peak response across all
folds and bins:

$$\hat{P}^{(k)} = P^{(k)} / \max_{b,k} [P^{(k)}]_{n,b}$$

### 12.3 Smoothing variants

The system runs five parallel CVPCA variants that differ only in what smoothing
is applied before fitting:

| Variant | Fit data | Score data |
|---|---|---|
| `org` (no smooth) | $P^{(0)}$ raw | $P^{(1)}, P^{(2)}$ raw |
| `org_smooth` (optimized smooth, all folds) | $P^{(0)}$ smoothed | $P^{(1)}, P^{(2)}$ smoothed |
| `org_fixed_smooth` (fixed smooth, all folds) | $P^{(0)}$ fixed smooth | $P^{(1)}, P^{(2)}$ fixed smooth |
| `reg` (R-CVPCA, optimized smooth on fit only) | $P^{(0)}$ optimized | $P^{(1)}, P^{(2)}$ raw |
| `reg_fixed` (R-CVPCA, fixed smooth on fit only) | $P^{(0)}$ fixed smooth | $P^{(1)}, P^{(2)}$ raw |

Smoothing uses a Gaussian kernel applied along the spatial dimension (axis 1).
For `reg` variants, the smoothing width is optimized by `RegularizedCVPCA` to
maximize CV score on the validation fold.

### 12.4 CVPCA scoring

For each fold assignment (0→fit, 1→score, 2→score), the CVPCA model:

1. **Fit:** Compute PCA on $P^{(0)}$ (or its smoothed version). Extract
   eigenvectors $\{v_i\}$.
2. **Score:** For each PC $i$, compute the **covariance** between projections on
   the two held-out folds:
   $$\hat{\lambda}_i = \mathrm{cov}(P^{(1)} v_i,\; P^{(2)} v_i)$$
   This equals $v_i^T P^{(1)} P^{(2)T} v_i / (B-1)^2$ and is an unbiased
   estimator of the true per-component variance $\lambda_i$ of the place field
   covariance.

3. **Train/test variances:** Also recorded separately:
   $\mathrm{var}(P^{(0)} v_i)$ (train) and $\mathrm{var}(P^{(1)} v_i)$, $\mathrm{var}(P^{(2)} v_i)$ (test).

Results are averaged over all three fold rotations (each fold takes a turn as
the fit fold).

### 12.5 PCA (non-CV baseline)

Standard PCA eigenvalues from a single fold, stored as `pca_covariances`.
These are upward-biased because overfitting to the fit fold is not penalized.

### 12.6 Spatial eigenvectors mode

When `use_spatial_eigenvectors=True`, PCA is applied to $P^T$ (bins × neurons)
instead of $P$ (neurons × bins). The resulting eigenvectors live in the bin
domain rather than the neuron domain, and the score measures which spatial
patterns are reproducible across trials.

---

## 13. StimSpaceSubspace (`StimSpaceConfig`)

This is the empirical application of the stimulus-space kappa framework to
real neural data. It maps directly onto the atlas `_stim_full_empirical_result`
logic but operates on behavioral position instead of discrete stimulus indices.

### 13.1 Place field construction

The pipeline uses four data splits: `train0`, `train1` (= `cv1`), `validation`
(= `cv2`), and `test`. For each split, the place field matrix is computed as
average activity per spatial bin, producing $P^{(k)} \in \mathbb{R}^{N \times S}$
where $S = \mathrm{num\_bins} \times \mathrm{num\_environments}$.

### 13.2 Pre-covariance matrix `_make_G`

$$G^{(k)} = \frac{P^{(k)} - \bar{P}^{(k)}}{\sqrt{S-1}} \in \mathbb{R}^{N \times S}$$

where $\bar{P}^{(k)}$ subtracts the mean firing rate across bins for each neuron.
This satisfies $G^{(k)} (G^{(k)})^T = \mathrm{cov}_s(P^{(k)})$ exactly.

### 13.3 Kernels computed at fit

| Kernel | Expression | Meaning |
|---|---|---|
| `pf_full_kernel` | $G_\mathrm{train}^T \,\Sigma^\mathrm{data}_\mathrm{test}\, G_\mathrm{train}$ | stim-space projection of full data covariance |
| `pf_pf_kernel` | $G_\mathrm{train}^T \,\Sigma^\mathrm{pf}_\mathrm{test}\, G_\mathrm{train}$ | stim-space projection of place-field covariance |

`u_pf_full` = eigenvectors of `pf_full_kernel` (directions for stim→full alignment).

`u_pf_pf` = eigenvectors of `G_train^T G_train` (when `directions_from_placefield_only=True`,
the default) or eigenvectors of `pf_pf_kernel` otherwise.

### 13.4 CV scores at score time

**`cv_variance_squared_placefields`** (corresponds to cvSER in atlas):

$$[\mathrm{cv\_var\_pf}]_i = u_i^T \,(G_\mathrm{cv1}^T \,\Sigma^\mathrm{data}_\mathrm{test}\, G_\mathrm{cv2})\, u_i$$

where $u_i$ are the `u_pf_full` directions learned at fit time. This is the
bilinear cross-validated estimator of $\lambda_i(K_{\Sigma^\mathrm{full}}(\Sigma^\mathrm{stim}))$.

**`cv_variance_squared_placefield_placefield`** (corresponds to cvStimStim in atlas):

$$[\mathrm{cv\_var\_pfpf}]_i = u_i^T \,(G_\mathrm{cv1}^T \,\Sigma^\mathrm{pf}_\mathrm{test}\, G_\mathrm{cv2})\, u_i$$

where $u_i$ are the `u_pf_pf` directions. This measures self-consistency of the
spatial map using place field covariance as the inner product.

### 13.5 Kappa-style variance (non-CV, amplitude scale)

The inner-block approach computes non-CV kappa modes by projecting through PCA
components:

$$M_\mathrm{activity} = \Lambda_A^{1/2} V_A^T \,\Sigma^\mathrm{data}_\mathrm{test}\, V_A \Lambda_A^{1/2}$$

$$[\mathrm{variance\_activity}]_i = \sqrt{\max(\lambda_i(M_\mathrm{activity}), 0)}$$

where $V_A$, $\Lambda_A$ come from the PCA of the train activity. This is the
finite-sample kappa spectrum in the neural-activity domain. The analogous
`variance_placefields` and `variance_placefield_placefield` use place field PCA
components.

---

## 14. Key Differences Between Metrics

### 14.1 Scale comparison

| Metric | What it measures | Scale | Unbiased? |
|---|---|---|---|
| Population SVR | True fraction of $\Sigma_\mathrm{full}$ structure in $\Sigma_\mathrm{stim}$ | Amplitude ($\sqrt{\lambda}$) | N/A (population) |
| Empirical SVR | Train stimulus vs test full, normalized by train-test full reliability | Amplitude | No (upward biased) |
| cvSER | CV stimulus energy vs full energy | Energy ($\lambda$) | Yes (numerator) |
| CV-Kappa | CV singular values of stim-root × full-root product | Amplitude | Yes |
| cvStimStim | Self-consistency of stim estimate | Energy | Yes (numerator) |
| CKA | Cosine similarity of covariance matrices | Amplitude (relative) | No |

### 14.2 When metrics diverge

**Aligned nuisance** (`stim_full.aligned_nuisance`): Population SVR $\approx 0.24$
but population CKA $\approx 1.0$. This is because the nuisance subspace is *aligned*
with the stimulus subspace — the eigenvectors of $\Sigma_\mathrm{full}$ and
$\Sigma_\mathrm{stim}$ are identical, so CKA (which is normalized out of scale
differences) sees perfect alignment. But SVR accounts for the extra variance in
$\Sigma_\mathrm{full}$ from the nuisance, so it correctly reflects that the
stimulus explains only ~24% of the reproducible structure.

**High diagonal noise** (`stim_full.high_diagonal_noise`): All metrics are
depressed because per-neuron noise inflates $\Sigma_\mathrm{full}$ isotropically
across all directions, diluting the stimulus signal in both kappa and CKA.

**Orthogonal high nuisance** (`stim_full.orthogonal_high_nuisance`): SVR $\approx 0.06$,
cvSER $\approx 0.01$. The discrepancy is because SVR uses amplitude scale
(square root of eigenvalues) while cvSER uses energy scale — a few modes of
stimulus signal look comparatively larger in amplitude than in energy.

**Context spectrum mismatch** (`context.spectrum_mismatch`): SVR > 1 because the
candidate has a steeper spectrum ($\alpha = 0.25$) than the reference ($\alpha = 2.0$),
so the top modes of the candidate are *much larger* than the top modes of the
reference. The Loewner nesting argument that bounds SVR to $[0,1]$ does not apply
when candidate and reference are from different conditions.

### 14.3 cvSER vs CKA

Both use the trace inner product $\mathrm{tr}(A B)$. The difference is the denominator:
- CKA divides by $\sqrt{\mathrm{tr}(A^2)\mathrm{tr}(B^2)}$ — the geometric mean
  of total energies, making it scale-invariant.
- cvSER divides by $\mathrm{tr}(\tilde{\Sigma}_\mathrm{full}^\mathrm{train}\tilde{\Sigma}_\mathrm{full}^\mathrm{test})$ — the
  reliability of the reference, penalizing for noise in $\Sigma_\mathrm{full}$.
  The numerator is also cross-validated (unbiased), while CKA's numerator has
  a bias from the single-estimate stimulus covariance.

---

## 15. Named Atlas Cases

All atlas cases use `num_neurons=200` and (for stim-full) `num_stimuli=40`, `stim_dim=10`.

### Stim-full cases

| Name | nuisance_dim | nuisance_scale | nuisance_alignment | noise_scale | Key behavior |
|---|---|---|---|---|---|
| `stim_full.identity` | 0 | 0.0 | — | 0.0 | $\Sigma_\mathrm{stim} = \Sigma_\mathrm{full}$; SVR = 1 |
| `stim_full.orthogonal_low_nuisance` | 10 | 0.25 | orthogonal | 0.05 | Modest orthogonal nuisance |
| `stim_full.orthogonal_high_nuisance` | 40 | 3.0 | orthogonal | 0.05 | Nuisance dominates volume |
| `stim_full.aligned_nuisance` | 10 | 3.0 | aligned | 0.05 | Nuisance axes = stimulus axes |
| `stim_full.angled_nuisance_45` | 10 | 3.0 | angle (π/4) | 0.05 | 45° principal angle |
| `stim_full.random_nuisance` | 40 | 2.0 | random | 0.05 | Independent random subspace |
| `stim_full.*_nodiagonal` | same as above | same | same | **0.0** | No per-neuron noise |
| `stim_full.high_diagonal_noise` | 20 | 0.2 | orthogonal | 2.0 | Diagonal noise dominates |

### Context-pair cases

| Name | `geometry` | Key behavior |
|---|---|---|
| `context.identical` | `"same"` | SVR = 1, CKA = 1 |
| `context.rotated_45` | `"angle"`, π/4 | SVR = $\cos(\pi/4) = 0.707$ |
| `context.orthogonal` | `"orthogonal"` | SVR = 0, CKA = 0 |
| `context.spectrum_mismatch` | `"same"`, $\alpha_A=0.25$, $\alpha_B=2.0$ | SVR > 1 |
| `context.partial_overlap` | `"partial"`, `shared_rank=5` | 5 shared, 15 private |
| `context.random` | `"random"` | Expected overlap from ambient dimension |

### Shared-space cases

| Name | `private_ratio` | Key behavior |
|---|---|---|
| `shared_space.shared_dominant` | 0.5 | Shared subspace dominates; SVR ≈ 0.77 |
| `shared_space.private_dominant` | 3.0 | Private variance dominates; SVR ≈ 0.08 |

For `context.rotated_45`, the SVR = $\cos(\pi/4)$ result is exact in the
population when both spectra are identical: all singular values of
$U_A^T U_B$ equal $\cos(\theta)$, so $\kappa(A,B) = \cos(\theta) \kappa(B,B)$.

---

## 16. Important Caveats and Numerical Considerations

**Symmetrization:** Every covariance matrix is symmetrized (`0.5*(A + A.T)`)
before eigendecomposition to prevent spurious complex eigenvalues from
floating-point asymmetry.

**Eigenvalue clamping:** Negative eigenvalues from numerical noise are clamped to
zero in `sqrtm_spd` and in all `maximum(..., 0)` calls. This is correct behavior
for PSD matrices but means that slightly negative CV scores are hard-floored to
zero when converting to amplitude scale.

**Negative CV modes:** The bilinear cv estimators ($w_i^\mathrm{cv}$) can be
negative. When averaging over modes (computing the total ratio) negative values
are included (they are real signal cancellation). When plotting on a log scale
(the notebook uses `np.maximum(v, min_val)`) negative values are floored.

**Test rotation:** The `test_rotation_angle` parameter rotates the stim and
nuisance subspaces before generating test data, simulating a session-to-session
coordinate change. This reduces the test-set SVR because the train directions
are no longer aligned with the test stimulus geometry.

**Tight frame vs diagonal:** The stim_latents matrix is a tight frame
($\mathbf{L}\mathbf{L}^T = S\mathbf{I}$) rather than a diagonal scale matrix.
This means the per-stimulus responses span the full $D$-dimensional stim space
rather than being concentrated on $D$ orthogonal axes. When $D = S$ the frame
is square and the construction is exact.

**Trial balancing in cvSER:** The stimulus-balanced fold assignment guarantees
each fold has approximately $T/3K$ trials per stimulus (where $K$ is the number
of stimuli). When some stimuli have very few trials, the fold sizes become unequal,
which introduces mild correlation between folds that slightly biases the CV estimate.

**Empirical SVR upward bias and sample size:** With few trials per stimulus, the
nuisance-averaging effect in $\bar{x}_s$ is incomplete, so the empirical
$\tilde{\Sigma}_\mathrm{stim}$ retains substantial nuisance contamination.
This is why empirical SVR is systematically higher than population SVR across
all atlas cases in the notebook output.
