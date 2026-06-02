# Placefield Simulation & Noise Generation — Design Document

## Goal

Build a composable system for generating synthetic neural population data from a
placefield mean matrix `P` (neurons × positions) plus structured noise. The
system must support:

1. Fitting noise models directly from real session data
2. Sampling noise from parametric distributions (for controlled experiments)
3. Adding position-independent "nuisance" structure on top of residual noise
4. Running spectrum-estimation methods on the result to validate estimators

---

## Data Shapes — Conventions

| Symbol | Shape      | Description                                           |
|--------|------------|-------------------------------------------------------|
| `P`    | `(N, K)`   | Placefield mean matrix (neurons × positions)          |
| `Y`    | `(N, K, T)`| Full data tensor (neurons × positions × trials)       |
| `R`    | `(N, K, T)`| Residual tensor: `Y - P[:, :, None]`                  |
| `X`    | `(N, S)`   | Unfolded data, `S = K * T` (neurons × samples)        |

---

## Core Abstractions

### 1. `ResidualConfig` (abstract base)

Knows how to:
- **Be constructed** from either real data or a parametric distribution
- **Generate residuals** `R` of shape `(N, K, T)` given a mean `P` and trial count

The key design decision: the config encapsulates the noise *structure*, and `P`
is passed at sample time because some noise models (heteroscedastic, Poisson-like)
need to know the mean to set local variance.

```python
class ResidualConfig(ABC):

    @classmethod
    def from_data(cls, Y: np.ndarray) -> "ResidualConfig":
        """
        Fit noise model from (N, K, T) data tensor.
        Computes residuals R = Y - Y.mean(axis=-1, keepdims=True),
        then fits model parameters.
        """
        ...

    @classmethod
    def from_distribution(cls, **params) -> "ResidualConfig":
        """
        Construct from explicit distribution parameters
        (e.g. a known variance profile, Fano factor, etc.)
        """
        ...

    @abstractmethod
    def sample(self, P: np.ndarray, n_trials: int) -> np.ndarray:
        """
        P: (N, K) mean placefield matrix
        returns R: (N, K, T)  — residuals ready to broadcast-add to P
        """
        ...
```

### 2. `NuisanceConfig` (abstract base)

Position-independent noise added in the **unfolded** `(N, S)` space. Captures
structure that isn't locked to spatial position — shared brain-state
fluctuations, low-rank drift, global gain modulation, etc.

```python
class NuisanceConfig(ABC):

    @abstractmethod
    def sample(self, n_neurons: int, n_samples: int) -> np.ndarray:
        """
        Returns Z: (N, S) — additive nuisance term in unfolded space
        """
        ...
```

### 3. `PlacefieldDataGenerator`

Composes a mean `P`, a `ResidualConfig`, and an optional `NuisanceConfig`
into a callable that produces synthetic datasets.

```python
class PlacefieldDataGenerator:

    def __init__(
        self,
        P: np.ndarray,                         # (N, K)
        residual_config: ResidualConfig,
        nuisance_config: NuisanceConfig | None = None,
    ): ...

    def sample(
        self,
        n_trials: int,
        return_unfolded: bool = False,
    ) -> np.ndarray:
        """
        1. R = residual_config.sample(P, n_trials)         # (N, K, T)
        2. Y = P[:, :, None] + R                           # (N, K, T)
        3. X = Y.reshape(N, K * n_trials)                  # (N, S)
        4. if nuisance_config: X += nuisance_config.sample(N, S)
        5. return X if return_unfolded else X.reshape(N, K, n_trials)
        """
        ...
```

---

## Residual Config Implementations

Listed in order of complexity. All share the same `sample(P, n_trials)` interface.

### `StationaryDiagonalResidualConfig`

Each neuron has a single scalar variance `σ²_n`, constant across positions.

- **from_data**: `σ²_n = mean over (k, t) of R[n, k, t]²`
- **sample**: `R[n, k, t] ~ N(0, σ²_n)` i.i.d.
- **params**: `sigma: np.ndarray` shape `(N,)`

### `StationaryLowRankResidualConfig`

Shared noise (neuropil, hemodynamics) captured by a low-rank factor model:
`Σ = W Wᵀ + Ψ` where `Ψ` is diagonal.

- **from_data**: pool residuals across positions → `sklearn.decomposition.FactorAnalysis`
- **sample**: draw from `N(0, Σ)` for each (k, t)
- **params**: `W: (N, r)`, `psi: (N,)`

### `HeteroscedasticDiagonalResidualConfig`

Variance-per-neuron-per-position. Captures Poisson-like structure without
requiring a full covariance matrix.

- **from_data**: `σ²[n, k] = var over t of R[n, k, :]`
- **sample**: `R[n, k, t] ~ N(0, σ²[n, k])` independently across neurons
- **params**: `sigma2: (N, K)`

### `PoissonLikeResidualConfig` *(recommended default for CA imaging)*

Parametric heteroscedastic model: `σ²[n, k] = a_n * P[n, k] + b_n`.

- **from_data**: for each neuron, regress `σ²[n, k]` over `k` against `P[n, k]`
- **from_distribution**: pass `a` and `b` directly (e.g. `a=1.0, b=0.1`)
- **sample**: draw `N(0, a_n * P[n,k] + b_n)` for each (n, k, t)
- **params**: `a: (N,)`, `b: (N,)`
- **note**: `P` is required at sample time — the mean matters here

### `FullCovariancePerPositionResidualConfig`

Fits a full `(N, N)` covariance matrix per position. Only tractable when `T >> N`.

- **from_data**: `Σ_k = (1/T) R[:,k,:] @ R[:,k,:].T`, regularized (Ledoit-Wolf)
- **sample**: draw from `N(0, Σ_k)` for each k, t
- **params**: `sigmas: (K, N, N)`
- **warning**: requires `T > N` per-position; flag if underdetermined

---

## Nuisance Config Implementations

### `LowRankNuisanceConfig`

Adds low-rank structure `Z = U V` where `U: (N, r)`, `V: (r, S)`.
Models global brain-state fluctuations or task-correlated signals
that aren't place-field-locked.

- **params**: `rank: int`, `variance: float`
- **sample**: draw `U ~ N(0, I)`, `V ~ N(0, I)`, rescale to target variance

### `StationaryGaussianNuisanceConfig`

Simple i.i.d. or correlated Gaussian in sample space. Baseline nuisance.

---

## Composition Example (pseudocode)

```python
# --- From real data ---
residual_cfg = PoissonLikeResidualConfig.from_data(Y_real)   # fit from session

# --- From known params (for controlled sweep) ---
residual_cfg = PoissonLikeResidualConfig.from_distribution(a=1.0, b=0.05)

# --- Optional nuisance ---
nuisance_cfg = LowRankNuisanceConfig(rank=3, variance=0.2)

# --- Generator ---
gen = PlacefieldDataGenerator(P, residual_cfg, nuisance_cfg)

Y_sim = gen.sample(n_trials=50)            # (N, K, 50)
X_sim = gen.sample(n_trials=50, return_unfolded=True)  # (N, K*50)

# --- True spectrum (target for estimators) ---
U, s, Vt = np.linalg.svd(P, full_matrices=False)
true_spectrum = s ** 2
```

---

## Intended Experimental Loop

```
for residual_cfg in [diagonal, poisson_like, full_cov_per_pos]:
    gen = PlacefieldDataGenerator(P, residual_cfg)
    for trial in range(n_bootstrap):
        Y_sim = gen.sample(n_trials=T)
        estimated_spectrum = run_cv_estimator(Y_sim)   # cvPCA, SVCA, SVR, ...
        compare(estimated_spectrum, true_spectrum)
```

The simulation validates whether a given CV estimator is **unbiased** for the
true spectrum of `P` under each noise regime.

---

## Open Questions / TODO

- [ ] Should `ResidualConfig.from_data` accept a pre-computed `P` (mean), or
      compute it internally from `Y`? Probably the latter for convenience, but
      expose `P_` as a fitted attribute for inspection.
- [ ] `PoissonLikeResidualConfig`: handle neurons with essentially zero mean
      rate (avoid dividing by tiny `P[n,k]` values) — clamp or add floor.
- [ ] Decide whether `NuisanceConfig` lives in `(N, S)` unfolded space only,
      or whether a position-structured nuisance (e.g. "extra place cells not in
      `P`") is worth a separate abstraction.
- [ ] Serialization: configs fitted from data should be saveable (dataclass +
      numpy `.npz`, or just pickleable dataclasses) so a fit from one session
      can be reused in another.
- [ ] Consider a `ResidualConfig.log_prob(R, P)` method for later model
      comparison / goodness-of-fit testing.
