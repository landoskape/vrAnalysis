"""Shared-variance analysis: direct config API and named atlas registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional

import numpy as np
import numpy.typing as npt

from .generators import (
    CovariancePairConfig,
    CovariancePairGenerator,
    SharedSpaceConfig,
    SharedSpaceGenerator,
    StimFullConfig,
    StimFullGenerator,
)
from .operators import sqrtm_spd

AtlasKind = Literal["stim_full", "context_pair"]
AtlasPipeline = Literal["stimulus_space", "covariance"]
AtlasBuilder = Callable[[np.random.Generator, npt.DTypeLike], tuple[Any, Any]]

_CONFIG_DISPATCH: dict[type, tuple[AtlasKind, AtlasPipeline, type]] = {
    StimFullConfig: ("stim_full", "stimulus_space", StimFullGenerator),
    CovariancePairConfig: ("context_pair", "covariance", CovariancePairGenerator),
    SharedSpaceConfig: ("context_pair", "covariance", SharedSpaceGenerator),
}


@dataclass(frozen=True)
class AtlasSpec:
    """Registry entry for one named atlas condition."""

    name: str
    kind: AtlasKind
    pipeline: AtlasPipeline
    description: str
    builder: AtlasBuilder

    def build(
        self,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> "AtlasBuild":
        """Instantiate this atlas condition."""
        if seed is not None and rng is not None:
            raise ValueError("Provide seed or rng, not both.")
        _rng = np.random.default_rng(seed) if rng is None else rng
        _dtype = np.dtype(dtype)
        config, generator = self.builder(_rng, _dtype)
        return AtlasBuild(
            name=self.name,
            kind=self.kind,
            pipeline=self.pipeline,
            description=self.description,
            config=config,
            generator=generator,
            dtype=_dtype,
        )


@dataclass(frozen=True)
class AtlasBuild:
    """Instantiated atlas condition with its config and generator."""

    name: str
    kind: AtlasKind
    pipeline: AtlasPipeline
    description: str
    config: Any
    generator: Any
    dtype: np.dtype


MetricKind = Literal["kappa", "energy"]


@dataclass(frozen=True)
class ModeComparison:
    """Candidate vs reference mode vectors and scalar summaries."""

    candidate_modes: npt.NDArray[np.floating]
    reference_modes: npt.NDArray[np.floating]
    ratio: float
    cumulative_ratio: npt.NDArray[np.floating]
    metric: MetricKind

    @property
    def candidate_variance_scale_modes(self) -> npt.NDArray[np.floating]:
        """Variance-scale modes (sqrt energy); identity for kappa comparisons."""
        if self.metric == "energy":
            return np.sqrt(np.maximum(self.candidate_modes, 0.0))
        return self.candidate_modes

    @property
    def reference_variance_scale_modes(self) -> npt.NDArray[np.floating]:
        """Variance-scale modes (sqrt energy); identity for kappa comparisons."""
        if self.metric == "energy":
            return np.sqrt(np.maximum(self.reference_modes, 0.0))
        return self.reference_modes

    def as_variance_scale(self) -> "ModeComparison":
        """Energy comparison mapped to variance scale with ratio recomputed."""
        return _comparison(
            self.candidate_variance_scale_modes,
            self.reference_variance_scale_modes,
            metric="kappa",
        )


@dataclass(frozen=True)
class SubspaceGeometry:
    """Eigenstructure and cross-subspace overlap for candidate vs reference."""

    candidate_spectrum: npt.NDArray[np.floating]
    reference_spectrum: npt.NDArray[np.floating]
    candidate_eigenvectors: npt.NDArray[np.floating]
    reference_eigenvectors: npt.NDArray[np.floating]
    reference_on_candidate_overlap: npt.NDArray[np.floating]
    candidate_on_reference_overlap: npt.NDArray[np.floating]
    cka: float
    trace_candidate: float
    trace_reference: float
    trace_nuisance: Optional[float] = None
    trace_eps: Optional[float] = None


@dataclass(frozen=True)
class PopulationBlock:
    """Population-level kappa, energy, and geometry diagnostics."""

    kappa: ModeComparison
    energy: ModeComparison
    geometry: SubspaceGeometry
    stimstim: Optional[ModeComparison] = None


@dataclass(frozen=True)
class EmpiricalDiagnostics:
    """Scalar empirical diagnostics not carried in mode comparisons."""

    cka: float
    cv_cka: Optional[float] = None


@dataclass(frozen=True)
class EmpiricalBlock:
    """Finite-sample comparisons and CV variants."""

    kappa: ModeComparison
    diagnostics: EmpiricalDiagnostics
    cv_energy: Optional[ModeComparison] = None
    cv_kappa: Optional[ModeComparison] = None
    cv_stimstim: Optional[ModeComparison] = None


@dataclass(frozen=True)
class AnalysisProvenance:
    """Parameters needed to reproduce empirical draws."""

    dtype: np.dtype
    num_samples: Optional[int]
    sample_seed: Optional[int]
    noise_variance: float
    test_rotation_angle: float


@dataclass(frozen=True)
class AtlasAnalysisResult:
    """Structured output for stimulus-full and context-pair analyses."""

    name: str
    kind: AtlasKind
    pipeline: AtlasPipeline
    description: str
    config: StimFullConfig | CovariancePairConfig | SharedSpaceConfig | None

    population: PopulationBlock
    empirical: Optional[EmpiricalBlock] = None
    provenance: AnalysisProvenance = field(default_factory=lambda: AnalysisProvenance(np.dtype(np.float64), None, None, 0.0, 0.0))
    extras: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Matrix / covariance helpers
# ---------------------------------------------------------------------------


def _precov(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return centered data scaled so G @ G.T equals np.cov(data)."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    n_samples = data.shape[1]
    if n_samples < 2:
        raise ValueError("Need at least two samples to compute covariance.")
    dtype = data.dtype
    centered = data - np.mean(data, axis=1, keepdims=True, dtype=dtype)
    scale = np.sqrt(np.array(n_samples - 1, dtype=dtype))
    return (centered / scale).astype(dtype, copy=False)


def _cov(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return row-wise covariance while preserving the input floating dtype."""
    data = np.asarray(data)
    return np.cov(data, rowvar=True, dtype=data.dtype).astype(data.dtype, copy=False)


def _symmetrize(A: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return 0.5 * (A + A.T)


def _sorted_eigenvalues(A: npt.NDArray[np.floating], *, symmetrize: bool = True) -> npt.NDArray[np.floating]:
    matrix = _symmetrize(A) if symmetrize else A
    if symmetrize:
        evals = np.linalg.eigvalsh(matrix)
    else:
        evals = np.linalg.eigvals(matrix)
        if np.max(np.abs(np.imag(evals))) > 1e-8:
            raise ValueError("Non-symmetric matrix has complex eigenvalues; provide directions instead.")
        evals = np.real(evals)
    return np.sort(evals)[::-1]


def _sqrt_sorted_eigenvalues(A: npt.NDArray[np.floating], *, symmetrize: bool = True) -> npt.NDArray[np.floating]:
    evals = _sorted_eigenvalues(A, symmetrize=symmetrize)
    return np.sqrt(np.maximum(evals, 0.0))


def _sorted_eigenvalues_and_eigenvectors(
    A: npt.NDArray[np.floating],
    *,
    symmetrize: bool = True,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    matrix = _symmetrize(A) if symmetrize else A
    if symmetrize:
        evals, evecs = np.linalg.eigh(matrix)
    else:
        evals, evecs = np.linalg.eig(matrix)
        if np.max(np.abs(np.imag(evals))) > 1e-8:
            raise ValueError("Non-symmetric matrix has complex eigenvalues; provide directions instead.")
        evals = np.real(evals)
        evecs = np.real(evecs)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


# ---------------------------------------------------------------------------
# Public operator functions
# ---------------------------------------------------------------------------


def kappa_modes(A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Mode-wise kappa(A, B) from covariance matrices."""
    Aroot = sqrtm_spd(_symmetrize(A))
    return _sqrt_sorted_eigenvalues(Aroot @ _symmetrize(B) @ Aroot)


def energy_modes(A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Mode-wise shared-energy eigenvalues whose sum is tr(A B), without the square root used by kappa."""
    Aroot = sqrtm_spd(_symmetrize(A))
    return np.maximum(_sorted_eigenvalues(Aroot @ _symmetrize(B) @ Aroot), 0.0)


def centered_kernel_alignment(A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]) -> float:
    """Centered kernel alignment between covariance matrices."""
    numerator = np.trace(A @ B)
    denominator = np.sqrt(np.trace(A @ A) * np.trace(B @ B))
    if denominator <= 0.0:
        return np.nan
    return float(numerator / denominator)


def stimulus_space_kappa_modes(G_A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Mode-wise kappa(A, B) using a pre-covariance G_A where A = G_A @ G_A.T."""
    return _sqrt_sorted_eigenvalues(G_A.T @ _symmetrize(B) @ G_A)


def stimulus_space_energy_modes(G_A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Mode-wise shared-energy eigenvalues using a pre-covariance G_A where A = G_A @ G_A.T."""
    return np.maximum(_sorted_eigenvalues(G_A.T @ _symmetrize(B) @ G_A), 0.0)


def cv_kappa_modes(
    root_A: npt.NDArray[np.floating],
    root_B: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Singular values of root_A @ root_B (cross-validated kappa modes).

    root_A = sqrtm_spd(Sigma_A) and root_B = sqrtm_spd(Sigma_B) must be
    estimated from independent data. Singular values equal kappa_modes(Sigma_A,
    Sigma_B) in the population, but removing the squared finite-sample bias that
    arises when both roots come from the same data.
    """
    _, s, _ = np.linalg.svd(root_A @ root_B, full_matrices=False)
    return s


# ---------------------------------------------------------------------------
# Internal analysis helpers
# ---------------------------------------------------------------------------


def _energy_directions(kernel: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    evals, directions = np.linalg.eigh(_symmetrize(kernel))
    order = np.argsort(evals)[::-1]
    return np.maximum(evals[order], 0.0), directions[:, order]


def _project_energy_modes(
    kernel: npt.NDArray[np.floating],
    directions: npt.NDArray[np.floating],
    *,
    symmetrize: bool = True,
) -> npt.NDArray[np.floating]:
    matrix = _symmetrize(kernel) if symmetrize else kernel
    return np.einsum("ij,ij->j", directions, matrix @ directions)


def _cv_kappa_fit_score(
    root_A_train: npt.NDArray[np.floating],
    root_B_train: npt.NDArray[np.floating],
    root_A_test: npt.NDArray[np.floating],
    root_B_test: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Learn shared subspace on train roots, score on test roots.

    Mirrors SVCA.fit(root_A_train, root_B_train) then SVCA.score(root_A_test,
    root_B_test, normalize=False). The score for mode k is:
        u_k.T @ root_A_test @ root_B_test @ v_k
    which is the projection of the test cross-product onto the train-learned mode.
    Scores can be negative; their sum is a cross-validated estimate of shared signal.
    """
    U, _, Vt = np.linalg.svd(root_A_train @ root_B_train, full_matrices=False)
    return np.diag(U.T @ root_A_test @ root_B_test @ Vt.T)


def _ratio(candidate_modes: npt.NDArray[np.floating], reference_modes: npt.NDArray[np.floating]) -> float:
    denom = float(np.sum(reference_modes))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(candidate_modes) / denom)


def _cumulative_ratio(
    candidate_modes: npt.NDArray[np.floating],
    reference_modes: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    n_modes = max(len(candidate_modes), len(reference_modes))
    candidate = np.zeros(n_modes, dtype=float)
    reference = np.zeros(n_modes, dtype=float)
    candidate[: len(candidate_modes)] = candidate_modes
    reference[: len(reference_modes)] = reference_modes
    reference_cumulative = np.cumsum(reference)
    return np.divide(
        np.cumsum(candidate),
        reference_cumulative,
        out=np.full(n_modes, np.nan, dtype=float),
        where=reference_cumulative > 0.0,
    )


def _comparison(
    candidate_modes: npt.NDArray[np.floating],
    reference_modes: npt.NDArray[np.floating],
    *,
    metric: MetricKind,
) -> ModeComparison:
    """Build a ModeComparison from candidate and reference mode vectors."""
    return ModeComparison(
        candidate_modes=candidate_modes,
        reference_modes=reference_modes,
        ratio=_ratio(candidate_modes, reference_modes),
        cumulative_ratio=_cumulative_ratio(candidate_modes, reference_modes),
        metric=metric,
    )


def _build_geometry(
    candidate_covariance: npt.NDArray[np.floating],
    reference_covariance: npt.NDArray[np.floating],
    *,
    trace_nuisance: Optional[float] = None,
    trace_eps: Optional[float] = None,
) -> SubspaceGeometry:
    """Eigenstructure and overlap diagnostics for a candidate/reference pair."""
    w_candidate, v_candidate = _sorted_eigenvalues_and_eigenvectors(candidate_covariance)
    w_reference, v_reference = _sorted_eigenvalues_and_eigenvectors(reference_covariance)
    reference_on_candidate = ((v_candidate.T @ v_reference) ** 2) @ w_reference
    candidate_on_reference = ((v_reference.T @ v_candidate) ** 2) @ w_candidate
    return SubspaceGeometry(
        candidate_spectrum=w_candidate,
        reference_spectrum=w_reference,
        candidate_eigenvectors=v_candidate,
        reference_eigenvectors=v_reference,
        reference_on_candidate_overlap=reference_on_candidate,
        candidate_on_reference_overlap=candidate_on_reference,
        cka=centered_kernel_alignment(candidate_covariance, reference_covariance),
        trace_candidate=float(np.trace(candidate_covariance)),
        trace_reference=float(np.trace(reference_covariance)),
        trace_nuisance=trace_nuisance,
        trace_eps=trace_eps,
    )


def _stimulus_means(
    data: npt.NDArray[np.floating],
    stim_indices: npt.NDArray[np.integer],
    num_stimuli: int,
) -> npt.NDArray[np.floating]:
    counts = np.bincount(stim_indices, minlength=num_stimuli)
    missing = np.flatnonzero(counts[:num_stimuli] == 0)
    if missing.size > 0:
        raise ValueError(f"Cannot estimate stimulus means; missing stimuli in sample: {missing.tolist()}")

    means = np.empty((data.shape[0], num_stimuli), dtype=data.dtype)
    for istim in range(num_stimuli):
        means[:, istim] = np.mean(data[:, stim_indices == istim], axis=1, dtype=data.dtype)
    return means


def _stimulus_balanced_folds(
    stim_indices: npt.NDArray[np.integer],
    num_stimuli: int,
    num_folds: int,
    rng: np.random.Generator,
) -> list[npt.NDArray[np.integer]]:
    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2, got {num_folds}")

    folds: list[list[int]] = [[] for _ in range(num_folds)]
    for istim in range(num_stimuli):
        indices = np.flatnonzero(stim_indices == istim)
        if len(indices) < num_folds:
            raise ValueError(f"Cannot make {num_folds} stimulus-balanced folds; stimulus {istim} has only {len(indices)} samples.")
        for ifold, split in enumerate(np.array_split(rng.permutation(indices), num_folds)):
            folds[ifold].extend(int(index) for index in split)

    return [np.array(sorted(fold), dtype=int) for fold in folds]


# ---------------------------------------------------------------------------
# Per-pipeline population and empirical result functions (take generator)
# ---------------------------------------------------------------------------


def _population_stimstim_comparison(
    stim_precov: npt.NDArray[np.floating],
    sigma_stim: npt.NDArray[np.floating],
) -> ModeComparison:
    """Population stim×stim shared-energy modes.

    Oracle analog of ``_stim_full_cv_stimstim_result``: learn directions from
    ``G.T @ Sigma_stim @ G`` and score ``G.T @ Sigma_stim @ G`` with independent
    pre-covariances collapsed to the same true stimulus means. The reference
    side matches the CV symmetric stimstim normalization,
    ``stimulus_space_energy_modes(G, Sigma_stim)``.

    Parameters
    ----------
    stim_precov
        Pre-covariance ``G`` from true per-stimulus responses (stimulus means).
    sigma_stim
        Population stimulus covariance ``Sigma_stim``.

    Returns
    -------
    ModeComparison
        Energy-mode comparison with ``metric="energy"``.
    """
    kernel = stim_precov.T @ _symmetrize(sigma_stim) @ stim_precov
    _, directions = _energy_directions(kernel)

    candidate_energy = _project_energy_modes(kernel, directions, symmetrize=False)
    reference_energy = stimulus_space_energy_modes(stim_precov, sigma_stim)

    return _comparison(candidate_energy, reference_energy, metric="energy")


def _stim_full_population_block(gen: StimFullGenerator) -> tuple[PopulationBlock, dict[str, Any]]:
    sigma_stim, sigma_nuisance, sigma_eps = gen.true_covariance()
    sigma_full = sigma_stim + sigma_nuisance + sigma_eps

    stim_responses = gen.stim_space @ np.diag(np.sqrt(gen.stim_spectrum)) @ gen.stim_latents
    stim_precov = _precov(stim_responses)
    candidate_kappa = stimulus_space_kappa_modes(stim_precov, sigma_full)
    reference_kappa = kappa_modes(sigma_full, sigma_full)
    candidate_energy = stimulus_space_energy_modes(stim_precov, sigma_full)
    reference_energy = energy_modes(sigma_full, sigma_full)

    geometry = _build_geometry(
        sigma_stim,
        sigma_full,
        trace_nuisance=float(np.trace(sigma_nuisance)),
        trace_eps=float(np.trace(sigma_eps)),
    )
    extras = {
        "stimulus_space_modes_match_covariance_modes": bool(
            np.allclose(candidate_kappa, kappa_modes(sigma_stim, sigma_full)[: len(candidate_kappa)], atol=1e-8)
        ),
    }
    return (
        PopulationBlock(
            kappa=_comparison(candidate_kappa, reference_kappa, metric="kappa"),
            energy=_comparison(candidate_energy, reference_energy, metric="energy"),
            geometry=geometry,
            stimstim=_population_stimstim_comparison(stim_precov, sigma_stim),
        ),
        extras,
    )


def _stim_full_cvser_result(
    data_train: npt.NDArray[np.floating],
    stim_indices: npt.NDArray[np.integer],
    num_stimuli: int,
    full_train: npt.NDArray[np.floating],
    full_test: npt.NDArray[np.floating],
    rng: np.random.Generator,
) -> tuple[ModeComparison, float, tuple[int, ...]]:
    folds = _stimulus_balanced_folds(stim_indices, num_stimuli, num_folds=3, rng=rng)
    fold_precovs = [_precov(_stimulus_means(data_train[:, fold], stim_indices[fold], num_stimuli)) for fold in folds]

    direction_kernel = fold_precovs[0].T @ _symmetrize(full_test) @ fold_precovs[0]
    _, directions = _energy_directions(direction_kernel)

    cv_kernel = fold_precovs[1].T @ full_test @ fold_precovs[2]
    candidate_energy = _project_energy_modes(cv_kernel, directions, symmetrize=False)
    reference_energy = energy_modes(full_train, full_test)

    cv_stim_cov_estimate = fold_precovs[1] @ fold_precovs[2].T
    cv_cka = centered_kernel_alignment(cv_stim_cov_estimate, full_test)
    fold_sizes = tuple(int(len(fold)) for fold in folds)
    return _comparison(candidate_energy, reference_energy, metric="energy"), cv_cka, fold_sizes


def _stim_full_cv_kappa_result(
    gen: "StimFullGenerator",
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
    test_rotation_angle: float,
) -> tuple[ModeComparison, dict[str, Any]]:
    """Cross-validated kappa for the stim-full pipeline.

    Four independent draws: (train0, train1) to learn the subspace, (test0, test1)
    to score it. train0 and test0 provide the stimulus covariance root (candidate);
    train1 and test1 provide the full-activity covariance root (reference).
    """
    data_train0, _, extras_train0 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_train1, _, _ = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_test0, _, extras_test0 = gen.generate(
        num_samples, noise_variance=noise_variance, rotation_angle=test_rotation_angle, rng=rng, return_extras=True
    )
    data_test1, _, _ = gen.generate(num_samples, noise_variance=noise_variance, rotation_angle=test_rotation_angle, rng=rng, return_extras=True)

    stim_means_train = _stimulus_means(data_train0, extras_train0["stim_indices"], gen.config.num_stimuli)
    stim_means_test = _stimulus_means(data_test0, extras_test0["stim_indices"], gen.config.num_stimuli)

    root_stim_train = sqrtm_spd(_cov(stim_means_train))
    root_stim_test = sqrtm_spd(_cov(stim_means_test))
    root_full_train1 = sqrtm_spd(_cov(data_train1))
    root_full_test1 = sqrtm_spd(_cov(data_test1))
    root_full_train0 = sqrtm_spd(_cov(data_train0))
    root_full_test0 = sqrtm_spd(_cov(data_test0))

    candidate_kappa = _cv_kappa_fit_score(root_stim_train, root_full_train1, root_stim_test, root_full_test1)
    reference_kappa = _cv_kappa_fit_score(root_full_train0, root_full_train1, root_full_test0, root_full_test1)

    return _comparison(candidate_kappa, reference_kappa, metric="kappa"), {}


def _stim_full_cv_stimstim_result(
    gen: "StimFullGenerator",
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
    test_rotation_angle: float,
) -> tuple[ModeComparison, dict[str, Any]]:
    """CV stimulus-stimstim energy modes.

    Mirrors cv_variance_squared_pf_pf from StimSpaceSubspace:
      directions: G(s_0).T @ cov(s_3) @ G(s_0)
      score:      G(s_1).T @ cov(s_3) @ G(s_2)
    s_3 is the shared reference fold; s_0, s_1, s_2 are independent draws.
    Reference: symmetric stimstim energy from a fresh draw s_t.
    """
    data_0, _, extras_0 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_1, _, extras_1 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_2, _, extras_2 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_3, _, extras_3 = gen.generate(num_samples, noise_variance=noise_variance, rotation_angle=test_rotation_angle, rng=rng, return_extras=True)
    data_t, _, extras_t = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)

    ns = gen.config.num_stimuli
    s_0 = _stimulus_means(data_0, extras_0["stim_indices"], ns)
    s_1 = _stimulus_means(data_1, extras_1["stim_indices"], ns)
    s_2 = _stimulus_means(data_2, extras_2["stim_indices"], ns)
    s_3 = _stimulus_means(data_3, extras_3["stim_indices"], ns)
    s_t = _stimulus_means(data_t, extras_t["stim_indices"], ns)

    G_0 = _precov(s_0)
    G_1 = _precov(s_1)
    G_2 = _precov(s_2)
    G_t = _precov(s_t)
    cov_3 = _cov(s_3)

    direction_kernel = G_0.T @ cov_3 @ G_0
    _, directions = _energy_directions(direction_kernel)

    cv_kernel = G_1.T @ cov_3 @ G_2
    candidate_energy = _project_energy_modes(cv_kernel, directions, symmetrize=False)
    reference_energy = stimulus_space_energy_modes(G_t, cov_3)

    return _comparison(candidate_energy, reference_energy, metric="energy"), {}


def _context_cv_kappa_result(
    gen,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
) -> tuple[ModeComparison, dict[str, Any]]:
    """Cross-validated kappa for the context-pair pipeline."""
    if isinstance(gen, CovariancePairGenerator):
        candidate_train0 = gen.generate(num_samples, which="candidate", noise_variance=noise_variance, rng=rng)
        candidate_test0 = gen.generate(num_samples, which="candidate", noise_variance=noise_variance, rng=rng)
        reference_train0 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
        reference_train1 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
        reference_test0 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
        reference_test1 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
    elif isinstance(gen, SharedSpaceGenerator):
        candidate_train0, reference_train0 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        candidate_test0, reference_test0 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        _, reference_train1 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        _, reference_test1 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
    else:
        raise TypeError(f"Unsupported context-pair generator type: {type(gen).__name__}")

    root_candidate_train0 = sqrtm_spd(_cov(candidate_train0))
    root_candidate_test0 = sqrtm_spd(_cov(candidate_test0))
    root_reference_train0 = sqrtm_spd(_cov(reference_train0))
    root_reference_train1 = sqrtm_spd(_cov(reference_train1))
    root_reference_test0 = sqrtm_spd(_cov(reference_test0))
    root_reference_test1 = sqrtm_spd(_cov(reference_test1))

    candidate_kappa = _cv_kappa_fit_score(root_candidate_train0, root_reference_train1, root_candidate_test0, root_reference_test1)
    reference_kappa = _cv_kappa_fit_score(root_reference_train0, root_reference_train1, root_reference_test0, root_reference_test1)

    return _comparison(candidate_kappa, reference_kappa, metric="kappa"), {}


def _stim_full_empirical_result(
    gen: StimFullGenerator,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
    test_rotation_angle: float,
) -> tuple[ModeComparison, ModeComparison, EmpiricalDiagnostics, dict[str, Any]]:
    data_train, _, extras_train = gen.generate(
        num_samples,
        noise_variance=noise_variance,
        rng=rng,
        return_extras=True,
    )
    data_test, _, _ = gen.generate(
        num_samples,
        noise_variance=noise_variance,
        rotation_angle=test_rotation_angle,
        rng=rng,
        return_extras=True,
    )

    stim_means = _stimulus_means(data_train, extras_train["stim_indices"], gen.config.num_stimuli)
    stim_precov = _precov(stim_means)
    full_train = _cov(data_train)
    full_test = _cov(data_test)

    candidate_kappa = stimulus_space_kappa_modes(stim_precov, full_test)
    reference_kappa = kappa_modes(full_train, full_test)
    cv_energy, cv_cka, fold_sizes = _stim_full_cvser_result(
        data_train,
        extras_train["stim_indices"],
        gen.config.num_stimuli,
        full_train,
        full_test,
        rng,
    )
    return (
        _comparison(candidate_kappa, reference_kappa, metric="kappa"),
        cv_energy,
        EmpiricalDiagnostics(
            cka=centered_kernel_alignment(_cov(stim_means), full_test),
            cv_cka=cv_cka,
        ),
        {"empirical_cv_fold_sizes": fold_sizes},
    )


def _context_population_block(gen) -> PopulationBlock:
    if isinstance(gen, CovariancePairGenerator):
        candidate_covariance, reference_covariance = gen.expected_covariances()
    elif isinstance(gen, SharedSpaceGenerator):
        candidate_covariance, reference_covariance = gen.true_covariance()
    else:
        raise TypeError(f"Unsupported context-pair generator type: {type(gen).__name__}")

    candidate_kappa = kappa_modes(candidate_covariance, reference_covariance)
    reference_kappa = kappa_modes(reference_covariance, reference_covariance)
    candidate_energy = energy_modes(candidate_covariance, reference_covariance)
    reference_energy = energy_modes(reference_covariance, reference_covariance)

    return PopulationBlock(
        kappa=_comparison(candidate_kappa, reference_kappa, metric="kappa"),
        energy=_comparison(candidate_energy, reference_energy, metric="energy"),
        geometry=_build_geometry(candidate_covariance, reference_covariance),
    )


def _context_empirical_block(
    gen,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
) -> EmpiricalBlock:
    if isinstance(gen, CovariancePairGenerator):
        candidate_train = gen.generate(num_samples, which="candidate", noise_variance=noise_variance, rng=rng)
        reference_train = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
        reference_test = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
    elif isinstance(gen, SharedSpaceGenerator):
        candidate_train, reference_train = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        _, reference_test = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
    else:
        raise TypeError(f"Unsupported context-pair generator type: {type(gen).__name__}")

    candidate_covariance = _cov(candidate_train)
    reference_train_covariance = _cov(reference_train)
    reference_test_covariance = _cov(reference_test)

    candidate_kappa = kappa_modes(candidate_covariance, reference_test_covariance)
    reference_kappa = kappa_modes(reference_train_covariance, reference_test_covariance)

    return EmpiricalBlock(
        kappa=_comparison(candidate_kappa, reference_kappa, metric="kappa"),
        diagnostics=EmpiricalDiagnostics(
            cka=centered_kernel_alignment(candidate_covariance, reference_test_covariance),
        ),
    )


# ---------------------------------------------------------------------------
# Core analysis engine (takes a generator directly)
# ---------------------------------------------------------------------------


def _run_analysis(
    gen,
    kind: AtlasKind,
    pipeline: AtlasPipeline,
    *,
    dtype: np.dtype,
    num_samples: Optional[int] = None,
    sample_seed: Optional[int] = None,
    noise_variance: float = 0.0,
    test_rotation_angle: float = 0.0,
    name: str = "",
    description: str = "",
    config: StimFullConfig | CovariancePairConfig | SharedSpaceConfig | None = None,
) -> AtlasAnalysisResult:
    provenance = AnalysisProvenance(
        dtype=dtype,
        num_samples=num_samples,
        sample_seed=sample_seed,
        noise_variance=noise_variance,
        test_rotation_angle=test_rotation_angle,
    )
    extras: dict[str, Any] = {}

    if kind == "stim_full":
        population, pop_extras = _stim_full_population_block(gen)
    elif kind == "context_pair":
        population = _context_population_block(gen)
        pop_extras = {}
    else:
        raise ValueError(f"Unknown atlas kind: {kind}")
    extras.update(pop_extras)

    empirical: Optional[EmpiricalBlock] = None
    if num_samples is not None:
        rng = np.random.default_rng(sample_seed)
        if kind == "stim_full":
            empirical_kappa, cv_energy, diagnostics, emp_extras = _stim_full_empirical_result(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
                test_rotation_angle=test_rotation_angle,
            )
            cv_kappa, _ = _stim_full_cv_kappa_result(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
                test_rotation_angle=test_rotation_angle,
            )
            cv_stimstim, _ = _stim_full_cv_stimstim_result(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
                test_rotation_angle=test_rotation_angle,
            )
            empirical = EmpiricalBlock(
                kappa=empirical_kappa,
                cv_energy=cv_energy,
                cv_kappa=cv_kappa,
                cv_stimstim=cv_stimstim,
                diagnostics=diagnostics,
            )
        else:
            empirical = _context_empirical_block(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
            )
            cv_kappa, _ = _context_cv_kappa_result(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
            )
            empirical = EmpiricalBlock(
                kappa=empirical.kappa,
                cv_kappa=cv_kappa,
                diagnostics=empirical.diagnostics,
            )
        extras.update(emp_extras if kind == "stim_full" else {})

    return AtlasAnalysisResult(
        name=name,
        kind=kind,
        pipeline=pipeline,
        description=description,
        config=config,
        population=population,
        empirical=empirical,
        provenance=provenance,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process(
    config: StimFullConfig | CovariancePairConfig | SharedSpaceConfig,
    *,
    dtype: npt.DTypeLike = np.float64,
    num_samples: Optional[int] = None,
    sample_seed: Optional[int] = None,
    noise_variance: float = 0.0,
    test_rotation_angle: float = 0.0,
    name: str = "",
    description: str = "",
) -> AtlasAnalysisResult:
    """
    Analyze a simulation config directly.

    Parameters
    ----------
    config
        One of StimFullConfig, CovariancePairConfig, or SharedSpaceConfig.
        The config rng field controls the geometric structure of the simulation.
    dtype
        Floating dtype for the generator.
    num_samples
        If provided, draw empirical samples and compute empirical SVR/CVSER.
    sample_seed
        Seed for empirical sampling (independent of config construction rng).
    noise_variance
        Added isotropic noise variance for empirical samples.
    test_rotation_angle
        Test-set rotation angle (stim-full pipeline only).
    name
        Label stored in the result (optional).
    description
        Description stored in the result (optional).
    """
    config_type = type(config)
    if config_type not in _CONFIG_DISPATCH:
        raise TypeError(f"Unsupported config type: {config_type.__name__}. " f"Expected one of: {', '.join(t.__name__ for t in _CONFIG_DISPATCH)}")
    kind, pipeline, gen_class = _CONFIG_DISPATCH[config_type]
    _dtype = np.dtype(dtype)
    gen = gen_class(config, dtype=_dtype)
    return _run_analysis(
        gen,
        kind,
        pipeline,
        dtype=_dtype,
        num_samples=num_samples,
        sample_seed=sample_seed,
        noise_variance=noise_variance,
        test_rotation_angle=test_rotation_angle,
        name=name,
        description=description,
        config=config,
    )


def analyze_build(
    build: AtlasBuild,
    *,
    num_samples: Optional[int] = None,
    sample_seed: Optional[int] = None,
    noise_variance: float = 0.0,
    test_rotation_angle: float = 0.0,
) -> AtlasAnalysisResult:
    """
    Analyze an instantiated atlas condition.

    The population result is always computed. If num_samples is provided, an
    empirical train/test result is also computed using the appropriate pipeline:
    stimulus-space K_B(A) for stim-full cases and covariance kappa for context
    pairs.
    """
    return _run_analysis(
        build.generator,
        build.kind,
        build.pipeline,
        dtype=build.dtype,
        num_samples=num_samples,
        sample_seed=sample_seed,
        noise_variance=noise_variance,
        test_rotation_angle=test_rotation_angle,
        name=build.name,
        description=build.description,
        config=build.config,
    )


def analyze_atlas_case(
    name: str,
    *,
    seed: Optional[int] = None,
    dtype: npt.DTypeLike = np.float64,
    num_samples: Optional[int] = None,
    sample_seed: Optional[int] = None,
    noise_variance: float = 0.0,
    test_rotation_angle: float = 0.0,
) -> AtlasAnalysisResult:
    """Build and analyze one named atlas case."""
    build = build_atlas_case(name, seed=seed, dtype=dtype)
    return analyze_build(
        build,
        num_samples=num_samples,
        sample_seed=sample_seed,
        noise_variance=noise_variance,
        test_rotation_angle=test_rotation_angle,
    )


# ---------------------------------------------------------------------------
# Atlas registry helpers
# ---------------------------------------------------------------------------


def _stim_config(
    rng: np.random.Generator,
    *,
    nuisance_dim: int,
    nuisance_scale: float,
    nuisance_alignment: Literal["orthogonal", "random", "aligned", "angle"] = "orthogonal",
    nuisance_angle: float = 0.0,
    noise_scale: float = 0.05,
    alpha_stim: float = 1.0,
    alpha_nuisance: float = 1.0,
) -> StimFullConfig:
    return StimFullConfig(
        num_neurons=200,
        num_stimuli=40,
        stim_dim=10,
        alpha_stim=alpha_stim,
        nuisance_dim=nuisance_dim,
        alpha_nuisance=alpha_nuisance,
        nuisance_scale=nuisance_scale,
        nuisance_alignment=nuisance_alignment,
        nuisance_angle=nuisance_angle,
        noise_scale=noise_scale,
        rng=rng,
    )


def _stim_generator(config: StimFullConfig, dtype: npt.DTypeLike = np.float64) -> StimFullGenerator:
    return StimFullGenerator(config, dtype=np.dtype(dtype))


def _cov_pair_config(
    rng: np.random.Generator,
    *,
    geometry: Literal["same", "random", "orthogonal", "angle", "partial"],
    alpha_candidate: float = 1.0,
    alpha_reference: float = 1.0,
    angle: float = 0.0,
    shared_rank: Optional[int] = None,
    reference_scale: float = 1.0,
) -> CovariancePairConfig:
    return CovariancePairConfig(
        num_neurons=200,
        candidate_rank=20,
        reference_rank=20,
        alpha_candidate=alpha_candidate,
        alpha_reference=alpha_reference,
        reference_scale=reference_scale,
        geometry=geometry,
        angle=angle,
        shared_rank=shared_rank,
        rng=rng,
    )


def _cov_pair_generator(config: CovariancePairConfig, dtype: npt.DTypeLike = np.float64) -> CovariancePairGenerator:
    return CovariancePairGenerator(config, dtype=np.dtype(dtype))


def _shared_space_config(
    rng: np.random.Generator,
    *,
    private_ratio: float,
    shuffle_shared: bool = False,
) -> SharedSpaceConfig:
    return SharedSpaceConfig(
        num_neurons=200,
        shared_dimensions=10,
        private_dimensions=(20, 20),
        alpha_shared_1=1.0,
        alpha_shared_2=1.0,
        shuffle_shared=shuffle_shared,
        alpha_private_1=1.0,
        alpha_private_2=1.0,
        private_ratio=private_ratio,
        rng=rng,
    )


def _shared_space_generator(config: SharedSpaceConfig, dtype: npt.DTypeLike = np.float64) -> SharedSpaceGenerator:
    return SharedSpaceGenerator(config, dtype=np.dtype(dtype))


def _make_specs() -> tuple[AtlasSpec, ...]:
    return (
        AtlasSpec(
            name="stim_full.identity",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus covariance is the full covariance; nuisance and diagonal noise are absent.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=0, nuisance_scale=0.0, noise_scale=0.0),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.orthogonal_low_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus and nuisance subspaces are orthogonal; nuisance variance is modest.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=0.25),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.orthogonal_high_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus and nuisance subspaces are orthogonal; full covariance is dominated by nuisance variance.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=40, nuisance_scale=3.0),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.aligned_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance lies on the same axes as the stimulus covariance.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=3.0, nuisance_alignment="aligned"),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.angled_nuisance_45",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance has 45 degree principal angles from stimulus covariance.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=3.0, nuisance_alignment="angle", nuisance_angle=np.pi / 4),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.random_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance is an independent random subspace with expected ambient overlap.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=40, nuisance_scale=2.0, nuisance_alignment="random"),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.aligned_nuisance_nodiagonal",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance lies on the same axes as the stimulus covariance.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=3.0, noise_scale=0.0, nuisance_alignment="aligned"),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.angled_nuisance_45_nodiagonal",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance has 45 degree principal angles from stimulus covariance.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=3.0, noise_scale=0.0, nuisance_alignment="angle", nuisance_angle=np.pi / 4),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.random_nuisance_nodiagonal",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance is an independent random subspace with expected ambient overlap.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=40, nuisance_scale=2.0, noise_scale=0.0, nuisance_alignment="random"),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="stim_full.high_diagonal_noise",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus and nuisance are present, but independent neuron-specific variance dominates the full covariance.",
            builder=lambda rng, dtype: (
                cfg := _stim_config(rng, nuisance_dim=20, nuisance_scale=0.2, noise_scale=2.0),
                _stim_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="context.identical",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances have identical eigenvectors and spectra.",
            builder=lambda rng, dtype: (
                cfg := _cov_pair_config(rng, geometry="same"),
                _cov_pair_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="context.rotated_45",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances have matched spectra but 45 degree principal-angle rotation.",
            builder=lambda rng, dtype: (
                cfg := _cov_pair_config(rng, geometry="angle", angle=np.pi / 4),
                _cov_pair_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="context.orthogonal",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances occupy orthogonal subspaces.",
            builder=lambda rng, dtype: (
                cfg := _cov_pair_config(rng, geometry="orthogonal"),
                _cov_pair_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="context.spectrum_mismatch",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target share eigenvectors but assign variance differently across modes.",
            builder=lambda rng, dtype: (
                cfg := _cov_pair_config(rng, geometry="same", alpha_candidate=0.25, alpha_reference=2.0),
                _cov_pair_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="context.partial_overlap",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target share only the first five dimensions; remaining target axes are private.",
            builder=lambda rng, dtype: (
                cfg := _cov_pair_config(rng, geometry="partial", shared_rank=5),
                _cov_pair_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="context.random",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances are independent random subspaces.",
            builder=lambda rng, dtype: (
                cfg := _cov_pair_config(rng, geometry="random"),
                _cov_pair_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="shared_space.shared_dominant",
            kind="context_pair",
            pipeline="covariance",
            description="Two contexts share a common subspace and have weak private covariance.",
            builder=lambda rng, dtype: (
                cfg := _shared_space_config(rng, private_ratio=0.5),
                _shared_space_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="shared_space.private_dominant",
            kind="context_pair",
            pipeline="covariance",
            description="Two contexts share a common subspace but private covariance dominates each context.",
            builder=lambda rng, dtype: (
                cfg := _shared_space_config(rng, private_ratio=3.0),
                _shared_space_generator(cfg, dtype),
            ),
        ),
    )


ATLAS: Mapping[str, AtlasSpec] = {spec.name: spec for spec in _make_specs()}


def list_atlas_cases(kind: Optional[AtlasKind] = None) -> list[str]:
    """List registered atlas case names."""
    if kind is None:
        return sorted(ATLAS)
    return sorted(name for name, spec in ATLAS.items() if spec.kind == kind)


def get_atlas_spec(name: str) -> AtlasSpec:
    """Return one atlas spec by name."""
    try:
        return ATLAS[name]
    except KeyError as exc:
        available = ", ".join(list_atlas_cases())
        raise KeyError(f"Unknown atlas case '{name}'. Available cases: {available}") from exc


def build_atlas_case(
    name: str,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    dtype: npt.DTypeLike = np.float64,
) -> AtlasBuild:
    """Instantiate one named atlas case and return its config and generator."""
    return get_atlas_spec(name).build(seed=seed, rng=rng, dtype=dtype)


__all__ = [
    "ATLAS",
    "AnalysisProvenance",
    "AtlasAnalysisResult",
    "AtlasBuild",
    "AtlasKind",
    "AtlasPipeline",
    "AtlasSpec",
    "EmpiricalBlock",
    "EmpiricalDiagnostics",
    "ModeComparison",
    "PopulationBlock",
    "SubspaceGeometry",
    "analyze_atlas_case",
    "analyze_build",
    "build_atlas_case",
    "cv_kappa_modes",
    "energy_modes",
    "get_atlas_spec",
    "kappa_modes",
    "list_atlas_cases",
    "process",
    "stimulus_space_energy_modes",
    "stimulus_space_kappa_modes",
]
