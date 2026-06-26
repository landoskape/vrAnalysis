"""Shared-variance analysis: direct config API and named atlas registry.

NOTE (deferred estimator): the place-field ``stim_full`` condition (``PlacefieldFullConfig`` /
``PlacefieldFullGenerator``) routes through this engine and gets population/CV kappa, energy,
cvSER, cv-stimstim, and an rCVPCA estimator (``_stim_full_rcvpca_result``). It does NOT yet
implement the canonical cross-validated place-field-kernel estimator from
``dimensionality_manuscript/subspace_analysis/stimspace.py`` (``StimSpaceSubspace``, config
``configs/stimspace.py``). Porting that estimator is future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional
import itertools

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import torch

from dimilibi.cvpca import CVPCA
from dimilibi.helpers import gaussian_filter

from .generators import (
    CovariancePairConfig,
    CovariancePairGenerator,
    SharedSpaceConfig,
    SharedSpaceGenerator,
    StimFullConfig,
    StimFullGenerator,
)
from .operators import sqrtm_spd
from .placefield_full import (
    PlacefieldFullConfig,
    PlacefieldFullGenerator,
    SmoothGPFieldConfig,
    ThresholdedGPFieldConfig,
    TilburyFieldConfig,
)

AtlasKind = Literal["stim_full", "context_pair"]
AtlasPipeline = Literal["stimulus_space", "covariance"]
AtlasBuilder = Callable[[np.random.Generator, npt.DTypeLike], tuple[Any, Any]]

_CONFIG_DISPATCH: dict[type, tuple[AtlasKind, AtlasPipeline, type]] = {
    StimFullConfig: ("stim_full", "stimulus_space", StimFullGenerator),
    PlacefieldFullConfig: ("stim_full", "stimulus_space", PlacefieldFullGenerator),
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


MetricKind = Literal["kappa", "energy", "energy_signed"]


@dataclass(frozen=True)
class ModeComparison:
    """Candidate vs reference mode vectors and scalar summaries."""

    candidate_modes: npt.NDArray[np.number]
    reference_modes: npt.NDArray[np.number]
    ratio: float
    cumulative_ratio: npt.NDArray[np.number]
    metric: MetricKind

    @property
    def candidate_variance_scale_modes(self) -> npt.NDArray[np.number]:
        """Variance-scale modes (sqrt energy); identity for kappa comparisons.

        ``energy_signed`` modes may be negative (unbiased CV estimate dipping below
        the noise floor) or complex (the asymmetric round-the-house kernel is not
        guaranteed to have real eigenvalues); ``np.sign``/``np.abs`` both handle
        complex input natively, so the sign/magnitude split stays meaningful without
        any extra branching here.
        """
        if self.metric == "energy":
            return np.sqrt(np.maximum(self.candidate_modes, 0.0))
        if self.metric == "energy_signed":
            return np.sign(self.candidate_modes) * np.sqrt(np.abs(self.candidate_modes))
        return self.candidate_modes

    @property
    def reference_variance_scale_modes(self) -> npt.NDArray[np.number]:
        """Variance-scale modes (sqrt energy); identity for kappa comparisons.

        ``energy_signed`` modes may be negative (unbiased CV estimate dipping below
        the noise floor) or complex (the asymmetric round-the-house kernel is not
        guaranteed to have real eigenvalues); ``np.sign``/``np.abs`` both handle
        complex input natively, so the sign/magnitude split stays meaningful without
        any extra branching here.
        """
        if self.metric == "energy":
            return np.sqrt(np.maximum(self.reference_modes, 0.0))
        if self.metric == "energy_signed":
            return np.sign(self.reference_modes) * np.sqrt(np.abs(self.reference_modes))
        return self.reference_modes

    def as_variance_scale(self) -> "ModeComparison":
        """Energy comparison mapped to variance scale with ratio recomputed."""
        return _comparison(
            self.candidate_variance_scale_modes,
            self.reference_variance_scale_modes,
            metric="kappa",
        )


@dataclass(frozen=True)
class CVPCAComparison:
    """rCVPCA computed in position space and neuron space."""

    modes_position: npt.NDArray[np.floating]
    modes_neuron: npt.NDArray[np.floating]
    num_neurons: int
    num_positions: int

    @property
    def p2n_ratio(self) -> float:
        return float((self.num_neurons - 1) / (self.num_positions - 1))


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
    reference_no_private_noise_spectrum: Optional[npt.NDArray[np.floating]] = None


@dataclass(frozen=True)
class NeuronSplit:
    """Source/target neuron index groups for round-the-house estimation."""

    source_idx: npt.NDArray[np.integer]
    target_idx: npt.NDArray[np.integer]


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
    cv_variance_scale: Optional[ModeComparison] = None
    cv_rcvpca: Optional[CVPCAComparison] = None
    roundhouse: Optional[ModeComparison] = None
    roundhouse_sym: Optional[ModeComparison] = None
    mtfa: Optional[ModeComparison] = None


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
    config: StimFullConfig | PlacefieldFullConfig | CovariancePairConfig | SharedSpaceConfig | None

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


def _mtfa_shrink(sigma: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Minimum-trace factor analysis: maximal private-variance diagonal s.t. Sigma - diag(d) is PSD."""
    p = sigma.shape[0]
    d = cp.Variable(p, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(d)), [sigma - cp.diag(d) >> 0])
    prob.solve(solver=cp.SCS)
    shared = sigma - np.diag(d.value)
    return shared.astype(sigma.dtype, copy=False), d.value


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


def _ratio(candidate_modes: npt.NDArray[np.number], reference_modes: npt.NDArray[np.number]) -> float:
    """Sum-of-modes ratio. Eigenvalue sums of a real matrix are exactly real (trace), so the
    full sum's imaginary part is float noise even when individual modes are complex; only the
    aggregate, not the individual eigenvalues, is forced real here."""
    denom = float(np.sum(reference_modes).real)
    if denom <= 0.0:
        return np.nan
    return float(np.sum(candidate_modes).real / denom)


def _cumulative_ratio(
    candidate_modes: npt.NDArray[np.number],
    reference_modes: npt.NDArray[np.number],
) -> npt.NDArray[np.number]:
    """Cumulative sum-of-modes ratio. Partial sums over a subset of eigenvalues (e.g. half of a
    complex-conjugate pair) can be genuinely complex, so the result dtype follows the inputs;
    only the where-mask (deciding where division is attempted) needs a real comparison."""
    n_modes = max(len(candidate_modes), len(reference_modes))
    dtype = complex if (np.iscomplexobj(candidate_modes) or np.iscomplexobj(reference_modes)) else float
    candidate = np.zeros(n_modes, dtype=dtype)
    reference = np.zeros(n_modes, dtype=dtype)
    candidate[: len(candidate_modes)] = candidate_modes
    reference[: len(reference_modes)] = reference_modes
    reference_cumulative = np.cumsum(reference)
    return np.divide(
        np.cumsum(candidate),
        reference_cumulative,
        out=np.full(n_modes, np.nan, dtype=dtype),
        where=reference_cumulative.real > 0.0,
    )


def _comparison(
    candidate_modes: npt.NDArray[np.number],
    reference_modes: npt.NDArray[np.number],
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
    reference_no_private_noise_covariance: Optional[npt.NDArray[np.floating]] = None,
) -> SubspaceGeometry:
    """Eigenstructure and overlap diagnostics for a candidate/reference pair."""
    w_candidate, v_candidate = _sorted_eigenvalues_and_eigenvectors(candidate_covariance)
    w_reference, v_reference = _sorted_eigenvalues_and_eigenvectors(reference_covariance)
    reference_on_candidate = ((v_candidate.T @ v_reference) ** 2) @ w_reference
    candidate_on_reference = ((v_reference.T @ v_candidate) ** 2) @ w_candidate
    reference_no_private_noise_spectrum = (
        _sorted_eigenvalues(reference_no_private_noise_covariance) if reference_no_private_noise_covariance is not None else None
    )
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
        
        reference_no_private_noise_spectrum=reference_no_private_noise_spectrum,
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


def _neuron_split(num_neurons: int) -> NeuronSplit:
    """Deterministic midpoint neuron split for round-the-house estimation."""
    if num_neurons < 2:
        raise ValueError(f"Need at least 2 neurons for round-the-house split, got {num_neurons}")
    midpoint = num_neurons // 2
    if midpoint == 0:
        raise ValueError(f"Midpoint split leaves empty source group for num_neurons={num_neurons}")
    return NeuronSplit(
        source_idx=np.arange(midpoint, dtype=np.intp),
        target_idx=np.arange(midpoint, num_neurons, dtype=np.intp),
    )


def _center_rows(matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Per-neuron centering: each row has zero mean across columns (time/stimuli)."""
    matrix = np.asarray(matrix)
    return matrix - np.mean(matrix, axis=1, keepdims=True, dtype=matrix.dtype)


def _roundhouse_kernel(
    m00: npt.NDArray[np.floating],
    m10: npt.NDArray[np.floating],
    m11: npt.NDArray[np.floating],
    m01: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Build the round-the-house kernel from four neuron/time blocks.

    Parameters
    ----------
    m00, m10
        Draw 0 blocks with shapes ``(N0, T0)`` and ``(N1, T0)``.
    m11, m01
        Draw 1 blocks with shapes ``(N1, T1)`` and ``(N0, T1)``.

    Notes
    -----
    Each pair sharing a sample axis (m00/m10 over T0, m11/m01 over T1) is scaled
    via ``_precov`` (center, divide by ``sqrt(n_samples - 1)``) so the resulting
    cross-products are in covariance units, matching ``kappa_modes``/``energy_modes``
    elsewhere in this module.
    """
    g00, g10 = _precov(m00), _precov(m10)
    g11, g01 = _precov(m11), _precov(m01)
    return g00 @ g10.T @ g11 @ g01.T


def _roundhouse_modes(kernel: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Energy-scale modes from a round-the-house kernel (symmetrized, clipped nonnegative)."""
    return np.maximum(_sorted_eigenvalues(kernel, symmetrize=True), 0.0)


def _roundhouse_modes_asymmetric(kernel: npt.NDArray[np.floating]) -> npt.NDArray[np.number]:
    """Signed, possibly-complex energy-scale modes from the literal (non-symmetrized)
    round-the-house kernel.

    The kernel is a product of two distinct cross-covariance blocks, so it isn't
    symmetric by construction, and its eigenvalues are not guaranteed real. Both
    negative and complex eigenvalues are kept as-is: they're expected outputs of
    this asymmetric estimator (negative values reflect legitimate cross-validation
    unbiasedness; complex-conjugate pairs reflect genuine 2D oscillatory structure
    in the kernel), not numerical noise to be clipped, dropped, or guarded against.
    """
    return np.sort(np.linalg.eigvals(kernel))[::-1]


def _roundhouse_modes_from_matrices(
    split: NeuronSplit,
    matrix00: npt.NDArray[np.floating],
    matrix10: npt.NDArray[np.floating],
    matrix11: npt.NDArray[np.floating],
    matrix01: npt.NDArray[np.floating],
    *,
    symmetric: bool = True,
) -> npt.NDArray[np.floating]:
    """Slice full-neuron matrices into source/target groups and return roundhouse modes."""
    source_idx = split.source_idx
    target_idx = split.target_idx
    kernel = _roundhouse_kernel(
        matrix00[source_idx],
        matrix10[target_idx],
        matrix11[target_idx],
        matrix01[source_idx],
    )
    return _roundhouse_modes(kernel) if symmetric else _roundhouse_modes_asymmetric(kernel)


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
        reference_no_private_noise_covariance=sigma_stim + sigma_nuisance,
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


def _stim_full_cv_variance_scale_result(
    gen: "StimFullGenerator",
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
) -> tuple[ModeComparison, dict[str, Any]]:
    """CV stimulus-full modes on variance scale.

    Aquires a cross-validated estimator of stim-full covariance on the variance scale.
      directions: stim1.T @ data3
      score:      stim2.T @ data3
    Since stim[n] share stimuli in each column but data matrices don't have matched time samples
    we need to keep one data3 for each reference in the train/test. This way we measure directions
    (with SVD) comparing one stim repeat with a reference data, then test that reference data with
    another stim repeat. We do it multiple times for each fold pairing for a better estimate.

    The reference is the full/full cross-validated kappa, constructed identically to
    ``reference_kappa`` in ``_stim_full_cv_kappa_result`` (two draws learn the subspace, two score
    it), so the candidate/reference ratio is a variance-scale SVR rather than a CV-survival fraction.
    """

    def cvsvd_stimfull(stim1, stim2, data3):
        stim1 = stim1 - np.mean(stim1, axis=1, keepdims=True)
        stim2 = stim2 - np.mean(stim2, axis=1, keepdims=True)
        data3 = data3 - np.mean(data3, axis=1, keepdims=True)
        sf_cross_train = stim1.T @ data3
        sf_cross_test = stim2.T @ data3
        Usf_train, _, Vsft_train = np.linalg.svd(sf_cross_train, full_matrices=False)
        ssf_test = np.sum(Usf_train * (sf_cross_test @ Vsft_train.T), axis=0)
        norm = np.sqrt((stim1.shape[1] - 1) * (data3.shape[1] - 1))
        return ssf_test / norm

    data_0, _, extras_0 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_1, _, extras_1 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_2, _, extras_2 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_3, _, extras_3 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)

    ns = gen.config.num_stimuli
    s_0 = _stimulus_means(data_0, extras_0["stim_indices"], ns)
    s_1 = _stimulus_means(data_1, extras_1["stim_indices"], ns)
    s_2 = _stimulus_means(data_2, extras_2["stim_indices"], ns)
    s_3 = _stimulus_means(data_3, extras_3["stim_indices"], ns)

    stim_list = [s_0, s_1, s_2, s_3]
    data_list = [data_0, data_1, data_2, data_3]
    candidate_modes = []
    for i, j, k in itertools.combinations(range(4), 3):
        candidate_modes.append(cvsvd_stimfull(stim_list[i], stim_list[j], data_list[k]))
    candidate_modes = np.stack(candidate_modes, axis=0)
    candidate_modes = np.mean(candidate_modes, axis=0)

    # Reference: full/full CV kappa, identical construction to reference_kappa in
    # _stim_full_cv_kappa_result. Two draws learn the subspace, two score it.
    root_full_train0 = sqrtm_spd(_cov(data_0))
    root_full_train1 = sqrtm_spd(_cov(data_1))
    root_full_test0 = sqrtm_spd(_cov(data_2))
    root_full_test1 = sqrtm_spd(_cov(data_3))
    reference_modes = _cv_kappa_fit_score(root_full_train0, root_full_train1, root_full_test0, root_full_test1)

    return _comparison(candidate_modes, reference_modes, metric="kappa"), {}


def _stim_full_rcvpca_result(
    gen: "PlacefieldFullGenerator",
    rng: np.random.Generator,
    noise_variance: float,
) -> tuple[ModeComparison, dict[str, Any]]:
    """Regularized cvPCA estimator for the place-field stim-full pipeline.

    Uses the generator's explicit repeat structure (``extras["repeat_maps"]``): the first four
    repeats play the r0/r1/r2/r3 roles of the place-field study. The candidate spectrum is the
    stimulus(position)-space cvPCA covariance from a smoothed training repeat r0, scored on the
    held-out repeats r2/r3. The reference is the neuron-space cvPCA covariance fit on the same r0
    (unsmoothed) and scored on r2/r3 — i.e. the total reproducible covariance. Their ratio is an
    SVR analog: fraction of reproducible variance carried by the place-field (position) subspace.

    These are signed cross-validated covariances (variance scale), so ``metric="kappa"`` is used
    for scale bookkeeping; individual modes may be negative at the noise floor.

    NOTE: this is rCVPCA, NOT the canonical cross-validated place-field-kernel estimator in
    ``subspace_analysis/stimspace.py`` (``StimSpaceSubspace``), which is not yet ported into the
    atlas.
    """
    cfg = gen.config
    P = cfg.num_positions
    n_repeats = cfg.n_repeats
    if n_repeats < 4:
        raise ValueError(f"rCVPCA needs n_repeats >= 4 (r0/r1/r2/r3), got {n_repeats}")

    _, _, extras = gen.generate(n_repeats * P, noise_variance=noise_variance, rng=rng, return_extras=True)
    repeat_maps = extras["repeat_maps"]
    r0, _r1, r2, r3 = (torch.as_tensor(repeat_maps[i]) for i in range(4))

    n_comp = cfg.rcvpca_num_components if cfg.rcvpca_num_components is not None else min(cfg.num_neurons - 1, P - 1)
    n_comp = min(n_comp, cfg.num_neurons - 1, P - 1)

    r0_smooth = gaussian_filter(r0, cfg.rcvpca_smooth_width, axis=1) if cfg.rcvpca_smooth_width is not None else r0

    # Candidate: position-space (on_stimuli) cvPCA from the smoothed training repeat.
    cvpca_position = CVPCA(num_components=n_comp, center=cfg.rcvpca_center, on_stimuli=True).fit(r0_smooth)
    modes_position = cvpca_position.score(r2, r3).cpu().numpy()

    # Reference: neuron-space cvPCA from the unsmoothed training repeat = total reproducible cov.
    cvpca_neuron = CVPCA(num_components=n_comp, center=cfg.rcvpca_center, on_stimuli=False).fit(r0_smooth)
    modes_neuron = cvpca_neuron.score(r2, r3).cpu().numpy()

    return CVPCAComparison(
        modes_position,
        modes_neuron,
        num_neurons=cfg.num_neurons,
        num_positions=cfg.num_positions,
    )


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


def _mtfa_kappa_comparison(
    candidate_cov: npt.NDArray[np.floating],
    reference_cov1: npt.NDArray[np.floating],
    reference_cov2: npt.NDArray[np.floating],
) -> ModeComparison:
    """Kappa comparison between MTFA-shrunk (shared-only) covariances.

    Candidate: shared(candidate) vs shared(reference draw 2).
    Reference: shared(reference draw 1) vs shared(reference draw 2).
    """
    shared_candidate, _ = _mtfa_shrink(candidate_cov)
    shared_reference1, _ = _mtfa_shrink(reference_cov1)
    shared_reference2, _ = _mtfa_shrink(reference_cov2)

    candidate_modes = kappa_modes(shared_candidate, shared_reference2)
    reference_modes = kappa_modes(shared_reference1, shared_reference2)
    return _comparison(candidate_modes, reference_modes, metric="kappa")


def _stim_full_mtfa_result(
    gen: "StimFullGenerator",
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
) -> tuple[ModeComparison, dict[str, Any]]:
    """MTFA shared-only kappa for the stim-full pipeline: stim_means vs two independent full draws."""
    data_stim, _, extras_stim = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_1, _, _ = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)
    data_2, _, _ = gen.generate(num_samples, noise_variance=noise_variance, rng=rng, return_extras=True)

    stim_means = _stimulus_means(data_stim, extras_stim["stim_indices"], gen.config.num_stimuli)

    comparison = _mtfa_kappa_comparison(_cov(stim_means), _cov(data_1), _cov(data_2))
    return comparison, {}


def _context_mtfa_result(
    gen,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
) -> tuple[ModeComparison, dict[str, Any]]:
    """MTFA shared-only kappa for the context-pair pipeline: candidate vs two independent reference draws."""
    if isinstance(gen, CovariancePairGenerator):
        candidate = gen.generate(num_samples, which="candidate", noise_variance=noise_variance, rng=rng)
        reference_1 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
        reference_2 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
    elif isinstance(gen, SharedSpaceGenerator):
        candidate, reference_1 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        _, reference_2 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
    else:
        raise TypeError(f"Unsupported context-pair generator type: {type(gen).__name__}")

    comparison = _mtfa_kappa_comparison(_cov(candidate), _cov(reference_1), _cov(reference_2))
    return comparison, {}


def _stim_full_roundhouse_result(
    gen,
    split: NeuronSplit,
    *,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
    test_rotation_angle: float,
) -> tuple[ModeComparison, ModeComparison]:
    """
    Empirical round-the-house energy comparison for the stim-full pipeline.

    Reference uses raw data from two full draws; candidate replaces the first
    cross-leg with per-stimulus means from draw 0 and keeps draw 1 full data
    for the trail leg.

    Returns the symmetrized comparison (clipped nonnegative, ``metric="energy"``)
    and the asymmetric/signed comparison (unclipped, ``metric="energy_signed"``),
    both built from the same draws so they're directly comparable.
    """
    data_ref0, _, extras0 = gen.generate(
        num_samples,
        noise_variance=noise_variance,
        rng=rng,
        return_extras=True,
    )
    data_ref1, _, _ = gen.generate(
        num_samples,
        noise_variance=noise_variance,
        rotation_angle=test_rotation_angle,
        rng=rng,
        return_extras=True,
    )
    stim_means = _stimulus_means(data_ref0, extras0["stim_indices"], gen.config.num_stimuli)

    reference_sym = _roundhouse_modes_from_matrices(split, data_ref0, data_ref0, data_ref1, data_ref1, symmetric=True)
    candidate_sym = _roundhouse_modes_from_matrices(split, stim_means, stim_means, data_ref1, data_ref1, symmetric=True)
    reference_asym = _roundhouse_modes_from_matrices(split, data_ref0, data_ref0, data_ref1, data_ref1, symmetric=False)
    candidate_asym = _roundhouse_modes_from_matrices(split, stim_means, stim_means, data_ref1, data_ref1, symmetric=False)

    return (
        _comparison(candidate_sym, reference_sym, metric="energy"),
        _comparison(candidate_asym, reference_asym, metric="energy_signed"),
    )


def _context_roundhouse_result(
    gen,
    split: NeuronSplit,
    *,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
) -> tuple[ModeComparison, ModeComparison]:
    """
    Empirical round-the-house energy comparison for the context-pair pipeline.

    Reference uses two reference draws; candidate uses candidate draw 0 for
    the lead leg and reference draw 1 for the trail leg.

    Returns the symmetrized comparison (clipped nonnegative, ``metric="energy"``)
    and the asymmetric/signed comparison (unclipped, ``metric="energy_signed"``),
    both built from the same draws so they're directly comparable.
    """
    if isinstance(gen, CovariancePairGenerator):
        ref0 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
        ref1 = gen.generate(num_samples, which="reference", noise_variance=noise_variance, rng=rng)
        cand0 = gen.generate(num_samples, which="candidate", noise_variance=noise_variance, rng=rng)
    elif isinstance(gen, SharedSpaceGenerator):
        _, ref0 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        _, ref1 = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        cand0, _ = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
    else:
        raise TypeError(f"Unsupported context-pair generator type: {type(gen).__name__}")

    reference_sym = _roundhouse_modes_from_matrices(split, ref0, ref0, ref1, ref1, symmetric=True)
    candidate_sym = _roundhouse_modes_from_matrices(split, cand0, cand0, ref1, ref1, symmetric=True)
    reference_asym = _roundhouse_modes_from_matrices(split, ref0, ref0, ref1, ref1, symmetric=False)
    candidate_asym = _roundhouse_modes_from_matrices(split, cand0, cand0, ref1, ref1, symmetric=False)

    return (
        _comparison(candidate_sym, reference_sym, metric="energy"),
        _comparison(candidate_asym, reference_asym, metric="energy_signed"),
    )


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
    config: StimFullConfig | PlacefieldFullConfig | CovariancePairConfig | SharedSpaceConfig | None = None,
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
        split = _neuron_split(gen.config.num_neurons)
        extras["neuron_split"] = split
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
            cv_variance_scale, _ = _stim_full_cv_variance_scale_result(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
            )
            # rCVPCA: only for generators that opt in and emit per-repeat maps (place fields).
            # StimFullGenerator lacks the repeat structure, so cv_rcvpca stays None there.
            cv_rcvpca: Optional[CVPCAComparison] = None
            if getattr(gen, "supports_rcvpca", False):
                cv_rcvpca = _stim_full_rcvpca_result(gen, rng=rng, noise_variance=noise_variance)

            roundhouse_sym, roundhouse = _stim_full_roundhouse_result(
                gen,
                split,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
                test_rotation_angle=test_rotation_angle,
            )
            mtfa, _ = _stim_full_mtfa_result(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
            )

            empirical = EmpiricalBlock(
                kappa=empirical_kappa,
                cv_energy=cv_energy,
                cv_kappa=cv_kappa,
                cv_stimstim=cv_stimstim,
                cv_variance_scale=cv_variance_scale,
                cv_rcvpca=cv_rcvpca,
                roundhouse=roundhouse,
                roundhouse_sym=roundhouse_sym,
                mtfa=mtfa,
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
            roundhouse_sym, roundhouse = _context_roundhouse_result(
                gen,
                split,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
            )
            mtfa, _ = _context_mtfa_result(
                gen,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
            )
            empirical = EmpiricalBlock(
                kappa=empirical.kappa,
                cv_kappa=cv_kappa,
                roundhouse=roundhouse,
                roundhouse_sym=roundhouse_sym,
                mtfa=mtfa,
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
    config: StimFullConfig | PlacefieldFullConfig | CovariancePairConfig | SharedSpaceConfig,
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
    if config_type in _CONFIG_DISPATCH:
        kind, pipeline, gen_class = _CONFIG_DISPATCH[config_type]
    else:
        # Fall back to name-based lookup so autoreload doesn't break type identity.
        _by_name = {k.__name__: v for k, v in _CONFIG_DISPATCH.items()}
        if config_type.__name__ not in _by_name:
            raise TypeError(f"Unsupported config type: {config_type.__name__}. Expected one of: {', '.join(t.__name__ for t in _CONFIG_DISPATCH)}")
        kind, pipeline, gen_class = _by_name[config_type.__name__]
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


def _placefield_config(rng: np.random.Generator, *, field_model: Any) -> PlacefieldFullConfig:
    return PlacefieldFullConfig(
        field_model=field_model,
        num_neurons=200,
        num_positions=50,
        n_repeats=4,
        rng=rng,
    )


def _placefield_generator(config: PlacefieldFullConfig, dtype: npt.DTypeLike = np.float64) -> PlacefieldFullGenerator:
    return PlacefieldFullGenerator(config, dtype=np.dtype(dtype))


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
        AtlasSpec(
            name="placefield_full.smooth",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Smooth (non-thresholded) RBF Gaussian-process place fields; positions act as stimuli.",
            builder=lambda rng, dtype: (
                cfg := _placefield_config(rng, field_model=SmoothGPFieldConfig()),
                _placefield_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="placefield_full.thresholded",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Thresholded + rectified RBF-GP place fields (sparse, localized); positions act as stimuli.",
            builder=lambda rng, dtype: (
                cfg := _placefield_config(rng, field_model=ThresholdedGPFieldConfig()),
                _placefield_generator(cfg, dtype),
            ),
        ),
        AtlasSpec(
            name="placefield_full.tilbury",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Per-neuron double generalized-Gaussian (Tilbury) place fields; positions act as stimuli.",
            builder=lambda rng, dtype: (
                cfg := _placefield_config(rng, field_model=TilburyFieldConfig()),
                _placefield_generator(cfg, dtype),
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
    "NeuronSplit",
    "PlacefieldFullConfig",
    "PlacefieldFullGenerator",
    "PopulationBlock",
    "SmoothGPFieldConfig",
    "SubspaceGeometry",
    "ThresholdedGPFieldConfig",
    "TilburyFieldConfig",
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
