"""Named simulation atlas for shared-variance operator examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional

import numpy as np
import numpy.typing as npt
from vrAnalysis.helpers import smart_pca

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
AtlasBuilder = Callable[[np.random.Generator], tuple[Any, Any]]


@dataclass(frozen=True)
class AtlasSpec:
    """Registry entry for one named atlas condition."""

    name: str
    kind: AtlasKind
    pipeline: AtlasPipeline
    description: str
    builder: AtlasBuilder

    def build(self, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> "AtlasBuild":
        """Instantiate this atlas condition."""
        if seed is not None and rng is not None:
            raise ValueError("Provide seed or rng, not both.")
        _rng = np.random.default_rng(seed) if rng is None else rng
        config, generator = self.builder(_rng)
        return AtlasBuild(
            name=self.name,
            kind=self.kind,
            pipeline=self.pipeline,
            description=self.description,
            config=config,
            generator=generator,
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


@dataclass(frozen=True)
class AtlasAnalysisResult:
    """Shared output shape for stimulus-full and context-pair analyses."""

    name: str
    kind: AtlasKind
    pipeline: AtlasPipeline
    description: str

    population_svr: float
    population_candidate_modes: npt.NDArray[np.floating]
    population_target_modes: npt.NDArray[np.floating]
    population_cumulative_svr: npt.NDArray[np.floating]

    empirical_svr: Optional[float] = None
    empirical_candidate_modes: Optional[npt.NDArray[np.floating]] = None
    empirical_target_modes: Optional[npt.NDArray[np.floating]] = None
    empirical_cumulative_svr: Optional[npt.NDArray[np.floating]] = None

    empirical_cvser: Optional[float] = None
    empirical_cv_candidate_energy_modes: Optional[npt.NDArray[np.floating]] = None
    empirical_cv_target_energy_modes: Optional[npt.NDArray[np.floating]] = None
    empirical_cv_cumulative_ser: Optional[npt.NDArray[np.floating]] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)


def _precov(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return centered data scaled so G @ G.T equals np.cov(data)."""
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    n_samples = data.shape[1]
    if n_samples < 2:
        raise ValueError("Need at least two samples to compute covariance.")
    centered = data - np.mean(data, axis=1, keepdims=True)
    return centered / np.sqrt(n_samples - 1)


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


def _svr(candidate_modes: npt.NDArray[np.floating], target_modes: npt.NDArray[np.floating]) -> float:
    denom = float(np.sum(target_modes))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(candidate_modes) / denom)


def _cumulative_svr(candidate_modes: npt.NDArray[np.floating], target_modes: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    n_modes = max(len(candidate_modes), len(target_modes))
    candidate = np.zeros(n_modes, dtype=float)
    target = np.zeros(n_modes, dtype=float)
    candidate[: len(candidate_modes)] = candidate_modes
    target[: len(target_modes)] = target_modes
    target_cumulative = np.cumsum(target)
    return np.divide(
        np.cumsum(candidate),
        target_cumulative,
        out=np.full(n_modes, np.nan, dtype=float),
        where=target_cumulative > 0.0,
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
        means[:, istim] = np.mean(data[:, stim_indices == istim], axis=1)
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


def _stim_full_population_result(build: AtlasBuild) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gen: StimFullGenerator = build.generator
    sigma_stim, sigma_nuisance, sigma_eps = gen.true_covariance()
    sigma_full = sigma_stim + sigma_nuisance + sigma_eps

    stim_responses = gen.stim_space @ np.diag(np.sqrt(gen.stim_spectrum)) @ gen.stim_latents
    stim_precov = _precov(stim_responses)
    candidate_modes = stimulus_space_kappa_modes(stim_precov, sigma_full)
    target_modes = kappa_modes(sigma_full, sigma_full)
    candidate_energy_modes = stimulus_space_energy_modes(stim_precov, sigma_full)
    target_energy_modes = energy_modes(sigma_full, sigma_full)

    # Also measure eigenvalues and overlap energy of stim and full-to-stim
    w_stim, v_stim = smart_pca(sigma_stim)
    w_full, v_full = smart_pca(sigma_full)
    energy_on_stim = ((v_stim.T @ v_full) ** 2) @ w_full
    energy_on_full = ((v_full.T @ v_stim) ** 2) @ w_stim

    metadata = {
        "sigma_stim": sigma_stim,
        "sigma_nuisance": sigma_nuisance,
        "sigma_eps": sigma_eps,
        "sigma_full": sigma_full,
        "trace_stim": float(np.trace(sigma_stim)),
        "trace_nuisance": float(np.trace(sigma_nuisance)),
        "trace_eps": float(np.trace(sigma_eps)),
        "trace_full": float(np.trace(sigma_full)),
        "cka": centered_kernel_alignment(sigma_stim, sigma_full),
        "ser": _svr(candidate_energy_modes, target_energy_modes),
        "candidate_energy_modes": candidate_energy_modes,
        "target_energy_modes": target_energy_modes,
        "stimulus_space_modes_match_covariance_modes": bool(
            np.allclose(candidate_modes, kappa_modes(sigma_stim, sigma_full)[: len(candidate_modes)], atol=1e-8)
        ),
        # Using names shared with context* version for simplicity
        "candidate_spectrum": w_stim,
        "target_spectrum": w_full,
        "target_on_candidate_overlap": energy_on_stim,
        "candidate_on_target_overlap": energy_on_full,
    }
    return candidate_modes, target_modes, metadata


def _stim_full_cvser_result(
    data_train: npt.NDArray[np.floating],
    stim_indices: npt.NDArray[np.integer],
    num_stimuli: int,
    full_train: npt.NDArray[np.floating],
    full_test: npt.NDArray[np.floating],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], dict[str, Any]]:
    folds = _stimulus_balanced_folds(stim_indices, num_stimuli, num_folds=3, rng=rng)
    fold_precovs = [_precov(_stimulus_means(data_train[:, fold], stim_indices[fold], num_stimuli)) for fold in folds]

    direction_kernel = fold_precovs[0].T @ _symmetrize(full_test) @ fold_precovs[0]
    train_energy_modes, directions = _energy_directions(direction_kernel)

    cv_kernel = fold_precovs[1].T @ full_test @ fold_precovs[2]
    candidate_energy_modes = _project_energy_modes(cv_kernel, directions, symmetrize=False)
    target_energy_modes = energy_modes(full_train, full_test)

    metadata = {
        "empirical_cv_candidate_train_energy_modes": train_energy_modes,
        "empirical_cv_full_energy": float(np.trace(full_train @ full_test)),
        "empirical_cv_stim_full_energy": float(np.sum(candidate_energy_modes)),
        "empirical_cv_fold_sizes": tuple(int(len(fold)) for fold in folds),
    }
    return candidate_energy_modes, target_energy_modes, metadata


def _stim_full_empirical_result(
    build: AtlasBuild,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
    test_rotation_angle: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gen: StimFullGenerator = build.generator
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
    full_train = np.cov(data_train)
    full_test = np.cov(data_test)

    candidate_modes = stimulus_space_kappa_modes(stim_precov, full_test)
    target_modes = kappa_modes(full_train, full_test)
    cv_candidate_energy_modes, cv_target_energy_modes, cv_metadata = _stim_full_cvser_result(
        data_train,
        extras_train["stim_indices"],
        gen.config.num_stimuli,
        full_train,
        full_test,
        rng,
    )
    metadata = {
        "empirical_full_train": full_train,
        "empirical_full_test": full_test,
        "empirical_stim_means": stim_means,
        "empirical_cka": centered_kernel_alignment(np.cov(stim_means), full_test),
        "empirical_cv_candidate_energy_modes": cv_candidate_energy_modes,
        "empirical_cv_target_energy_modes": cv_target_energy_modes,
        "empirical_cvser": _svr(cv_candidate_energy_modes, cv_target_energy_modes),
        **cv_metadata,
    }
    return candidate_modes, target_modes, metadata


def _context_population_covariances(build: AtlasBuild) -> tuple[np.ndarray, np.ndarray]:
    gen = build.generator
    if isinstance(gen, CovariancePairGenerator):
        return gen.expected_covariances()
    if isinstance(gen, SharedSpaceGenerator):
        return gen.true_covariance()
    if hasattr(gen, "true_covariance"):
        covariances = gen.true_covariance()
        if len(covariances) != 2:
            raise ValueError(f"Expected two covariances from {type(gen).__name__}.true_covariance()")
        return covariances
    raise TypeError(f"Unsupported context-pair generator type: {type(gen).__name__}")


def _context_population_result(build: AtlasBuild) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_covariance, target_covariance = _context_population_covariances(build)
    candidate_modes = kappa_modes(candidate_covariance, target_covariance)
    target_modes = kappa_modes(target_covariance, target_covariance)

    # Also measure eigenvalues and overlap energy of stim and full-to-stim
    w_candidate, v_candidate = smart_pca(candidate_covariance)
    w_target, v_target = smart_pca(target_covariance)
    energy_target_on_candidate = ((v_candidate.T @ v_target) ** 2) @ w_target
    energy_candidate_on_target = ((v_target.T @ v_candidate) ** 2) @ w_candidate

    metadata = {
        "candidate_covariance": candidate_covariance,
        "target_covariance": target_covariance,
        "trace_candidate": float(np.trace(candidate_covariance)),
        "trace_target": float(np.trace(target_covariance)),
        "cka": centered_kernel_alignment(candidate_covariance, target_covariance),
        "candidate_spectrum": w_candidate,
        "target_spectrum": w_target,
        "target_on_candidate_overlap": energy_target_on_candidate,
        "candidate_on_target_overlap": energy_candidate_on_target,
    }
    return candidate_modes, target_modes, metadata


def _context_empirical_result(
    build: AtlasBuild,
    num_samples: int,
    rng: np.random.Generator,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gen = build.generator
    if isinstance(gen, CovariancePairGenerator):
        candidate_train = gen.generate(num_samples, which="candidate", noise_variance=noise_variance, rng=rng)
        target_train = gen.generate(num_samples, which="target", noise_variance=noise_variance, rng=rng)
        target_test = gen.generate(num_samples, which="target", noise_variance=noise_variance, rng=rng)
    elif isinstance(gen, SharedSpaceGenerator):
        candidate_train, target_train = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
        _, target_test = gen.generate(num_samples, noise_variance=noise_variance, rng=rng)
    else:
        raise TypeError(f"Unsupported context-pair generator type: {type(gen).__name__}")

    candidate_covariance = np.cov(candidate_train)
    target_train_covariance = np.cov(target_train)
    target_test_covariance = np.cov(target_test)

    candidate_modes = kappa_modes(candidate_covariance, target_test_covariance)
    target_modes = kappa_modes(target_train_covariance, target_test_covariance)
    metadata = {
        "empirical_candidate_covariance": candidate_covariance,
        "empirical_target_train_covariance": target_train_covariance,
        "empirical_target_test_covariance": target_test_covariance,
        "empirical_cka": centered_kernel_alignment(candidate_covariance, target_test_covariance),
    }
    return candidate_modes, target_modes, metadata


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
    if build.kind == "stim_full":
        population_candidate, population_target, metadata = _stim_full_population_result(build)
    elif build.kind == "context_pair":
        population_candidate, population_target, metadata = _context_population_result(build)
    else:
        raise ValueError(f"Unknown atlas kind: {build.kind}")

    empirical_candidate = None
    empirical_target = None
    empirical_cumulative = None
    empirical_value = None
    empirical_cv_candidate_energy = None
    empirical_cv_target_energy = None
    empirical_cv_cumulative = None
    empirical_cv_value = None
    if num_samples is not None:
        rng = np.random.default_rng(sample_seed)
        if build.kind == "stim_full":
            empirical_candidate, empirical_target, empirical_metadata = _stim_full_empirical_result(
                build,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
                test_rotation_angle=test_rotation_angle,
            )
        else:
            empirical_candidate, empirical_target, empirical_metadata = _context_empirical_result(
                build,
                num_samples=num_samples,
                rng=rng,
                noise_variance=noise_variance,
            )
        metadata = {**metadata, **empirical_metadata}
        empirical_value = _svr(empirical_candidate, empirical_target)
        empirical_cumulative = _cumulative_svr(empirical_candidate, empirical_target)
        empirical_cv_candidate_energy = empirical_metadata.get("empirical_cv_candidate_energy_modes")
        empirical_cv_target_energy = empirical_metadata.get("empirical_cv_target_energy_modes")
        if empirical_cv_candidate_energy is not None and empirical_cv_target_energy is not None:
            empirical_cv_value = _svr(empirical_cv_candidate_energy, empirical_cv_target_energy)
            empirical_cv_cumulative = _cumulative_svr(empirical_cv_candidate_energy, empirical_cv_target_energy)

    return AtlasAnalysisResult(
        name=build.name,
        kind=build.kind,
        pipeline=build.pipeline,
        description=build.description,
        population_svr=_svr(population_candidate, population_target),
        population_candidate_modes=population_candidate,
        population_target_modes=population_target,
        population_cumulative_svr=_cumulative_svr(population_candidate, population_target),
        empirical_svr=empirical_value,
        empirical_candidate_modes=empirical_candidate,
        empirical_target_modes=empirical_target,
        empirical_cumulative_svr=empirical_cumulative,
        empirical_cvser=empirical_cv_value,
        empirical_cv_candidate_energy_modes=empirical_cv_candidate_energy,
        empirical_cv_target_energy_modes=empirical_cv_target_energy,
        empirical_cv_cumulative_ser=empirical_cv_cumulative,
        metadata=metadata,
    )


def analyze_atlas_case(
    name: str,
    *,
    seed: Optional[int] = None,
    num_samples: Optional[int] = None,
    sample_seed: Optional[int] = None,
    noise_variance: float = 0.0,
    test_rotation_angle: float = 0.0,
) -> AtlasAnalysisResult:
    """Build and analyze one named atlas case."""
    build = build_atlas_case(name, seed=seed)
    return analyze_build(
        build,
        num_samples=num_samples,
        sample_seed=sample_seed,
        noise_variance=noise_variance,
        test_rotation_angle=test_rotation_angle,
    )


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


def _stim_generator(config: StimFullConfig) -> StimFullGenerator:
    return StimFullGenerator(config)


def _cov_pair_config(
    rng: np.random.Generator,
    *,
    geometry: Literal["same", "random", "orthogonal", "angle", "partial"],
    alpha_candidate: float = 1.0,
    alpha_target: float = 1.0,
    angle: float = 0.0,
    shared_rank: Optional[int] = None,
    target_scale: float = 1.0,
) -> CovariancePairConfig:
    return CovariancePairConfig(
        num_neurons=200,
        candidate_rank=20,
        target_rank=20,
        alpha_candidate=alpha_candidate,
        alpha_target=alpha_target,
        target_scale=target_scale,
        geometry=geometry,
        angle=angle,
        shared_rank=shared_rank,
        rng=rng,
    )


def _cov_pair_generator(config: CovariancePairConfig) -> CovariancePairGenerator:
    return CovariancePairGenerator(config)


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


def _shared_space_generator(config: SharedSpaceConfig) -> SharedSpaceGenerator:
    return SharedSpaceGenerator(config)


def _make_specs() -> tuple[AtlasSpec, ...]:
    return (
        AtlasSpec(
            name="stim_full.identity",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus covariance is the full covariance; nuisance and diagonal noise are absent.",
            builder=lambda rng: (cfg := _stim_config(rng, nuisance_dim=0, nuisance_scale=0.0, noise_scale=0.0), _stim_generator(cfg)),
        ),
        AtlasSpec(
            name="stim_full.orthogonal_low_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus and nuisance subspaces are orthogonal; nuisance variance is modest.",
            builder=lambda rng: (cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=0.25), _stim_generator(cfg)),
        ),
        AtlasSpec(
            name="stim_full.orthogonal_high_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus and nuisance subspaces are orthogonal; full covariance is dominated by nuisance variance.",
            builder=lambda rng: (cfg := _stim_config(rng, nuisance_dim=40, nuisance_scale=3.0), _stim_generator(cfg)),
        ),
        AtlasSpec(
            name="stim_full.aligned_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance lies on the same axes as the stimulus covariance.",
            builder=lambda rng: (
                cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=3.0, nuisance_alignment="aligned"),
                _stim_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="stim_full.angled_nuisance_45",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance has 45 degree principal angles from stimulus covariance.",
            builder=lambda rng: (
                cfg := _stim_config(rng, nuisance_dim=10, nuisance_scale=3.0, nuisance_alignment="angle", nuisance_angle=np.pi / 4),
                _stim_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="stim_full.random_nuisance",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Nuisance covariance is an independent random subspace with expected ambient overlap.",
            builder=lambda rng: (
                cfg := _stim_config(rng, nuisance_dim=40, nuisance_scale=2.0, nuisance_alignment="random"),
                _stim_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="stim_full.high_diagonal_noise",
            kind="stim_full",
            pipeline="stimulus_space",
            description="Stimulus and nuisance are present, but independent neuron-specific variance dominates the full covariance.",
            builder=lambda rng: (
                cfg := _stim_config(rng, nuisance_dim=20, nuisance_scale=0.5, noise_scale=2.0),
                _stim_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="context.identical",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances have identical eigenvectors and spectra.",
            builder=lambda rng: (cfg := _cov_pair_config(rng, geometry="same"), _cov_pair_generator(cfg)),
        ),
        AtlasSpec(
            name="context.rotated_45",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances have matched spectra but 45 degree principal-angle rotation.",
            builder=lambda rng: (
                cfg := _cov_pair_config(rng, geometry="angle", angle=np.pi / 4),
                _cov_pair_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="context.orthogonal",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances occupy orthogonal subspaces.",
            builder=lambda rng: (cfg := _cov_pair_config(rng, geometry="orthogonal"), _cov_pair_generator(cfg)),
        ),
        AtlasSpec(
            name="context.spectrum_mismatch",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target share eigenvectors but assign variance differently across modes.",
            builder=lambda rng: (
                cfg := _cov_pair_config(rng, geometry="same", alpha_candidate=0.25, alpha_target=2.0),
                _cov_pair_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="context.partial_overlap",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target share only the first five dimensions; remaining target axes are private.",
            builder=lambda rng: (
                cfg := _cov_pair_config(rng, geometry="partial", shared_rank=5),
                _cov_pair_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="context.random",
            kind="context_pair",
            pipeline="covariance",
            description="Candidate and target covariances are independent random subspaces.",
            builder=lambda rng: (cfg := _cov_pair_config(rng, geometry="random"), _cov_pair_generator(cfg)),
        ),
        AtlasSpec(
            name="shared_space.shared_dominant",
            kind="context_pair",
            pipeline="covariance",
            description="Two contexts share a common subspace and have weak private covariance.",
            builder=lambda rng: (
                cfg := _shared_space_config(rng, private_ratio=0.5),
                _shared_space_generator(cfg),
            ),
        ),
        AtlasSpec(
            name="shared_space.private_dominant",
            kind="context_pair",
            pipeline="covariance",
            description="Two contexts share a common subspace but private covariance dominates each context.",
            builder=lambda rng: (
                cfg := _shared_space_config(rng, private_ratio=3.0),
                _shared_space_generator(cfg),
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
) -> AtlasBuild:
    """Instantiate one named atlas case and return its config and generator."""
    return get_atlas_spec(name).build(seed=seed, rng=rng)


__all__ = [
    "ATLAS",
    "AtlasAnalysisResult",
    "AtlasBuild",
    "AtlasKind",
    "AtlasPipeline",
    "AtlasSpec",
    "analyze_atlas_case",
    "analyze_build",
    "build_atlas_case",
    "energy_modes",
    "get_atlas_spec",
    "kappa_modes",
    "list_atlas_cases",
    "stimulus_space_energy_modes",
    "stimulus_space_kappa_modes",
]
