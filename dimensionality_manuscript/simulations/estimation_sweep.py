"""SVR estimation sweep: oracle vs. empirical ratio curves across atlas conditions.

Two entry points:
  run_named_sweep  -- sweep existing ATLAS entries, varying geometry per seed
  run_param_sweep  -- user-defined parametric axis (e.g. nuisance_scale 0→5)

Both return SweepResults with shape (n_cases, n_seeds) arrays for each oracle
metric and each empirical estimation method, enabling estimation-curve plots of
true SVR vs. estimated SVR per method.

WARNING — noise_variance affects empirical draws only, NOT the oracle
--------------------------------------------------------------------
The oracle (PopulationBlock) is always computed from the exact true covariances
via gen.true_covariance() — zero noise, regardless of noise_variance.

The empirical block (EmpiricalBlock) is computed from gen.generate(...,
noise_variance=noise_variance), so added noise degrades empirical estimates.

Consequence: a nonzero noise_variance creates a DELIBERATE MISMATCH between
oracle and empirical ratios.  This is sometimes what you want:

  noise_variance=0.0  → "can the estimator recover the true SVR from
                         finite samples alone?" (sampling noise only)
  noise_variance>0.0  → "how does estimation hold up under added measurement
                         noise?" (useful for a robustness sweep)

Never mix these two interpretations in the same plot without labelling which
regime you are in.  The canonical comparison (oracle vs. empirical) should use
noise_variance=0.0 unless you are explicitly studying noise robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from tqdm import tqdm

import numpy as np
import numpy.typing as npt

from .generators import StimFullConfig
from .placefield_full import PlacefieldFullConfig
from .shared_variance import (
    AtlasAnalysisResult,
    CVPCAComparison,
    get_atlas_spec,
    process,
)

AnyConfig = StimFullConfig | PlacefieldFullConfig
ConfigFactory = Callable[[np.random.Generator], AnyConfig]

_ORACLE_ATTR: dict[str, str] = {
    "kappa": "oracle_kappa",
    "energy": "oracle_energy",
    "stimstim": "oracle_stimstim",
}
_EMPIRICAL_ATTR: dict[str, str] = {
    "kappa": "empirical_kappa",
    "cv_energy": "empirical_cv_energy",
    "cv_kappa": "empirical_cv_kappa",
    "cv_stimstim": "empirical_cv_stimstim",
    "cv_variance_scale": "empirical_cv_variance_scale",
    "cv_rcvpca": "empirical_cv_rcvpca",
}


@dataclass(frozen=True)
class SweepSpec:
    """Parameterized sweep: one config factory per parameter value.

    Parameters
    ----------
    param_name
        Label for the swept axis (e.g. ``"nuisance_scale"``).
    param_values
        Shape ``(n_params,)`` numeric values for the x-axis of estimation curves.
    factories
        One callable per param value. Each takes an ``np.random.Generator`` and
        returns a config. Geometry varies across seeds because each call receives
        a fresh rng derived from a SeedSequence.
    """

    param_name: str
    param_values: npt.NDArray[np.floating]
    factories: Sequence[ConfigFactory]

    def __post_init__(self) -> None:
        if len(self.param_values) != len(self.factories):
            raise ValueError(f"len(param_values)={len(self.param_values)} must equal " f"len(factories)={len(self.factories)}")


@dataclass
class SweepResults:
    """Collected oracle and empirical SVR ratios, shape ``(n_cases, n_seeds)``.

    NaN entries indicate the metric was unavailable for that case (e.g.
    ``cv_rcvpca`` is only computed for ``PlacefieldFullGenerator`` cases).

    Use ``oracle(metric)`` and ``empirical(method)`` for named access.
    """

    case_labels: list[str]
    n_seeds: int
    spec: Optional[SweepSpec]

    # Oracle (population) ratios
    oracle_kappa: npt.NDArray[np.floating]
    oracle_energy: npt.NDArray[np.floating]
    oracle_stimstim: npt.NDArray[np.floating]

    # Empirical estimation ratios
    empirical_kappa: npt.NDArray[np.floating]
    empirical_cv_energy: npt.NDArray[np.floating]
    empirical_cv_kappa: npt.NDArray[np.floating]
    empirical_cv_stimstim: npt.NDArray[np.floating]
    empirical_cv_variance_scale: npt.NDArray[np.floating]
    empirical_cv_rcvpca: npt.NDArray[np.floating]

    @property
    def n_cases(self) -> int:
        return len(self.case_labels)

    def oracle(self, metric: str = "kappa") -> npt.NDArray[np.floating]:
        """Return oracle ratio array. metric: 'kappa', 'energy', or 'stimstim'."""
        if metric not in _ORACLE_ATTR:
            raise KeyError(f"Unknown oracle metric {metric!r}. Options: {sorted(_ORACLE_ATTR)}")
        return getattr(self, _ORACLE_ATTR[metric])

    def empirical(self, method: str) -> npt.NDArray[np.floating]:
        """Return empirical ratio array by method name."""
        if method not in _EMPIRICAL_ATTR:
            raise KeyError(f"Unknown empirical method {method!r}. Options: {sorted(_EMPIRICAL_ATTR)}")
        return getattr(self, _EMPIRICAL_ATTR[method])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rcvpca_ratio(cv_rcvpca: CVPCAComparison) -> float:
    denom = float(np.sum(cv_rcvpca.modes_neuron))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(cv_rcvpca.modes_position) / denom)


def _extract_ratios(result: AtlasAnalysisResult) -> dict[str, float]:
    pop = result.population
    emp = result.empirical
    out: dict[str, float] = {
        "oracle_kappa": pop.kappa.ratio,
        "oracle_energy": pop.energy.ratio,
        "oracle_stimstim": pop.stimstim.ratio if pop.stimstim is not None else np.nan,
        "empirical_kappa": np.nan,
        "empirical_cv_energy": np.nan,
        "empirical_cv_kappa": np.nan,
        "empirical_cv_stimstim": np.nan,
        "empirical_cv_variance_scale": np.nan,
        "empirical_cv_rcvpca": np.nan,
    }
    if emp is None:
        return out
    out["empirical_kappa"] = emp.kappa.ratio
    if emp.cv_energy is not None:
        out["empirical_cv_energy"] = emp.cv_energy.ratio
    if emp.cv_kappa is not None:
        out["empirical_cv_kappa"] = emp.cv_kappa.ratio
    if emp.cv_stimstim is not None:
        out["empirical_cv_stimstim"] = emp.cv_stimstim.ratio
    if emp.cv_variance_scale is not None:
        out["empirical_cv_variance_scale"] = emp.cv_variance_scale.ratio
    if emp.cv_rcvpca is not None:
        out["empirical_cv_rcvpca"] = _rcvpca_ratio(emp.cv_rcvpca)
    return out


def _run_factories(
    factories: Sequence[ConfigFactory],
    case_labels: list[str],
    n_seeds: int,
    base_seed: int,
    num_samples: int,
    noise_variance: float,
    test_rotation_angle: float,
    dtype: np.dtype,
    spec: Optional[SweepSpec],
) -> SweepResults:
    n_cases = len(factories)
    all_keys = list(_ORACLE_ATTR.values()) + list(_EMPIRICAL_ATTR.values())
    arrays: dict[str, npt.NDArray[np.floating]] = {k: np.full((n_cases, n_seeds), np.nan) for k in all_keys}

    # SeedSequence derives two independent child seeds per (case, seed):
    #   children[2*i]   → geometry rng for the config factory
    #   children[2*i+1] → sample_seed for process()
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(n_cases * n_seeds * 2)

    for i_case, factory in tqdm(enumerate(factories), desc="Running factories"):
        for i_seed in tqdm(range(n_seeds), desc="Running seeds"):
            idx = (i_case * n_seeds + i_seed) * 2
            geo_rng = np.random.default_rng(children[idx])
            sample_seed = int(children[idx + 1].generate_state(1)[0])

            cfg = factory(geo_rng)
            result = process(
                cfg,
                dtype=dtype,
                num_samples=num_samples,
                sample_seed=sample_seed,
                noise_variance=noise_variance,
                test_rotation_angle=test_rotation_angle,
            )
            ratios = _extract_ratios(result)
            for k in all_keys:
                arrays[k][i_case, i_seed] = ratios[k]

    return SweepResults(case_labels=case_labels, n_seeds=n_seeds, spec=spec, **arrays)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_named_sweep(
    case_names: Sequence[str],
    *,
    n_seeds: int = 20,
    num_samples: int,
    noise_variance: float = 0.0,
    test_rotation_angle: float = 0.0,
    base_seed: int = 0,
    dtype: npt.DTypeLike = np.float64,
) -> SweepResults:
    """Run each named atlas case n_seeds times with fresh geometry per seed.

    Each seed uses an independent rng (derived from a SeedSequence) so geometry
    varies across seeds. Use this to estimate mean/variance of oracle and
    empirical SVR ratios for each atlas condition.

    Parameters
    ----------
    case_names
        Names of atlas cases; must be ``stim_full`` or ``placefield`` kind.
        See ``shared_variance.list_atlas_cases()`` for available names.
    n_seeds
        Independent geometry realizations per case.
    num_samples
        Trial count passed to ``process()`` for empirical estimation.
    noise_variance
        Isotropic noise added to empirical draws only. The oracle PopulationBlock
        is ALWAYS computed from the exact true covariances (no noise). Use 0.0
        for the canonical "does the estimator recover true SVR?" question; use
        a nonzero value to study noise robustness. See module docstring.
    test_rotation_angle
        Test-set rotation angle (stim-full pipeline only).
    base_seed
        Root for the SeedSequence that derives all geometry and sample seeds.
    dtype
        Floating dtype for generation.

    Returns
    -------
    SweepResults
        Ratio arrays of shape ``(n_cases, n_seeds)``.
    """
    _dtype = np.dtype(dtype)
    factories: list[ConfigFactory] = []
    for name in tqdm(case_names, desc="Building factories"):
        atlas_spec = get_atlas_spec(name)

        def _make_factory(s: object) -> ConfigFactory:
            def factory(rng: np.random.Generator) -> AnyConfig:
                # Call atlas builder to get (config, generator); discard generator.
                # config.rng stores the geometry seed; process() builds its own
                # generator from it, keeping population and empirical consistent.
                config, _ = s.builder(rng, _dtype)  # type: ignore[attr-defined]
                return config

            return factory

        factories.append(_make_factory(atlas_spec))

    return _run_factories(
        factories=factories,
        case_labels=list(case_names),
        n_seeds=n_seeds,
        base_seed=base_seed,
        num_samples=num_samples,
        noise_variance=noise_variance,
        test_rotation_angle=test_rotation_angle,
        dtype=_dtype,
        spec=None,
    )


def run_param_sweep(
    spec: SweepSpec,
    *,
    n_seeds: int = 20,
    num_samples: int,
    noise_variance: float = 0.0,
    test_rotation_angle: float = 0.0,
    base_seed: int = 0,
    dtype: npt.DTypeLike = np.float64,
) -> SweepResults:
    """Run a parametric sweep: for each param value call its factory n_seeds times.

    Enables estimation curves where the x-axis is a continuous config parameter
    (e.g. ``nuisance_scale``) and each curve is one empirical estimation method.

    Parameters
    ----------
    spec
        :class:`SweepSpec` defining the param axis and one factory per value.
    n_seeds
        Independent geometry realizations per parameter value.
    num_samples
        Trial count for empirical estimation.
    noise_variance
        Isotropic noise added to empirical draws only. The oracle PopulationBlock
        is ALWAYS computed from the exact true covariances (no noise). Use 0.0
        for the canonical "does the estimator recover true SVR?" question; use
        a nonzero value to study noise robustness. See module docstring.
    test_rotation_angle
        Test-set rotation angle (stim-full pipeline only).
    base_seed
        Root for the SeedSequence.
    dtype
        Floating dtype for generation.

    Returns
    -------
    SweepResults
        Ratio arrays of shape ``(n_params, n_seeds)``.
    """
    _dtype = np.dtype(dtype)
    labels = [f"{spec.param_name}={v}" for v in spec.param_values]
    return _run_factories(
        factories=list(spec.factories),
        case_labels=labels,
        n_seeds=n_seeds,
        base_seed=base_seed,
        num_samples=num_samples,
        noise_variance=noise_variance,
        test_rotation_angle=test_rotation_angle,
        dtype=_dtype,
        spec=spec,
    )


__all__ = [
    "AnyConfig",
    "ConfigFactory",
    "SweepSpec",
    "SweepResults",
    "run_named_sweep",
    "run_param_sweep",
]
