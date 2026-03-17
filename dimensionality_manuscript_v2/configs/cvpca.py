"""CVPCAConfig — concrete analysis config porting measure_cvpca.py logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch
from dimilibi import gaussian_filter
from dimilibi.cvpca import CVPCA, RegularizedCVPCA
from dimilibi.pca import PCA
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.workflows.compare_old_cvpca import get_legacy_cvpca
from vrAnalysis.helpers import cross_validate_trials, reliability_loo
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors.placefields import get_placefield
from vrAnalysis.sessions import B2Session

from ..pipeline.base import AnalysisConfigBase


def nanmax(tensor: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """Max ignoring NaNs by replacing them with dtype minimum."""
    min_value = torch.finfo(tensor.dtype).min
    return tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim).values


@dataclass(frozen=True)
class CVPCAConfig(AnalysisConfigBase):
    """Configuration for cross-validated PCA analysis.

    Parameters
    ----------
    center : bool
        Whether to center data before PCA.
    normalize : bool
        Whether to normalize placefields by max firing rate.
    use_fast_sampling : bool
        Whether to use fast sampling for placefield computation.
    reliability_threshold : float or None
        Minimum reliability for neuron inclusion.
    fraction_active_threshold : float or None
        Minimum fraction active for neuron inclusion.
    fixed_smooth_width : float
        Width for fixed Gaussian smoothing (cm).
    num_bins : int
        Number of spatial bins.
    """

    schema_version: str = "v2"

    center: bool = True
    normalize: bool = True
    use_fast_sampling: bool = True
    reliability_threshold: Optional[float] = None
    fraction_active_threshold: Optional[float] = None
    fixed_smooth_width: float = 3.0
    num_bins: int = 100
    use_spatial_eigenvectors: bool = False  # Whether to use spatial eigenvectors instead of neural eigenvectors

    display_name: ClassVar[str] = "cvpca"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "center": [True, False],
            "use_fast_sampling": [True, False],
            "normalize": [True, False],
            "reliability_threshold": [None, 0.2],
            "fraction_active_threshold": [None, 0.05],
            "use_spatial_eigenvectors": [False, True],
        }

    def summary(self) -> str:
        parts = [
            self.display_name,
            f"center={self.center}",
            f"norm={self.normalize}",
            f"fast={self.use_fast_sampling}",
            f"rel={self.reliability_threshold}",
            f"frac={self.fraction_active_threshold}",
            f"smooth={self.fixed_smooth_width}",
            f"spatial_eig={self.use_spatial_eigenvectors}",
            f"bins={self.num_bins}",
            self.schema_version,
        ]
        return "_".join(parts)

    def process(self, session: B2Session, registry: PopulationRegistry, return_data: bool = False, return_models: bool = False) -> dict:
        """Run CVPCA analysis on a session.

        Ports logic from ``measure_cvpca.process_session()``.
        """
        num_per_env = {i: np.sum(session.trial_environment == i) for i in session.environments}
        best_env = max(num_per_env, key=num_per_env.get)
        best_env_idx = np.where(session.environments == best_env)[0][0]

        env_length = session.env_length[0]
        dist_edges = np.linspace(0, env_length, self.num_bins + 1)
        population, frame_behavior = registry.get_population(session)

        trial_folds = cross_validate_trials(session.trial_environment, [1, 1, 1])
        data = np.array(population.data[population.idx_neurons][:, population.idx_samples]).T

        # Filter neurons by reliability / fraction active
        if self.reliability_threshold is not None or self.fraction_active_threshold is not None:
            _all_trials = get_placefield(
                data, frame_behavior, dist_edges, average=False, use_fast_sampling=True, session=session
            ).filter_by_environment(best_env)
            _pf_data = np.transpose(_all_trials.placefield, (2, 0, 1))

            idx_keep = np.ones(_pf_data.shape[0], dtype=bool)
            if self.reliability_threshold is not None:
                _reliable = reliability_loo(_pf_data)
                idx_keep = idx_keep & (_reliable > self.reliability_threshold)

            if self.fraction_active_threshold is not None:
                _fraction_active = FractionActive.compute(
                    _pf_data,
                    activity_axis=2,
                    fraction_axis=1,
                    activity_method="rms",
                    fraction_method="participation",
                )
                idx_keep = idx_keep & (_fraction_active > self.fraction_active_threshold)

            data = data[:, idx_keep]

        # Compute placefields per fold
        placefields = [
            get_placefield(
                data,
                frame_behavior,
                dist_edges,
                trial_filter=trial_fold,
                zero_to_nan=True,
                use_fast_sampling=self.use_fast_sampling,
                session=session,
            )
            for trial_fold in trial_folds
        ]
        torch_pfs = [torch.tensor(pf.placefield[best_env_idx].T) for pf in placefields]

        # Handle missing bin counts
        if any(np.any(pf.count[best_env_idx] == 0) for pf in placefields):
            bad_locations = [np.where(pf.count[best_env_idx] == 0)[0] for pf in placefields]
            bad_locations = np.unique(np.concatenate(bad_locations))
            good_idx = np.setdiff1d(np.arange(self.num_bins), bad_locations)
            if not np.all(np.diff(good_idx) == 1):
                raise ValueError(f"Non-sequential missing counts at locations: {bad_locations}")
            torch_pfs = [pf[:, good_idx] for pf in torch_pfs]

        # Normalize by max
        if self.normalize:
            _max_neuron = nanmax(torch.concatenate(torch_pfs, dim=1), dim=1, keepdim=True)
            _max_neuron[_max_neuron == 0] = 1
            torch_pfs = [pf / _max_neuron for pf in torch_pfs]

        if any(torch.any(torch.isnan(pf)) for pf in torch_pfs):
            raise ValueError("Some placefields have NaNs!")

        # Run CVPCA/PCA variants across folds
        pca_covariances = []
        pca_smooth_covariances = []
        pca_fixed_smooth_covariances = []
        reg_covariances = []
        reg_fixed_smooth_covariances = []
        org_covariances = []
        org_smooth_covariances = []
        org_fixed_smooth_covariances = []
        smoothing_widths = []

        # Where covariance measures covariance across transforms of repeats,
        # variances measure just the variance of each transform (for train - which
        # was used for fitting eigenvectors, and test, which is used for c-v testing).
        reg_variances_train = []
        reg_fixed_smooth_variances_train = []
        org_variances_train = []
        org_smooth_variances_train = []
        org_fixed_smooth_variances_train = []
        reg_variances_test = []
        reg_fixed_smooth_variances_test = []
        org_variances_test = []
        org_smooth_variances_test = []
        org_fixed_smooth_variances_test = []

        if return_models:
            models_dict = {
                "reg_cvpca": [],
                "reg_cvpca_fixed": [],
                "cvpca": [],
                "cvpca_smooth": [],
                "cvpca_smooth_fixed": [],
            }

        _cvpca_args = {"center": self.center, "on_stimuli": self.use_spatial_eigenvectors}
        _opt_transpose = lambda x: x.T if self.use_spatial_eigenvectors else x
        for ref_fold in range(len(trial_folds)):
            c0 = torch_pfs[ref_fold]
            c1 = torch_pfs[(ref_fold + 1) % len(trial_folds)]
            c2 = torch_pfs[(ref_fold + 2) % len(trial_folds)]

            c0_fixed = gaussian_filter(c0, self.fixed_smooth_width, axis=1)
            c1_fixed = gaussian_filter(c1, self.fixed_smooth_width, axis=1)
            c2_fixed = gaussian_filter(c2, self.fixed_smooth_width, axis=1)

            # R-CVPCA with fixed smoothing on fit data only
            cvpca_fixed_fit = CVPCA(**_cvpca_args).fit(c0_fixed)
            reg_fixed_smooth_covariances.append(cvpca_fixed_fit.score(c1, c2))
            reg_fixed_smooth_variances_train.append(torch.var(cvpca_fixed_fit.pca.transform(_opt_transpose(c0_fixed)), dim=1))
            reg_fixed_smooth_variances_test.append(torch.var(cvpca_fixed_fit.pca.transform(_opt_transpose(c1)), dim=1))
            reg_fixed_smooth_variances_test.append(torch.var(cvpca_fixed_fit.pca.transform(_opt_transpose(c2)), dim=1))

            # Regularized CVPCA with optimized smoothing
            reg_cvpca = RegularizedCVPCA(**_cvpca_args)
            reg_cvpca = reg_cvpca.fit_smoothing(c0, c1, c2)
            reg_cvpca = reg_cvpca.fit(c0)
            reg_covariances.append(reg_cvpca.score(c1, c2))

            # Get smoothed data
            c0_smooth = gaussian_filter(c0, reg_cvpca.smoothing_widths, axis=1)
            c1_smooth = gaussian_filter(c1, reg_cvpca.smoothing_widths, axis=1)
            c2_smooth = gaussian_filter(c2, reg_cvpca.smoothing_widths, axis=1)

            reg_variances_train.append(torch.var(reg_cvpca.pca.transform(_opt_transpose(c0_smooth)), dim=1))
            reg_variances_test.append(torch.var(reg_cvpca.pca.transform(_opt_transpose(c1)), dim=1))
            reg_variances_test.append(torch.var(reg_cvpca.pca.transform(_opt_transpose(c2)), dim=1))

            # Standard CVPCA (no smoothing)
            cvpca = CVPCA(**_cvpca_args).fit(c0)
            org_covariances.append(cvpca.score(c1, c2))
            org_variances_train.append(torch.var(cvpca.pca.transform(_opt_transpose(c0)), dim=1))
            org_variances_test.append(torch.var(cvpca.pca.transform(_opt_transpose(c1)), dim=1))
            org_variances_test.append(torch.var(cvpca.pca.transform(_opt_transpose(c2)), dim=1))

            # CVPCA with optimized smoothing applied to all folds
            cvpca_smooth = CVPCA(**_cvpca_args).fit(c0_smooth)
            org_smooth_covariances.append(cvpca_smooth.score(c1_smooth, c2_smooth))
            org_smooth_variances_train.append(torch.var(cvpca_smooth.pca.transform(_opt_transpose(c0_smooth)), dim=1))
            org_smooth_variances_test.append(torch.var(cvpca_smooth.pca.transform(_opt_transpose(c1_smooth)), dim=1))
            org_smooth_variances_test.append(torch.var(cvpca_smooth.pca.transform(_opt_transpose(c2_smooth)), dim=1))

            # CVPCA with fixed smoothing applied to all folds
            cvpca_fixed_all = CVPCA(**_cvpca_args).fit(c0_fixed)
            org_fixed_smooth_covariances.append(cvpca_fixed_all.score(c1_fixed, c2_fixed))
            org_fixed_smooth_variances_train.append(torch.var(cvpca_fixed_all.pca.transform(_opt_transpose(c0_fixed)), dim=1))
            org_fixed_smooth_variances_test.append(torch.var(cvpca_fixed_all.pca.transform(_opt_transpose(c1_fixed)), dim=1))
            org_fixed_smooth_variances_test.append(torch.var(cvpca_fixed_all.pca.transform(_opt_transpose(c2_fixed)), dim=1))

            # PCA variants
            pca_covariances.append(PCA(center=self.center).fit(c0).get_eigenvalues())
            pca_smooth_covariances.append(PCA(center=self.center).fit(c0_smooth).get_eigenvalues())
            pca_fixed_smooth_covariances.append(PCA(center=self.center).fit(c0_fixed).get_eigenvalues())

            smoothing_widths.append(reg_cvpca.smoothing_widths)

            if return_models:
                models_dict["reg_cvpca"].append(reg_cvpca)
                models_dict["reg_cvpca_fixed"].append(cvpca_fixed_fit)
                models_dict["cvpca"].append(cvpca)
                models_dict["cvpca_smooth"].append(cvpca_smooth)
                models_dict["cvpca_smooth_fixed"].append(cvpca_fixed_all)

        # Legacy CVPCA results
        try:
            saved_leg_result = get_legacy_cvpca(session, best_env_idx=best_env_idx)
        except Exception as e:
            print(f"Error getting legacy CVPCA for {session.session_uid}: {e}")
            saved_leg_result = None

        results_dict = {
            "trial_folds": trial_folds,
            "reg_covariances": np.mean(np.stack(reg_covariances, axis=0), axis=0),
            "reg_covariances_fixed": np.mean(np.stack(reg_fixed_smooth_covariances, axis=0), axis=0),
            "org_covariances": np.mean(np.stack(org_covariances, axis=0), axis=0),
            "org_smooth_covariances": np.mean(np.stack(org_smooth_covariances, axis=0), axis=0),
            "org_fixed_smooth_covariances": np.mean(np.stack(org_fixed_smooth_covariances, axis=0), axis=0),
            "pca_covariances": np.mean(np.stack(pca_covariances, axis=0), axis=0),
            "pca_smooth_covariances": np.mean(np.stack(pca_smooth_covariances, axis=0), axis=0),
            "pca_fixed_smooth_covariances": np.mean(np.stack(pca_fixed_smooth_covariances, axis=0), axis=0),
            "saved_leg_covariances": saved_leg_result["cv_by_env_all"] if saved_leg_result is not None else None,
            "smoothing_widths": np.mean(np.stack(smoothing_widths, axis=0), axis=0),
            "reg_variances_train": np.mean(np.stack(reg_variances_train, axis=0), axis=0),
            "reg_fixed_smooth_variances_train": np.mean(np.stack(reg_fixed_smooth_variances_train, axis=0), axis=0),
            "org_variances_train": np.mean(np.stack(org_variances_train, axis=0), axis=0),
            "org_smooth_variances_train": np.mean(np.stack(org_smooth_variances_train, axis=0), axis=0),
            "org_fixed_smooth_variances_train": np.mean(np.stack(org_fixed_smooth_variances_train, axis=0), axis=0),
            "reg_variances_test": np.mean(np.stack(reg_variances_test, axis=0), axis=0),
            "reg_fixed_smooth_variances_test": np.mean(np.stack(reg_fixed_smooth_variances_test, axis=0), axis=0),
            "org_variances_test": np.mean(np.stack(org_variances_test, axis=0), axis=0),
            "org_smooth_variances_test": np.mean(np.stack(org_smooth_variances_test, axis=0), axis=0),
            "org_fixed_smooth_variances_test": np.mean(np.stack(org_fixed_smooth_variances_test, axis=0), axis=0),
        }

        output = (results_dict,)

        if return_models:
            output += (models_dict,)

        if return_data:
            data_dict = {
                "torch_pfs": torch_pfs,
                "dist_edges": dist_edges,
                "best_env_idx": best_env_idx,
                "trial_folds": trial_folds,
            }
            output += (data_dict,)

        return output
