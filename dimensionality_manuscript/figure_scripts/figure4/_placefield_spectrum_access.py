"""Shared session-level spectrum access for placefield population and example panels."""

import numpy as np

from ._alpha_config import AdaptiveAlphaConfig
from ._param_axes import FIT_KEY_PARAM_KEYS, SOURCE_OF_KEY, encode_param, sel_params
from ._spectrum_math import _clip_at_first_negative, _eig_to_ss_scale, _smooth_spectrum


class PlacefieldSpectrumAccess:
    """Read spectra using the same selection, covariance scale, and preprocessing."""

    def encode_param(self, name, value):
        return encode_param(self._tuple_labels, name, value)

    def _sel_params(self, state, source):
        return sel_params(state, self._tuple_labels, self._agg[source].param_axes)

    def _fit_sel_params(self, state):
        return sel_params(state, self._tuple_labels, self._fit_axes)

    @staticmethod
    def _cfg_from_state(state, prefix):
        return AdaptiveAlphaConfig(
            smooth_method=state[f"{prefix}_smooth_method"],
            smooth_width=state[f"{prefix}_smooth_width"],
            fpd_window_size=int(state[f"{prefix}_fpd_window_size"]),
            adaptive_buffer=int(state[f"{prefix}_adaptive_buffer"]),
            minimum_window_size=int(state[f"{prefix}_minimum_window_size"]),
        )

    def _preprocess(self, spec, state, cfg):
        spec = np.atleast_2d(np.asarray(spec, dtype=float))
        if state.get("clip_negative", False):
            spec = _clip_at_first_negative(spec)
        if state["normalize"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                spec = spec / np.nansum(spec, axis=1)[:, None]
        return spec, _smooth_spectrum(spec, cfg.smooth_method, cfg.smooth_width)

    def _spectrum_sessions(self, state, key, cfg):
        source = SOURCE_OF_KEY[key]
        agg = self._agg[source]
        spec = agg.sel(keys=[key], avg_by_mouse=False, **self._sel_params(state, source))[key]
        return (*self._preprocess(spec, state, cfg), agg.mouse_names, agg.session_ids)

    def _fit_spectrum_raw_sessions(self, state, key):
        """Convert fitted PCA eigenvalues to sample-covariance units, P / (P - 1)."""
        fit_params = self._fit_sel_params(state)
        param_keys = FIT_KEY_PARAM_KEYS[key]
        selected = self.results_fit.sel(keys=[key, *param_keys], squeeze_ones=False, **fit_params)
        centers = self.results_fit.sel_objects(keys=["dist_centers"], **fit_params)["dist_centers"]
        return np.stack([
            _eig_to_ss_scale(eig, list(params_row), theta)
            for eig, theta, *params_row in zip(
                np.asarray(selected[key], dtype=float), centers, *(selected[pk] for pk in param_keys)
            )
        ])

    def _fit_spectrum_sessions(self, state, key, cfg):
        spec = self._fit_spectrum_raw_sessions(state, key)
        return (*self._preprocess(spec, state, cfg), self.results_fit.mouse_names, self.results_fit.session_ids)
