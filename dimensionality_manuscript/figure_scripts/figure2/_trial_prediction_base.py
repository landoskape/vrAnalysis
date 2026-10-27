"""Shared selection and prediction-cache controls for the trial time viewer."""

import numpy as np

from dimensionality_manuscript.configs.placefield_structure import CrossValidatedPlacefieldsConfig
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts import session_cache
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets
from dimensionality_manuscript.pipeline import ResultsAggregator

from .trial_performance import (
    TRIAL_PERFORMANCE_MODEL_NAMES, TRIAL_PERFORMANCE_MODEL_LABELS, TRIAL_PERFORMANCE_MODEL_COLORS,
    SQRT_GAIN_MODEL_NAME, SQRT_GAIN_MODEL_LABEL, SQRT_GAIN_MODEL_COLOR,
)
from .performance import (
    STRUCTURED_ADDITIVE_MODEL_NAME, STRUCTURED_ADDITIVE_MODEL_LABEL, STRUCTURED_ADDITIVE_MODEL_COLOR,
)
from .placefield_gain import least_squares_gain_matrix


class _TrialPredictionBase(FigureViewer):
    """Select trial examples and cache scored predictions for a concrete viewer."""

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        mouse: str | None = None,
        example_session: str | None = None,
        roi: int = 0,
        env: int = 0,
        trial: int = 0,
        prediction_results: ResultsAggregator | None = None,
        models: tuple[str, ...] = (),
        filter_to_target_set: bool = False,
        fontsize: float = 12.0,
        figsize: tuple[float, float] = (8.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.prediction_results = prediction_results
        if self.prediction_results is None and getattr(results.config_class, "display_name", None) == "trial_placecell_regression":
            self.prediction_results = results
        if self.prediction_results is not None and self.prediction_results.config_class.display_name != "trial_placecell_regression":
            raise ValueError("prediction_results must use TrialPlacecellRegressionConfig")
        self._prediction_cache = {}
        self._prediction_geometry = {}
        self._unfiltered_selections = {}
        self._filter_active = False
        self._empty_target_set = False
        self._prediction_styles = dict(zip(
            TRIAL_PERFORMANCE_MODEL_NAMES,
            zip(TRIAL_PERFORMANCE_MODEL_LABELS, TRIAL_PERFORMANCE_MODEL_COLORS),
        ))
        self._prediction_styles[SQRT_GAIN_MODEL_NAME] = (SQRT_GAIN_MODEL_LABEL, SQRT_GAIN_MODEL_COLOR)
        self._prediction_styles["external_placefield_1d"] = ("External Placefield", "#666666")
        self._prediction_styles[STRUCTURED_ADDITIVE_MODEL_NAME] = (
            STRUCTURED_ADDITIVE_MODEL_LABEL, STRUCTURED_ADDITIVE_MODEL_COLOR,
        )
        self.selection_names = ()
        options = list(self._prediction_styles)
        if self.prediction_results is not None:
            options = [name for name in options if name in self.prediction_results.param_axes["model_name"]]
            self.selection_names = add_data_selection_widgets(
                self, self.prediction_results, skip=("model_name",), defaults=selection_defaults,
            )
        if models and self.prediction_results is None:
            raise ValueError("Supply trial-regression prediction_results to select models")
        self.add_multiple_selection("models", value=list(models), options=options)
        self.add_boolean("filter_to_target_set", value=filter_to_target_set)
        self._example_config = CrossValidatedPlacefieldsConfig()
        mouse_names = np.asarray(results.mouse_names)
        self.mice = sorted({str(name) for name in mouse_names})
        if not self.mice:
            raise ValueError("results must contain at least one session")
        self._sessions_by_mouse = {
            name: {
                results.sessions[row].session_print(): results.sessions[row]
                for row in sorted(
                    np.flatnonzero(mouse_names == name),
                    key=lambda row: _session_sort_key(results.sessions[row]),
                )
            }
            for name in self.mice
        }
        initial_mouse = mouse if mouse is not None else self.mice[0]
        labels = list(self._sessions_by_mouse[initial_mouse])
        initial_session = example_session if example_session is not None else labels[0]
        self.add_selection("mouse", value=initial_mouse, options=self.mice)
        self.add_selection("example_session", value=initial_session, options=labels)
        self.add_integer("env", value=env, min=0, max=max(env, 0))
        self.add_selection("roi", value=roi, options=[roi])
        self.add_selection("trial", value=trial, options=[trial])
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.on_change("mouse", self.update_example_session)
        self.on_change(["example_session", "env"], self.update_example_bounds)
        self.on_change(["roi", "trial", "models", "filter_to_target_set", *self.selection_names], self.refresh_data)
        self.update_example_session(self.state)

    def update_example_session(self, state):
        """Offer this mouse's sessions in chronological order."""
        labels = list(self._sessions_by_mouse[state["mouse"]])
        current = state["example_session"] if state["example_session"] in labels else labels[0]
        self.update_selection("example_session", value=current, options=labels)
        self.update_example_bounds({**state, "example_session": current})

    def _get_env_maps(self, state):
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        previous_spks_type = session.params.spks_type
        session.params.spks_type = self._example_config.spks_type
        try:
            return session_cache.get_env_maps(session)
        finally:
            session.params.spks_type = previous_spks_type

    def update_example_bounds(self, state):
        """Clamp selections when the session or environment has fewer examples."""
        maps = self._get_env_maps(state)
        if not len(maps.environments):
            raise ValueError("Selected session has no environment maps")
        env = min(state["env"], len(maps.environments) - 1)
        nroi, ntrial, _ = maps.spkmap[env].shape
        if not nroi or not ntrial:
            raise ValueError("Selected environment has no ROIs or full trials")
        self.update_integer("env", value=env, max=len(maps.environments) - 1)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        maps = self._get_env_maps(state)
        self.distcenters = maps.distcenters
        self.environment = maps.environments[state["env"]]
        self.prediction_curves = {}
        self.prediction_positions = {}
        self.prediction_notes = []
        prediction_sets = {}
        if state["models"]:
            prediction_sets, raw_rois, trial_ids = self._load_prediction_sets(state)

        nroi, ntrial, _ = maps.spkmap[state["env"]].shape
        context = (state["mouse"], state["example_session"], state["env"])
        filtering = state["filter_to_target_set"] and bool(state["models"])
        roi, trial = state["roi"], state["trial"]
        if not filtering and self._filter_active:
            roi, trial = self._unfiltered_selections.get(context, (roi, trial))
        roi_options = list(range(nroi))
        trial_options = list(range(ntrial))
        self._empty_target_set = False
        if filtering:
            # Intersect pairs, not separate ROI/trial lists: each offered pair
            # must have an overlay for every selected model.
            pairs = set.intersection(*(set(curves) for curves in prediction_sets.values()))
            roi_lookup = {int(raw): i for i, raw in enumerate(raw_rois)}
            trial_lookup = {int(raw): i for i, raw in enumerate(trial_ids)}
            pairs = {(roi_lookup[r], trial_lookup[t]) for r, t in pairs
                     if r in roi_lookup and t in trial_lookup}
            roi_options = sorted({r for r, _ in pairs})
            roi = roi if roi in roi_options else next(iter(roi_options), None)
            trial_options = sorted(t for r, t in pairs if r == roi)
            self._empty_target_set = not roi_options

        roi = roi if roi in roi_options else next(iter(roi_options), None)
        trial = trial if trial in trial_options else next(iter(trial_options), None)
        # Syd requires a nonempty option list. None represents no available data,
        # rather than offering an ROI/trial which has no prediction.
        self.update_selection("roi", value=roi, options=roi_options or [None])
        self.update_selection("trial", value=trial, options=trial_options or [None])
        self._filter_active = filtering
        if not filtering:
            self._unfiltered_selections[context] = (roi, trial)
        self._selected_trial = trial
        if self._empty_target_set:
            self.prediction_notes.append("No ROI/trial pairs have predictions for every selected model.")
            return

        self.spkmap = np.asarray(maps.spkmap[state["env"]][roi], dtype=float)
        self.placefield = np.nanmean(self.spkmap, axis=0)
        self.trial_activity = self.spkmap[trial]
        gains, _ = least_squares_gain_matrix(self.trial_activity[None, None, :], self.placefield[None, :])
        self.gain = float(gains[0, 0])
        for name, curves in prediction_sets.items():
            curve = curves.get((int(raw_rois[roi]), int(trial_ids[trial])))
            if curve is None:
                label = self._prediction_styles[name][0].replace("\n", " ")
                self.prediction_notes.append(f"{label}: ROI/trial outside scored target set")
            else:
                self.prediction_curves[name] = curve

    def clear_prediction_cache(self):
        """Discard replayed fits after changing saved results or model code."""
        self._prediction_cache.clear()
        self._prediction_geometry.clear()

    def _replay_predictions(self, config, saved, session, edges):
        raise NotImplementedError

    def _load_prediction_sets(self, state):
        results = self.prediction_results
        if results is None:
            raise ValueError("Supply trial-regression prediction_results to select models")
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        row = results._session_index.get(session.session_uid)
        if row is None:
            raise ValueError("Selected session is absent from prediction_results")
        previous_spks_type = session.params.spks_type
        prediction_sets = {name: {} for name in state["models"]}
        try:
            session.params.spks_type = self._example_config.spks_type
            raw_rois = np.flatnonzero(session.idx_rois)
            smp = session_cache.get_smp(session)
            geometry_key = (session.session_uid, self._example_config.spks_type)
            if geometry_key not in self._prediction_geometry:
                processed = smp.get_processed_maps()
                required = smp._idx_required_position_bins()
                self._prediction_geometry[geometry_key] = np.all(
                    ~np.isnan(processed.occmap[:, required]), axis=1,
                )
            full = self._prediction_geometry[geometry_key]
            trial_ids = np.flatnonzero(full & (np.asarray(session.trial_environment) == self.environment))
            for name in state["models"]:
                selection = {key: state[key] for key in self.selection_names}
                selection["model_name"] = name
                indices = tuple(results.param_axes[key].index(selection[key]) for key in results._param_names)
                ref = results._cells.get((row, indices))
                label = self._prediction_styles[name][0].replace("\n", " ")
                if ref is None:
                    self.prediction_notes.append(f"{label}: no stored result")
                    continue
                config = results._key_to_config[ref.row["analysis_key"]]
                session.params.spks_type = config.spks_type
                if config.spks_type != self._example_config.spks_type:
                    raise ValueError("Prediction and example activity must use the same spks_type")
                key = (session.session_uid, config.key())
                if key not in self._prediction_cache:
                    saved = config.get_result(results.store, ref.row)
                    if saved is None:
                        raise ValueError(f"Could not load saved result for {name}")
                    self._prediction_cache[key] = self._replay_predictions(config, saved, session, smp.dist_edges)
                prediction_sets[name] = self._prediction_cache[key]
                self.prediction_positions[name] = (smp.dist_edges[:-1] + smp.dist_edges[1:]) / 2
            return prediction_sets, raw_rois, trial_ids
        finally:
            session.params.spks_type = previous_spks_type

