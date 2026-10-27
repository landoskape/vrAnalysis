"""Observed and predicted activity versus retained time within one trial."""

import numpy as np

from dimensionality_manuscript.figure_scripts import session_cache
from vrAnalysis.helpers.plotting import format_spines
from vrAnalysis.processors.placefields import get_frame_behavior

from ._trial_prediction_base import _TrialPredictionBase
from .trial_predictions import replay_predictions


class TrialTimeFocus(_TrialPredictionBase):
    """A single time-axis panel of observed and predicted trial activity.

    Model curves are the scored temporal predictions, without spatial binning,
    smoothing, or interpolation. Activity, reference, and predictions share
    the trial registry's valid-behavior, environment, and speed mask. Shared
    excluded samples are removed, and retained frames are treated as consecutive
    at the session's median imaging-frame interval. Additional undefined
    model/reference values remain NaN gaps. Values use session sigma units.

    The black reference uses the saved internal-placefield model's training-fold
    template and optimized parameters, evaluated at true position by default.
    ``internal4reference=True`` evaluates the same template at internally decoded
    position instead. Both reference modes are cached; toggling does not refit.

    Select overlays with the ``models`` multi-select. Supply a trial-regression
    results aggregator as ``results`` or ``prediction_results``. Fits are cached
    per session/config; changing ROI/trial or removing and re-adding a model
    reuses the cache. Call ``clear_prediction_cache()`` after changing results
    or model code.

    ``roi`` uses session-filtered indices. ``env`` indexes the environment maps,
    and ``trial`` indexes that environment's retained full trials. Set
    ``filter_to_target_set=True`` to restrict ROI/trial choices to predictions
    available for every selected model. Turning it off restores the full choices
    and previous unfiltered selection. With no models selected, choices remain
    unfiltered. The time axis measures retained trial time, with shared gaps removed.

    In a notebook::

        viewer = TrialTimeFocus(trial_results, models=("rrr",),
                               filter_to_target_set=True)
        viewer.show()
    """

    def __init__(self, results, *, internal4reference=False, figsize=(8.0, 3.0), **kwargs):
        self._temporal_session_cache = {}
        self._reference_cache = {}
        self.add_boolean("internal4reference", value=internal4reference)
        super().__init__(results, figsize=figsize, **kwargs)
        self.on_change("internal4reference", self.refresh_data)

    def _replay_predictions(self, config, saved, session, edges):
        return replay_predictions(config, saved, session, edges, temporal=True)

    def clear_prediction_cache(self):
        super().clear_prediction_cache()
        self._temporal_session_cache.clear()
        self._reference_cache.clear()

    def _get_reference(self, state, session):
        results = self.prediction_results
        if results is None:
            raise ValueError("TrialTimeFocus requires trial-regression results or prediction_results")
        selection = {name: state[name] for name in self.selection_names}
        selection["model_name"] = "internal_placefield_1d"
        indices = tuple(results.param_axes[name].index(selection[name]) for name in results._param_names)
        row = results._session_index.get(session.session_uid)
        ref = results._cells.get((row, indices))
        config = next(cfg for cfg in results._key_to_config.values()
                      if all(getattr(cfg, name) == value for name, value in selection.items()))
        if config.spks_type != self._example_config.spks_type:
            raise ValueError("Reference and example activity must use the same spks_type")
        if ref is None:
            self.prediction_notes.append("Training placefield reference: no stored result")
            return config, {}
        key = (session.session_uid, config.key())
        if key not in self._reference_cache:
            saved = config.get_result(results.store, ref.row)
            if saved is None:
                raise ValueError("Could not load saved internal-placefield reference result")
            edges = session_cache.get_smp(session).dist_edges
            self._reference_cache[key] = replay_predictions(
                config, saved, session, edges, temporal=True, reference=True,
            )
        return config, self._reference_cache[key]

    def _trace_on_trial(self, indices, values):
        trace = np.full(len(self._trial_frame_indices), np.nan)
        offsets = np.asarray(indices, dtype=int) - self._trial_frame_indices[0]
        if np.any((offsets < 0) | (offsets >= len(trace))):
            raise ValueError("Prediction frames fall outside the selected trial")
        trace[offsets] = values
        return trace[self.registry_mask]

    def refresh_data(self, state):
        super().refresh_data(state)
        if self._empty_target_set:
            return
        session = self._sessions_by_mouse[state["mouse"]][state["example_session"]]
        previous_spks_type = session.params.spks_type
        try:
            session.params.spks_type = self._example_config.spks_type
            key = (session.session_uid, session.params.spks_type)
            if key not in self._temporal_session_cache:
                self._temporal_session_cache[key] = (
                    get_frame_behavior(session), np.asarray(session.timestamps),
                    session_cache.get_spks(session),
                )
            behavior, timestamps, activity = self._temporal_session_cache[key]
            full = self._prediction_geometry.get(key)
            if full is None:
                smp = session_cache.get_smp(session)
                processed = smp.get_processed_maps()
                required = smp._idx_required_position_bins()
                full = np.all(~np.isnan(processed.occmap[:, required]), axis=1)
                self._prediction_geometry[key] = full
            trial_ids = np.flatnonzero(full & (np.asarray(session.trial_environment) == self.environment))
            self.absolute_trial = int(trial_ids[self._selected_trial])
            assigned = np.flatnonzero(behavior.trial == self.absolute_trial)
            if not len(assigned):
                raise ValueError("Selected trial has no assigned imaging frames")
            # Keep original frame coordinates for aligning cached predictions,
            # then remove shared exclusions from every displayed trace.
            self._trial_frame_indices = np.arange(assigned[0], assigned[-1] + 1)
            config, reference = self._get_reference(state, session)
            common = behavior.valid_frames(full_check=True)
            common &= behavior.speed >= config.data_config.speed_threshold
            common &= behavior.environment == self.environment
            common &= behavior.trial == self.absolute_trial
            self.registry_mask = common[self._trial_frame_indices]
            self.frame_indices = self._trial_frame_indices[self.registry_mask]
            frame_interval = float(np.median(np.diff(timestamps)))
            self.time = np.arange(len(self.frame_indices)) * frame_interval
            if not len(self.frame_indices):
                self.prediction_notes.append("No trial frames pass the registry mask.")
            self.observed_activity = np.asarray(
                activity[self.frame_indices, self.state["roi"]], dtype=float,
            ).copy()
            raw_roi = int(np.flatnonzero(session.idx_rois)[self.state["roi"]])
            reference_trace = reference.get((raw_roi, self.absolute_trial))
            self.placefield_activity = np.full(len(self.frame_indices), np.nan)
            if reference_trace is None:
                self.prediction_notes.append("Training placefield reference: ROI/trial outside scored target set")
            else:
                indices, external, internal = reference_trace
                values = internal if state["internal4reference"] else external
                self.placefield_activity = self._trace_on_trial(indices, values)
            self.temporal_predictions = {
                name: self._trace_on_trial(indices, values)
                for name, (indices, values) in self.prediction_curves.items()
            }
        finally:
            session.params.spks_type = previous_spks_type

    def plot(self, state):
        fontsize = state["fontsize"]
        fig, ax = self.new_subplots(figsize=self.figsize, layout="constrained")
        if self._empty_target_set:
            ax.set_axis_off()
            ax.text(.5, .5, "\n".join(self.prediction_notes), transform=ax.transAxes,
                    ha="center", va="center", wrap=True, fontsize=fontsize)
            return fig
        ax.plot(self.time, self.placefield_activity, color="black", lw=1.5,
                label=("Training place field at decoded position" if state["internal4reference"]
                       else "Training place field at true position"))
        ax.plot(self.time, self.observed_activity, color="#c00000", lw=1.5,
                label=f"Trial {self._selected_trial}")
        for name, trace in self.temporal_predictions.items():
            label, color = self._prediction_styles[name]
            ax.plot(self.time, trace, color=color, lw=1.5,
                    label=label.replace("\n", " ") + " (prediction)")
        if self.prediction_notes:
            ax.text(.02, .02, "\n".join(self.prediction_notes), transform=ax.transAxes,
                    va="bottom", fontsize=max(6, fontsize - 3))
        ax.set_xlabel("Retained trial time (s)", fontsize=fontsize)
        ax.set_ylabel(r"Fluorescence ($\sigma$)", fontsize=fontsize)
        ax.legend(frameon=False, fontsize=fontsize)
        format_spines(ax, x_pos=-.02, y_pos=-.02, spines_visible=["left", "bottom"],
                      tick_length=2, tick_fontsize=fontsize)
        return fig

