"""Find tracked examples of changes in spatial and amplitude reliability."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from vrAnalysis.tracking import Tracker

from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets, data_selection

from .placefield import PF_RMS_COMPARISONS, PlaceFieldPredictionFocus, _transform_amplitude


class TrackedPlaceFieldPredictionFocus(FigureViewer):
    """Two stacked PlaceFieldPredictionFocus panels for the same tracked neuron.

    Pass the same schema-v11 ``pfpred_quality`` ResultsAggregator as the single-session
    viewer. ``msm`` optionally supplies a MultiSessionSpkmaps instance or a mapping
    of mouse names to instances; otherwise Trackers are loaded lazily per mouse.
    Tracking is matched to result sessions by identity, never by list position.

    ``environment`` is the actual environment ID (not its slot/index). Session
    selectors offer only tracked sessions in that environment. Either session order
    is accepted: the panels and late-minus-early changes always use chronological
    order. Session and quality filters must pass in BOTH sessions. Quality uses
    strict > cutoffs, matching HolisticPlacefieldFamiliarityViewer.

    Amplitude reliability is always 1 / (1 + trial_rms_cv**2). The default ordering
    is the largest late-minus-early change first. ``candidate`` is a zero-based rank
    in this ordering; ``candidate_table()`` returns the full sorted list, including
    tracking cluster IDs, original and filtered ROI indices, and both sessions'
    measurements. Missing measurements are excluded. Threshold/order changes reset
    the rank to zero. No-match selections produce an explanatory empty figure.

    Example::

        viewer = TrackedPlaceFieldPredictionFocus(
            quality_results, msm=msm, environment=1,
            early_session="ATL027/2022-01-01/1",
            late_session="ATL027/2022-01-10/1",
            reliability_threshold=0.3, fraction_active_threshold=0.1,
        )
        viewer.show()
        candidates = viewer.candidate_table()
        fig = viewer.plot(viewer.state)

    As with PlaceFieldPredictionFocus, spike maps must be available locally and
    must use the ROI filters with which the quality results were computed.
    ``show_correlation=True`` displays PF-Var Alignment above both position curves,
    with an underlined label and the value on the next line.
    """

    _SORT_METRICS = (
        "amplitude_change", "spatial_change", "amplitude_reliability_early",
        "amplitude_reliability_late", "spatial_reliability_early",
        "spatial_reliability_late", "cluster_id",
    )

    def __init__(
        self, results, *, msm=None, mouse=None, environment=None,
        early_session=None, late_session=None, reliability_threshold=0.3,
        fraction_active_threshold=0.1, sort_by="amplitude_change", descending=True,
        candidate=0, vmax=5.0, pf_rms_comparison="f2_from_f1", show_correlation=False,
        fontsize=10.0, show_consistency=True, show_prediction=False,
        include_amplitude_summary=True, nbins=10, row_ratio=2.0,
        use_trial_weights=True, figsize=(10.0, 10.0), **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self._trackers = {}
        if msm is not None:
            instances = msm.values() if isinstance(msm, Mapping) else [msm]
            self._trackers = {instance.tracker.mouse_name: instance.tracker for instance in instances}
        self._sessions = {
            name: {s.session_print(): s for s in sorted(results.sessions, key=_session_sort_key) if s.mouse_name == name}
            for name in sorted(set(results.mouse_names))
            if msm is None or name in self._trackers
        }
        if not self._sessions:
            raise ValueError("No result mice match the supplied tracking objects.")
        mouse = mouse if mouse is not None else next(iter(self._sessions))
        if mouse not in self._sessions:
            raise ValueError(f"Mouse {mouse!r} is absent from the available results/tracking.")
        self._pair_cache = None
        self._focus_panels = {}

        # Keep quality cutoffs independent of the stored result configuration.
        self.selection_names = add_data_selection_widgets(
            self, results, skip=("reliability_threshold", "fraction_active_threshold"), defaults=selection_defaults,
        )
        self._stored_thresholds = {
            name: getattr(results.config_class, name, values[0])
            if getattr(results.config_class, name, None) in values else values[0]
            for name, values in results.param_axes.items()
            if name in ("reliability_threshold", "fraction_active_threshold")
        }
        self.add_selection("mouse", value=mouse, options=list(self._sessions))
        environments = self._environments(mouse)
        environment = environments[0] if environment is None else environment
        if environment not in environments:
            raise ValueError(f"Environment {environment} does not have two tracked result sessions for {mouse}.")
        labels = self._session_labels(mouse, environment)
        self.add_selection("environment", value=environment, options=environments)
        self.add_selection("early_session", value=early_session or labels[0], options=labels)
        self.add_selection("late_session", value=late_session or labels[-1], options=labels)
        self.add_float("reliability_threshold", value=reliability_threshold, min=-1.0, max=1.0, step=0.01)
        self.add_float("fraction_active_threshold", value=fraction_active_threshold, min=0.0, max=1.0, step=0.01)
        self.add_selection("sort_by", value=sort_by, options=list(self._SORT_METRICS))
        self.add_boolean("descending", value=descending)
        self.add_integer("candidate", value=candidate, min=0, max=max(candidate, 0))
        self.add_float("vmax", value=vmax, min=1.0, max=20.0)
        self.add_selection("pf_rms_comparison", value=pf_rms_comparison, options=list(PF_RMS_COMPARISONS))
        self.add_boolean("show_correlation", value=show_correlation)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.add_boolean("show_consistency", value=show_consistency)
        self.add_boolean("show_prediction", value=show_prediction)
        self.add_boolean("include_amplitude_summary", value=include_amplitude_summary)
        self.add_integer("nbins", value=nbins, min=1, max=100)
        self.add_float("row_ratio", value=row_ratio, min=0.25, max=10.0)
        self.add_boolean("use_trial_weights", value=use_trial_weights)
        self.on_change("mouse", self.update_mouse)
        self.on_change("environment", self.update_environment)
        self.on_change(
            ["early_session", "late_session", "reliability_threshold", "fraction_active_threshold",
             "sort_by", "descending", *self.selection_names], self.refresh_candidates,
        )
        table = self.candidate_table()
        self.update_integer("candidate", value=min(candidate, max(len(table) - 1, 0)), max=max(len(table) - 1, 0))

    def _tracker(self, mouse):
        if mouse not in self._trackers:
            self._trackers[mouse] = Tracker(mouse)
        return self._trackers[mouse]

    def _tracked_sessions(self, mouse):
        identities = {session.session_uid for session in self._tracker(mouse).sessions}
        return {label: session for label, session in self._sessions[mouse].items() if session.session_uid in identities}

    def _environments(self, mouse):
        counts = {}
        for session in self._tracked_sessions(mouse).values():
            for env in set(session.environments):
                if env >= 0:
                    counts[int(env)] = counts.get(int(env), 0) + 1
        environments = sorted(env for env, count in counts.items() if count >= 2)
        if not environments:
            raise ValueError(f"{mouse} has no environment with two tracked result sessions.")
        return environments

    def _session_labels(self, mouse, environment):
        return [label for label, session in self._tracked_sessions(mouse).items() if environment in session.environments]

    def update_mouse(self, state):
        options = self._environments(state["mouse"])
        environment = state["environment"] if state["environment"] in options else options[0]
        self.update_selection("environment", value=environment, options=options)
        self.update_environment({**state, "environment": environment})

    def update_environment(self, state):
        labels = self._session_labels(state["mouse"], state["environment"])
        early = state["early_session"] if state["early_session"] in labels else labels[0]
        late = state["late_session"] if state["late_session"] in labels else labels[-1]
        self.update_selection("early_session", value=early, options=labels)
        self.update_selection("late_session", value=late, options=labels)
        self.refresh_candidates({**state, "early_session": early, "late_session": late})

    def refresh_candidates(self, state):
        table = self.candidate_table(state)
        self.update_integer("candidate", value=0, max=max(len(table) - 1, 0))

    def _config(self, state):
        params = data_selection(state, self.results, self.selection_names)
        params.update(self._stored_thresholds)
        if getattr(self.results, "_schema_version", None) is not None:
            params["schema_version"] = self.results._schema_version
        return self.results.config_class(**params)

    def _ordered_pair(self, state):
        sessions = self._sessions[state["mouse"]]
        return sorted([sessions[state["early_session"]], sessions[state["late_session"]]], key=_session_sort_key)

    def _pair_measurements(self, state):
        sessions = self._ordered_pair(state)
        config = self._config(state)
        key = (tuple(session.session_uid for session in sessions), config.key())
        if self._pair_cache is not None and self._pair_cache[0] == key:
            return self._pair_cache[1]
        tracker = self._tracker(state["mouse"])
        indices = {s.session_uid: i for i, s in enumerate(tracker.sessions)}
        tracked, extras = tracker.get_tracked_idx(
            idx_ses=[indices[s.session_uid] for s in sessions], use_session_filters=False,
        )
        table = {"cluster_id": extras["cluster_ids"]}
        selected_results = []
        for i, (session, suffix) in enumerate(zip(sessions, ("early", "late"))):
            previous = session.params.spks_type
            session.params.spks_type = config.spks_type
            try:
                mask = np.asarray(session.idx_rois, dtype=bool)
            finally:
                session.params.spks_type = previous
            # Tracking indices refer to all original ROIs; metrics refer to filtered ROIs.
            lookup = np.full(mask.size, -1, dtype=int)
            lookup[mask] = np.arange(np.count_nonzero(mask))
            table[f"original_roi_{suffix}"] = tracked[i]
            table[f"roi_{suffix}"] = lookup[tracked[i]]
            selected = self.results.store.get(session.session_uid, config)
            if selected is None:
                raise KeyError(f"No pfpred_quality result for {session.session_print()} -- rerun the schema v11 sweep.")
            keys = ("env_slot_ids", "reliability_slot", "fraction_active_slot", "trial_rms_cv")
            missing = [name for name in keys if name not in selected]
            if missing:
                raise KeyError(f"Missing {missing} for {session.session_print()} -- schema v11 results required.")
            n_rois = np.count_nonzero(mask)
            if any(np.asarray(selected[name]).shape[-1] != n_rois for name in keys[1:]):
                raise ValueError(f"ROI filters for {session.session_print()} differ from the saved quality results.")
            selected_results.append(selected)
        table = pd.DataFrame(table)
        table = table[(table.roi_early >= 0) & (table.roi_late >= 0)].copy()
        value = (table, selected_results)
        self._pair_cache = (key, value)
        return value

    def candidate_table(self, state=None):
        """Sorted eligible neurons, with original/filtered ROI indices and paired metrics."""
        state = self.state if state is None else state
        if state["early_session"] == state["late_session"]:
            return pd.DataFrame()
        base, selected_results = self._pair_measurements(state)
        table = base.copy()
        for selected, suffix in zip(selected_results, ("early", "late")):
            slots = np.flatnonzero(np.asarray(selected["env_slot_ids"]) == state["environment"])
            if not slots.size:
                raise ValueError(f"Environment {state['environment']} is missing from the saved quality results.")
            slot = int(slots[0])
            roi = table[f"roi_{suffix}"].to_numpy(dtype=int)
            for column, name in (("spatial_reliability", "reliability_slot"), ("fraction_active", "fraction_active_slot")):
                table[f"{column}_{suffix}"] = np.asarray(selected[name])[slot, roi]
            table[f"amplitude_reliability_{suffix}"] = _transform_amplitude(
                np.asarray(selected["trial_rms_cv"])[slot, roi], "bounded",
            )
        valid = np.ones(len(table), dtype=bool)
        for suffix in ("early", "late"):
            for name in ("spatial_reliability", "fraction_active", "amplitude_reliability"):
                valid &= np.isfinite(table[f"{name}_{suffix}"])
            valid &= table[f"spatial_reliability_{suffix}"] > state["reliability_threshold"]
            valid &= table[f"fraction_active_{suffix}"] > state["fraction_active_threshold"]
        table = table.loc[valid].copy()
        table["amplitude_change"] = table.amplitude_reliability_late - table.amplitude_reliability_early
        table["spatial_change"] = table.spatial_reliability_late - table.spatial_reliability_early
        return table.sort_values(state["sort_by"], ascending=not state["descending"], kind="stable").reset_index(drop=True)

    def plot(self, state):
        table = self.candidate_table(state)
        fig = self.new_figure(figsize=self.figsize, layout="constrained")
        if table.empty:
            ax = fig.add_subplot()
            ax.set_axis_off()
            reason = "Choose two different sessions." if state["early_session"] == state["late_session"] else (
                "No tracked neurons pass both quality thresholds in both sessions.\nTry lowering the thresholds or choosing another pair."
            )
            ax.text(0.5, 0.5, reason, ha="center", va="center", transform=ax.transAxes, fontsize=state["fontsize"])
            return fig
        rank = min(state["candidate"], len(table) - 1)
        selected = table.iloc[rank]
        config = self._config(state)
        outer = fig.add_gridspec(2, 1)
        for row, (session, suffix) in enumerate(zip(self._ordered_pair(state), ("early", "late"))):
            roi = int(selected[f"roi_{suffix}"])
            # Cache just the two current panels/results, rather than all mice/sessions.
            panel_key = (session.session_uid, config.key())
            cached = self._focus_panels.get(suffix)
            if cached is None or cached[0] != panel_key:
                params = data_selection(state, self.results, self.selection_names)
                params.update(self._stored_thresholds)
                panel = PlaceFieldPredictionFocus(
                    self.results, mouse=state["mouse"], example_session=session.session_print(), roi=roi, **params,
                )
                self._focus_panels[suffix] = (panel_key, panel)
            else:
                panel = cached[1]
            env_maps = panel._get_env_maps(panel.state)
            env = int(np.flatnonzero(np.asarray(env_maps.environments) == state["environment"])[0])
            panel_state = {**panel.state, **state, "roi": roi, "env": env, "example_session": session.session_print(),
                           "amplitude_metric": "trial_rms", "amplitude_transform": "bounded", **self._stored_thresholds}
            panel.refresh_data(panel_state)
            first_axis = len(fig.axes)
            panel.draw(panel_state, fig, outer[row])
            for ax in fig.axes[first_axis:]:
                if row == 0:
                    ax.set_xlabel("")
                elif ax.get_subplotspec().rowspan.start == 0:
                    ax.set_title("")
        return fig
