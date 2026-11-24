"""Compare placefield panels for the same tracked neuron across sessions."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from vrAnalysis.tracking import Tracker

from dimensionality_manuscript.configs.pfpred_quality import TRIAL_ACTIVITY_METRICS, TRIAL_ACTIVITY_STATISTICS
from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import FigureViewer, add_data_selection_widgets, data_selection

from .placefield import PF_RMS_COMPARISONS, PlaceFieldPredictionFocus, _AMPLITUDE_TRANSFORMS, _transform_amplitude
from .placefield_breakdown import PlacefieldBreakdownFocus, _add_breakdown_widgets


class _TrackedPlacefieldFocus(FigureViewer):
    """Shared session matching, candidate ranking, and two-session layout."""

    _SORT_METRICS = (
        "amplitude_change",
        "spatial_change",
        "amplitude_reliability_early",
        "amplitude_reliability_late",
        "spatial_reliability_early",
        "spatial_reliability_late",
        "cluster_id",
    )

    def __init__(
        self,
        results,
        *,
        msm=None,
        mouse=None,
        environment=None,
        early_session=None,
        late_session=None,
        reliability_threshold=0.3,
        fraction_active_threshold=0.1,
        sort_by="amplitude_change",
        descending=True,
        candidate=None,
        vmax=5.0,
        fontsize=10.0,
        amplitude_metric="trial_gain_least_squares",
        amplitude_statistic="cv",
        amplitude_transform="bounded",
        use_trial_weights=True,
        figsize=(10.0, 10.0),
        **selection_defaults,
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
            self,
            results,
            skip=("reliability_threshold", "fraction_active_threshold"),
            defaults=selection_defaults,
        )
        self._stored_thresholds = {
            name: getattr(results.config_class, name, values[0]) if getattr(results.config_class, name, None) in values else values[0]
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
        self.add_selection("candidate", value="No matching ROIs", options=["No matching ROIs"])
        self.add_float("vmax", value=vmax, min=1.0, max=20.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=30.0)
        self.add_selection("amplitude_metric", value=amplitude_metric, options=list(TRIAL_ACTIVITY_METRICS))
        self.add_selection("amplitude_statistic", value=amplitude_statistic, options=list(TRIAL_ACTIVITY_STATISTICS))
        self.add_selection("amplitude_transform", value=amplitude_transform, options=list(_AMPLITUDE_TRANSFORMS))
        self.add_boolean("use_trial_weights", value=use_trial_weights)
        self.on_change("mouse", self.update_mouse)
        self.on_change("environment", self.update_environment)
        self.on_change(
            [
                "early_session",
                "late_session",
                "reliability_threshold",
                "fraction_active_threshold",
                "sort_by",
                "descending",
                "amplitude_metric",
                "amplitude_statistic",
                "amplitude_transform",
                *self.selection_names,
            ],
            self.refresh_candidates,
        )
        table = self.candidate_table()
        options = self._candidate_options(table)
        self.update_selection("candidate", value=candidate if candidate in options else options[0], options=options)

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

    @staticmethod
    def _candidate_options(table):
        """Identify each tracked pair by its filtered early/late ROI numbers."""
        return [f"ROI {int(row.roi_early)} → ROI {int(row.roi_late)}" for row in table.itertuples()] or ["No matching ROIs"]

    def refresh_candidates(self, state):
        table = self.candidate_table(state)
        options = self._candidate_options(table)
        candidate = state["candidate"]
        self.update_selection("candidate", value=candidate if candidate in options else options[0], options=options)

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
            idx_ses=[indices[s.session_uid] for s in sessions],
            use_session_filters=False,
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
            keys = ("env_slot_ids", "reliability_slot", "fraction_active_slot")
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
            amplitude_key = f"{state['amplitude_metric']}_{state['amplitude_statistic']}"
            if amplitude_key not in selected:
                raise KeyError(f"Missing {amplitude_key} -- schema v11 results required.")
            values = np.asarray(selected[amplitude_key])
            if values.shape != np.asarray(selected["reliability_slot"]).shape:
                raise ValueError("ROI filters for the amplitude summary differ from the saved quality results.")
            table[f"amplitude_reliability_{suffix}"] = _transform_amplitude(
                values[slot, roi],
                state["amplitude_transform"],
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
            reason = (
                "Choose two different sessions."
                if state["early_session"] == state["late_session"]
                else ("No tracked neurons pass both quality thresholds in both sessions.\nTry lowering the thresholds or choosing another pair.")
            )
            ax.text(0.5, 0.5, reason, ha="center", va="center", transform=ax.transAxes, fontsize=state["fontsize"])
            return fig
        options = self._candidate_options(table)
        selected = table.iloc[options.index(state["candidate"]) if state["candidate"] in options else 0]
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
                panel = self._panel_class(
                    self.results,
                    mouse=state["mouse"],
                    example_session=session.session_print(),
                    roi=roi,
                    **params,
                )
                self._focus_panels[suffix] = (panel_key, panel)
            else:
                panel = cached[1]
            env_maps = panel._get_env_maps(panel.state)
            env = int(np.flatnonzero(np.asarray(env_maps.environments) == state["environment"])[0])
            panel_state = {**panel.state, **state, "roi": roi, "env": env, "example_session": session.session_print(), **self._stored_thresholds}
            panel.refresh_data(panel_state)
            first_axis = len(fig.axes)
            panel.draw(panel_state, fig, outer[row])
            for ax in fig.axes[first_axis:]:
                if row == 0:
                    ax.set_xlabel("")
                elif ax.get_subplotspec().rowspan.start == 0:
                    ax.set_title("")
        return fig


class TrackedPlaceFieldPredictionFocus(_TrackedPlacefieldFocus):
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

    ``amplitude_metric``, ``amplitude_statistic``, and ``amplitude_transform`` control
    both candidate ranking and plotted summaries. Defaults are least-squares trial
    gain, CV, and the bounded transform: 1 / (1 + CV**2). The default ordering
    is the largest late-minus-early change first. ``candidate`` selects a label
    ``ROI early → ROI late`` using session-filtered ROI indices, in preference order; ``candidate_table()`` returns the full sorted list, including
    tracking cluster IDs, original and filtered ROI indices, and both sessions'
    measurements. Missing measurements are excluded. The selected pair is kept when
    it remains eligible after a filter or sort change. If it drops out, the first
    eligible pair is selected. No-match selections produce an explanatory empty figure.

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
    ``show_correlation=True`` displays r(M,S) inside both position panels.
    ``colorbar_x/y/width/height`` set inset bounds in axes coordinates, and
    ``bottom_text_location`` selects "upper left" or "upper right" for lower-panel text.
    """

    _panel_class = PlaceFieldPredictionFocus

    def __init__(
        self,
        results,
        *,
        msm=None,
        mouse=None,
        environment=None,
        early_session=None,
        late_session=None,
        reliability_threshold=0.3,
        fraction_active_threshold=0.1,
        sort_by="amplitude_change",
        descending=True,
        candidate=None,
        vmax=5.0,
        pf_rms_comparison="f2_from_f1",
        show_correlation=False,
        fontsize=10.0,
        show_consistency=True,
        show_prediction=False,
        include_amplitude_summary=True,
        amplitude_metric="trial_gain_least_squares",
        amplitude_statistic="cv",
        amplitude_transform="bounded",
        nbins=10,
        row_ratio=2.0,
        colorbar_x=0.225,
        colorbar_y=0.15,
        colorbar_width=0.125,
        colorbar_height=0.7,
        bottom_text_location="upper left",
        use_trial_weights=True,
        figsize=(10.0, 10.0),
        **selection_defaults,
    ):
        super().__init__(
            results,
            msm=msm,
            mouse=mouse,
            environment=environment,
            early_session=early_session,
            late_session=late_session,
            reliability_threshold=reliability_threshold,
            fraction_active_threshold=fraction_active_threshold,
            sort_by=sort_by,
            descending=descending,
            candidate=candidate,
            vmax=vmax,
            fontsize=fontsize,
            amplitude_metric=amplitude_metric,
            amplitude_statistic=amplitude_statistic,
            amplitude_transform=amplitude_transform,
            use_trial_weights=use_trial_weights,
            figsize=figsize,
            **selection_defaults,
        )
        self.add_selection("pf_rms_comparison", value=pf_rms_comparison, options=list(PF_RMS_COMPARISONS))
        self.add_boolean("show_correlation", value=show_correlation)
        self.add_boolean("show_consistency", value=show_consistency)
        self.add_boolean("show_prediction", value=show_prediction)
        self.add_boolean("include_amplitude_summary", value=include_amplitude_summary)
        self.add_integer("nbins", value=nbins, min=1, max=100)
        self.add_float("row_ratio", value=row_ratio, min=0.25, max=10.0)
        PlaceFieldPredictionFocus._add_layout_controls(
            self,
            colorbar_x,
            colorbar_y,
            colorbar_width,
            colorbar_height,
            bottom_text_location,
        )


class TrackedPlacefieldBreakdownFocus(_TrackedPlacefieldFocus):
    """Stack two PlacefieldBreakdownFocus rows for the same tracked neuron.

    Tracking, quality cutoffs, and candidate ranking follow
    :class:`TrackedPlaceFieldPredictionFocus`: pass the same quality results and
    select an environment ID and two sessions. The earlier session appears on
    top. The amplitude controls rank the candidates; each row still displays
    the all-trial least-squares gain used in its prediction.

    The map and summary controls follow :class:`PlacefieldBreakdownFocus`.
    Each session shows activity, gain × place field, and the signed residual.
    The spatial similarity, trial gain, and residual summaries are optional.

    Example::

        viewer = TrackedPlacefieldBreakdownFocus(
            quality_results, msm=msm, environment=1,
            include_trial_gain=True, include_residual_rms=True,
            residual_metric="unexplained_fraction",
        )
        viewer.show()
    """

    _panel_class = PlacefieldBreakdownFocus

    def __init__(
        self,
        results,
        *,
        fontsize=12.0,
        figsize=(12.0, 8.0),
        include_spatial_reliability=False,
        include_trial_gain=False,
        include_residual_rms=False,
        residual_metric="rms",
        summary_width=0.5,
        markersize=1.75,
        colorbar_x=0.82,
        colorbar_y=0.12,
        colorbar_width=0.045,
        colorbar_height=0.32,
        legend_x=0.02,
        legend_y=0.98,
        legend_ha="left",
        legend_va="top",
        **tracking_defaults,
    ):
        super().__init__(results, fontsize=fontsize, figsize=figsize, **tracking_defaults)
        _add_breakdown_widgets(
            self,
            include_spatial_reliability=include_spatial_reliability,
            include_trial_gain=include_trial_gain,
            include_residual_rms=include_residual_rms,
            residual_metric=residual_metric,
            summary_width=summary_width,
            markersize=markersize,
            colorbar_x=colorbar_x,
            colorbar_y=colorbar_y,
            colorbar_width=colorbar_width,
            colorbar_height=colorbar_height,
            legend_x=legend_x,
            legend_y=legend_y,
            legend_ha=legend_ha,
            legend_va=legend_va,
        )
