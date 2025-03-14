import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import joblib
from syd import Viewer
from ..tracking import Tracker
from ..helpers import vectorCorrelation
from ..multisession import MultiSessionSpkmaps
from ..helpers import errorPlot
from vrAnalysis2.files import analysis_path


class ReliabilityStabilitySummary(Viewer):
    def __init__(self, tracked_mice: list[str], use_cache: bool = True):
        self.tracked_mice = list(tracked_mice)
        self.multisessions = {mouse: None for mouse in self.tracked_mice}
        self.use_cache = use_cache

        # Set up syd parameters
        if use_cache:
            self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
            self.add_selection("environment", value=1, options=[1])
            self.add_selection("reliability_threshold", value=0.5, options=[0.3, 0.5, 0.7, 0.9])
            self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out"])
            self.add_selection("smooth_width", value=5, options=[5])
            self.add_selection("use_session_filters", value=True, options=[True])
            self.add_boolean("continuous", value=False)
        else:
            self.add_selection("mouse", value=self.tracked_mice[0], options=self.tracked_mice)
            self.add_selection("environment", value=1, options=[1])
            self.add_float("reliability_threshold", value=0.5, min=0.1, max=1)
            self.add_selection("reliability_method", value="leave_one_out", options=["leave_one_out", "mse", "correlation"])
            self.add_selection("smooth_width", value=5, options=[1, 5])
            self.add_boolean("use_session_filters", value=True)
            self.add_boolean("continuous", value=True)
            self.add_boolean("max_session_diff", value=6)

        self.on_change("mouse", self.update_mouse)
        self.update_mouse(self.state)

    def update_mouse(self, state: dict):
        results = self.gather_data(state, try_cache=True)
        environments = [envnum for envnum in results if results[envnum] is not None]
        self.update_selection("environment", options=environments)

    @classmethod
    def get_results_name(cls, state: dict) -> str:
        return f"{state['mouse']}-{state['reliability_method']}-Threshold{state['reliability_threshold']}-SmoothWidth{state['smooth_width']}-Continuous{state['continuous']}-GoodROIOnly{state['use_session_filters']}-results.joblib"

    def get_multisession(self, mouse: str) -> MultiSessionSpkmaps:
        if self.multisessions[mouse] is None:
            track = Tracker(mouse)
            self.multisessions[mouse] = MultiSessionSpkmaps(track)
        return self.multisessions[mouse]

    def get_environments(self, mouse: str) -> np.ndarray:
        """Get all environments represented in tracked sessions"""
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        return environments

    def spkmap_kwargs(self, idx_ses: list[int], state: dict):
        return dict(
            idx_ses=idx_ses,
            use_session_filters=state["use_session_filters"],
            tracked=True,
            average=True,
            pop_nan=True,
            reliability_method=state["reliability_method"],
            smooth=float(state["smooth_width"]),
        )

    def _gather_combo_data(
        self,
        state: dict,
        envnum: int,
        msm: MultiSessionSpkmaps,
        all_idx_ses: list[int],
        idx_ses_subset: list[int],
        idx_ref: int,
    ):
        c_idx_ref = idx_ses_subset.index(idx_ref)
        if c_idx_ref != 0:
            raise ValueError("c_idx_ref is not 0 -- it should be for both forward and backward analyses!")
        spkmaps, extras = msm.get_spkmaps(envnum, **self.spkmap_kwargs(idx_ses_subset, state))
        reliability = np.stack(extras["reliability"])
        spkmaps = np.stack(spkmaps)
        pflocs = np.stack(extras["pfloc"])
        target_rois = reliability[c_idx_ref] > state["reliability_threshold"]
        idx_red = np.any(np.stack(extras["idx_red"]), axis=0)
        num_ctl = np.sum(~idx_red & target_rois)
        num_red = np.sum(idx_red & target_rois)
        ctl_reliable = reliability[:, target_rois & ~idx_red]
        red_reliable = reliability[:, target_rois & idx_red]
        ctl_pflocs = pflocs[:, target_rois & ~idx_red]
        red_pflocs = pflocs[:, target_rois & idx_red]
        ctl_spkmaps = spkmaps[:, target_rois & ~idx_red]
        red_spkmaps = spkmaps[:, target_rois & idx_red]
        ctl_stability = np.full(ctl_reliable.shape, False, dtype=bool)
        red_stability = np.full(red_reliable.shape, False, dtype=bool)
        ctl_stability[c_idx_ref] = True
        red_stability[c_idx_ref] = True
        # Works for continuous and discontinuous, because in discontinuous we're only
        # considering the two sessions in question!!
        for idx in range(1, len(idx_ses_subset)):
            ctl_stability[idx] = (ctl_reliable[idx] > state["reliability_threshold"]) & ctl_stability[idx - 1]
            red_stability[idx] = (red_reliable[idx] > state["reliability_threshold"]) & red_stability[idx - 1]
        num_stable_ctl = np.sum(ctl_stability, axis=1)
        num_stable_red = np.sum(red_stability, axis=1)
        fraction_stable_ctl = num_stable_ctl / num_ctl
        fraction_stable_red = num_stable_red / num_red
        stable_reliability_ctl = np.sum(ctl_stability * ctl_reliable, axis=1) / np.sum(ctl_stability, axis=1)
        stable_reliability_red = np.sum(red_stability * red_reliable, axis=1) / np.sum(red_stability, axis=1)
        pfloc_changes_ctl = np.abs(ctl_pflocs - ctl_pflocs[c_idx_ref])
        pfloc_changes_red = np.abs(red_pflocs - red_pflocs[c_idx_ref])
        pfloc_changes_ctl = np.sum(pfloc_changes_ctl * ctl_stability, axis=1) / np.sum(ctl_stability, axis=1)
        pfloc_changes_red = np.sum(pfloc_changes_red * red_stability, axis=1) / np.sum(red_stability, axis=1)
        spkmap_correlations_ctl = np.full(ctl_spkmaps.shape[:2], np.nan)
        spkmap_correlations_red = np.full(red_spkmaps.shape[:2], np.nan)
        for ii in range(len(idx_ses_subset)):
            spkmap_correlations_ctl[ii] = vectorCorrelation(ctl_spkmaps[ii], ctl_spkmaps[c_idx_ref], axis=1)
            spkmap_correlations_red[ii] = vectorCorrelation(red_spkmaps[ii], red_spkmaps[c_idx_ref], axis=1)
        spkmap_correlations_ctl = np.sum(spkmap_correlations_ctl * ctl_stability, axis=1) / np.sum(ctl_stability, axis=1)
        spkmap_correlations_red = np.sum(spkmap_correlations_red * red_stability, axis=1) / np.sum(red_stability, axis=1)

        inputs = dict(
            num_stable_ctl=num_stable_ctl,
            num_stable_red=num_stable_red,
            fraction_stable_ctl=fraction_stable_ctl,
            fraction_stable_red=fraction_stable_red,
            stable_reliability_ctl=stable_reliability_ctl,
            stable_reliability_red=stable_reliability_red,
            pfloc_changes_ctl=pfloc_changes_ctl,
            pfloc_changes_red=pfloc_changes_red,
            spkmap_correlations_ctl=spkmap_correlations_ctl,
            spkmap_correlations_red=spkmap_correlations_red,
        )
        outputs = {input_key: np.full(state["max_session_diff"], np.nan) for input_key in inputs}
        for ikey, input in inputs.items():
            for cidx, cidxses in enumerate(idx_ses_subset):
                if cidx == c_idx_ref:
                    # Don't add in the reference session
                    continue
                # Relative session difference (if a session is skipped, e.g. we have 1, 2, 4, 5; then the 2 4 combo has a difference of 1)
                # Ignore direction --- if forward or backward we only care about the number of sessions between the two!
                session_difference = abs(all_idx_ses.index(cidxses) - all_idx_ses.index(idx_ref))
                outputs[ikey][session_difference - 1] = input[cidx]
        return outputs

    def gather_data(self, state: dict, try_cache: bool = True):
        if try_cache:
            results_name = self.get_results_name(state)
            results_path = analysis_path() / "before_the_reveal_temp_data" / results_name
            if results_path.exists():
                return joblib.load(results_path)

        if self.use_cache:
            raise RuntimeError("Using cache mode but the loading failed!")

        mouse = state["mouse"]
        msm = self.get_multisession(mouse)
        environments = np.unique(np.concatenate([session.environments for session in msm.tracker.sessions]))
        continuous = state["continuous"]

        results = {env: {} for env in environments}
        for envnum in environments:
            all_idx_ses = msm.idx_ses_with_env(envnum)
            num_sessions = len(all_idx_ses)
            if num_sessions == 1:
                results[envnum] = None
                continue

            forward_results = []
            backward_results = []
            forward_combos = []
            backward_combos = []
            for idx, idx_ref in enumerate(all_idx_ses):
                # Forward analysis
                last_forward_compare = min(num_sessions, idx + state["max_session_diff"] + 1)
                for idx_compare in range(idx + 1, last_forward_compare):
                    if continuous:
                        idx_ses_subset = [all_idx_ses[i] for i in range(idx, idx_compare + 1)]
                    else:
                        idx_ses_subset = [all_idx_ses[i] for i in [idx, idx_compare]]
                    forward_results.append(self._gather_combo_data(state, envnum, msm, all_idx_ses, idx_ses_subset, idx_ref))
                    forward_combos.append(idx_ses_subset)

                # Backward analysis
                first_backward_compare = max(-1, idx - state["max_session_diff"] - 1)
                for idx_compare in range(idx - 1, first_backward_compare, -1):
                    if continuous:
                        idx_ses_subset = [all_idx_ses[i] for i in range(idx, idx_compare - 1, -1)]
                    else:
                        idx_ses_subset = [all_idx_ses[i] for i in [idx, idx_compare]]
                    backward_results.append(self._gather_combo_data(state, envnum, msm, all_idx_ses, idx_ses_subset, idx_ref))
                    backward_combos.append(idx_ses_subset)

            output_keys = list(forward_results[0].keys())
            stacked_forward = {k: np.stack([d[k] for d in forward_results]) for k in output_keys}
            stacked_backward = {k: np.stack([d[k] for d in backward_results]) for k in output_keys}
            results[envnum] = dict(
                forward=stacked_forward,
                backward=stacked_backward,
                forward_combos=forward_combos,
                backward_combos=backward_combos,
            )

        return results

    def define_state(
        self,
        mouse_name: str,
        envnum: int,
        reliability_threshold: float = 0.5,
        reliability_method: str = "leave_one_out",
        smooth_width: int = 5,
        use_session_filters: bool = True,
        continuous: bool = True,
        max_session_diff: int = 6,
    ):
        """For use from independent of the viewer method to create a state for plotting!"""
        state = dict(
            mouse=mouse_name,
            environment=envnum,
            reliability_threshold=reliability_threshold,
            reliability_method=reliability_method,
            smooth_width=smooth_width,
            use_session_filters=use_session_filters,
            continuous=continuous,
            max_session_diff=max_session_diff,
        )
        return state

    def plot(self, state, results=None):
        if results is None:
            results = self.gather_data(state, try_cache=True)

        environments = [envnum for envnum in results if results[envnum] is not None]
        output_keys = list(results[environments[0]]["forward"].keys())

        # Create figure with three subplots
        figwidth = 3
        figheight = 3.5

        envnum = state["environment"]
        max_diffs = results[envnum]["forward"]["num_stable_ctl"].shape[1]
        forward_ticks = np.arange(1, max_diffs + 1)
        backward_ticks = -forward_ticks[::-1]
        xticks = [backward_ticks, forward_ticks]

        fb_data = {}
        for output_key in output_keys:
            cforward = results[envnum]["forward"][output_key]
            cbackward = np.fliplr(results[envnum]["backward"][output_key])
            fb_data[output_key] = [cbackward, cforward]

        fig = plt.figure(figsize=(3 * figwidth, figheight), layout="constrained")
        gs = fig.add_gridspec(2, 3)
        ax_frac = fig.add_subplot(gs[0, 0])
        ax_rel = fig.add_subplot(gs[0, 1])
        ax_pfloc = fig.add_subplot(gs[1, 0])
        ax_spkmap = fig.add_subplot(gs[1, 1])
        ax_num_ctl = fig.add_subplot(gs[0, 2])
        ax_num_red = fig.add_subplot(gs[1, 2])

        error_plot_kwargs = lambda color: dict(axis=0, se=True, color=color, alpha=0.2, handle_nans=True)

        # Fraction Stably Reliable Plot
        for ii in range(2):
            errorPlot(xticks[ii], fb_data["fraction_stable_ctl"][ii], **error_plot_kwargs("k"), ax=ax_frac)
            errorPlot(xticks[ii], fb_data["fraction_stable_red"][ii], **error_plot_kwargs("r"), ax=ax_frac)
        ax_frac.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax_frac.set_ylim(-0.05, 1.05)
        ax_frac.set_xlabel("Session Difference")
        ax_frac.set_ylabel("Fraction Stable")

        # Of fraction stable, reliability values plot
        for ii in range(2):
            errorPlot(xticks[ii], fb_data["stable_reliability_ctl"][ii], **error_plot_kwargs("k"), ax=ax_rel)
            errorPlot(xticks[ii], fb_data["stable_reliability_red"][ii], **error_plot_kwargs("r"), ax=ax_rel)
        ax_rel.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax_rel.set_xlabel("Session Difference")
        ax_rel.set_ylabel("Reliability (mean)")

        # Num stable control cells plot
        for ii in range(2):
            errorPlot(xticks[ii], fb_data["num_stable_ctl"][ii], **error_plot_kwargs("k"), ax=ax_num_ctl)
        ax_num_ctl.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ylim = ax_num_ctl.get_ylim()
        ax_num_ctl.set_ylim(0, ylim[1])
        ax_num_ctl.set_xlabel("Session Difference")
        ax_num_ctl.set_ylabel("# Cells (CTL)")

        # Num stable red cells plot
        for ii in range(2):
            errorPlot(xticks[ii], fb_data["num_stable_red"][ii], **error_plot_kwargs("r"), ax=ax_num_red)
        ax_num_red.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ylim = ax_num_red.get_ylim()
        ax_num_red.set_ylim(0, ylim[1])
        ax_num_red.set_xlabel("Session Difference")
        ax_num_red.set_ylabel("# Cells (RED)")

        # PFloc changes plot
        for ii in range(2):
            errorPlot(xticks[ii], fb_data["pfloc_changes_ctl"][ii], **error_plot_kwargs("k"), ax=ax_pfloc)
            errorPlot(xticks[ii], fb_data["pfloc_changes_red"][ii], **error_plot_kwargs("r"), ax=ax_pfloc)
        ax_pfloc.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ylim = ax_pfloc.get_ylim()
        ax_pfloc.set_ylim(0, ylim[1])
        ax_pfloc.set_xlabel("Session Difference")
        ax_pfloc.set_ylabel("PFloc Change")

        # Spkmap correlation plot
        for ii in range(2):
            errorPlot(xticks[ii], fb_data["spkmap_correlations_ctl"][ii], **error_plot_kwargs("k"), ax=ax_spkmap)
            errorPlot(xticks[ii], fb_data["spkmap_correlations_red"][ii], **error_plot_kwargs("r"), ax=ax_spkmap)
        ax_spkmap.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax_spkmap.set_xlabel("Session Difference")
        ax_spkmap.set_ylabel("Spkmap Correlation")

        title_parts = dict(
            mouse=state["mouse"],
            envnum=state["environment"],
            relthresh=state["reliability_threshold"],
            relmethod=state["reliability_method"],
            addreturn=None,
            smoothsig=state["smooth_width"],
            goodroionly=state["use_session_filters"],
            continuous=state["continuous"],
        )
        title = ""
        for key, value in title_parts.items():
            if key == "addreturn":
                title += "\n"
                continue
            title += f"{key}{value}-"
        title = title[:-1]
        fig.suptitle(title)

        return fig


class ReliabilityMasterSummary(Viewer):
    def __init__(self, tracked_mice: list[str]):
        self.tracked_mice = list(tracked_mice)

        self.add_selection("reliability_threshold", value=0.5, options=[0.3, 0.5, 0.7, 0.9])
        self.add_boolean("continuous", value=True)
        self.add_selection("forward_backward", value="both", options=["forward", "backward", "both"])
        self.add_boolean("split_environments", value=True)

        self.msms = {mouse: MultiSessionSpkmaps(Tracker(mouse)) for mouse in self.tracked_mice}
        self.summary_viewer = ReliabilityStabilitySummary(self.tracked_mice)

    def plot(self, state: dict):
        reliability_method = "leave_one_out"
        smooth_width = float(5.0)
        use_session_filters = True
        max_session_diff = 6
        permitted_thresholds = [0.3, 0.5, 0.7, 0.9]
        permitted_continuous = [True, False]
        if state["reliability_threshold"] not in permitted_thresholds:
            raise ValueError(f"Invalid reliability threshold: {state['reliability_threshold']}")
        if state["continuous"] not in permitted_continuous:
            raise ValueError(f"Invalid continuous: {state['continuous']}")

        results = {}
        for mouse in self.tracked_mice:
            results_state = self.summary_viewer.define_state(
                mouse_name=mouse,
                envnum=1,
                reliability_threshold=state["reliability_threshold"],
                reliability_method=reliability_method,
                smooth_width=int(smooth_width),
                use_session_filters=use_session_filters,
                continuous=state["continuous"],
                max_session_diff=max_session_diff,
            )
            results[mouse] = self.summary_viewer.gather_data(results_state, try_cache=True)

        output_keys = [
            "num_stable_ctl",
            "num_stable_red",
            "fraction_stable_ctl",
            "fraction_stable_red",
            "stable_reliability_ctl",
            "stable_reliability_red",
            "pfloc_changes_ctl",
            "pfloc_changes_red",
            "spkmap_correlations_ctl",
            "spkmap_correlations_red",
        ]
        output_names = ["fraction_stable", "pfloc_changes", "spkmap_correlations"]

        max_environments = 3

        data = {key: np.full((len(results), max_environments, max_session_diff), np.nan) for key in output_keys}

        for imouse, mouse in enumerate(results):
            envstats = self.msms[mouse].env_stats()
            env_in_order = sorted(envstats, key=lambda x: envstats[x][0])
            env_in_order = [env for env in env_in_order if env != -1]
            for ienv, env in enumerate(env_in_order):
                if ienv >= max_environments:
                    continue
                if results[mouse][env] is None:
                    continue
                for key in output_keys:
                    if state["forward_backward"] == "forward":
                        data[key][imouse, ienv] = np.nanmean(results[mouse][env]["forward"][key], axis=0)
                    elif state["forward_backward"] == "backward":
                        data[key][imouse, ienv] = np.nanmean(results[mouse][env]["backward"][key], axis=0)
                    elif state["forward_backward"] == "both":
                        forward_data = np.nanmean(results[mouse][env]["forward"][key], axis=0)
                        backward_data = np.nanmean(results[mouse][env]["backward"][key], axis=0)
                        data[key][imouse, ienv] = np.nanmean(np.stack([forward_data, backward_data]), axis=0)

        num_cols = len(output_names)
        num_rows = max_environments if state["split_environments"] else 1

        cmap_mice = mpl.colormaps["tab10"]
        colors_mice = cmap_mice(np.linspace(0, 1, len(results)))
        colors_mice = {mouse: colors_mice[imouse] for imouse, mouse in enumerate(results)}
        colors_mice["CR_Hippocannula6"] = "black"
        colors_mice["CR_Hippocannula7"] = "dimgrey"
        linewidth = [1.5 if mouse in ["CR_Hippocannula6", "CR_Hippocannula7"] else 0.75 for mouse in results]
        zorder = [1 if mouse in ["CR_Hippocannula6", "CR_Hippocannula7"] else 0 for mouse in results]

        figwidth = 3
        figheight = 3
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(figwidth * num_cols, figheight * num_rows), layout="constrained")
        if num_rows == 1:
            ax = np.reshape(ax, (1, num_cols))
        for icol, col_name in enumerate(output_names):
            ctl_data = data[col_name + "_ctl"]
            red_data = data[col_name + "_red"]
            if not state["split_environments"]:
                ctl_data = np.nanmean(ctl_data, axis=1, keepdims=True)
                red_data = np.nanmean(red_data, axis=1, keepdims=True)
            diff_data = red_data - ctl_data

            for irow in range(num_rows):
                for imouse, mouse in enumerate(results):
                    ax[irow, icol].plot(
                        range(1, max_session_diff + 1),
                        diff_data[imouse, irow],
                        color=colors_mice[mouse],
                        linewidth=linewidth[imouse],
                        zorder=zorder[imouse],
                    )
                ax[irow, icol].set_ylabel(f"$\Delta$ {col_name} (red-ctl)")

            ax[-1, icol].set_xlabel("$\Delta$ Session")
            ax[0, icol].set_title(col_name)

            ax_titles = [(f"{col_name}" if irow == 0 else "") for irow in range(num_rows)]
            ax_titles = [title + ("\n" if irow == 0 else "") + f"Env#{irow+1}" for irow, title in enumerate(ax_titles)]
            for irow, title in enumerate(ax_titles):
                ax[irow, icol].set_title(title)

        suptitle = (
            f"Full Summary: Threshold={state['reliability_threshold']}, Continuous={state['continuous']}, ForwardBackward={state['forward_backward']}"
        )
        fig.suptitle(suptitle)

        return fig
