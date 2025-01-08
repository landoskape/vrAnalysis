import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from vrAnalysis import analysis, database, tracking, helpers


class ReliabilityViewer:
    def __init__(self, fast_mode=False):
        mousedb = database.vrDatabase("vrMice")
        df = mousedb.getTable(trackerExists=True)
        self.mouse_names = df["mouseName"].unique()
        if fast_mode:
            self.mouse_names = self.mouse_names[:2]
        print(self.mouse_names)
        self.track = {}
        self.env_selection = {}
        self.idx_ses_selection = {}
        self.rel_idx_ses = {}
        self.idx_ses_first = {}
        self.idx_ses_second = {}
        self.rel_idx_ses_first = {}
        self.rel_idx_ses_second = {}
        self.idx_red = {}
        self.relcor_first = {}
        self.relcor_second = {}
        self.relmse_first = {}
        self.relmse_second = {}
        self.idx_tracked = {}

        for mouse_name in tqdm(self.mouse_names, desc="Preparing mouse data", leave=True):
            self._prepare_mouse_data(mouse_name, fast_mode)

    def _prepare_mouse_data(self, mouse_name, fast_mode):
        self.keep_planes = [1] if fast_mode else [1, 2, 3, 4]
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=self.keep_planes)
        env_stats = pcm.env_stats()
        envnum_first = pcm.env_selector(envmethod="first")
        envnum_second = pcm.env_selector(envmethod="second")
        idx_ses_first = env_stats[envnum_first]
        idx_ses_second = env_stats[envnum_second]
        idx_ses = sorted(list(set(idx_ses_first) | set(idx_ses_second)))

        idx_ses_first_re_second = [ses - idx_ses_second[0] for ses in idx_ses_first]
        idx_ses_second_re_second = [ses - idx_ses_second[0] for ses in idx_ses_second]
        idx_ses_re_second = [ses - idx_ses_second[0] for ses in idx_ses]
        idx_red = [pcm.pcss[i].vrexp.getRedIdx(keep_planes=self.keep_planes) for i in idx_ses]
        relmse_first, relcor_first = helpers.named_transpose([pcm.pcss[i].get_reliability_values(envnum=envnum_first) for i in idx_ses_first])
        relmse_first = list(map(lambda x: x[0], relmse_first))
        relcor_first = list(map(lambda x: x[0], relcor_first))
        relmse_second, relcor_second = helpers.named_transpose([pcm.pcss[i].get_reliability_values(envnum=envnum_second) for i in idx_ses_second])
        relmse_second = list(map(lambda x: x[0], relmse_second))
        relcor_second = list(map(lambda x: x[0], relcor_second))
        idx_tracked = track.get_tracked_idx(idx_ses=idx_ses, keep_planes=self.keep_planes)

        self.track[mouse_name] = track
        self.env_selection[mouse_name] = (envnum_first, envnum_second)
        self.idx_ses_selection[mouse_name] = idx_ses
        self.idx_ses_first[mouse_name] = idx_ses_first
        self.idx_ses_second[mouse_name] = idx_ses_second
        self.rel_idx_ses_first[mouse_name] = idx_ses_first_re_second
        self.rel_idx_ses_second[mouse_name] = idx_ses_second_re_second
        self.rel_idx_ses[mouse_name] = idx_ses_re_second
        self.idx_red[mouse_name] = idx_red
        self.relcor_first[mouse_name] = relcor_first
        self.relcor_second[mouse_name] = relcor_second
        self.relmse_first[mouse_name] = relmse_first
        self.relmse_second[mouse_name] = relmse_second
        self.idx_tracked[mouse_name] = idx_tracked

    def _make_reliability_trajectory(self, mouse_name, tracked, use_relcor, min_session=None, max_session=None):
        # Get the indices of sessions that are used for this mouse
        idx_ses = self.idx_ses_selection[mouse_name]
        idx_ses_first = self.idx_ses_first[mouse_name]
        idx_ses_second = self.idx_ses_second[mouse_name]

        # Filter sessions based on min_session and max_session
        ses_min = min(self.rel_idx_ses[mouse_name]) if min_session is None else min_session
        ses_max = max(self.rel_idx_ses[mouse_name]) if max_session is None else max_session

        # Get indices for sessions within the range
        idx_first = [i for i, x in enumerate(self.rel_idx_ses_first[mouse_name]) if ses_min <= x <= ses_max]
        idx_second = [i for i, x in enumerate(self.rel_idx_ses_second[mouse_name]) if ses_min <= x <= ses_max]

        # Filter session indices
        filtered_ses_first = [idx_ses_first[i] for i in idx_first]
        filtered_ses_second = [idx_ses_second[i] for i in idx_second]
        filtered_ses = sorted(list(set(filtered_ses_first) | set(filtered_ses_second)))

        # Get the reliability values for the filtered sessions
        relvalues_first = self.relcor_first[mouse_name] if use_relcor else self.relmse_first[mouse_name]
        relvalues_second = self.relcor_second[mouse_name] if use_relcor else self.relmse_second[mouse_name]
        relvalues_first = [relvalues_first[i] for i in idx_first]
        relvalues_second = [relvalues_second[i] for i in idx_second]

        # Get the indices of red cells for the filtered sessions
        idx_red = self.idx_red[mouse_name]
        idx_red_first = [idx_red[idx_ses.index(i)] for i in filtered_ses_first]
        idx_red_second = [idx_red[idx_ses.index(i)] for i in filtered_ses_second]

        # Get the indices of tracked cells for the first and second environments
        min_changed = min_session is not None and min_session != min(self.rel_idx_ses[mouse_name])
        max_changed = max_session is not None and max_session != max(self.rel_idx_ses[mouse_name])
        if min_changed or max_changed:
            idx_tracked = self.track[mouse_name].get_tracked_idx(idx_ses=filtered_ses, keep_planes=self.keep_planes)
        else:
            idx_tracked = [self.idx_tracked[mouse_name][idx_ses.index(i)] for i in filtered_ses]

        # Get the indices of tracked cells for the first and second environments
        idx_tracked_first = [idx_tracked[filtered_ses.index(i)] for i in filtered_ses_first]
        idx_tracked_second = [idx_tracked[filtered_ses.index(i)] for i in filtered_ses_second]

        if tracked:
            # If tracked, filter everything by tracked indices first then divide by red / ctl
            relvalues_first = [rvs[i] for rvs, i in zip(relvalues_first, idx_tracked_first)]
            relvalues_second = [rvs[i] for rvs, i in zip(relvalues_second, idx_tracked_second)]
            idx_red_first = [ir[i] for ir, i in zip(idx_red_first, idx_tracked_first)]
            idx_red_second = [ir[i] for ir, i in zip(idx_red_second, idx_tracked_second)]
            idx_red_first = np.any(np.stack(idx_red_first), axis=0)
            idx_red_second = np.any(np.stack(idx_red_second), axis=0)
            ctl_relvalues_first = [rvs[~idx_red_first] for rvs in relvalues_first]
            red_relvalues_first = [rvs[idx_red_first] for rvs in relvalues_first]
            ctl_relvalues_second = [rvs[~idx_red_second] for rvs in relvalues_second]
            red_relvalues_second = [rvs[idx_red_second] for rvs in relvalues_second]

        else:
            # If not tracked, divide by red/ctl per session
            ctl_relvalues_first = [rvs[~ir] for rvs, ir in zip(relvalues_first, idx_red_first)]
            red_relvalues_first = [rvs[ir] for rvs, ir in zip(relvalues_first, idx_red_first)]
            ctl_relvalues_second = [rvs[~ir] for rvs, ir in zip(relvalues_second, idx_red_second)]
            red_relvalues_second = [rvs[ir] for rvs, ir in zip(relvalues_second, idx_red_second)]

        ctl_rel_avg_first = list(map(np.nanmean, ctl_relvalues_first))
        red_rel_avg_first = list(map(np.nanmean, red_relvalues_first))
        ctl_rel_avg_second = list(map(np.nanmean, ctl_relvalues_second))
        red_rel_avg_second = list(map(np.nanmean, red_relvalues_second))
        rel_averages = dict(
            ctl_rel_avg_first=ctl_rel_avg_first,
            red_rel_avg_first=red_rel_avg_first,
            ctl_rel_avg_second=ctl_rel_avg_second,
            red_rel_avg_second=red_rel_avg_second,
        )
        rel_values = dict(
            ctl_rel_first=ctl_relvalues_first,
            red_rel_first=red_relvalues_first,
            ctl_rel_second=ctl_relvalues_second,
            red_rel_second=red_relvalues_second,
        )
        return rel_averages, rel_values

    def _plot_averages(self, mouse_name, relname, tracked, rel_averages, min_session=None, max_session=None):
        plt.rcParams["font.size"] = 14
        fig = plt.figure(1, figsize=(8, 7), layout="constrained")
        fig.clf()

        # Get session limits for plot display
        ses_min = min(self.rel_idx_ses[mouse_name]) if min_session is None else min_session
        ses_max = max(self.rel_idx_ses[mouse_name]) if max_session is None else max_session

        # Get filtered data (already filtered in _make_reliability_trajectory)
        ctl_avg_first = rel_averages["ctl_rel_avg_first"]
        red_avg_first = rel_averages["red_rel_avg_first"]
        ctl_avg_second = rel_averages["ctl_rel_avg_second"]
        red_avg_second = rel_averages["red_rel_avg_second"]

        # Get filtered session numbers
        ses_first = [x for x in self.rel_idx_ses_first[mouse_name] if ses_min <= x <= ses_max]
        ses_second = [x for x in self.rel_idx_ses_second[mouse_name] if ses_min <= x <= ses_max]

        min_y = min(ctl_avg_first + red_avg_first + ctl_avg_second + red_avg_second)
        max_y = max(ctl_avg_first + red_avg_first + ctl_avg_second + red_avg_second)

        # Rest of plotting code remains the same, but using filtered data directly
        ax = fig.add_subplot(311)
        ax.cla()
        ax.plot(ses_first, ctl_avg_first, color="k", linewidth=2, marker=".", markersize=10, markerfacecolor="k", markeredgecolor="k", label="Ctl")
        ax.plot(ses_first, red_avg_first, color="r", linewidth=2, marker=".", markersize=10, markerfacecolor="r", markeredgecolor="r", label="Red")
        ax.set_xlim(ses_min - 0.5, ses_max + 0.5)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel("Session Number Re: First in Novel")
        ax.set_ylabel(relname)
        ax.set_title("Familiar Environment")
        ax.legend(loc="best")

        ax = fig.add_subplot(312)
        ax.cla()
        ax.plot(ses_second, ctl_avg_second, color="k", linewidth=2, marker=".", markersize=10, markerfacecolor="k", markeredgecolor="k", label="Ctl")
        ax.plot(ses_second, red_avg_second, color="r", linewidth=2, marker=".", markersize=10, markerfacecolor="r", markeredgecolor="r", label="Red")
        ax.set_xlim(ses_min - 0.5, ses_max + 0.5)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel("Session Number Re: First in Novel")
        ax.set_ylabel(relname)
        ax.set_title("Novel Environment")
        ax.legend(loc="best")

        ax = fig.add_subplot(313)
        ax.cla()
        difference_first = [rf - cf for rf, cf in zip(red_avg_first, ctl_avg_first)]
        difference_second = [rf - cf for rf, cf in zip(red_avg_second, ctl_avg_second)]
        ax.plot(
            ses_first,
            difference_first,
            color="k",
            linewidth=2,
            marker=".",
            markersize=10,
            markerfacecolor="k",
            markeredgecolor="k",
            label="Familiar Environment",
        )
        ax.plot(
            ses_second,
            difference_second,
            color="g",
            linewidth=2,
            marker=".",
            markersize=10,
            markerfacecolor="g",
            markeredgecolor="g",
            label="Novel Environment",
        )
        ax.axhline(0, color="k", linewidth=1, linestyle="--")
        ax.set_xlim(ses_min - 0.5, ses_max + 0.5)
        ax.set_xlabel("Session Number Re: First in Novel")
        ax.set_ylabel(f"$\Delta$ {relname}")
        ax.set_title("Red - Ctl")
        ax.legend(loc="best")

        fig.suptitle(f"Mouse: {mouse_name}, Reliability: {relname}, Tracked: {tracked}")
        return fig

    def _plot_distributions(self, mouse_name, relname, tracked, rel_values, rel_averages, min_session=None, max_session=None):
        plt.rcParams["font.size"] = 14
        fig = plt.figure(1, figsize=(8, 7), layout="constrained")
        fig.clf()

        # Get session limits for plot display
        ses_min = min(self.rel_idx_ses[mouse_name]) if min_session is None else min_session
        ses_max = max(self.rel_idx_ses[mouse_name]) if max_session is None else max_session

        ctl_rel_avg_first = rel_averages["ctl_rel_avg_first"]
        red_rel_avg_first = rel_averages["red_rel_avg_first"]
        ctl_rel_avg_second = rel_averages["ctl_rel_avg_second"]
        red_rel_avg_second = rel_averages["red_rel_avg_second"]

        # Get filtered data (already filtered in _make_reliability_trajectory)
        ctl_rel_first = rel_values["ctl_rel_first"]
        red_rel_first = rel_values["red_rel_first"]
        ctl_rel_second = rel_values["ctl_rel_second"]
        red_rel_second = rel_values["red_rel_second"]

        # Get filtered session numbers
        ses_first = [x for x in self.rel_idx_ses_first[mouse_name] if ses_min <= x <= ses_max]
        ses_second = [x for x in self.rel_idx_ses_second[mouse_name] if ses_min <= x <= ses_max]

        if relname == "relcor":
            all_values = np.concatenate([np.concatenate(l) for l in [ctl_rel_first, red_rel_first, ctl_rel_second, red_rel_second]])
            min_y = np.nanmin(all_values)
            max_y = np.nanmax(all_values)
        else:
            min_y = -3.5
            max_y = 1

        fig = plt.figure(1, figsize=(8, 7), layout="constrained")
        fig.clf()

        ax = fig.add_subplot(311)
        ax.cla()
        labels = []
        for idx, rel in enumerate(ses_first):
            ctl_data = ctl_rel_first[idx][~np.isnan(ctl_rel_first[idx])]
            red_data = red_rel_first[idx][~np.isnan(red_rel_first[idx])]
            ctl = ax.violinplot(ctl_data, positions=[rel], showmeans=True, showmedians=False, showextrema=False)
            red = ax.violinplot(red_data, positions=[rel], showmeans=True, showmedians=False, showextrema=False)
            if idx == 0:
                labels.append((mpatches.Patch(color="k"), "Ctl"))
                labels.append((mpatches.Patch(color="r"), "Red"))
            ctl["bodies"][0].set_facecolor("k")
            red["bodies"][0].set_facecolor("r")
            ctl["bodies"][0].set_edgecolor(None)
            red["bodies"][0].set_edgecolor(None)
            ctl["bodies"][0].set_alpha(0.3)
            red["bodies"][0].set_alpha(0.3)
            ctl["cmeans"].set_edgecolor("k")
            red["cmeans"].set_edgecolor("r")
            ctl["cmeans"].set_linewidth(2)
            red["cmeans"].set_linewidth(2)
        ax.set_xlim(ses_min - 0.5, ses_max + 0.5)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel("Session Number Re: First in Novel")
        ax.set_ylabel(relname)
        ax.set_title("Familiar Environment")
        ax.legend(*zip(*labels), loc="best")

        ax = fig.add_subplot(312)
        ax.cla()
        labels = []
        for idx, rel in enumerate(ses_second):
            ctl_data = ctl_rel_second[idx][~np.isnan(ctl_rel_second[idx])]
            red_data = red_rel_second[idx][~np.isnan(red_rel_second[idx])]
            ctl = ax.violinplot(ctl_data, positions=[rel], showmeans=True, showmedians=False, showextrema=False)
            red = ax.violinplot(red_data, positions=[rel], showmeans=True, showmedians=False, showextrema=False)
            if idx == 0:
                labels.append((mpatches.Patch(color="k"), "Ctl"))
                labels.append((mpatches.Patch(color="r"), "Red"))
            ctl["bodies"][0].set_facecolor("k")
            red["bodies"][0].set_facecolor("r")
            ctl["bodies"][0].set_edgecolor(None)
            red["bodies"][0].set_edgecolor(None)
            ctl["bodies"][0].set_alpha(0.3)
            red["bodies"][0].set_alpha(0.3)
            ctl["cmeans"].set_edgecolor("k")
            red["cmeans"].set_edgecolor("r")
            ctl["cmeans"].set_linewidth(2)
            red["cmeans"].set_linewidth(2)
        ax.set_xlim(ses_min - 0.5, ses_max + 0.5)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel("Session Number Re: First in Novel")
        ax.set_ylabel(relname)
        ax.set_title("Novel Environment")
        ax.legend(*zip(*labels), loc="best")

        ax = fig.add_subplot(313)
        ax.cla()
        # Filter the differences for the bottom plot
        filtered_idx_first = [i for i, x in enumerate(self.rel_idx_ses_first[mouse_name]) if ses_min <= x <= ses_max]
        filtered_idx_second = [i for i, x in enumerate(self.rel_idx_ses_second[mouse_name]) if ses_min <= x <= ses_max]

        difference_first = [red_rel_avg_first[i] - ctl_rel_avg_first[i] for i in filtered_idx_first]
        difference_second = [red_rel_avg_second[i] - ctl_rel_avg_second[i] for i in filtered_idx_second]

        ax.plot(
            ses_first,
            difference_first,
            color="k",
            linewidth=2,
            marker=".",
            markersize=10,
            markerfacecolor="k",
            markeredgecolor="k",
            label="Familiar Environment",
        )
        ax.plot(
            ses_second,
            difference_second,
            color="g",
            linewidth=2,
            marker=".",
            markersize=10,
            markerfacecolor="g",
            markeredgecolor="g",
            label="Novel Environment",
        )
        ax.axhline(0, color="k", linewidth=1, linestyle="--")
        ax.set_xlim(ses_min - 0.5, ses_max + 0.5)
        ax.set_xlabel("Session Number Re: First in Novel")
        ax.set_ylabel(f"$\Delta$ {relname}")
        ax.set_title("Red - Ctl")
        ax.legend(loc="best")

        fig.suptitle(f"Mouse: {mouse_name}, Reliability: {relname}, Tracked: {tracked}")
        return fig

    def get_plot(self, mouse_name, use_relcor=True, tracked=True, average=True, min_session=None, max_session=None):
        rel_averages, rel_values = self._make_reliability_trajectory(mouse_name, tracked, use_relcor, min_session, max_session)

        relname = "relcor" if use_relcor else "relmse"
        if average:
            return self._plot_averages(mouse_name, relname, tracked, rel_averages, min_session, max_session)
        else:
            return self._plot_distributions(mouse_name, relname, tracked, rel_values, rel_averages, min_session, max_session)
