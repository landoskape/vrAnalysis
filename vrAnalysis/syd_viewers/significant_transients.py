from typing import Optional, Dict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from vrAnalysis import helpers
from vrAnalysis import database
from vrAnalysis import session
from syd import Viewer

sessiondb = database.vrDatabase("vrSessions")


class SigTransientLoader:
    def __init__(self, ses: session.vrExperiment, percentile: int = 30, window_duration: int = 60):
        self.ses = ses
        self.percentile = percentile
        self.window_duration = window_duration
        self.fluor_times = ses.loadone("mpci.times")
        self.fluor_corrected = ses.loadfcorr().T
        self.num_rois = self.fluor_corrected.shape[1]
        self.fluorescence_buffer = {}
        self.buffer_roi_idx = {}

    def _load_fluorescence(self, percentile: int, window_duration: int, roi_idx: Optional[np.ndarray] = None):
        """Load fluorescence and percentile filtered standardized fluorescence

        Can optionally specify the filtering properties (percentile, window_duration)
        Can also specify which ROIs to keep (to speed things up for example)

        Will use a buffer intelligently to only process data that hasn't been processed with the current settings

        roi_idx = None means all ROIs
        """
        buffer_id = self._get_buffer_id(percentile, window_duration)
        roi_idx_needed = self._roi_idx_diff(buffer_id, roi_idx)

        fcorr = self.fluor_corrected
        if roi_idx is not None:
            fcorr = fcorr[:, roi_idx_needed]

        dff_std = helpers.get_standardized_dff(fcorr, self.ses.opts["fs"], percentile, window_duration)

        # Get cached ROI idx, and concatenate with just loaded roi idx
        roi_idx_cached = self._get_cached_roi_idx(buffer_id)
        all_roi_idx = np.concatenate((roi_idx_cached, roi_idx_needed))

        # Get cached fluorescence data and concatenate with just loaded new fluorescence data
        fluor_data_cached = self._get_cached_fluor_data(buffer_id)
        all_data = np.concatenate((fluor_data_cached, dff_std), axis=1)

        # Sort to keep ROIs in order
        idx_roi_ordered = np.argsort(all_roi_idx)
        sorted_roi_idx = all_roi_idx[idx_roi_ordered]
        sorted_dff_std = all_data[:, idx_roi_ordered]

        # Cache new data
        self.fluorescence_buffer[buffer_id] = sorted_dff_std
        self.buffer_roi_idx[buffer_id] = sorted_roi_idx

        # Figure out which part was just requested
        idx_keep_rois = np.isin(sorted_roi_idx, roi_idx)
        return sorted_dff_std[:, idx_keep_rois]

    def _get_cached_roi_idx(self, buffer_id: str):
        return self.buffer_roi_idx.get(buffer_id, np.array([]))

    def _get_cached_fluor_data(self, buffer_id: str):
        return self.fluorescence_buffer.get(buffer_id, np.empty((len(self.fluor_times), 0)))

    def _get_buffer_id(self, percentile: int, window_duration: int):
        return f"percentile{percentile}_window{window_duration}"

    def _roi_idx_diff(self, buffer_id: str, roi_idx: Optional[np.ndarray] = None):
        # Identify rois in roi_idx that aren't in buffer_id
        if roi_idx is None:
            roi_idx = np.arange(self.num_rois)
        roi_idx_cached = self._get_cached_roi_idx(buffer_id)
        diff = roi_idx[~np.isin(roi_idx, roi_idx_cached)]
        return diff

    def load_fluorescence(self, percentile: Optional[int] = None, window_duration: Optional[int] = None, roi_idx: Optional[np.ndarray] = None):
        percentile = percentile or self.percentile
        window_duration = window_duration or self.window_duration

        fcorr = self.fluor_corrected
        if roi_idx is not None:
            fcorr = fcorr[:, roi_idx]

        buffer_id = self._get_buffer_id(percentile, window_duration)
        if buffer_id in self.buffer_roi_idx and not len(self._roi_idx_diff(buffer_id, roi_idx)) > 0:
            dff_std = self.fluorescence_buffer[buffer_id]
        else:
            dff_std = self._load_fluorescence(percentile, window_duration, roi_idx)

        return self.fluor_times, fcorr, dff_std

    def load_significant_transients(self, dff: np.ndarray, min_threshold: float, max_threshold: float, step_threshold: float):
        thresholds = np.arange(min_threshold, max_threshold + step_threshold, step_threshold)
        thresholds = np.round(thresholds * 100) / 100
        significant_transients, stats = helpers.get_significant_transients(dff, threshold_levels=thresholds, verbose=False, return_stats=True)
        return significant_transients, thresholds, stats

    def load_plot_data(self, state: dict):
        percentile = state["percentile"]
        window_duration = state["window_duration"]
        # roi_idx_range = state["roi_idx_range"]
        # roi_idx = np.arange(roi_idx_range[0], roi_idx_range[1])
        roi = np.array(state["roi"])
        min_threshold, max_threshold = state["threshold_range"]
        step_threshold = state["step_threshold"]
        ftimes, fcorr, dff = self.load_fluorescence(percentile, window_duration, roi)
        significant_transients, thresholds, stats = self.load_significant_transients(dff, min_threshold, max_threshold, step_threshold)
        keep_frames = np.any(significant_transients, axis=2)[:, 0]
        significant_fluorescence = np.zeros_like(fcorr)
        significant_fluorescence[keep_frames] = fcorr[keep_frames]
        data = dict(
            ftimes=ftimes,
            fcorr=fcorr,
            dff=dff,
            significant_transients=significant_transients,
            significant_fluorescence=significant_fluorescence,
            thresholds=thresholds,
            stats=stats,
        )
        return data

    def plot(self, _, state: dict):
        data = self.load_plot_data(state)
        num_threshold = data["significant_transients"].shape[2]
        xmin = data["ftimes"][0]
        xmax = data["ftimes"][-1]
        dx = xmax / (data["significant_transients"].shape[0] - 1)  # width of each pixel
        extent = [xmin - dx / 2, xmax + dx / 2, -0.5, num_threshold - 0.5]

        fig = plt.figure(figsize=(8, 6), layout="constrained")
        fig.clf()
        subfigs = fig.subfigures(1, 2, wspace=0.05)

        # Create two separate GridSpec objects
        gs_left = subfigs[0].add_gridspec(3, 1)
        gs_right = subfigs[1].add_gridspec(3, 1, height_ratios=[1, 1, 0.2])

        # Left column plots (evenly distributed)
        ax_lineplot = fig.add_subplot(gs_left[0])
        ax_dff = fig.add_subplot(gs_left[1])
        ax_sigplot = fig.add_subplot(gs_left[2])

        # Right section plots
        ax_pos_durs = fig.add_subplot(gs_right[0])
        ax_neg_durs = fig.add_subplot(gs_right[1])
        ax_cbar = fig.add_subplot(gs_right[2])

        ax_lineplot.plot(data["ftimes"], data["fcorr"], color="k", linewidth=1, label="fcorr")
        ax_lineplot.plot(data["ftimes"], data["significant_fluorescence"], color="b", linewidth=1, label="sig.transients")
        ax_lineplot.set_ylabel("Activity")
        ax_lineplot.set_title(f"ROI: {state['roi']}")
        ax_lineplot.set_xlim(0, xmax)

        ax_dff.plot(data["ftimes"], data["dff"], color="k", linewidth=1, label="dff")
        ax_dff.axhline(0, color="r", linestyle="--", linewidth=0.5)
        ax_dff.set_ylabel("DFF")
        ax_dff.set_xlim(0, xmax)

        ax_sigplot.imshow(data["significant_transients"][:, 0].T, interpolation="none", aspect="auto", vmin=0, vmax=1, cmap="gray", extent=extent)
        ax_sigplot.set_xlim(0, xmax)
        ax_sigplot.set_yticks(np.arange(num_threshold))
        ax_sigplot.set_yticklabels(data["thresholds"])

        ax_dff.sharex(ax_lineplot)
        ax_sigplot.sharex(ax_lineplot)

        max_duration = 0
        for i, threshold in enumerate(data["thresholds"]):
            pos_durs = data["stats"][i]["positive_durations"]
            neg_durs = data["stats"][i]["negative_durations"]
            if len(pos_durs) > 0:
                max_duration = max(max_duration, np.max(pos_durs))
            if len(neg_durs) > 0:
                max_duration = max(max_duration, np.max(neg_durs))

        cmap = mpl.colormaps["cividis"]
        norm = mpl.colors.Normalize(vmin=min(data["thresholds"]), vmax=max(data["thresholds"]))
        colors = [cmap(norm(threshold)) for threshold in data["thresholds"]]

        max_counts = 0
        binedges = np.arange(-0.5, max_duration + 0.5, 1)
        for i, threshold in enumerate(data["thresholds"]):
            c_pos_counts = np.cumsum(np.histogram(data["stats"][i]["positive_durations"], bins=binedges)[0])
            max_counts = max(max_counts, np.max(c_pos_counts))
            ax_pos_durs.plot(binedges[:-1], c_pos_counts, color=colors[i])

            c_neg_counts = np.cumsum(np.histogram(data["stats"][i]["negative_durations"], bins=binedges)[0])
            max_counts = max(max_counts, np.max(c_neg_counts))
            ax_neg_durs.plot(binedges[:-1], c_neg_counts, color=colors[i])

        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation="horizontal")
        cbar.set_label("Threshold")

        ax_pos_durs.set_xlim(0, max_duration)
        ax_neg_durs.set_xlim(0, max_duration)
        ax_pos_durs.set_ylim(0, max_counts)
        ax_neg_durs.set_ylim(0, max_counts)
        ax_neg_durs.set_xlabel("Duration (s)")
        ax_neg_durs.set_ylabel("Cumulative Counts")
        ax_pos_durs.set_xlabel("Duration (s)")
        ax_pos_durs.set_ylabel("Cumulative Counts")
        ax_pos_durs.set_title("Positive Transients")
        ax_neg_durs.set_title("Negative Transients")

        return fig


class SigTransientViewer(Viewer):
    def __init__(self, loader: SigTransientLoader):
        self.loader = loader
        self.add_integer("percentile", value=30, min_value=10, max_value=90)
        self.add_integer("window_duration", value=60, min_value=10, max_value=180)
        # viewer.add_integer_range("roi_idx_range", value=(0, min(100, loader.num_rois)), min_value=0, max_value=loader.num_rois)
        self.add_integer("roi", value=0, min_value=0, max_value=loader.num_rois - 1)
        self.add_float_range("threshold_range", value=(1.0, 4.0), min_value=0.2, max_value=5.0, step=0.1)
        self.add_selection("step_threshold", value=0.25, options=[0.1, 0.2, 0.25, 0.5, 1.0])
        self.set_plot(loader.plot)
