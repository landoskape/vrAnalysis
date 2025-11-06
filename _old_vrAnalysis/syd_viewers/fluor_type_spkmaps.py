from typing import List, Dict, Any
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from _old_vrAnalysis import session
from _old_vrAnalysis.analysis import placeCellSingleSession
from syd import Viewer


class FluorTypeLoader:
    def __init__(self, ises: List[session.vrExperiment]):
        self.ises = ises

        self.fluorescence_types = ["fcorr", "significant", "deconvolved", "oasis"]
        self.fluorescence_colors = ["black", "blue", "red", "orange"]
        self.fluor_colors = dict(zip(self.fluorescence_types, self.fluorescence_colors))

        self.mice_list = list(np.unique([ses.mouseName for ses in ises]))
        self.mouse_sessions = {mouse: [ses for ses in ises if ses.mouseName == mouse] for mouse in self.mice_list}
        self.session_environments = {mouse: [self.get_session_environments(ses) for ses in self.mouse_sessions[mouse]] for mouse in self.mice_list}

        # For caching spkmaps... will clear if the mouse changes
        self.current_mouse = None
        self.loaded_fluor_data = {}
        self.loaded_spkmaps = {}
        self.loaded_reliable_rois = {}

    def get_session_environments(self, ses: session.vrExperiment):
        trial_envnum = ses.loadone("trials.environmentIndex")
        return np.unique(trial_envnum)

    def register_spkmap(self, ses: session.vrExperiment, fluor_type: str, envnum: int):
        loaded_spkmap_id = f"{ses}_{envnum}_{fluor_type}"
        if loaded_spkmap_id not in self.loaded_spkmaps:
            c_keep_idx = ses.idxToPlanes(keep_planes=[1, 2, 3, 4])
            c_loaded_spkmap = self.load_spkmap(ses, fluor_type, envnum)
            c_loaded_spkmap = c_loaded_spkmap[c_keep_idx]
            self.loaded_spkmaps[loaded_spkmap_id] = c_loaded_spkmap
        return self.loaded_spkmaps[loaded_spkmap_id]

    def load_spkmap(self, ses: session.vrExperiment, fluor_type: str, envnum: int):
        return ses.load_spkmaps(self.fluorescence_data_name(fluor_type), envnum=envnum)[0]

    def register_fluorescence_data(self, ses: session.vrExperiment, fluor_type: str):
        loaded_fluor_data_id = f"{ses}_{fluor_type}"
        if loaded_fluor_data_id not in self.loaded_fluor_data:
            c_keep_idx = ses.idxToPlanes(keep_planes=[1, 2, 3, 4])
            c_loaded_data = self.load_fluorescence_data(ses, fluor_type)
            c_loaded_data = c_loaded_data[:, c_keep_idx]
            c_loaded_data_max = np.nanmax(c_loaded_data, axis=0, keepdims=True)
            c_loaded_data_max[c_loaded_data_max == 0] = 1
            c_loaded_data = c_loaded_data / c_loaded_data_max
            self.loaded_fluor_data[loaded_fluor_data_id] = c_loaded_data
        return self.loaded_fluor_data[loaded_fluor_data_id]

    def load_fluorescence_data(self, ses: session.vrExperiment, fluor_type: str):
        if fluor_type == "fcorr":
            return ses.loadfcorr().T
        elif fluor_type == "deconvolved":
            return ses.loadone("mpci.roiActivityDeconvolved")
        elif fluor_type == "oasis":
            return ses.loadone("mpci.roiActivityDeconvolvedOasis")
        elif fluor_type == "significant":
            return ses.loadone("mpci.roiSignificantFluorescence", sparse=True).toarray()
        else:
            raise ValueError(f"Didn't recognized fluor_type! Recieved: {fluor_type}")

    def fluorescence_data_name(self, fluor_type: str):
        if fluor_type == "fcorr":
            return "fcorr"
        elif fluor_type == "deconvolved":
            return "mpci.roiActivityDeconvolved"
        elif fluor_type == "oasis":
            return "mpci.roiActivityDeconvolvedOasis"
        elif fluor_type == "significant":
            return "mpci.roiSignificantFluorescence"
        else:
            raise ValueError(f"Didn't recognized fluor_type! Recieved: {fluor_type}")

    def get_plotting_data(self, state: Dict[str, Any]):
        mouse = state["mouse"]
        if mouse != self.current_mouse:
            self.loaded_spkmaps = {}
            self.loaded_fluor_data = {}
            self.current_mouse = mouse

        ses = self.mouse_sessions[mouse][state["session"]]
        roi = state["roi"]
        fluor_data = {ft: data[:, roi] for ft, data in self.get_fluorescence_data(ses, state).items()}
        spkmap = self.get_spkmap(ses, state)[roi]
        position_data = self.get_position_data(ses, state)
        return fluor_data, spkmap, position_data

    def get_spkmap(self, ses: session.vrExperiment, state: dict):
        fluor_type = state["spkmap_fluor_type"]
        envnum = state["envnum"]
        return self.register_spkmap(ses, fluor_type, envnum)

    def get_fluorescence_data(self, ses: session.vrExperiment, state: Dict[str, Any]):
        fluor_types = state["fluor_type"]
        fluor_data = {ft: self.register_fluorescence_data(ses, ft) for ft in fluor_types}
        return fluor_data

    def get_position_data(self, ses: session.vrExperiment, state: dict):
        frame_pos_index_by_env, _, environments = ses.get_position_by_env(speedThreshold=1, use_average=True)
        idx_to_env = environments == state["envnum"]
        frame_pos = frame_pos_index_by_env[idx_to_env]
        frame_pos = frame_pos.astype(float)
        frame_pos[frame_pos == -100] = np.nan
        return frame_pos

    def identify_reliable_rois(self, state: Dict[str, Any]):
        mouse = state["mouse"]
        ses = self.mouse_sessions[mouse][state["session"]]
        envnum = state["envnum"]
        fluor_type = state["spkmap_fluor_type"]
        reliable_id = f"{mouse}_{state['session']}_{envnum}_{fluor_type}"
        if reliable_id not in self.loaded_reliable_rois:
            pcss = placeCellSingleSession(ses, autoload=False, keep_planes=[1, 2, 3, 4])
            # Take relloo (the third output) and take the first value (the environment -- it's a list of a single environment)
            use_spkmap = self.get_spkmap(ses, state)
            num_trials = pcss.occmap.shape[0]
            rawspkmap = np.zeros((num_trials, pcss.occmap.shape[1], use_spkmap.shape[0]))
            env_trials = pcss.idxFullTrialEachEnv[pcss.envnum_to_idx(envnum)[0]]
            rawspkmap[env_trials] = np.transpose(use_spkmap, (1, 2, 0))
            relloo = pcss.get_reliability_values(envnum=envnum, rawspkmap=rawspkmap)[2][0]
            self.loaded_reliable_rois[reliable_id] = relloo
        return self.loaded_reliable_rois[reliable_id]


class FluorTypeSpkmapsViewer(Viewer):
    def __init__(self, fluor_type_loader: FluorTypeLoader):
        self.fluor_type_loader = fluor_type_loader
        self.add_selection("mouse", value=self.fluor_type_loader.mice_list[0], options=self.fluor_type_loader.mice_list)
        self.add_integer("session", value=12, min_value=0, max_value=20)
        self.add_selection("roi", value=16, options=[16])
        self.add_selection("envnum", value=3, options=[3])
        self.add_float("vmax", value=1.0, min_value=0.5, max_value=20.0, step=0.5)
        self.add_multiple_selection("fluor_type", value=self.fluor_type_loader.fluorescence_types, options=self.fluor_type_loader.fluorescence_types)
        self.add_selection("spkmap_fluor_type", value=self.fluor_type_loader.fluorescence_types[0], options=self.fluor_type_loader.fluorescence_types)
        self.add_integer_range("reliable_percentile", value=(90, 100), min_value=0, max_value=100)

        self.update_viewer(self.get_state())
        self.on_change(["mouse", "session", "roi", "envnum", "reliable_percentile"], self.update_viewer)

    def update_viewer(self, state: Dict[str, Any]):
        mouse = state["mouse"]
        reliable_percentile = state["reliable_percentile"]
        num_sessions = len(self.fluor_type_loader.mouse_sessions[mouse])
        envnums = list(self.fluor_type_loader.session_environments[mouse][state["session"]])
        self.update_integer("session", max_value=num_sessions - 1)
        self.update_selection("envnum", options=envnums)

        reliability = self.fluor_type_loader.identify_reliable_rois(self.get_state())
        lower_threshold = np.percentile(reliability, reliable_percentile[0])
        upper_threshold = np.percentile(reliability, reliable_percentile[1])
        reliable_rois = list(np.where((reliability >= lower_threshold) & (reliability <= upper_threshold))[0])
        self.update_selection("roi", options=reliable_rois)

    def plot(self, state: Dict[str, Any]):
        roi = state["roi"]
        title = f"Mouse: {state['mouse']}, Session: {state['session']}, ROI: {roi}"

        fluor_data, spkmap, position_data = self.fluor_type_loader.get_plotting_data(state)
        edges = np.arange(0, spkmap.shape[1] + 1)

        plt.close("all")

        fig = plt.figure(figsize=(8, 4), layout="constrained")
        gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1, 0.5, 1])

        ax_lineplot = fig.add_subplot(gs[0, 0])
        for ft, data in fluor_data.items():
            ax_lineplot.plot(data, color=self.fluor_type_loader.fluor_colors[ft], label=ft)
        ax_lineplot.set_title(title)
        ax_lineplot.set_ylabel("Activity")
        ax_lineplot.legend(loc="best", fontsize=8)

        ax_position = fig.add_subplot(gs[1, 0])
        ax_position.sharex(ax_lineplot)
        ax_position.plot(position_data[0], color="black")
        ax_position.set_ylabel("Pos")

        ax_imdata = fig.add_subplot(gs[2, 0])
        ax_imdata.sharex(ax_lineplot)
        imdata = np.stack([d for d in fluor_data.values()], axis=0)
        ax_imdata.imshow(
            imdata,
            cmap="gray_r",
            interpolation="none",
            aspect="auto",
            vmin=0,
            vmax=1,
            extent=[0, imdata.shape[1], imdata.shape[0], 0],
        )
        ax_imdata.set_xlabel("Frame")
        ax_imdata.set_yticks(0.5 + np.arange(imdata.shape[0]))
        ax_imdata.set_yticklabels(state["fluor_type"])

        cmap = mpl.colormaps["gray_r"]
        cmap.set_bad(("r", 0.2))
        ax_spkmaps = fig.add_subplot(gs[:, 1])
        ax_spkmaps.imshow(
            spkmap,
            interpolation="none",
            aspect="auto",
            vmin=0,
            vmax=state["vmax"],
            extent=[edges[0], edges[-1], spkmap.shape[0], 0],
            cmap=cmap,
        )
        ax_spkmaps.set_xlabel("VR Pos")
        ax_spkmaps.set_ylabel("Trial")
        ax_spkmaps.set_title(f"Env: {state['envnum']}")

        return fig
