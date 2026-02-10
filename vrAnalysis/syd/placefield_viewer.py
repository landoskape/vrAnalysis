from dataclasses import dataclass
from vrAnalysis.sessions.b2session import B2Session
from vrAnalysis.processors.placefields import get_frame_behavior, get_placefield, Placefield
from vrAnalysis.helpers import edge2center, reliability_loo, get_placefield_location
from vrAnalysis.metrics import FractionActive
from syd.viewer import Viewer
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlacefieldType:
    normalize: bool = True
    use_fast_sampling: bool = True
    by_sample_duration: bool = False
    average: bool = True
    use_smoothing: bool = True
    smooth_width: float = 0.1

    def get_default_state(self, env_idx: int, neuron_idx: int = 0) -> dict:
        return dict(
            normalize=self.normalize,
            use_fast_sampling=self.use_fast_sampling,
            by_sample_duration=self.by_sample_duration,
            average=self.average,
            use_smoothing=self.use_smoothing,
            smooth_width=self.smooth_width,
            env_idx=env_idx,
            neuron_idx=neuron_idx,
        )

    def get_cache_name(self, state: dict) -> str:
        return "_".join(
            [
                "norm" if state["normalize"] else "nonorm",
                "fast_samples" if state["use_fast_sampling"] else "frame_samples",
                "by_sample_duration" if state["by_sample_duration"] else "",
                "preaverage" if state["average"] else "postaverage",
                f"smooth{state['smooth_width']:.1f}" if state["use_smoothing"] else "nosmooth",
                f"env_{state['env_idx']}",
            ]
        )


class PlacefieldViewer(Viewer):
    def __init__(self, session: B2Session, num_bins: int = 100):
        """Initialize the placefield viewer with data.

        Parameters
        ----------
        session : B2Session
            Session object to get all data
        num_bins : int
            Number of bins for the placefield
        """
        self.session = session
        self.spks = session.spks[:, session.idx_rois]
        self.frame_behavior, idx_valid = get_frame_behavior(session).filter_valid_frames()
        self.spks = self.spks[idx_valid]
        self.dist_edges = np.linspace(0, session.env_length[0], num_bins + 1)
        self.num_bins = num_bins
        self.placefield_default = PlacefieldType()
        self.placefield_cache = dict()

        self.num_environments = len(self.session.environments)

        self.vmax_no_norm = np.nanpercentile(self.spks, 80)

        # Add parameters
        self.add_boolean("normalize", value=self.placefield_default.normalize)
        self.add_boolean("use_fast_sampling", value=self.placefield_default.use_fast_sampling)
        self.add_boolean("by_sample_duration", value=self.placefield_default.by_sample_duration)
        self.add_boolean("average", value=self.placefield_default.average)
        self.add_boolean("use_smoothing", value=self.placefield_default.use_smoothing)
        self.add_float("smooth_width", min=0.0, max=8.0, value=self.placefield_default.smooth_width, step=0.1)
        self.add_integer("env_idx", min=0, max=self.num_environments - 1, value=0)
        self.add_float("vmax", min=0.0, max=10.0, value=1.0, step=0.1)
        self.add_integer("neuron_idx", min=0, max=self.spks.shape[1] - 1, value=0)
        self.on_change("use_smoothing", self._toggle_smoothing)
        self.on_change("normalize", self._update_vmax)
        self._toggle_smoothing(self.state)
        self._update_vmax(self.state)

        # Sort by neurons by reliability and fraction_active, also get maximum value for each neuron
        self._idx_sort_reliable = []
        self._idx_max = []
        for env_idx in range(self.num_environments):
            _init_state = self.placefield_default.get_default_state(env_idx)
            _init_state["normalize"] = False
            _all_trials = np.transpose(self._get_all_trials(_init_state, filter=True)[0], (0, 2, 1))
            _idx_reliable = reliability_loo(_all_trials)
            _fraction_active = FractionActive.compute(
                _all_trials,
                activity_axis=2,
                fraction_axis=1,
                activity_method="rms",
                fraction_method="participation",
            )
            _integrated_reliability = _idx_reliable * _fraction_active
            self._idx_sort_reliable.append(np.argsort(_integrated_reliability)[::-1])

            _idx_max = np.nanmax(np.nanmean(_all_trials, axis=2), axis=1)
            _idx_max[_idx_max == 0] = 1.0
            _idx_max = np.nan_to_num(_idx_max, 1.0)
            self._idx_max.append(_idx_max)

    def _get_all_trials(self, state: dict, filter: bool = False):
        current_state = state.copy()
        current_state["average"] = False
        return self._get_placefield(current_state, average=False, filter=filter)

    def _toggle_smoothing(self, state):
        if state["use_smoothing"]:
            self.update_float("smooth_width", max=8.0, value=0.1)
        else:
            self.update_float("smooth_width", max=0.0, value=0.0)

    def _update_vmax(self, state):
        if state["normalize"]:
            self.update_float("vmax", max=1.0)
        else:
            self.update_float("vmax", max=self.vmax_no_norm)

    def _get_placefield(self, state, average: bool = True, filter: bool = False) -> Placefield:
        """Get placefield, using cache if available."""

        cache_name = self.placefield_default.get_cache_name(state)
        if cache_name in self.placefield_cache:
            placefield = self.placefield_cache[cache_name]

        else:
            # Compute placefield with current parameters
            kwargs = {
                "smooth_width": None if state["smooth_width"] == 0.0 else state["smooth_width"],
                "use_fast_sampling": state["use_fast_sampling"],
                "by_sample_duration": state["by_sample_duration"],
                "average": state["average"],
                "session": self.session,
                "zero_to_nan": True,
            }
            placefield = get_placefield(self.spks, self.frame_behavior, self.dist_edges, **kwargs)

            self.placefield_cache[cache_name] = placefield

        if filter:
            placefield = placefield.filter_by_coverage(start_bins=3, end_bins=3, filter_positions=True)

        if state["average"]:
            pf_data = placefield[state["env_idx"]].T
        else:
            env = self.session.environments[state["env_idx"]]
            pf_data = np.transpose(placefield.filter_by_environment(env).placefield, (2, 1, 0))

            if average:
                pf_data = np.nanmean(pf_data, axis=2)

        if state["normalize"]:
            _idx_max = self._idx_max[state["env_idx"]]
            if average:
                _idx_max = _idx_max[:, None]
            else:
                _idx_max = _idx_max[:, None, None]
            pf_data = pf_data / _idx_max

        return pf_data, placefield.idx_positions

    def plot(self, state):
        """Plot the placefield."""
        # Get placefield from cache
        pf_data = self._get_placefield(state)[0]
        default_state = self.placefield_default.get_default_state(state["env_idx"])
        _idx_sort = get_placefield_location(self._get_placefield(default_state)[0])[1]
        _idx_sort_reliable = self._idx_sort_reliable[state["env_idx"]]
        _all_trials, _idx_positions = self._get_all_trials(state)
        single_neuron = _all_trials[_idx_sort_reliable[state["neuron_idx"]]]
        single_from_average = pf_data[_idx_sort_reliable][state["neuron_idx"]]

        # Create figure
        fig, ax = plt.subplots(2, 2, figsize=(8, 6), height_ratios=[1, 0.1], layout="constrained", sharex=True)
        extent = [self.dist_edges[0], self.dist_edges[-1], 0, pf_data.shape[0]]
        ax[0, 0].imshow(pf_data[_idx_sort], aspect="auto", extent=extent, cmap="gray_r", vmin=0, vmax=state["vmax"])
        ax[0, 0].set_ylabel("ROI Index")
        ax[1, 0].plot(edge2center(self.dist_edges), np.nanmean(pf_data, axis=0), color="black")
        ax[1, 0].set_xlabel("Position (cm)")
        ax[1, 0].set_ylabel("Pop. Average")

        vmax = np.nanmax(single_neuron)
        centers = edge2center(self.dist_edges)[_idx_positions]
        extent = [self.dist_edges[0], self.dist_edges[-1], 0, single_neuron.shape[1]]
        ax[0, 1].imshow(single_neuron.T, aspect="auto", extent=extent, cmap="gray_r", vmin=0, vmax=vmax)
        ax[1, 1].plot(centers, np.nanmean(single_neuron, axis=1), color="black")
        ax[1, 1].plot(centers, single_from_average[_idx_positions], color="blue")
        ax[1, 1].set_xlabel("Position (cm)")
        ax[1, 1].set_ylabel("Trial Average")

        return fig
