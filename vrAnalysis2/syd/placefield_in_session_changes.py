import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from syd import Viewer


class BTSPViewer(Viewer):
    def __init__(self, spkmap, com, reliability, fraction_active, positions):
        self.spkmap = spkmap
        self.com = com
        self.reliability = reliability
        self.fraction_active = fraction_active
        self.positions = positions
        bin_diff = positions[1] - positions[0]
        self.position_edges = np.append(positions[1:] - bin_diff / 2, positions[-1] + bin_diff / 2)
        self.active_trial_com = [com[~np.isnan(com)] for com in self.com]
        self.difference_com = [np.diff(com) for com in self.active_trial_com]
        self.difference_com_mean = np.array([np.mean(com) for com in self.difference_com])
        self.difference_com_std = np.array([np.std(com) for com in self.difference_com])

        self.reliability_default = (0.5, 1.0)
        self.fraction_active_default = (0.1, 1.0)

        self.add_float_range("reliability", value=self.reliability_default, min=-1.0, max=1.0, step=0.01)
        self.add_float_range("fraction_active", value=self.fraction_active_default, min=0.0, max=1.0, step=0.01)
        self.add_float_range("diffcomstd_percentile", value=(0.0, 100.0), min=0.0, max=100.0, step=0.1)

        self.add_integer("roi", value=0, min=0, max=1)
        self.add_float("vmax", value=5.0, min=0.0, max=20.0, step=0.1)

        self.on_change(["reliability", "fraction_active", "diffcomstd_percentile"], self.reset_selection)
        self.reset_selection(self.state)

    def get_r2(self, x, y, slope, intercept):
        # Calculate predicted values
        r2 = np.zeros(len(slope))
        for i, (ss, ii) in enumerate(zip(slope, intercept)):
            mask = ~y[i].mask
            x_valid = x[mask]
            y_valid = y[i][mask]
            y_pred = ss * x_valid + ii

            # Calculate R²
            ss_total = np.sum((y_valid - np.mean(y_valid)) ** 2)
            ss_residual = np.sum((y_valid - y_pred) ** 2)
            r2[i] = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        return r2

    def reset_selection(self, state):
        valid_reliability = (self.reliability >= state["reliability"][0]) & (self.reliability < state["reliability"][1])
        valid_fraction_active = (self.fraction_active >= state["fraction_active"][0]) & (self.fraction_active < state["fraction_active"][1])
        min_prctile = np.nanpercentile(self.difference_com_std, state["diffcomstd_percentile"][0])
        max_prctile = np.nanpercentile(self.difference_com_std, state["diffcomstd_percentile"][1])
        valid_diffcomstd = (self.difference_com_std >= min_prctile) & (self.difference_com_std <= max_prctile)
        self._valid_idx = np.where(valid_reliability & valid_fraction_active & valid_diffcomstd)[0]
        if len(self._valid_idx) == 0:
            self.update_float_range("reliability", value=self.reliability_default)
            self.update_float_range("fraction_active", value=self.fraction_active_default)
            self.reset_selection(self.state)

        c_coms = ma.masked_invalid(self.com[self._valid_idx])
        c_trials = np.arange(c_coms.shape[1])
        coeffs = np.stack([np.ma.polyfit(c_trials, com, deg=1, full=False) for com in c_coms])

        self.slopes = coeffs[:, 0]
        self.intercepts = coeffs[:, 1]
        self.r2 = self.get_r2(c_trials, c_coms, self.slopes, self.intercepts)

        num_valid = len(self._valid_idx)
        self.update_integer("roi", max=num_valid - 1)

    def plot(self, state):
        idx = self._valid_idx[state["roi"]]

        intercept = self.intercepts[state["roi"]]
        slope = self.slopes[state["roi"]]
        r2 = self.r2[state["roi"]]
        trials = np.arange(self.com.shape[1])
        com_fit = slope * trials + intercept

        extent = [self.position_edges[0], self.position_edges[-1], self.spkmap.shape[1], 0]
        fig, ax = plt.subplots(1, 4, figsize=(10, 3.5), layout="constrained")
        ax[0].imshow(self.spkmap[idx], aspect="auto", interpolation="none", cmap="gray_r", vmin=0, vmax=state["vmax"], origin="upper", extent=extent)
        ax[0].plot(com_fit, trials, c="r", linewidth=0.5, linestyle="--")
        ax[1].scatter(self.com[idx], trials, s=15, c="r")
        ax[1].plot(com_fit, trials, c="k", linewidth=0.5, linestyle="--")
        for axx in [ax[0], ax[1]]:
            axx.set_xlim(self.position_edges[0], self.position_edges[-1])
            axx.set_ylim(self.spkmap.shape[1], 0)

        not_nan_diff = ~np.isnan(self.difference_com_std)
        keep_diff = self.difference_com_std[not_nan_diff]
        counts, bins = np.histogram(keep_diff, bins=100)
        centers = (bins[:-1] + bins[1:]) / 2
        widths = bins[1] - bins[0]
        counts = counts / counts.sum()
        counts_selected = np.histogram(self.difference_com_std[self._valid_idx][not_nan_diff[self._valid_idx]], bins=bins)[0]
        counts_selected = counts_selected / counts_selected.sum()

        ax[2].bar(centers, counts, color="k", width=widths)
        ax[2].bar(centers, counts_selected, color=("b", 0.5), width=widths)
        ax[3].scatter(self.slopes, self.r2, c="b", alpha=0.5, s=15)
        ax[3].scatter(slope, r2, c="r", alpha=1.0, s=15)

        fig.suptitle(f"ROI: {idx}, Slope: {slope:.2g}, Intercept: {int(intercept)}, R2: {r2:.2g}")
        return fig
