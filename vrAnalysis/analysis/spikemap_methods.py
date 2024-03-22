from tqdm import tqdm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from .. import helpers
from ..analysis import placeCellSingleSession


class SpikemapMethods(placeCellSingleSession):
    """
    Some analysis methods for comparing and analyzing how to make spike maps in different ways
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SpikemapMethods"

    def setting_as_string(self, setting):
        """convert setting to string representation"""
        return f"Resoution:{setting['distStep']}, FiltWidth:{setting['smooth']}"

    def compare_methods(self, settings, verbose=True):
        """
        compare spkmaps and reliability made with different spkmap settings.

        settings should be a list of dictionaries where each contains the keys "distStep" and "smooth"
        which refer to the spatial resolution and spatial smoothing width, respectively.
        """
        spkmaps = []
        rawspkmaps = []
        relmse, relcor = [], []
        distcenters = []

        iterate = tqdm(settings) if verbose else settings
        for setting in iterate:
            if verbose:
                iterate.set_description(self.setting_as_string(setting))
            self.load_data(distStep=setting["distStep"], full_trial_flexibility=3)
            spkmaps.append([np.squeeze(sm) for sm in self.get_spkmap(smooth=setting["smooth"])])
            rawspkmaps.append([np.mean(rsm, axis=1) for rsm in self.get_spkmap(average=False, smooth=setting["smooth"])])
            relmse.append(self.relmse)
            relcor.append(self.relcor)
            distcenters.append(self.distcenters)

        return spkmaps, rawspkmaps, relmse, relcor, distcenters, settings

    def plot_examples(
        self,
        spkmaps,
        rawspkmaps,
        relmse,
        relcor,
        distcenters,
        settings,
        envidx=-1,
        num_to_plot=1,
        same=True,
        ses4rel=None,
        cutoffs=(0.4, 0.7),
        maxcutoffs=None,
        withSave=False,
        withShow=True,
    ):
        """
        Method for plotting example ROIs (either the same or different) with different methods of
        making spkmaps, along with showing the differences between averaging before or after dividing
        """
        idxrel = self.get_reliable(relmse, relcor, envidx, cutoffs=cutoffs, maxcutoffs=maxcutoffs)
        if same:
            # Use same ROI from each spkmap method
            if ses4rel is not None:
                # Use reliability criterion from requested sessions
                icell = np.random.choice(np.where(idxrel[ses4rel])[0], num_to_plot)
            else:
                idxrel_all = np.all(np.stack(idxrel), axis=0)
                icell = np.random.choice(np.where(idxrel_all)[0], num_to_plot)
        else:
            icell = [np.random.choice(np.where(irel)[0], num_to_plot) for irel in idxrel]

        for ii in range(num_to_plot):
            num_settings = len(spkmaps)
            figsize = 3
            fig, ax = plt.subplots(1, num_settings, figsize=(num_settings * figsize, figsize), layout="constrained")
            for idx, setting in enumerate(settings):
                if not same:
                    ic = icell[idx][ii]
                else:
                    ic = icell[ii]
                ax[idx].plot(distcenters[idx], spkmaps[idx][envidx][ic], label="preaveraging")
                ax[idx].plot(distcenters[idx], rawspkmaps[idx][envidx][ic], label="trialbytrial")
                ax[idx].set_title(self.setting_as_string(setting))
                ax[idx].legend(loc="best")

            if withSave:
                self.saveFigure(fig.number, f"example_spkmaps_from_spkmap_method_comparison", extra_name=f"ROI{ic}")

            # Show figure if requested
            plt.show() if withShow else plt.close()

    def plot_reliability_distribution(self, relmse, relcor, settings, withSave=False, withShow=True):
        """
        Method for plotting example ROIs (either the same or different) with different methods of
        making spkmaps, along with showing the differences between averaging before or after dividing
        """
        figsize = 3
        fig, ax = plt.subplots(1, 3, figsize=(figsize * 3, figsize), layout="constrained")

        bins = [np.linspace(-4, 1, 51), np.linspace(-1, 1, 51)]  # predefined bins for relmse and relcor
        centers = [helpers.edge2center(b) for b in bins]
        cmap = mpl.colormaps["Dark2"].resampled(len(settings))

        for idx, setting in enumerate(settings):
            mse_counts = np.histogram(relmse[idx], bins=bins[0])[0]
            cor_counts = np.histogram(relcor[idx], bins=bins[1])[0]
            ax[0].plot(centers[0], mse_counts, color=cmap(idx))
            ax[1].plot(centers[1], cor_counts, color=cmap(idx))
            ax[2].plot(centers[0], centers[0], color=cmap(idx), label=self.setting_as_string(setting))
        ax[2].legend(loc="best")
        ax[0].set_title("MSE Reliability Histogram")
        ax[1].set_title("CORR Reliability Histogram")

        if withSave:
            self.saveFigure(fig.number, f"reliability_distribution_from_spkmap_method_comparison")

        # Show figure if requested
        plt.show() if withShow else plt.close()

    def get_reliable(self, relmse, relcor, envidx, cutoffs=None, maxcutoffs=None):
        """
        simple method to get idx of reliable cells from each spkmethod and requested environment
        using cutoffs and maxcutoffs
        """
        cutoffs = (-np.inf, -np.inf) if cutoffs is None else cutoffs
        maxcutoffs = (np.inf, np.inf) if maxcutoffs is None else maxcutoffs
        idx_reliable = [
            (mse[envidx] >= cutoffs[0]) & (cor[envidx] >= cutoffs[1]) & (mse[envidx] <= maxcutoffs[0]) & (cor[envidx] <= maxcutoffs[1])
            for mse, cor in zip(relmse, relcor)
        ]
        return idx_reliable
