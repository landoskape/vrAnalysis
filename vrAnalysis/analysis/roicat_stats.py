import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

from .placeCellMultiSession import placeCellMultiSession, handle_idx_ses
from .. import helpers

class RoicatStats(placeCellMultiSession):
    def __init__(self, *args, num_bins=31, sim_cutoffs=[0.3, 0.7], **kwargs):
        super().__init__(*args, **kwargs)
        self.set_sim_hist_edges(num_bins=num_bins)
        self.set_sim_cutoffs(cutoffs=sim_cutoffs)
        self.set_corr_hist_edges(num_bins=num_bins)

        self.name = 'ROICaT_Stats'

    def set_sim_hist_edges(self, num_bins=31):
        """set bin edges for histogram of similarity data"""
        self.sim_hist_edges = np.linspace(0, 1, num_bins)
    
    def set_sim_cutoffs(self, cutoffs=[0.3, 0.7]):
        """set cutoffs for similarity of roicat_diff and roicat_same"""
        self.sim_cutoffs = cutoffs

    def set_corr_hist_edges(self, num_bins=31):
        """set bin edges for histogram of correlation data"""
        self.corr_hist_edges = np.linspace(-1, 1, num_bins)
    
    def session_pair_names(self, prms):
        """helper for converting session pairs in prms to list of strings"""
        return [','.join([str(p) for p in pair]) for pair in prms['idx_ses_pairs']]

    def similarity_stats(self, sim, corr):
        """
        analysis method for getting statistics about the similarity data in **sim** and **corr**
        """
        sim_counts = [np.histogram(s, bins=self.sim_hist_edges) for s in sim]
        idx_roicat_diff = [s < self.sim_cutoffs[0] for s in sim]
        idx_roicat_same = [s > self.sim_cutoffs[1] for s in sim]
        pass

    @handle_idx_ses
    def make_roicat_comparison(self, envnum, sim_name='sConj', tracked=False, idx_ses=None, cutoffs=(0.4, 0.7), both_reliable=False):        
        """
        load ROICaT comparison data
        
        returns sim, corr, and prms, where sim and corr are lists of values associated with each pair of ROIs in a pair of sessions
        sim contains the ROICaT similarity value named by **sim_name**
        corr contains the place field correlation value

        prms is a dictionary describing the data parameters (which sessions, which session pairs are used, which environment)

        uses data from sessions with a specific environment and ROIs according to the criteria in the kwargs
        """
        # get all pairs of sessions for idx_ses
        idx_ses_pairs = helpers.all_pairs(idx_ses)

        # get all spkmaps from requested sessions
        spkmaps, relmse, relcor, pfloc, _, _ = self.get_spkmaps(envnum, trials='full', average=True, tracked=tracked, idx_ses=idx_ses, by_plane=True)

        # define reliability metric
        idx_reliable = [[(mse>cutoffs[0]) & (cor>cutoffs[1]) for mse, cor in zip(rmse, rcor)] for rmse, rcor in zip(relmse, relcor)]

        # for each source/target pair in idx_ses, do: 
        sim, corr = [], []       
        for source, target in idx_ses_pairs:
            isource, itarget = helpers.index_in_target(source, idx_ses)[1][0], helpers.index_in_target(target, idx_ses)[1][0]

            # get similarity data from source/target
            sim_paired = self.track.get_similarity_paired(sim_name, source=source, target=target, symmetric=True, tracked=tracked, cat_planes=False, keep_planes=self.keep_planes)
            
            # compute correlation between source and target
            corrs = [helpers.crossCorrelation(spksource.T, spktarget.T) for spksource, spktarget in zip(spkmaps[isource], spkmaps[itarget])]
            
            # filter by reliability
            sim_paired = [sim[idx_source] for sim, idx_source in zip(sim_paired, idx_reliable[isource])]
            corrs = [cor[idx_source] for cor, idx_source in zip(corrs, idx_reliable[isource])]
            if both_reliable:
                sim_paired = [sim[:, idx_target] for sim, idx_target in zip(sim_paired, idx_reliable[itarget])]
                corrs = [cor[:, idx_target] for cor, idx_target in zip(corrs, idx_reliable[itarget])]

            # stack and flatten across planes
            sim.append(np.concatenate([s.toarray().flatten() for s in sim_paired]))
            corr.append(np.concatenate([c.flatten() for c in corrs]))

        # result parameters
        prms = dict(
            envnum=envnum,
            idx_ses=idx_ses,
            idx_ses_pairs=idx_ses_pairs,
        )

        return sim, corr, prms

    def split_by_roicat_assignment(self, sim, corr):
        """split corr values by roicat assignment (comparing sim to cutoffs)"""
        idx_roicat_diff = [s < self.sim_cutoffs[0] for s in sim]
        idx_roicat_same = [s > self.sim_cutoffs[1] for s in sim]
        corr_diff = [c[idx] for c, idx in zip(corr, idx_roicat_diff)]
        corr_same = [c[idx] for c, idx in zip(corr, idx_roicat_same)]
        return corr_same, corr_diff

    def plot_pfcorr_by_samediff(self, sim, corr, prms, with_show=True, with_save=False):
        """
        helper for plotting pfcorr values for same and different populations
        
        starts by making a dataframe object to plot distribution of place field correlation values
        for pairs of ROIs based on whether their roicat similarity value is in the "same" group or
        the "diff" group. 

        corr_same & corr_diff should be lists of np arrays of pf corr values for each session for each distribution
        prms should be a dictionary describing the data in sim and corrs
        """
        corr_same, corr_diff = self.split_by_roicat_assignment(sim, corr)
        
        dataframes = []
        for name, csame, cdiff in zip(self.session_pair_names(prms), corr_same, corr_diff):
            same_df = pd.DataFrame({'PF Correlation': csame, 'Session Pair': name, 'ROICaT Assignment': 'Same'})
            diff_df = pd.DataFrame({'PF Correlation': cdiff, 'Session Pair': name, 'ROICaT Assignment': 'Diff'})
            dataframes.extend([same_df, diff_df])
        
        data = pd.concat(dataframes, ignore_index=True)
        
        plt.close('all')
        fig, ax = plt.subplots()
        sns.boxenplot(data=data, x="Session Pair", y="PF Correlation", hue="ROICaT Assignment", ax=ax)
        plt.show()
        ax.legend(loc='lower right')

        if with_save: 
            sesidx = '_'.join([str(i) for i in prms['idx_ses']])
            save_name = f"pfcorr_by_samediff_envnum{prms['envnum']}_idxses{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if with_show else plt.close()

    def plot_similarity_histograms(self, sim, prms, with_show=True, with_save=False):
        """
        plot histogram of similarity values in each session
        """
        centers = helpers.edge2center(self.sim_hist_edges)
        sim_counts = [np.histogram(s[s!=0], bins=self.sim_hist_edges)[0] for s in sim]
        sim_zeros = [np.sum(s==0) for s in sim]
        
        cmap = mpl.colormaps['tab10']

        plt.close('all')
        fig, ax = plt.subplots()
        inset = fig.add_axes([0.2, 0.175, 0.1, 0.25])
        
        names = self.session_pair_names(prms)
        for idx, (name, count, zeros) in enumerate(zip(names, sim_counts, sim_zeros)):
            ax.plot(centers, count, c=cmap(idx), label=name)
            inset.plot([-0.2, 0.2], [zeros, zeros], c=cmap(idx), linewidth=2)

        ax.set_yscale('log')
        ax.legend(loc='upper right', title='Session Pair')
        ax.set_xlabel('ROICaT Similarity Value')
        ax.set_ylabel('Counts')

        inset.set_xlim(-0.5, 0.5)
        inset.set_xlabel('')
        inset.set_xticks([])
        inset.set_ylabel('sim=0', fontsize=10)

        if with_save: 
            sesidx = '_'.join([str(i) for i in prms['idx_ses']])
            save_name = f"similarity_histogram_idxses{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if with_show else plt.close()
