import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
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

        self.name = "ROICaT_Stats"

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
        return [",".join([str(p) for p in pair]) for pair in prms["idx_ses_pairs"]]

    def similarity_stats(self, sim, corr):
        """
        analysis method for getting statistics about the similarity data in **sim** and **corr**
        """
        sim_counts = [np.histogram(s, bins=self.sim_hist_edges) for s in sim]
        idx_roicat_diff = [s < self.sim_cutoffs[0] for s in sim]
        idx_roicat_same = [s > self.sim_cutoffs[1] for s in sim]
        # evidently this was an unnecessary work in progress
        return None

    @handle_idx_ses
    def make_roicat_comparison(
        self, envnum, sim_name="sConj", idx_ses=None, cutoffs=(0.4, 0.7), both_reliable=False
    ):
        """
        load ROICaT comparison data

        returns sim, corr, tracked, pwdist, nnpair, and prms, where:
        sim, corr, and tracked are lists of values associated with each pair of ROIs in a pair of sessions
        sim contains the ROICaT similarity value named by **sim_name**
        corr contains the place field correlation value
        tracked contains a 1 if the pair is tracked by ROICaT and a zero otherwise
        pwdist contains the pairwise distance of ROI centroid after alignment by ROICaT
        nnpair contains a 1 for whatever pair is closest by euclidean distance after alignment

        prms is a dictionary describing the data parameters (which sessions, which session pairs are used, which environment)

        uses data from sessions with a specific environment and ROIs according to the criteria in the kwargs
        """
        # get all pairs of sessions for idx_ses
        idx_ses_pairs = helpers.all_pairs(idx_ses)

        # get all spkmaps from requested sessions
        spkmaps, relmse, relcor, pfloc, _, _ = self.get_spkmaps(
            envnum, trials="full", average=True, tracked=False, idx_ses=idx_ses, by_plane=True
        )

        # define reliability metric
        idx_reliable = [
            [(mse > cutoffs[0]) & (cor > cutoffs[1]) for mse, cor in zip(rmse, rcor)]
            for rmse, rcor in zip(relmse, relcor)
        ]

        # get all labels from requested sessions
        tracked_labels = self.track.get_tracked_labels(
            cat_planes=False, nan_untracked=True, idx_ses=idx_ses, keep_planes=self.keep_planes
        )

        # get xy coordinates of aligned ROIs from requested sessions
        yxcentroids = self.track.get_centroids(
            combine=True, idx_ses=idx_ses, keep_planes=self.keep_planes
        )

        # for each source/target pair in idx_ses, do:
        sim, corr, tracked, pwdist, nnpair = [], [], [], [], []

        for source, target in idx_ses_pairs:
            isource, itarget = (
                helpers.index_in_target(source, idx_ses)[1][0],
                helpers.index_in_target(target, idx_ses)[1][0],
            )

            # get similarity data from source/target
            sim_paired = self.track.get_similarity_paired(
                sim_name,
                source=source,
                target=target,
                symmetric=True,
                tracked=False,
                cat_planes=False,
                keep_planes=self.keep_planes,
            )

            # compute correlation between source and target
            corrs = [
                helpers.crossCorrelation(spksource.T, spktarget.T)
                for spksource, spktarget in zip(spkmaps[isource], spkmaps[itarget])
            ]

            # retrieve source/target labels for each ROI by plane
            # 1 if pair is tracked (by label) and 0 if pair isn't tracked
            tracked_pair = [
                label_source.reshape(-1, 1) == label_target.reshape(1, -1)
                for label_source, label_target in zip(
                    tracked_labels[isource], tracked_labels[itarget]
                )
            ]

            # get pairwise distance
            pwdists = [
                cdist(yx_source, yx_target)
                for yx_source, yx_target in zip(yxcentroids[isource], yxcentroids[itarget])
            ]

            # get index of nearest neighbor by pair-wise distance post alignment
            nn_dist = [np.nanmin(pwd, axis=1) for pwd in pwdists]
            nn_pairs = [pwd == nn.reshape(-1, 1) for pwd, nn in zip(pwdists, nn_dist)]

            # measure pairwise distance between pairs
            # filter by reliability
            sim_paired = [
                sim[idx_source] for sim, idx_source in zip(sim_paired, idx_reliable[isource])
            ]
            corrs = [cor[idx_source] for cor, idx_source in zip(corrs, idx_reliable[isource])]
            tracked_pair = [
                pair[idx_source] for pair, idx_source in zip(tracked_pair, idx_reliable[isource])
            ]
            pwdists = [pwd[idx_source] for pwd, idx_source in zip(pwdists, idx_reliable[isource])]
            nn_pairs = [
                nnp[idx_source] for nnp, idx_source in zip(nn_pairs, idx_reliable[isource])
            ]
            if both_reliable:
                sim_paired = [
                    sim[:, idx_target]
                    for sim, idx_target in zip(sim_paired, idx_reliable[itarget])
                ]
                corrs = [
                    cor[:, idx_target] for cor, idx_target in zip(corrs, idx_reliable[itarget])
                ]
                tracked_pair = [
                    pair[:, idx_target]
                    for pair, idx_target in zip(tracked_pair, idx_reliable[itarget])
                ]
                pwdists = [
                    pwd[:, idx_target] for pwd, idx_target in zip(pwdists, idx_reliable[itarget])
                ]
                nn_pairs = [
                    nnp[:, idx_target] for nnp, idx_target in zip(nn_pairs, idx_reliable[itarget])
                ]

            # stack and flatten across planes
            sim.append(np.concatenate([s.toarray().flatten() for s in sim_paired]))
            corr.append(np.concatenate([c.flatten() for c in corrs]))
            tracked.append(np.concatenate([p.flatten() for p in tracked_pair]))
            pwdist.append(np.concatenate([d.flatten() for d in pwdists]))
            nnpair.append(np.concatenate([n.flatten() for n in nn_pairs]))

        # result parameters
        prms = dict(
            envnum=envnum,
            idx_ses=idx_ses,
            idx_ses_pairs=idx_ses_pairs,
            sim_name=sim_name,
            cutoffs=cutoffs,
            both_reliable=both_reliable,
        )

        return sim, corr, tracked, pwdist, nnpair, prms

    def split_by_roicat_similarity(self, data, sim):
        """
        split values by roicat similarity

        compares sim to cutoffs set as default attribute of self
        and splits data based on these cutoffs
        """
        assert all(
            [d.shape == s.shape for d, s in zip(data, sim)]
        ), "data and sim shapes must be equal"

        idx_roicat_same = [s > self.sim_cutoffs[1] for s in sim]
        idx_roicat_diff = [s < self.sim_cutoffs[0] for s in sim]

        data_same = [c[idx] for c, idx in zip(data, idx_roicat_same)]
        data_diff = [c[idx] for c, idx in zip(data, idx_roicat_diff)]
        return data_same, data_diff

    def split_by_roicat_assignment(self, data, tracked):
        """
        split values by roicat assignment in tracked

        or use tracked pair identities in tracked
        """
        assert all(
            [d.shape == t.shape for d, t in zip(data, tracked)]
        ), "data and tracked shapes must be equal"

        idx_roicat_same = [t == True for t in tracked]
        idx_roicat_diff = [t == False for t in tracked]

        data_same = [c[idx] for c, idx in zip(data, idx_roicat_same)]
        data_diff = [c[idx] for c, idx in zip(data, idx_roicat_diff)]
        return data_same, data_diff

    def plot_sim_vs_pfcorr(
        self, sim, corr, tracked, prms, color_mode=None, with_show=True, with_save=False
    ):
        """
        helper for making scatter plots between sim and corr

        inputs are sim, corr, tracked, and prms which come from make_roicat_comparison()

        if color_mode='density', will color plots by their local density with gaussian_kde
        (keep in mind that this is slow)
        if color_mode='tracked', will color points by whether they are tracked
        if color_mode=None, will color the points black
        """
        marginal_corr_edges = np.linspace(-1, 1, 51)
        marginal_corr_centers = helpers.edge2center(marginal_corr_edges)
        for s, c, t, name in zip(sim, corr, tracked, self.session_pair_names(prms)):
            idx_zero = s == 0
            s_nozero = s[~idx_zero]
            c_nozero = c[~idx_zero]

            c_zero = c[idx_zero]
            c_zero_counts = helpers.fractional_histogram(c_zero, bins=marginal_corr_edges)[0]

            if color_mode == "density":
                xy = np.stack((s_nozero, c_nozero))
                color = gaussian_kde(xy)(xy)  # color by local density
            elif color_mode == "tracked":
                t_nozero = t[~idx_zero]
                color = [("k", 0.1), ("r", 0.1)]
            else:
                color = ("k", 0.1)  # use black and alpha to see the density a bit

            fig, ax = plt.subplots()
            if color_mode == "tracked":
                ax.scatter(
                    s_nozero[t_nozero == 0],
                    c_nozero[t_nozero == 0],
                    c=color[0],
                    s=5,
                    label="non-tracked",
                )
                ax.scatter(
                    s_nozero[t_nozero == 1],
                    c_nozero[t_nozero == 1],
                    c=color[1],
                    s=5,
                    label="tracked",
                )
            else:
                ax.scatter(s_nozero, c_nozero, c=color, s=5, label="ROI pairs")

            ax.plot(
                c_zero_counts,
                marginal_corr_centers,
                c="r",
                linewidth=1.5,
                label=f"{prms['sim_name']}=0",
            )

            ax.set_xlabel(f"ROICaT Similarity {prms['sim_name']}")
            ax.set_ylabel("Place Field Correlation")
            ax.set_title(f"Session Pair: {name}")
            ax.legend(loc="lower right")

            if with_save:
                save_name = f"simcorr_comparison_envnum{prms['envnum']}_idxses{name}"
                self.saveFigure(fig.number, self.track.mouse_name, save_name)

        # Show figure if requested
        plt.show() if with_show else plt.close()

    def plot_pwdist_vs_pfcorr(
        self,
        pwdist,
        corr,
        tracked,
        prms,
        max_dist=100,
        color_mode=None,
        with_show=True,
        with_save=False,
    ):
        """
        helper for making scatter plots between pwdist and corr

        inputs are pwdist, corr, tracked, and prms which come from make_roicat_comparison()

        max_dist determines the maximum distance of pwdist to filter for plotting

        if color_mode='density', will color plots by their local density with gaussian_kde (this takes a long while)
        if color_mode='tracked', will color points by whether they are tracked
        if color_mode=None, will color the points black
        """
        for d, c, t, name in zip(pwdist, corr, tracked, self.session_pair_names(prms)):
            idx_within_range = d < max_dist
            d_in_range = d[idx_within_range]
            c_in_range = c[idx_within_range]

            if color_mode == "density":
                xy = np.stack((d_in_range, c_in_range))
                color = gaussian_kde(xy)(xy)  # color by local density
            elif color_mode == "tracked":
                t_in_range = t[idx_within_range]
                color = [("k", 0.1), ("b", 0.1)]
            else:
                color = ("k", 0.1)  # use black and alpha to see the density a bit

            fig, ax = plt.subplots()
            if color_mode == "tracked":
                ax.scatter(
                    d_in_range[t_in_range == 0],
                    c_in_range[t_in_range == 0],
                    c=color[0],
                    s=5,
                    label="non-tracked",
                )
                ax.scatter(
                    d_in_range[t_in_range == 1],
                    c_in_range[t_in_range == 1],
                    c=color[1],
                    s=5,
                    label="tracked",
                )

            else:
                ax.scatter(d_in_range, c_in_range, c=color, s=5, label="ROI pairs")

            ax.set_xlabel(f"Pair-wise Distance ROI Centroids")
            ax.set_ylabel("Place Field Correlation")
            ax.set_title(f"Session Pair: {name}")
            ax.legend(loc="lower left")

            if with_save:
                save_name = f"pwdistcorr_comparison_envnum{prms['envnum']}_idxses{name}"
                self.saveFigure(fig.number, self.track.mouse_name, save_name)

        # Show figure if requested
        plt.show() if with_show else plt.close()

    def plot_pfcorr_by_samediff(
        self, corr, tracked, nnpair, prms, with_show=True, with_save=False
    ):
        """
        helper for plotting pfcorr values for same and different populations

        starts by making a dataframe object to plot distribution of place field correlation values
        for pairs of ROIs based on whether their roicat similarity value is in the "same" group or
        the "diff" group.

        inputs are corr, tracked, nnpair, and prms which come from make_roicat_comparison()
        """
        corr_same, corr_diff = self.split_by_roicat_assignment(corr, tracked)
        corr_same_nn, _ = self.split_by_roicat_assignment(corr, nnpair)

        dataframes = []
        for name, csame, cdiff, csamenn in zip(
            self.session_pair_names(prms), corr_same, corr_diff, corr_same_nn
        ):
            same_df = pd.DataFrame(
                {"PF Correlation": csame, "Session Pair": name, "ROICaT Assignment": "Same"}
            )
            diff_df = pd.DataFrame(
                {"PF Correlation": cdiff, "Session Pair": name, "ROICaT Assignment": "Diff"}
            )
            nn_df = pd.DataFrame(
                {"PF Correlation": csamenn, "Session Pair": name, "ROICaT Assignment": "NN"}
            )
            dataframes.extend([same_df, diff_df, nn_df])

        data = pd.concat(dataframes, ignore_index=True)

        plt.close("all")
        fig, ax = plt.subplots()
        sns.boxenplot(
            data=data, x="Session Pair", y="PF Correlation", hue="ROICaT Assignment", ax=ax
        )
        plt.show()
        ax.legend(loc="lower right")

        if with_save:
            sesidx = "_".join([str(i) for i in prms["idx_ses"]])
            save_name = f"pfcorr_by_samediff_envnum{prms['envnum']}_idxses{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)

        # Show figure if requested
        plt.show() if with_show else plt.close()

    def plot_pfcorrmean_by_samediff(
        self,
        corr,
        tracked,
        nnpair,
        pwdist,
        prms,
        dist_limit=10,
        return_data=False,
        with_show=True,
        with_save=False,
    ):
        """
        helper for plotting pfcorr values for same and different populations (only plotting the mean!)

        gets the mean/se correlation for tracked, not-tracked, and nearest-neighbor (<dist_limit) pairs
        from each session pair.

        then plots each as a curve across session pairs with different colors

        optionally returns the data for across-mouse plotting

        inputs come from make_roicat_comparison() and are named identically to the outputs of that function
        """
        corr_same, corr_diff = self.split_by_roicat_assignment(corr, tracked)
        corr_same_nn, _ = self.split_by_roicat_assignment(corr, nnpair)
        pwdist_same_nn, _ = self.split_by_roicat_assignment(pwdist, nnpair)

        # filter nn by whether they are "close" (e.g. less than a threshold distance apart)
        corr_same_nn = [csn[psn < dist_limit] for csn, psn in zip(corr_same_nn, pwdist_same_nn)]

        # we need three means per session pair (corrsame, corrsame_nearestNeighbor, and corrdiff)
        same_mean = [np.nanmean(cs) for cs in corr_same]
        samenn_mean = [np.nanmean(csn) for csn in corr_same_nn]
        diff_mean = [np.nanmean(cd) for cd in corr_diff]

        # also get standard error
        same_se = [np.nanstd(cs) / np.sqrt(np.sum(~np.isnan(cs))) for cs in corr_same]
        samenn_se = [np.nanstd(csn) / np.sqrt(np.sum(~np.isnan(csn))) for csn in corr_same_nn]
        diff_se = [np.nanstd(cd) / np.sqrt(np.sum(~np.isnan(cd))) for cd in corr_diff]

        # stack means/standard errors for easy data handling
        means = np.stack([np.array(d) for d in (same_mean, samenn_mean, diff_mean)])
        serrors = np.stack([np.array(d) for d in (same_se, samenn_se, diff_se)])

        # define some label names and colors, etc
        type_names = ["tracked", f"nearest-neighbors (<{dist_limit}pix^2)", "random-pairs"]
        type_colors = ["b", "r", "k"]
        session_pair_names = self.session_pair_names(prms)
        num_session_pairs = len(session_pair_names)

        # make the plots
        plt.close("all")
        fig, ax = plt.subplots()
        for tname, tcolor, tdata, sdata in zip(type_names, type_colors, means, serrors):
            ax.plot(
                range(num_session_pairs), tdata, color=tcolor, linewidth=1, marker="o", label=tname
            )
            ax.fill_between(
                range(num_session_pairs), tdata + sdata, tdata - sdata, color=(tcolor, 0.3)
            )

        ax.set_xlabel("Session Pair")
        ax.set_xticks(range(num_session_pairs), labels=session_pair_names)
        ax.set_ylabel("PF Correlation (+/- se)")
        ax.legend(loc="best")

        if with_save:
            sesidx = "_".join([str(i) for i in prms["idx_ses"]])
            save_name = f"pfcorrmean_by_samediff_envnum{prms['envnum']}_idxses{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)

        # Show figure if requested
        plt.show() if with_show else plt.close()

        # return data if requested
        if return_data:
            return means, serrors

    def plot_pfcorr_vs_pwdist_by_group(
        self, corr, tracked, pwdist, nnpair, prms, with_show=True, with_save=False
    ):
        """
        hi
        """
        # split pairwise distances by roicat assignment
        pwsame, pwdiff = self.split_by_roicat_assignment(pwdist, tracked)
        corrsame, corrdiff = self.split_by_roicat_assignment(corr, tracked)

        pwsamenn, pwdiffnn = self.split_by_roicat_assignment(pwdist, nnpair)
        corr_nnsame, corr_nndiff = self.split_by_roicat_assignment(corr, nnpair)

        # figure out maximum pair-wise distance in dataset
        max_pwd = np.max(
            [np.maximum(np.nanmax(pws), np.nanmax(pwd)) for pws, pwd in zip(pwsame, pwdiff)]
        )

        # measure distribution of pair-wise distances
        max_pwdist = 10
        bins = np.linspace(0, max_pwdist, 61)
        centers = helpers.edge2center(bins)

        fractional = False
        hist_method = helpers.fractional_histogram if fractional else np.histogram
        same_counts = [hist_method(pws[~np.isnan(pws)], bins=bins)[0] for pws in pwsame]
        diff_counts = [hist_method(pwd[~np.isnan(pwd)], bins=bins)[0] for pwd in pwdiff]

        # nn
        samenn_counts = [hist_method(pws[~np.isnan(pws)], bins=bins)[0] for pws in pwsamenn]
        diffnn_counts = [hist_method(pwd[~np.isnan(pwd)], bins=bins)[0] for pwd in pwdiffnn]

        # corresponds to "centers" now (and can include -1 or len(centers), but we'll ignore those)
        same_bin_idx = [np.searchsorted(bins, pws, side="left") - 1 for pws in pwsame]
        diff_bin_idx = [np.searchsorted(bins, pwd, side="left") - 1 for pwd in pwdiff]

        # corresponds to "centers" now (and can include -1 or len(centers), but we'll ignore those)
        samenn_bin_idx = [np.searchsorted(bins, pws, side="left") - 1 for pws in pwsamenn]
        diffnn_bin_idx = [np.searchsorted(bins, pwd, side="left") - 1 for pwd in pwdiffnn]

        # get corrs for each bin and measure mean, std
        def corr_stats(corr, binidx, max_bin):
            mean = np.full(max_bin, np.nan)
            for ic in range(max_bin):
                c_data = corr[binidx == ic]
                if len(c_data) != 0:
                    mean[ic] = np.mean(c_data)
            return mean

        # get mean/std pfcorr given pw distance
        same_corr_by_dist_mean = [
            corr_stats(csame, sbi, len(centers)) for csame, sbi in zip(corrsame, same_bin_idx)
        ]
        diff_corr_by_dist_mean = [
            corr_stats(cdiff, dbi, len(centers)) for cdiff, dbi in zip(corrdiff, diff_bin_idx)
        ]
        samenn_corr_by_dist_mean = [
            corr_stats(csame, sbi, len(centers)) for csame, sbi in zip(corr_nnsame, samenn_bin_idx)
        ]
        diffnn_corr_by_dist_mean = [
            corr_stats(cdiff, dbi, len(centers)) for cdiff, dbi in zip(corr_nndiff, diffnn_bin_idx)
        ]

        # stack these
        same_corr_by_dist_mean = np.stack(same_corr_by_dist_mean)
        diff_corr_by_dist_mean = np.stack(diff_corr_by_dist_mean)
        samenn_corr_by_dist_mean = np.stack(samenn_corr_by_dist_mean)
        diffnn_corr_by_dist_mean = np.stack(diffnn_corr_by_dist_mean)

        # also stack these
        same_counts = np.stack(same_counts)
        diff_counts = np.stack(diff_counts)
        samenn_counts = np.stack(samenn_counts)
        diffnn_counts = np.stack(diffnn_counts)

        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")

        mn_count_diff = np.nanmean(diff_counts, axis=0)
        se_count_diff = np.nanstd(diff_counts, axis=0)
        mn_count_same = np.nanmean(same_counts, axis=0)
        se_count_same = np.nanstd(same_counts, axis=0)

        mn_count_diffnn = np.nanmean(diffnn_counts, axis=0)
        se_count_diffnn = np.nanstd(diffnn_counts, axis=0)
        mn_count_samenn = np.nanmean(samenn_counts, axis=0)
        se_count_samenn = np.nanstd(samenn_counts, axis=0)

        ax[0].plot(centers, mn_count_diff, color="k", linewidth=1.5, label="different")
        ax[0].fill_between(
            centers, mn_count_diff + se_count_diff, mn_count_diff - se_count_diff, color=("k", 0.3)
        )
        ax[0].plot(centers, mn_count_same, color="b", linewidth=1.5, label="same")
        ax[0].fill_between(
            centers, mn_count_same + se_count_same, mn_count_same - se_count_same, color=("b", 0.3)
        )
        ax[0].plot(centers, mn_count_samenn, color="r", linewidth=1.5, label="nearest neighbor")
        ax[0].fill_between(
            centers,
            mn_count_samenn + se_count_samenn,
            mn_count_samenn - se_count_samenn,
            color=("r", 0.3),
        )

        ax[0].set_xlim(0, max_pwdist)
        ax[0].set_ylim(0, np.max(same_counts) * 1.2)
        ax[0].set_xlabel("pair-wise distance ROI centroids")
        ax[0].set_ylabel("counts")
        ax[0].legend(loc="best")

        mn_cbd_diff = np.nanmean(diff_corr_by_dist_mean, axis=0)
        se_cbd_diff = np.nanstd(diff_corr_by_dist_mean, axis=0)
        mn_cbd_same = np.nanmean(same_corr_by_dist_mean, axis=0)
        se_cbd_same = np.nanstd(same_corr_by_dist_mean, axis=0)

        mn_cbd_diffnn = np.nanmean(diffnn_corr_by_dist_mean, axis=0)
        se_cbd_diffnn = np.nanstd(diffnn_corr_by_dist_mean, axis=0)
        mn_cbd_samenn = np.nanmean(samenn_corr_by_dist_mean, axis=0)
        se_cbd_samenn = np.nanstd(samenn_corr_by_dist_mean, axis=0)

        ax[1].plot(centers, mn_cbd_diff, color="k", linewidth=1.5, label="different")
        ax[1].fill_between(
            centers, mn_cbd_diff + se_cbd_diff, mn_cbd_diff - se_cbd_diff, color=("k", 0.3)
        )
        ax[1].plot(centers, mn_cbd_same, color="b", linewidth=1.5, label="same")
        ax[1].fill_between(
            centers, mn_cbd_same + se_cbd_same, mn_cbd_same - se_cbd_same, color=("b", 0.3)
        )
        ax[1].plot(centers, mn_cbd_samenn, color="r", linewidth=1.5, label="nearest neighbor")
        ax[1].fill_between(
            centers, mn_cbd_samenn + se_cbd_samenn, mn_cbd_samenn - se_cbd_samenn, color=("r", 0.3)
        )

        ax[1].set_xlim(0, max_pwdist)
        ax[1].set_xlabel("pair-wise distance ROI centroids")
        ax[1].set_ylabel("average place field correlation")
        ax[1].legend(loc="lower left")

        if with_save:
            sesidx = "_".join([str(i) for i in prms["idx_ses"]])
            save_name = f"pfcorr_asfof_pwdist_{prms['envnum']}_idxses{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)

        # Show figure if requested
        plt.show() if with_show else plt.close()

    @handle_idx_ses
    def plot_roi_diameter(self, idx_ses=None, keep_planes=None, with_show=True, with_save=False):
        """
        simple plot of roi diameters (average range of xpix to ypix in ROI)
        """
        yxrange = self.track.get_roi_range(
            combine=True, cat_planes=True, idx_ses=idx_ses, keep_planes=self.keep_planes
        )

        bins = np.linspace(0, 50, 51)
        centers = helpers.edge2center(bins)

        data = np.mean(np.concatenate(yxrange, axis=0), axis=1)
        counts = helpers.fractional_histogram(data, bins=bins)[0]

        plt.close("all")
        fig = plt.figure()
        plt.plot(centers, counts, linewidth=1.5, label="footprint")
        plt.xlabel("ROI Diameter (pixels)")
        plt.ylabel("fraction of ROIs")

        if with_save:
            sesidx = "_".join([str(i) for i in idx_ses])
            save_name = f"roi_diameter_idxses{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)

        # Show figure if requested
        plt.show() if with_show else plt.close()

    def plot_similarity_histograms(self, sim, prms, with_show=True, with_save=False):
        """
        plot histogram of similarity values in each session
        """
        centers = helpers.edge2center(self.sim_hist_edges)
        sim_counts = [np.histogram(s[s != 0], bins=self.sim_hist_edges)[0] for s in sim]
        sim_zeros = [np.sum(s == 0) for s in sim]

        cmap = mpl.colormaps["tab10"]

        plt.close("all")
        fig, ax = plt.subplots()
        inset = fig.add_axes([0.2, 0.175, 0.1, 0.25])

        names = self.session_pair_names(prms)
        for idx, (name, count, zeros) in enumerate(zip(names, sim_counts, sim_zeros)):
            ax.plot(centers, count, c=cmap(idx), label=name)
            inset.plot([-0.2, 0.2], [zeros, zeros], c=cmap(idx), linewidth=2)

        # ax.set_yscale('log')
        ax.legend(loc="upper right", title="Session Pair")
        ax.set_xlabel("ROICaT Similarity Value")
        ax.set_ylabel("Counts")

        inset.set_xlim(-0.5, 0.5)
        inset.set_xlabel("")
        inset.set_xticks([])
        inset.set_ylabel("sim=0", fontsize=10)

        if with_save:
            sesidx = "_".join([str(i) for i in prms["idx_ses"]])
            save_name = f"similarity_histogram_idxses{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)

        # Show figure if requested
        plt.show() if with_show else plt.close()
