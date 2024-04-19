from copy import copy
from tqdm import tqdm
import pickle
import numpy as np
import faststats as fs
import matplotlib as mpl
from matplotlib import pyplot as plt
from .. import helpers
from ..analysis import placeCellSingleSession

"""
Documentation:
Found in docs/analysis/variance_structure.md
"""


class VarianceStructure(placeCellSingleSession):
    """
    Child of placeCellSingleSession designed to specifically analyze the variance structure using:

    - cvPCA
    - fourier mode decomposition
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "varianceStructure"

    def prepare_spkmaps(self, envnum=None, cutoffs=(0.4, 0.7), maxcutoffs=None, smooth=0.1, reliable=False):
        """prepare spkmaps for cvPCA"""
        # get spkmap with particular smoothing settings (don't smooth because we need all the trials)
        spkmaps = self.get_spkmap(envnum=envnum, average=False, smooth=smooth, trials="full")

        # Filter by reliable cells if requested
        if reliable:
            idx_reliable = self.get_reliable(envnum=envnum, cutoffs=cutoffs, maxcutoffs=maxcutoffs)
            spkmaps = [spkmap[ir] for spkmap, ir in zip(spkmaps, idx_reliable)]

        # return data
        return spkmaps

    def _get_min_trials(self, spkmaps):
        """helper for getting minimum number of trials for a particular environment's spkmap"""
        return min([sm.shape[1] for sm in spkmaps])

    def _get_min_neurons(self, spkmaps):
        """helper for getting minimum number of neurons for a particular environment's spkmap"""
        return min([sm.shape[0] for sm in spkmaps])

    def filter_nans(self, spkmaps):
        """helper for filtering nans (does so across spkmaps, e.g. environments) for equal number of valid positions in each environment"""
        idx_with_nan = [np.any(np.isnan(s), axis=(0, 1)) for s in spkmaps]
        idx_with_nan_across_spkmaps = np.any(np.stack(idx_with_nan), axis=0)
        return [spkmap[:, :, ~idx_with_nan_across_spkmaps] for spkmap in spkmaps]

    def concatenate_envs(self, spkmaps, filter_nans=True):
        """helper for concatenating spkmaps across environments"""
        if filter_nans:
            spkmaps = self.filter_nans(spkmaps)

        # concatenate across positions in spikemaps (using random sets of trials to make equal but not always use first N trials)
        num_trials_to_use = self._get_min_trials(spkmaps)
        num_neurons_to_use = self._get_min_neurons(spkmaps)
        trials_to_use = [np.random.permutation(s.shape[1])[:num_trials_to_use] for s in spkmaps]
        neurons_to_use = [np.random.permutation(s.shape[0])[:num_neurons_to_use] for s in spkmaps]
        return np.concatenate([s[n2u][:, t2u] for s, n2u, t2u in zip(spkmaps, neurons_to_use, trials_to_use)], axis=2)

    def do_cvpca(self, spkmaps, by_trial=False, nshuff=5, max_trials=None, max_neurons=None):
        """helper for running the full cvPCA gamut on spkmaps"""
        spkmaps = self.filter_nans(spkmaps)
        allspkmaps = self.concatenate_envs(spkmaps)

        # get maximum number of trials / neurons for consistent rank of matrices across environments / all envs
        max_trials = max_trials or int(self._get_min_trials(spkmaps) // 2)
        max_neurons = max_neurons or int(self._get_min_neurons(spkmaps))

        # do cvpca
        cv_by_env = [helpers.cvpca(spkmap, by_trial=by_trial, max_trials=max_trials, max_neurons=max_neurons, nshuff=nshuff) for spkmap in spkmaps]
        cv_across = helpers.cvpca(allspkmaps, by_trial=by_trial, max_trials=max_trials, max_neurons=max_neurons, nshuff=nshuff)

        return cv_by_env, cv_across

    def do_cvfourier(self, spkmaps, by_trial=False, nshuff=3, center=False, covariance=False):
        """helper for running the full cvFOURIER gamut on spkmaps"""
        freqs, basis = helpers.get_fourier_basis(spkmaps[0].shape[2], Fs=self.distStep)

        # get maximum number of trials / neurons for consistent rank of matrices across environments / all envs
        max_trials = int(self._get_min_trials(spkmaps) // 2)
        max_neurons = int(self._get_min_neurons(spkmaps))

        # do cvpca
        cv_by_env = [
            helpers.cv_fourier(
                spkmap, basis, by_trial=by_trial, center=center, covariance=covariance, max_trials=max_trials, max_neurons=max_neurons, nshuff=nshuff
            )
            for spkmap in spkmaps
        ]

        return freqs, cv_by_env


def load_spectra_data(pcm, args, save_as_temp=True, reload=True):
    """
    load data for variance structure analysis of cvPCA and cvFOURIER spectra

    first looks for temporary data and checks if it matches
    """
    # create variance structure objects (will load data which takes time)
    vss = []
    for p in pcm.pcss:
        vss.append(VarianceStructure(p.vrexp, distStep=args.dist_step, autoload=False))

    # will only load data if a temp doesn't exist or if it does but it doesn't match the expected session data
    load_data = False

    # first check if temp data exists
    if not args.reload_spectra_data and (pcm.saveDirectory("temp") / f"{args.mouse_name}_spectra_data.pkl").exists():
        # load data and populate into variables
        with open(pcm.saveDirectory("temp") / f"{args.mouse_name}_spectra_data.pkl", "rb") as f:
            temp_files = pickle.load(f)

        # populate variables
        try:
            required_keys = [
                "names",
                "envstats",
                "cv_by_env_all",
                "cv_by_env_rel",
                "cv_across_all",
                "cv_across_rel",
                "cvf_freqs",
                "cvf_by_env_all",
                "cvf_by_env_rel",
                "cvf_by_env_cov_all",
                "cvf_by_env_cov_rel",
                "kernels",
                "cv_kernels",
                "rel_mse",
                "rel_cor",
                "all_pf_mean",
                "all_pf_var",
                "all_pf_cv",
                "all_pf_tdot_mean",
                "all_pf_tdot_std",
                "all_pf_tdot_cv",
                "all_pf_tcorr_mean",
                "all_pf_tcorr_std",
                "rel_pf_mean",
                "rel_pf_var",
                "rel_pf_cv",
                "rel_pf_tdot_mean",
                "rel_pf_tdot_std",
                "rel_pf_tdot_cv",
                "rel_pf_tcorr_mean",
                "rel_pf_tcorr_std",
                "svca_shared",
                "svca_total",
            ]
            for key in required_keys:
                if key not in temp_files:
                    load_data = True
                    break

        except KeyError:
            load_data = True

        if not load_data:
            # check if variables are correct
            for name, v in zip(temp_files["names"], vss):
                if name != v.vrexp.sessionPrint():
                    load_data = True
                    continue

        # check if envstats is correct
        if not load_data and (temp_files["envstats"] != pcm.env_stats()):
            load_data = True

        # check if arguments are correct
        if not load_data:
            check_args = args if isinstance(args, dict) else vars(args)
            for key in ["cutoffs", "maxcutoffs", "smooth", "mouse_name", "dist_step"]:
                if temp_files["args"][key] != check_args[key]:
                    load_data = True

    else:
        # if temp file doesn't exist we need to load the data
        load_data = True

    # if data doesn't exist or is incorrect, then load data
    if load_data and not reload:
        raise ValueError("No temporary data found (or args.reload_spectra_data is True) for variance structure analysis, but reload set to False.")

    if load_data:
        names = [v.vrexp.sessionPrint() for v in vss]
        envstats = pcm.env_stats()

        # first load session data (this can take a while)
        for v in tqdm(vss, leave=True, desc="loading session data"):
            v.load_data()

        # get spkmaps of all cells / just reliable cells
        allcell_maps = []
        relcell_maps = []
        rel_mse = []
        rel_cor = []
        all_pf_mean = []
        all_pf_var = []
        all_pf_cv = []
        all_pf_tdot_mean = []
        all_pf_tdot_std = []
        all_pf_tdot_cv = []
        all_pf_tcorr_mean = []
        all_pf_tcorr_std = []
        rel_pf_mean = []
        rel_pf_var = []
        rel_pf_cv = []
        rel_pf_tdot_mean = []
        rel_pf_tdot_std = []
        rel_pf_tdot_cv = []
        rel_pf_tcorr_mean = []
        rel_pf_tcorr_std = []
        kernels = []
        cv_kernels = []
        for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
            # get reliable cells (for each environment) and spkmaps for each environment (with all cells)
            c_idx_reliable = v.get_reliable(envnum=None, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs)
            c_spkmaps = v.prepare_spkmaps(envnum=None, smooth=args.smooth, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs, reliable=False)
            c_rel_spkmaps = [spkmap[cir] for spkmap, cir in zip(c_spkmaps, c_idx_reliable)]

            # get reliable values for each environment
            c_mse, c_cor = v.get_reliability_values(envnum=None, with_test=False)

            # get place field for each cell
            c_placefields_all = [np.nanmean(spkmap, axis=1) for spkmap in c_spkmaps]
            c_placefields_rel = [np.nanmean(spkmap[cir], axis=1) for spkmap, cir in zip(c_spkmaps, c_idx_reliable)]

            # make place field a unit vector
            c_all_unitpf = [placefield / np.linalg.norm(placefield, axis=1, keepdims=True) for placefield in c_placefields_all]
            c_rel_unitpf = [placefield / np.linalg.norm(placefield, axis=1, keepdims=True) for placefield in c_placefields_rel]

            # add each to list
            allcell_maps.append(c_spkmaps)
            relcell_maps.append(c_rel_spkmaps)
            rel_mse.append(c_mse)
            rel_cor.append(c_cor)
            all_pf_var.append([np.nanvar(placefield, axis=1) for placefield in c_placefields_all])
            rel_pf_var.append([np.nanvar(placefield, axis=1) for placefield in c_placefields_rel])

            # compute spatial kernel matrices
            train_idx, test_idx = helpers.named_transpose([helpers.cvFoldSplit(np.arange(spkmap.shape[1]), 2) for spkmap in c_spkmaps])
            c_pf_train = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, train_idx)]
            c_pf_test = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, test_idx)]
            c_pf_train_centered = [pf - np.nanmean(pf, axis=0) for pf in c_pf_train]
            c_pf_test_centered = [pf - np.nanmean(pf, axis=0) for pf in c_pf_test]

            kernels.append([np.cov(pf.T) for pf in c_placefields_all])
            cv_kernels.append([cpftrain.T @ cpftest / (cpftrain.shape[0] - 1) for cpftrain, cpftest in zip(c_pf_train_centered, c_pf_test_centered)])

            # get other place field statistics
            c_all_pf_mean = [fs.nanmean(placefield, axis=1) for placefield in c_placefields_all]
            c_all_pf_cv = [fs.nanstd(placefield, axis=1) / fs.nanmean(placefield, axis=1) for placefield in c_placefields_all]
            c_all_pf_amplitude = [fs.nansum(np.expand_dims(placefield, 1) * spkmap, axis=2) for placefield, spkmap in zip(c_all_unitpf, c_spkmaps)]
            c_all_pf_tdot_mean = [np.nanmean(amplitude, axis=1) for amplitude in c_all_pf_amplitude]
            c_all_pf_tdot_std = [np.nanstd(amplitude, axis=1) for amplitude in c_all_pf_amplitude]
            c_all_pf_tdot_cv = [fs.nanstd(amplitude, axis=1) / fs.nanmean(amplitude, axis=1) for amplitude in c_all_pf_amplitude]

            all_pf_mean.append(c_all_pf_mean)
            all_pf_cv.append(c_all_pf_cv)
            all_pf_tdot_mean.append(c_all_pf_tdot_mean)
            all_pf_tdot_std.append(c_all_pf_tdot_std)
            all_pf_tdot_cv.append(c_all_pf_tdot_cv)

            c_rel_pf_mean = [fs.nanmean(placefield, axis=1) for placefield in c_placefields_rel]
            c_rel_pf_cv = [fs.nanstd(placefield, axis=1) / fs.nanmean(placefield, axis=1) for placefield in c_placefields_rel]
            c_rel_pf_amplitude = [
                fs.nansum(np.expand_dims(placefield, 1) * spkmap, axis=2) for placefield, spkmap in zip(c_rel_unitpf, c_rel_spkmaps)
            ]
            c_rel_pf_tdot_mean = [np.nanmean(amplitude, axis=1) for amplitude in c_rel_pf_amplitude]
            c_rel_pf_tdot_std = [np.nanstd(amplitude, axis=1) for amplitude in c_rel_pf_amplitude]
            c_rel_pf_tdot_cv = [fs.nanstd(amplitude, axis=1) / fs.nanmean(amplitude, axis=1) for amplitude in c_rel_pf_amplitude]

            rel_pf_mean.append(c_rel_pf_mean)
            rel_pf_cv.append(c_rel_pf_cv)
            rel_pf_tdot_mean.append(c_rel_pf_tdot_mean)
            rel_pf_tdot_std.append(c_rel_pf_tdot_std)
            rel_pf_tdot_cv.append(c_rel_pf_tdot_cv)

            # get trial by trial correlation with the mean place field and the trial by trial place field
            c_all_pf_tcorr = [
                helpers.vectorCorrelation(spkmap, np.repeat(np.expand_dims(placefield, 1), spkmap.shape[1], 1), axis=2)
                for spkmap, placefield in zip(c_spkmaps, c_placefields_all)
            ]
            c_rel_pf_tcorr = [
                helpers.vectorCorrelation(spkmap, np.repeat(np.expand_dims(placefield, 1), spkmap.shape[1], 1), axis=2)
                for spkmap, placefield in zip(c_rel_spkmaps, c_placefields_rel)
            ]

            all_pf_tcorr_mean.append([np.nanmean(tcorr, axis=1) for tcorr in c_all_pf_tcorr])
            all_pf_tcorr_std.append([np.nanstd(tcorr, axis=1) for tcorr in c_all_pf_tcorr])
            rel_pf_tcorr_mean.append([np.nanmean(tcorr, axis=1) for tcorr in c_rel_pf_tcorr])
            rel_pf_tcorr_std.append([np.nanstd(tcorr, axis=1) for tcorr in c_rel_pf_tcorr])

        # make analyses consistent by using same (randomly subsampled) numbers of trials & neurons for each analysis
        all_max_trials = min([int(v._get_min_trials(allmap) // 2) for v, allmap in zip(vss, allcell_maps)])
        all_max_neurons = min([int(v._get_min_neurons(allmap)) for v, allmap in zip(vss, allcell_maps)])
        rel_max_trials = min([int(v._get_min_trials(relmap) // 2) for v, relmap in zip(vss, relcell_maps)])
        rel_max_neurons = min([int(v._get_min_neurons(relmap)) for v, relmap in zip(vss, relcell_maps)])

        # get cvPCA and cvFOURIER analyses for all cells / just reliable cells
        cv_by_env_all = []
        cv_by_env_rel = []
        cv_across_all = []
        cv_across_rel = []
        cvf_freqs = []
        cvf_by_env_all = []
        cvf_by_env_rel = []
        cvf_by_env_cov_all = []
        cvf_by_env_cov_rel = []
        for allmap, relmap, v in tqdm(zip(allcell_maps, relcell_maps, vss), leave=False, desc="running cvPCA and cvFOURIER", total=len(vss)):
            # get cvPCA for all cell spike maps (always do by_trial=False until we have a theory for all trial=True)
            c_env, c_acc = v.do_cvpca(allmap, by_trial=False, max_trials=all_max_trials, max_neurons=all_max_neurons)
            cv_by_env_all.append(c_env)
            cv_across_all.append(c_acc)

            # get cvPCA for rel cell spike maps (always do by_trial=False until we have a theory for all trial=True)
            c_env, c_acc = v.do_cvpca(relmap, by_trial=False, max_trials=rel_max_trials, max_neurons=rel_max_neurons)
            cv_by_env_rel.append(c_env)
            cv_across_rel.append(c_acc)

            # get cvFOURIER for all/rel cell spike maps using correlation (always do by_trial=False until we have a theory for all trial=True)
            c_freqs, c_all = v.do_cvfourier(allmap, by_trial=False, covariance=False)
            _, c_rel = v.do_cvfourier(relmap, by_trial=False, covariance=False)
            cvf_freqs.append(c_freqs)
            cvf_by_env_all.append(c_all)
            cvf_by_env_rel.append(c_rel)

            # get cvFOURIER for all/rel cell spike maps using covariance (always do by_trial=False until we have a theory for all trial=True)
            _, c_all = v.do_cvfourier(allmap, by_trial=False, covariance=True)
            _, c_rel = v.do_cvfourier(relmap, by_trial=False, covariance=True)
            cvf_by_env_cov_all.append(c_all)
            cvf_by_env_cov_rel.append(c_rel)

        # get spks of all cells (in time, not space) -- filter by good planes (which is defaulted to all but first, which is usually flyback)
        idx_rois = []
        for v in vss:
            v.get_plane_idx(keep_planes=[1, 2, 3, 4])
            idx_rois.append(v.idxUseROI)
        ospks = [v.vrexp.loadone("mpci.roiActivityDeconvolvedOasis")[:, idx] for v, idx in zip(vss, idx_rois)]

        # get spkmaps of all cells / just reliable cells
        svca_shared = []
        svca_total = []
        for ospk in tqdm(ospks, leave=False, desc="doing SVCA"):
            c_shared_var, c_tot_cov_space_var = helpers.split_and_svca(ospk.T, verbose=False)[:2]
            svca_shared.append(c_shared_var)
            svca_total.append(c_tot_cov_space_var)

        # save data as temporary files
        temp_save_args = args if type(args) == dict else args.asdict() if type(args) == helpers.AttributeDict else vars(args)
        temp_files = {
            "names": names,
            "envstats": envstats,
            "args": temp_save_args,
            "cv_by_env_all": cv_by_env_all,
            "cv_by_env_rel": cv_by_env_rel,
            "cv_across_all": cv_across_all,
            "cv_across_rel": cv_across_rel,
            "cvf_freqs": cvf_freqs,
            "cvf_by_env_all": cvf_by_env_all,
            "cvf_by_env_rel": cvf_by_env_rel,
            "cvf_by_env_cov_all": cvf_by_env_cov_all,
            "cvf_by_env_cov_rel": cvf_by_env_cov_rel,
            "kernels": kernels,
            "cv_kernels": cv_kernels,
            "rel_mse": rel_mse,
            "rel_cor": rel_cor,
            "all_pf_mean": all_pf_mean,
            "all_pf_var": all_pf_var,
            "all_pf_cv": all_pf_cv,
            "all_pf_tdot_mean": all_pf_tdot_mean,
            "all_pf_tdot_std": all_pf_tdot_std,
            "all_pf_tdot_cv": all_pf_tdot_cv,
            "all_pf_tcorr_mean": all_pf_tcorr_mean,
            "all_pf_tcorr_std": all_pf_tcorr_std,
            "rel_pf_mean": rel_pf_mean,
            "rel_pf_var": rel_pf_var,
            "rel_pf_cv": rel_pf_cv,
            "rel_pf_tdot_mean": rel_pf_tdot_mean,
            "rel_pf_tdot_std": rel_pf_tdot_std,
            "rel_pf_tdot_cv": rel_pf_tdot_cv,
            "rel_pf_tcorr_mean": rel_pf_tcorr_mean,
            "rel_pf_tcorr_std": rel_pf_tcorr_std,
            "svca_shared": svca_shared,
            "svca_total": svca_total,
        }
        if save_as_temp:
            pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")

    else:
        print("Successfully loaded temporary data for variance structure analysis.")

    # return spectra data
    return temp_files


def plot_svca_data(
    pcm,
    spectra_data,
    normalize=False,
    y_min=1e-5,
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    cmap_ses = mpl.colormaps["turbo"].resampled(num_sessions)
    cmap_env = mpl.colormaps["Set1"].resampled(num_envs)

    def norm(data):
        """helper for optionally normalizing data"""
        if normalize:
            data = data / np.nansum(data)
        data[data < y_min] = np.nan
        return data

    alpha = 1
    figdim = 3
    fig, ax = plt.subplots(2, 2, figsize=(2 * figdim, 2 * figdim), layout="constrained", sharex=True, sharey="row")
    for i in range(num_sessions):
        c_num_env = -1
        for seslist in spectra_data["envstats"].values():
            if i in seslist:
                c_num_env += 1
        cdata = spectra_data["svca_shared"][i]
        ax[0, 0].plot(range(1, len(cdata) + 1), norm(cdata), color=(cmap_ses(i), alpha))
        ax[0, 1].plot(range(1, len(cdata) + 1), norm(cdata), color=(cmap_env(c_num_env), alpha))
        ax[1, 0].plot(range(1, len(cdata) + 1), np.nancumsum(norm(cdata)), color=(cmap_ses(i), alpha))
        ax[1, 1].plot(range(1, len(cdata) + 1), np.nancumsum(norm(cdata)), color=(cmap_env(c_num_env), alpha))

    ax[0, 0].set_title("Shared Variance - (by session)")
    ax[0, 1].set_title("Shared Variance - (by #env/session)")
    ax[0, 0].set_ylabel("Variance")
    ax[1, 0].set_ylabel("Cumulative Variance")
    ax[1, 0].set_xlabel("Dimension")
    ax[1, 1].set_xlabel("Dimension")

    for ax_row in ax:
        for a in ax_row:
            a.set_xscale("log")
            a.set_yscale("log")

    axins0 = ax[0, 0].inset_axes([0.05, 0.18, 0.4, 0.075])
    axins0.xaxis.set_ticks_position("bottom")
    m = mpl.cm.ScalarMappable(cmap=cmap_ses)
    cb0 = fig.colorbar(m, cax=axins0, orientation="horizontal")
    cb0.set_label("session #", loc="center", y=10)

    axins1 = ax[0, 1].inset_axes([0.05, 0.18, 0.4, 0.075])
    axins1.xaxis.set_ticks_position("bottom")
    m = mpl.cm.ScalarMappable(cmap=cmap_env)
    fig.colorbar(m, cax=axins1, orientation="horizontal", label="#env/ses")

    if with_show:
        plt.show()

    if with_save:
        special_name = "_normalized" if normalize else ""
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "svca_spectra_" + special_name)


def add_to_spectra_data(pcm, args):
    """skeleton for adding something without reloading everything"""
    with open(pcm.saveDirectory("temp") / f"{args.mouse_name}_spectra_data.pkl", "rb") as f:
        temp_files = pickle.load(f)

    vss = []
    for p in pcm.pcss:
        vss.append(VarianceStructure(p.vrexp, distStep=args.dist_step, autoload=False))

    # first load session data (this can take a while)
    for v in tqdm(vss, leave=True, desc="loading session data"):
        v.load_data()

    # get spkmaps of all cells / just reliable cells
    variable = []
    for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
        # get ~variable~ for each session
        # variable.append(v.get_variable(...))
        pass

    # temp_files["variable"] = variable
    pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")


def plot_spectral_data(
    pcm,
    spectra_data,
    color_by_session=True,
    normalize=False,
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    cmap = mpl.colormaps["turbo"].resampled(num_sessions)

    def norm(data):
        """helper for optionally normalizing data"""
        if normalize:
            return data / np.sum(data)
        return data

    def get_color(env, sesnum):
        """helper for getting color based on color method"""
        if color_by_session:
            # color by absolute session number
            return cmap(sesnum)
        else:
            # color by relative session number (within environment)
            sesnum_for_env = spectra_data["envstats"][env].index(sesnum)
            return cmap(sesnum_for_env)

    figdim = 3
    fig, ax = plt.subplots(2, num_envs + 1, figsize=((num_envs + 1) * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = spectra_data["cv_by_env_all"][j][eidx]
                ax[0, i].plot(range(1, len(cdata) + 1), norm(cdata), color=get_color(c_env, j))

                cdata = spectra_data["cv_by_env_rel"][j][eidx]
                ax[1, i].plot(range(1, len(cdata) + 1), norm(cdata), color=get_color(c_env, j))

            ax[0, i].set_title(f"Environment {c_env}")
            ax[0, i].set_ylabel("Eigenspectrum")
            ax[1, i].set_ylabel("Eigenspectrum (reliable)")
            ax[1, i].set_xlabel("Dimension")

        ax[0, i].set_xscale("log")
        ax[1, i].set_xscale("log")

    for j in range(num_sessions):
        cdata = spectra_data["cv_across_all"][j]
        ax[0, -1].plot(range(1, len(cdata) + 1), norm(cdata), color=cmap(j))
        cdata = spectra_data["cv_across_rel"][j]
        ax[1, -1].plot(range(1, len(cdata) + 1), norm(cdata), color=cmap(j))
        ax[0, -1].set_title(f"All Environments")
        ax[1, -1].set_xlabel("Dimension")

    if with_show:
        plt.show()

    if with_save:
        special_name = "by_session" if color_by_session else "by_relative_session"
        special_name = special_name + "_normalized" if normalize else special_name
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "cv_spectra_" + special_name)


def plot_spectral_averages(
    pcm,
    spectra_data,
    do_xlog=False,
    do_ylog=False,
    ylog_min=1e-3,
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    norm = lambda x: x / np.nansum(x)

    if not do_ylog:
        ylog_min = -np.inf

    all_be = []
    all_across = []

    figdim = 3
    fig, ax = plt.subplots(2, 3, figsize=(3 * figdim, 2 * figdim), layout="constrained")
    for ii, cc in enumerate(spectra_data["cv_by_env_all"]):
        for jj, c in enumerate(cc):
            label = "Single Env" if (ii == 0) and (jj == 0) else None
            c_c = copy(c)
            c_c[c_c < ylog_min] = np.nan
            all_be.append(c_c)
            ax[0, 0].plot(range(1, len(c) + 1), norm(c_c), c=("k", 0.3), label=None)
            ax[1, 0].plot(range(1, len(c) + 1), np.cumsum(norm(c_c)), c=("k", 0.3))

    for ii, c in enumerate(spectra_data["cv_across_all"]):
        label = "Across Envs" if ii == 0 else None
        c_c = copy(c)
        c_c[c_c < ylog_min] = np.nan
        all_across.append(c_c)
        ax[0, 1].plot(range(1, len(c) + 1), norm(c_c), c=("r", 0.3), label=label)
        ax[1, 1].plot(range(1, len(c) + 1), np.cumsum(norm(c_c)), c=("r", 0.3))

    all_be = np.stack(all_be)
    all_across = np.stack(all_across)

    ax[0, 2].plot(range(1, all_be.shape[1] + 1), norm(np.nanmean(all_be, axis=0)), c="k", label="Average Single Env")
    ax[0, 2].plot(range(1, all_across.shape[1] + 1), norm(np.nanmean(all_across, axis=0)), c="r")

    ax[1, 2].plot(range(1, all_be.shape[1] + 1), np.cumsum(norm(np.nanmean(all_be, axis=0))), c="k", label="Average Across Envs")
    ax[1, 2].plot(range(1, all_across.shape[1] + 1), np.cumsum(norm(np.nanmean(all_across, axis=0))), c="r")

    ax[1, 0].set_xlabel("Dimension")
    ax[1, 1].set_xlabel("Dimension")
    ax[1, 2].set_xlabel("Dimension")
    ax[0, 0].set_ylabel("Variance")
    ax[1, 0].set_ylabel("Cumulative Variance")
    ax[0, 0].set_title("Single Environments")
    ax[0, 1].set_title("Across Environments")
    ax[0, 2].set_title("Averages")

    if do_xlog:
        for aa in ax:
            for a in aa:
                a.set_xscale("log")

    if do_ylog:
        for aa in ax:
            for a in aa:
                a.set_yscale("log")

    if with_show:
        plt.show()

    if with_save:
        special_name = "logx_" if do_xlog else "linx_"
        special_name = special_name + ("logy" if do_ylog else "liny")
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "cv_norm_spectra_" + special_name)


def plot_spectral_energy(
    pcm,
    spectra_data,
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    cmap = mpl.colormaps["Set1"].resampled(num_envs)

    # get total energy for each env/session
    var_by_env_all = np.full((num_sessions, num_envs), np.nan)
    var_by_env_rel = np.full((num_sessions, num_envs), np.nan)

    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = spectra_data["cv_by_env_all"][j][eidx]
                var_by_env_all[j, i] = np.sum(cdata)

                cdata = spectra_data["cv_by_env_rel"][j][eidx]
                var_by_env_rel[j, i] = np.sum(cdata)

    figdim = 3
    fig, ax = plt.subplots(1, 2, figsize=(2 * figdim, figdim), layout="constrained")
    for i in range(num_envs):
        ax[0].plot(range(num_sessions), var_by_env_all[:, i], color=cmap(i), marker=".", label=f"Environment {pcm.environments[i]}")
        ax[1].plot(range(num_sessions), var_by_env_rel[:, i], color=cmap(i), marker=".", label=f"Environment {pcm.environments[i]}")

    ax[0].set_title("All Cells")
    ax[1].set_title("Reliable Cells")
    ax[0].set_ylabel("Total Variance")
    ax[1].set_ylabel("Total Variance")
    ax[0].set_xlabel("Session")
    ax[1].set_xlabel("Session")
    ax[0].legend(fontsize=8)

    if with_show:
        plt.show()

    if with_save:
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "cv_total_variance")


def plot_fourier_data(
    pcm,
    spectra_data,
    covariance=False,
    color_by_session=True,
    ignore_dc=True,
    with_show=True,
    with_save=False,
):
    """
    plot fourier data for variance structure analysis
    """

    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    cmap = mpl.colormaps["turbo"].resampled(num_sessions)

    def get_color(env, sesnum):
        """helper for getting color based on color method"""
        if color_by_session:
            # color by absolute session number
            return cmap(sesnum)
        else:
            # color by relative session number (within environment)
            sesnum_for_env = spectra_data["envstats"][env].index(sesnum)
            return cmap(sesnum_for_env)

    cvf_all = spectra_data["cvf_by_env_cov_all" if covariance else "cvf_by_env_all"]
    cvf_rel = spectra_data["cvf_by_env_cov_rel" if covariance else "cvf_by_env_rel"]

    figdim = 3
    fig, ax = plt.subplots(2, num_envs * 2, figsize=(2 * num_envs * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            c_freqs = spectra_data["cvf_freqs"][j]
            if ignore_dc:
                c_freqs = c_freqs[1:]
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = cvf_all[j][eidx]
                if ignore_dc:
                    cdata = [c[1:] for c in cdata]
                ax[0, 2 * i].plot(c_freqs, cdata[0], color=get_color(c_env, j), linestyle="-")
                ax[0, 2 * i + 1].plot(c_freqs, cdata[1], color=get_color(c_env, j), linestyle="--")

                cdata = cvf_rel[j][eidx]
                if ignore_dc:
                    cdata = [c[1:] for c in cdata]
                ax[1, 2 * i].plot(c_freqs, cdata[0], color=get_color(c_env, j), linestyle="-")
                ax[1, 2 * i + 1].plot(c_freqs, cdata[1], color=get_color(c_env, j), linestyle="--")

            ax[0, 2 * i].set_title(f"Environment {c_env} (cosines)")
            ax[0, 2 * i + 1].set_title(f"Environment {c_env} (sines)")
            if i == 0:
                ax[0, i].set_ylabel("Fourier Reliability")
                ax[1, i].set_ylabel("Fourier Reliability (reliable)")
            ax[1, 2 * i].set_xlabel("1 / SpatialWidth (1/cm)")
            ax[1, 2 * i + 1].set_xlabel("1 / SpatialWidth (1/cm)")

        ax[0, i].set_xscale("log")
        ax[1, i].set_xscale("log")

    if with_show:
        plt.show()

    if with_save:
        special_name = "by_session" if color_by_session else "by_relative_session"
        special_name = special_name + "_covariance" if covariance else special_name + "_correlation"
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "cv_fourier_" + special_name)


def plot_reliability_data(pcm, spectra_data, color_by_session=True, with_show=True, with_save=False):
    """
    plot fourier data for variance structure analysis
    """

    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    # some plotting supporting variables
    cmap = mpl.colormaps["turbo"].resampled(num_sessions)
    mse_bins = np.linspace(-4, 1, 31)
    cor_bins = np.linspace(-1, 1, 31)
    mse_centers = helpers.edge2center(mse_bins)
    cor_centers = helpers.edge2center(cor_bins)

    def get_color(env, sesnum):
        """helper for getting color based on color method"""
        if color_by_session:
            # color by absolute session number
            return cmap(sesnum)
        else:
            # color by relative session number (within environment)
            sesnum_for_env = spectra_data["envstats"][env].index(sesnum)
            return cmap(sesnum_for_env)

    figdim = 3
    fig, ax = plt.subplots(2, num_envs, figsize=(num_envs * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = helpers.fractional_histogram(spectra_data["rel_mse"][j][eidx], bins=mse_bins)[0]
                ax[0, i].plot(mse_centers, cdata, color=get_color(c_env, j), linestyle="-")
                ax[0, i].axvline(np.mean(spectra_data["rel_mse"][j][eidx]), color=get_color(c_env, j))

                cdata = helpers.fractional_histogram(spectra_data["rel_cor"][j][eidx], bins=cor_bins)[0]
                ax[1, i].plot(cor_centers, cdata, color=get_color(c_env, j), linestyle="-")
                ax[1, i].axvline(np.mean(spectra_data["rel_cor"][j][eidx]), color=get_color(c_env, j))

            ax[0, i].set_title(f"Environment {c_env} Rel-MSE")
            ax[1, i].set_title(f"Environment {c_env} Rel-Corr")
            if i == 0:
                ax[0, i].set_ylabel("Counts")
                ax[1, i].set_ylabel("Counts")
            ax[0, i].set_xlabel("Reliability (MSE)")
            ax[1, i].set_xlabel("Reliability (Corr)")

    if with_show:
        plt.show()

    if with_save:
        special_name = "by_session" if color_by_session else "by_relative_session"
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "reliability_" + special_name)


def plot_pf_var_data(pcm, spectra_data, color_by_session=True, with_show=True, with_save=False):
    """
    plot place field variance data for variance structure analysis
    """

    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    # some plotting supporting variables
    cmap = mpl.colormaps["turbo"].resampled(num_sessions)
    # concatenate a list of lists of np arrays
    min_pf_var = np.nanmin(
        np.concatenate([np.concatenate(apf) for apf in spectra_data["all_pf_var"]] + [np.concatenate(rpf) for rpf in spectra_data["rel_pf_var"]])
    )
    max_pf_var = np.nanmax(
        np.concatenate([np.concatenate(apf) for apf in spectra_data["all_pf_var"]] + [np.concatenate(rpf) for rpf in spectra_data["rel_pf_var"]])
    )
    pf_bins = np.linspace(min_pf_var, max_pf_var, 21)
    pf_centers = helpers.edge2center(pf_bins)

    def get_color(env, sesnum):
        """helper for getting color based on color method"""
        if color_by_session:
            # color by absolute session number
            return cmap(sesnum)
        else:
            # color by relative session number (within environment)
            sesnum_for_env = spectra_data["envstats"][env].index(sesnum)
            return cmap(sesnum_for_env)

    figdim = 3
    fig, ax = plt.subplots(2, num_envs, figsize=(num_envs * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = helpers.fractional_histogram(spectra_data["all_pf_var"][j][eidx], bins=pf_bins)[0]
                ax[0, i].plot(pf_centers, cdata, color=get_color(c_env, j), linestyle="-")

                cdata = helpers.fractional_histogram(spectra_data["rel_pf_var"][j][eidx], bins=pf_bins)[0]
                ax[1, i].plot(pf_centers, cdata, color=get_color(c_env, j), linestyle="-")

            ax[0, i].set_title(f"Environment {c_env} All Cell PF Var")
            ax[1, i].set_title(f"Environment {c_env} Rel Cell PF Var")
            if i == 0:
                ax[0, i].set_ylabel("Counts")
                ax[1, i].set_ylabel("Counts")
            ax[0, i].set_xlabel("Variance")
            ax[1, i].set_xlabel("Variance")

        ax[0, i].set_yscale("log")
        ax[1, i].set_yscale("log")

    if with_show:
        plt.show()

    if with_save:
        special_name = "by_session" if color_by_session else "by_relative_session"
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "pf_variance_" + special_name)


def compare_exp_fits(pcm, spectra_data, amplitude=True, color_by_session=True, with_show=True, with_save=False):
    """
    First make exponential fits of the eigenspectra, then use place field and other properties to predict the fit parameters
    """
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    single_amplitude = []
    single_decay = []
    single_r2 = []
    across_amplitude = np.full(num_sessions, np.nan)
    across_decay = np.full(num_sessions, np.nan)
    across_r2 = np.full(num_sessions, np.nan)
    for ii, (c_by_env, c_across) in enumerate(zip(spectra_data["cv_by_env_all"], spectra_data["cv_across_all"])):
        # get exponential fits and r2 for single environment spectra
        (a, d), r = helpers.fit_exponentials(np.stack(c_by_env), bias=False)
        single_amplitude.append(a)
        single_decay.append(d)
        single_r2.append(r)

        # get exponential fits and r2 for across environment spectra
        (a, d), r = helpers.fit_exponentials(c_across.reshape(1, -1), bias=False)
        across_amplitude[ii] = a[0]
        across_decay[ii] = d[0]
        across_r2[ii] = r[0]

    # names of place-field related variables to compare with exponential fit data
    pfvars = [
        "rel_cor",
        "all_pf_mean",
        "all_pf_var",
        "all_pf_cv",
        "all_pf_tdot_mean",
        "all_pf_tdot_std",
        "all_pf_tdot_cv",
        "all_pf_tcorr_mean",
        "all_pf_tcorr_std",
    ]
    num_vars = len(pfvars)

    cmap = mpl.colormaps["turbo"].resampled(num_sessions)

    def get_color(env, sesnum):
        """helper for getting color based on color method"""
        if color_by_session:
            # color by absolute session number
            return cmap(sesnum)
        else:
            # color by relative session number (within environment)
            sesnum_for_env = spectra_data["envstats"][env].index(sesnum)
            return cmap(sesnum_for_env)

    # determine which parameter to plot based on amplitude kwarg
    single = single_amplitude if amplitude else single_decay

    # make the plot
    figdim = 1.5
    fig, ax = plt.subplots(num_envs, num_vars, figsize=(num_vars * figdim, num_envs * figdim), layout="constrained", sharex="col", sharey=True)
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]
                for ipf, c_pfvar in enumerate(pfvars):
                    cdata = np.nanmean(spectra_data[c_pfvar][j][eidx])
                    if np.all(np.isnan(spectra_data[c_pfvar][j][eidx])):
                        print(f"Warning: {c_pfvar} is all nan for {pcm.track.mouse_name} {c_env} {j}")

                    ax[i, ipf].scatter(cdata, single[j][eidx], color=get_color(c_env, j), marker=".")

                    if ipf == 0:
                        ax[i, ipf].set_ylabel(f"Environment {c_env}\n{'Amplitude' if amplitude else 'Decay'}")
                    else:
                        ax[i, ipf].set_ylabel("Amplitude" if amplitude else "Decay")
                    if i == 0:
                        ax[i, ipf].set_title(c_pfvar)
                    ax[i, ipf].set_xlabel(c_pfvar)
                    ax[i, ipf].set_yscale("linear")
                    if c_pfvar == "rel_mse":
                        ax[i, ipf].set_xlim(-4, 1)

    ax[0, 0].set_ylim(bottom=0)

    if with_show:
        plt.show()

    if with_save:
        special_name = "amplitude" if amplitude else "decay"
        special_name = special_name + ("_by_session" if color_by_session else "_by_relative_session")
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "comparison_pfvars_eigenspectra_" + special_name)


# =================================== code for comparing spectral data across mice =================================== #
def compare_spectral_averages(summary_dicts):
    """
    get average spectral data for each mouse (specifically the average eigenspectra for single / all environments)

    summary dicts is a list of dictionaries containing full spectra data for each mouse to be compared
    returns a list of lists, where the outer list corresponds to each mouse (e.g. each summary_dict)
    and the inner list contains the eigenspectra for each session / environment
    """
    # get average eigenspectra for each mouse
    single_env = []
    across_env = []
    for summary_dict in summary_dicts:
        all_single_env = []
        all_across_env = []
        # go through each session's single environment eigenspectra
        for cc in summary_dict["cv_by_env_all"]:
            # it'll be a list of eigenspectra for each environment, group them together nondiscriminately
            for c in cc:
                all_single_env.append(c)  # normalize each eigenspectrum and add to list

        # go through each session's across environment eigenspectra
        for c in summary_dict["cv_across_all"]:
            all_across_env.append(c)

        # add them all to master list
        single_env.append(all_single_env)
        across_env.append(all_across_env)

    return single_env, across_env


def plot_spectral_averages_comparison(pcms, single_env, across_env, do_xlog=False, do_ylog=False, ylog_min=1e-3, with_show=True, with_save=False):

    # if not using a y-log axis, then set the minimum to -inf to not change any data
    if not do_ylog:
        ylog_min = -np.inf

    # create processing method
    def _process(data):
        """internal function for processing a set of eigenspectra"""
        data = np.stack(data)
        # normalize each row so they sum to 1
        data = data / np.sum(data, axis=1, keepdims=True)
        # take the average across rows (across sessions / environments)
        data = np.mean(data, axis=0)
        # remove any values below the minimum for log scaling
        data[data < ylog_min] = np.nan
        # return processed data
        return data

    num_mice = len(pcms)
    mouse_names = [pcm.track.mouse_name for pcm in pcms]
    cmap = mpl.colormaps["turbo"].resampled(num_mice)

    figdim = 3
    fig, ax = plt.subplots(2, 2, figsize=(2 * figdim, 2 * figdim), layout="constrained", sharex=True, sharey="row")
    for imouse, (mouse_name, c_single_env, c_across_env) in enumerate(zip(mouse_names, single_env, across_env)):
        c_single_data = _process(c_single_env)
        c_across_data = _process(c_across_env)
        ax[0, 0].plot(range(1, len(c_single_data) + 1), c_single_data, color=cmap(imouse), label=mouse_name)
        ax[0, 1].plot(range(1, len(c_across_data) + 1), c_across_data, color=cmap(imouse), label=mouse_name)
        ax[1, 0].plot(range(1, len(c_single_data) + 1), np.nancumsum(c_single_data), color=cmap(imouse), label=mouse_name)
        ax[1, 1].plot(range(1, len(c_across_data) + 1), np.nancumsum(c_across_data), color=cmap(imouse), label=mouse_name)

    ax[1, 0].set_xlabel("Dimension")
    ax[1, 1].set_xlabel("Dimension")
    ax[0, 0].set_ylabel("Variance")
    ax[1, 0].set_ylabel("Cumulative Variance")
    ax[0, 0].set_title("Single Environments")
    ax[0, 1].set_title("Across Environments")
    ax[1, 1].legend(fontsize=8)

    if do_xlog:
        for aa in ax:
            for a in aa:
                a.set_xscale("log")

    if do_ylog:
        for aa in ax:
            for a in aa:
                a.set_yscale("log")

    if with_show:
        plt.show()

    if with_save:
        special_name = "logx_" if do_xlog else "linx_"
        special_name = special_name + ("logy" if do_ylog else "liny")
        pcms[0].saveFigure(fig.number, "comparisons", "cv_spectral_average_comparison_" + special_name)


def plot_all_exponential_fits(pcms, spectra_data, with_show=True, with_save=False):
    single_env, across_env = compare_spectral_averages(spectra_data)
    single_amp = []
    single_decay = []
    single_r2 = []
    across_amp = []
    across_decay = []
    across_r2 = []
    for senv, aenv in zip(single_env, across_env):
        # get exponential fits and r2 for single environment spectra
        (a, d), r = helpers.fit_exponentials(np.stack(senv), bias=False)
        single_amp.append(a)
        single_decay.append(d)
        single_r2.append(r)

        # get exponential fits and r2 for across environment spectra
        (a, d), r = helpers.fit_exponentials(np.stack(aenv), bias=False)
        across_amp.append(a)
        across_decay.append(d)
        across_r2.append(r)

    single_mouse_id = np.concatenate([i * np.ones(len(single_env[i])) for i in range(len(single_env))])
    across_mouse_id = np.concatenate([i * np.ones(len(across_env[i])) for i in range(len(across_env))])

    mouse_cmap = mpl.colormaps["Dark2"].resampled(len(single_env))
    r2_cmap = mpl.colormaps["plasma"]

    figdim = 3
    alpha = 0.7
    s = 25

    fig, ax = plt.subplots(2, 2, figsize=(2 * figdim, 2 * figdim), layout="constrained", sharex="col", sharey="col")
    ax[0, 0].scatter(
        np.concatenate(single_decay), np.concatenate(single_amp), s=s, c=single_mouse_id, cmap=mouse_cmap, alpha=alpha, lw=0.5, edgecolor="k"
    )
    ax[0, 1].scatter(
        np.concatenate(single_decay),
        np.concatenate(single_amp),
        s=s,
        c=np.concatenate(single_r2),
        cmap=r2_cmap,
        vmin=0,
        vmax=1,
        alpha=alpha,
        lw=0.5,
        edgecolor="k",
    )

    ax[1, 0].scatter(
        np.concatenate(across_decay), np.concatenate(across_amp), s=s, c=across_mouse_id, cmap=mouse_cmap, alpha=alpha, lw=0.5, edgecolor="k"
    )
    ax[1, 1].scatter(
        np.concatenate(across_decay),
        np.concatenate(across_amp),
        s=s,
        c=np.concatenate(across_r2),
        cmap=r2_cmap,
        vmin=0,
        vmax=1,
        alpha=alpha,
        lw=0.5,
        edgecolor="k",
    )

    ax[1, 0].set_xlabel("Decay")
    ax[1, 1].set_xlabel("Decay")
    ax[0, 0].set_ylabel("Amplitude")
    ax[1, 0].set_ylabel("Amplitude")
    ax[0, 0].set_title("Single Env Spectra")
    ax[0, 1].set_title("Single Env Spectra")
    ax[1, 0].set_title("Across Env Spectra")
    ax[1, 1].set_title("Across Env Spectra")

    for a in ax:
        for _a in a:
            _a.set_yscale("log")

    iax = ax[0, 0].inset_axes([0.6, 0.85, 0.29, 0.07])
    iax.xaxis.set_ticks_position("bottom")
    norm = mpl.colors.Normalize(vmin=0, vmax=len(single_env) - 1)
    m = mpl.cm.ScalarMappable(cmap=mouse_cmap, norm=norm)
    fig.colorbar(m, cax=iax, orientation="horizontal", label="Mouse ID")

    iax = ax[0, 1].inset_axes([0.6, 0.85, 0.29, 0.07])
    iax.xaxis.set_ticks_position("bottom")
    m = mpl.cm.ScalarMappable(cmap=r2_cmap)
    fig.colorbar(m, cax=iax, orientation="horizontal", label="R**2")

    if with_show:
        plt.show()

    if with_save:
        pcms[0].saveFigure(fig.number, "comparisons", "exponential_fit_results")
