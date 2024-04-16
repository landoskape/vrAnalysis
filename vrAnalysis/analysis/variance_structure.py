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
            names = temp_files["names"]
            envstats = temp_files["envstats"]
            cv_by_env_all = temp_files["cv_by_env_all"]
            cv_by_env_rel = temp_files["cv_by_env_rel"]
            cv_across_all = temp_files["cv_across_all"]
            cv_across_rel = temp_files["cv_across_rel"]
            cvf_freqs = temp_files["cvf_freqs"]
            cvf_by_env_all = temp_files["cvf_by_env_all"]
            cvf_by_env_rel = temp_files["cvf_by_env_rel"]
            cvf_by_env_cov_all = temp_files["cvf_by_env_cov_all"]
            cvf_by_env_cov_rel = temp_files["cvf_by_env_cov_rel"]
            rel_mse = temp_files["rel_mse"]
            rel_cor = temp_files["rel_cor"]
            all_pf_mean = temp_files["all_pf_mean"]
            all_pf_var = temp_files["all_pf_var"]
            all_pf_cv = temp_files["all_pf_cv"]
            all_pf_tcv = temp_files["all_pf_tcv"]
            rel_pf_mean = temp_files["rel_pf_mean"]
            rel_pf_var = temp_files["rel_pf_var"]
            rel_pf_cv = temp_files["rel_pf_cv"]
            rel_pf_tcv = temp_files["rel_pf_tcv"]

        except KeyError:
            load_data = True

        if not load_data:
            # check if variables are correct
            for name, v in zip(names, vss):
                if name != v.vrexp.sessionPrint():
                    load_data = True
                    continue

        # check if envstats is correct
        if not load_data and (envstats != pcm.env_stats()):
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
        all_pf_tcv = []
        rel_pf_mean = []
        rel_pf_var = []
        rel_pf_cv = []
        rel_pf_tcv = []
        for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
            # get reliable cells (for each environment) and spkmaps for each environment (with all cells)
            c_idx_reliable = v.get_reliable(envnum=None, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs)
            c_spkmaps = v.prepare_spkmaps(envnum=None, smooth=args.smooth, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs, reliable=False)
            c_rel_spkmaps = [spkmap[cir] for spkmap, cir in zip(c_spkmaps, c_idx_reliable)]

            # get reliable values for each environment
            c_mse, c_cor = v.get_reliability_values(envnum=None, with_test=False)

            # get place field for each cell
            c_placefields_all = [np.mean(spkmap, axis=1) for spkmap in c_spkmaps]
            c_placefields_rel = [np.mean(spkmap[cir], axis=1) for spkmap, cir in zip(c_spkmaps, c_idx_reliable)]
            # make place field a unit vector
            c_all_unitpf = [placefield / np.linalg.norm(placefield, axis=1, keepdims=True) for placefield in c_placefields_all]
            c_rel_unitpf = [placefield / np.linalg.norm(placefield, axis=1, keepdims=True) for placefield in c_placefields_rel]

            # add each to list
            allcell_maps.append(c_spkmaps)
            relcell_maps.append(c_rel_spkmaps)
            rel_mse.append(c_mse)
            rel_cor.append(c_cor)
            all_pf_var.append([np.var(placefield, axis=1) for placefield in c_placefields_all])
            rel_pf_var.append([np.var(placefield, axis=1) for placefield in c_placefields_rel])

            # get other place field statistics
            c_all_pf_mean = [fs.nanmean(placefield, axis=1) for placefield in c_placefields_all]
            c_all_pf_cv = [fs.nanstd(placefield, axis=1) / fs.nanmean(placefield, axis=1) for placefield in c_placefields_all]
            c_all_pf_amplitude = [fs.nansum(np.expand_dims(placefield, 1) * spkmap, axis=2) for placefield, spkmap in zip(c_all_unitpf, c_spkmaps)]
            c_all_pf_tcv = [fs.nanstd(amplitude, axis=1) / fs.nanmean(amplitude, axis=1) for amplitude in c_all_pf_amplitude]

            all_pf_mean.append(c_all_pf_mean)
            all_pf_cv.append(c_all_pf_cv)
            all_pf_tcv.append(c_all_pf_tcv)

            c_rel_pf_mean = [fs.nanmean(placefield, axis=1) for placefield in c_placefields_rel]
            c_rel_pf_cv = [fs.nanstd(placefield, axis=1) / fs.nanmean(placefield, axis=1) for placefield in c_placefields_rel]
            c_rel_pf_amplitude = [
                fs.nansum(np.expand_dims(placefield, 1) * spkmap, axis=2) for placefield, spkmap in zip(c_rel_unitpf, c_rel_spkmaps)
            ]
            c_rel_pf_tcv = [fs.nanstd(amplitude, axis=1) / fs.nanmean(amplitude, axis=1) for amplitude in c_rel_pf_amplitude]

            rel_pf_mean.append(c_rel_pf_mean)
            rel_pf_cv.append(c_rel_pf_cv)
            rel_pf_tcv.append(c_rel_pf_tcv)

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

        if save_as_temp:
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
                "rel_mse": rel_mse,
                "rel_cor": rel_cor,
                "all_pf_mean": all_pf_mean,
                "all_pf_var": all_pf_var,
                "all_pf_cv": all_pf_cv,
                "all_pf_tcv": all_pf_tcv,
                "rel_pf_mean": rel_pf_mean,
                "rel_pf_var": rel_pf_var,
                "rel_pf_cv": rel_pf_cv,
                "rel_pf_tcv": rel_pf_tcv,
            }
            pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")

    else:
        print("Successfully loaded temporary data for variance structure analysis.")

    # return all the variables
    return (
        names,
        envstats,
        cv_by_env_all,
        cv_by_env_rel,
        cv_across_all,
        cv_across_rel,
        cvf_freqs,
        cvf_by_env_all,
        cvf_by_env_rel,
        cvf_by_env_cov_all,
        cvf_by_env_cov_rel,
        rel_mse,
        rel_cor,
        all_pf_mean,
        all_pf_var,
        all_pf_cv,
        all_pf_tcv,
        rel_pf_mean,
        rel_pf_var,
        rel_pf_cv,
        rel_pf_tcv,
    )


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
    names,
    envstats,
    cv_by_env_all,
    cv_by_env_rel,
    cv_across_all,
    cv_across_rel,
    color_by_session=True,
    normalize=False,
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    num_sessions = len(names)
    num_envs = len(envstats)

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
            sesnum_for_env = envstats[env].index(sesnum)
            return cmap(sesnum_for_env)

    figdim = 3
    fig, ax = plt.subplots(2, num_envs + 1, figsize=((num_envs + 1) * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in envstats[c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = cv_by_env_all[j][eidx]
                ax[0, i].plot(range(1, len(cdata) + 1), norm(cdata), color=get_color(c_env, j))

                cdata = cv_by_env_rel[j][eidx]
                ax[1, i].plot(range(1, len(cdata) + 1), norm(cdata), color=get_color(c_env, j))

            ax[0, i].set_title(f"Environment {c_env}")
            ax[0, i].set_ylabel("Eigenspectrum")
            ax[1, i].set_ylabel("Eigenspectrum (reliable)")
            ax[1, i].set_xlabel("Dimension")

        ax[0, i].set_xscale("log")
        ax[1, i].set_xscale("log")

    for j in range(num_sessions):
        cdata = cv_across_all[j]
        ax[0, -1].plot(range(1, len(cdata) + 1), norm(cdata), color=cmap(j))
        cdata = cv_across_rel[j]
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
    cv_by_env_all,
    cv_across_all,
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
    for ii, cc in enumerate(cv_by_env_all):
        for jj, c in enumerate(cc):
            label = "Single Env" if (ii == 0) and (jj == 0) else None
            c_c = copy(c)
            c_c[c_c < ylog_min] = np.nan
            all_be.append(c_c)
            ax[0, 0].plot(range(1, len(c) + 1), norm(c_c), c=("k", 0.3), label=None)
            ax[1, 0].plot(range(1, len(c) + 1), np.cumsum(norm(c_c)), c=("k", 0.3))

    for ii, c in enumerate(cv_across_all):
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
    names,
    envstats,
    cv_by_env_all,
    cv_by_env_rel,
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    num_sessions = len(names)
    num_envs = len(envstats)

    cmap = mpl.colormaps["Set1"].resampled(num_envs)

    # get total energy for each env/session
    var_by_env_all = np.full((num_sessions, num_envs), np.nan)
    var_by_env_rel = np.full((num_sessions, num_envs), np.nan)

    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in envstats[c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = cv_by_env_all[j][eidx]
                var_by_env_all[j, i] = np.sum(cdata)

                cdata = cv_by_env_rel[j][eidx]
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
    names,
    envstats,
    cvf_freqs,
    cvf_by_env_all,
    cvf_by_env_rel,
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
    num_sessions = len(names)
    num_envs = len(envstats)

    cmap = mpl.colormaps["turbo"].resampled(num_sessions)

    def get_color(env, sesnum):
        """helper for getting color based on color method"""
        if color_by_session:
            # color by absolute session number
            return cmap(sesnum)
        else:
            # color by relative session number (within environment)
            sesnum_for_env = envstats[env].index(sesnum)
            return cmap(sesnum_for_env)

    figdim = 3
    fig, ax = plt.subplots(2, num_envs * 2, figsize=(2 * num_envs * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            c_freqs = cvf_freqs[j]
            if ignore_dc:
                c_freqs = c_freqs[1:]
            if j in envstats[c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = cvf_by_env_all[j][eidx]
                if ignore_dc:
                    cdata = [c[1:] for c in cdata]
                ax[0, 2 * i].plot(c_freqs, cdata[0], color=get_color(c_env, j), linestyle="-")
                ax[0, 2 * i + 1].plot(c_freqs, cdata[1], color=get_color(c_env, j), linestyle="--")

                cdata = cvf_by_env_rel[j][eidx]
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


def plot_reliability_data(pcm, names, envstats, rel_mse, rel_cor, color_by_session=True, with_show=True, with_save=False):
    """
    plot fourier data for variance structure analysis
    """

    # make plots of spectra data
    num_sessions = len(names)
    num_envs = len(envstats)

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
            sesnum_for_env = envstats[env].index(sesnum)
            return cmap(sesnum_for_env)

    figdim = 3
    fig, ax = plt.subplots(2, num_envs, figsize=(num_envs * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in envstats[c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = helpers.fractional_histogram(rel_mse[j][eidx], bins=mse_bins)[0]
                ax[0, i].plot(mse_centers, cdata, color=get_color(c_env, j), linestyle="-")
                ax[0, i].axvline(np.mean(rel_mse[j][eidx]), color=get_color(c_env, j))

                cdata = helpers.fractional_histogram(rel_cor[j][eidx], bins=cor_bins)[0]
                ax[1, i].plot(cor_centers, cdata, color=get_color(c_env, j), linestyle="-")
                ax[1, i].axvline(np.mean(rel_cor[j][eidx]), color=get_color(c_env, j))

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


def plot_pf_var_data(pcm, names, envstats, all_pf_var, rel_pf_var, color_by_session=True, with_show=True, with_save=False):
    """
    plot place field variance data for variance structure analysis
    """

    # make plots of spectra data
    num_sessions = len(names)
    num_envs = len(envstats)

    # some plotting supporting variables
    cmap = mpl.colormaps["turbo"].resampled(num_sessions)
    # concatenate a list of lists of np arrays
    min_pf_var = np.nanmin(np.concatenate([np.concatenate(apf) for apf in all_pf_var] + [np.concatenate(rpf) for rpf in rel_pf_var]))
    max_pf_var = np.nanmax(np.concatenate([np.concatenate(apf) for apf in all_pf_var] + [np.concatenate(rpf) for rpf in rel_pf_var]))
    pf_bins = np.linspace(min_pf_var, max_pf_var, 21)
    pf_centers = helpers.edge2center(pf_bins)

    def get_color(env, sesnum):
        """helper for getting color based on color method"""
        if color_by_session:
            # color by absolute session number
            return cmap(sesnum)
        else:
            # color by relative session number (within environment)
            sesnum_for_env = envstats[env].index(sesnum)
            return cmap(sesnum_for_env)

    figdim = 3
    fig, ax = plt.subplots(2, num_envs, figsize=(num_envs * figdim, 2 * figdim), layout="constrained", sharex="row", sharey="row")
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in envstats[c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = helpers.fractional_histogram(all_pf_var[j][eidx], bins=pf_bins)[0]
                ax[0, i].plot(pf_centers, cdata, color=get_color(c_env, j), linestyle="-")

                cdata = helpers.fractional_histogram(rel_pf_var[j][eidx], bins=pf_bins)[0]
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
        data = data / np.nansum(data, axis=1, keepdims=True)
        data[data < ylog_min] = np.nan
        return np.nanmean(data, axis=0)

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
        ax[1, 0].plot(range(1, len(c_single_data) + 1), np.cumsum(c_single_data), color=cmap(imouse), label=mouse_name)
        ax[1, 1].plot(range(1, len(c_across_data) + 1), np.cumsum(c_across_data), color=cmap(imouse), label=mouse_name)

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
