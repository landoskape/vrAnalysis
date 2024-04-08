from tqdm import tqdm
import pickle
import numpy as np
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

    def do_cvpca(self, spkmaps, by_trial=False, nshuff=3):
        """helper for running the full cvPCA gamut on spkmaps"""
        spkmaps = self.filter_nans(spkmaps)
        allspkmaps = self.concatenate_envs(spkmaps)

        # get maximum number of trials / neurons for consistent rank of matrices across environments / all envs
        max_trials = int(self._get_min_trials(spkmaps) // 2)
        max_neurons = int(self._get_min_neurons(spkmaps))

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


def load_spectra_data(pcm, args, save_as_temp=True):
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
    if load_data:
        # first load session data (this can take a while)
        for v in tqdm(vss, leave=True, desc="loading session data"):
            v.load_data()

        # get spkmaps of all cells / just reliable cells
        allcell_maps = []
        relcell_maps = []
        for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
            # get reliable cells (for each environment) and spkmaps for each environment (with all cells)
            c_idx_reliable = v.get_reliable(envnum=None, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs)
            c_spkmaps = v.prepare_spkmaps(envnum=None, smooth=args.smooth, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs, reliable=False)

            # add each to list
            allcell_maps.append(c_spkmaps)
            relcell_maps.append([spkmap[ir] for spkmap, ir in zip(c_spkmaps, c_idx_reliable)])

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
            c_env, c_acc = v.do_cvpca(allmap, by_trial=False)
            cv_by_env_all.append(c_env)
            cv_across_all.append(c_acc)

            # get cvPCA for rel cell spike maps (always do by_trial=False until we have a theory for all trial=True)
            c_env, c_acc = v.do_cvpca(relmap, by_trial=False)
            cv_by_env_rel.append(c_env)
            cv_across_rel.append(c_acc)

            # get cvFOURIER for all/rel cell spike maps using correlation (always do by_trial=False until we have a theory for all trial=True)
            c_freqs, c_all = v.do_cvfourier(allmap, by_trial=False, covariance=False)
            _, c_rel = v.do_cvfourier(relmap, by_trial=False, covariance=False)
            cvf_freqs.append(c_freqs)
            cvf_by_env_all.append(c_all)
            cvf_by_env_rel.append(c_rel)

            # get cvFOURIER for all/rel cell spike maps using covariance (always do by_trial=False until we have a theory for all trial=True)
            c_freqs, c_all = v.do_cvfourier(allmap, by_trial=False, covariance=True)
            _, c_rel = v.do_cvfourier(relmap, by_trial=False, covariance=True)
            cvf_freqs.append(c_freqs)
            cvf_by_env_cov_all.append(c_all)
            cvf_by_env_cov_rel.append(c_rel)

        if save_as_temp:
            # save data as temporary files
            temp_save_args = args if type(args) == dict else args.asdict() if type(args) == helpers.AttributeDict else vars(args)
            temp_files = {
                "names": [v.vrexp.sessionPrint() for v in vss],
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
    )


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
    fig, ax = plt.subplots(2, num_envs + 1, figsize=((num_envs + 1) * figdim, 2 * figdim), layout="constrained", sharex=True, sharey=True)
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


def plot_fourier_data(
    pcm, names, envstats, cvf_freqs, cvf_by_env_all, cvf_by_env_rel, covariance=False, color_by_session=True, with_show=True, with_save=False
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
    fig, ax = plt.subplots(2, num_envs * 2, figsize=(2 * num_envs * figdim, 2 * figdim), layout="constrained", sharex=True, sharey=True)
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            c_freqs = cvf_freqs[j]
            if j in envstats[c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                cdata = cvf_by_env_all[j][eidx]
                ax[0, 2 * i].plot(c_freqs, cdata[0], color=get_color(c_env, j), linestyle="-")
                ax[0, 2 * i + 1].plot(c_freqs, cdata[1], color=get_color(c_env, j), linestyle="--")

                cdata = cvf_by_env_rel[j][eidx]
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
