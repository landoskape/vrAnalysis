from copy import copy
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from .. import helpers
from ..analysis import placeCellSingleSession
from .. import database
from .. import faststats as fs
from dimilibi import Population, SVCA

mousedb = database.vrDatabase("vrMice")

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

    def prepare_spkmaps(self, envnum=None, average=False, trials="full", cutoffs=(0.4, 0.7), maxcutoffs=None, smooth=0.1, reliable=False):
        """prepare spkmaps for cvPCA"""
        # get spkmap with particular smoothing settings (don't smooth because we need all the trials)
        spkmaps = self.get_spkmap(envnum=envnum, average=average, smooth=smooth, trials=trials)

        # Filter by reliable cells if requested
        if reliable:
            idx_reliable = self.get_reliable(envnum=envnum, cutoffs=cutoffs, maxcutoffs=maxcutoffs)
            spkmaps = [spkmap[ir] for spkmap, ir in zip(spkmaps, idx_reliable)]

        # return data
        return spkmaps

    def get_frame_behavior(self, speedThreshold=-1, use_average=True, ignore_to_nan=True, return_speed=False):
        """get behavioral variables for session and convert to frame timing"""
        if return_speed:
            frame_position, idx_valid, ses_envs, frame_speed = self.vrexp.get_position_by_env(
                speedThreshold=speedThreshold, use_average=use_average, return_speed=return_speed
            )
        else:
            frame_position, idx_valid, ses_envs = self.vrexp.get_position_by_env(
                speedThreshold=speedThreshold, use_average=use_average, return_speed=return_speed
            )

        if ignore_to_nan:
            frame_position = frame_position.astype(float)
            frame_position[frame_position == -100] = np.nan
            if return_speed:
                frame_speed = frame_speed.astype(float)
                frame_speed[frame_speed == -100] = np.nan

        msg = f"pcss environments {self.environments} doesn't match session environments {ses_envs}"
        assert np.array_equal(self.environments, ses_envs), msg

        if return_speed:
            return frame_position, idx_valid, frame_speed

        return frame_position, idx_valid

    def generate_spks_prediction(self, spks, spkmaps, frame_position, idx_valid, background_value=0):
        """generate spks prediction from spkmaps and frame_position data"""
        spks_prediction = np.full(spks.shape, background_value, dtype=float)
        max_spkmap_bin = max([sm.shape[1] for sm in spkmaps])
        if np.any(frame_position > max_spkmap_bin):
            raise ValueError("frame_position is greater than the number of bins in the spikemap")
        if np.any(frame_position < 0):
            raise ValueError("frame_position is less than 0")
        if np.any(frame_position == max_spkmap_bin):
            # assume the binning was just a little off and bring it down by 1
            frame_position[frame_position == max_spkmap_bin] = max_spkmap_bin - 1
        for spkmap, frame_pos, idx in zip(spkmaps, frame_position, idx_valid):
            idx_notnan = ~np.isnan(frame_pos)
            spks_prediction[idx & idx_notnan] = spkmap[:, np.floor(frame_pos[idx & idx_notnan]).astype(int)].T
        return spks_prediction

    def get_traversals(self, spks, spks_prediction, spkmaps, frame_position, envidx, cellidx, width=10, fill_nan=False):
        """for every time the mouse passes the peak of the place field, get the traversal of the spks timecourse and the prediction"""
        pos_peak = self.distcenters[np.nanargmax(spkmaps[envidx][cellidx])]
        idx_traversal = find_peaks(-np.abs(frame_position[envidx] - pos_peak), distance=width)[0]

        traversals = np.zeros((len(idx_traversal), width * 2 + 1))
        pred_travs = np.zeros((len(idx_traversal), width * 2 + 1))
        for ii, it in enumerate(idx_traversal):
            istart = it - width
            iend = it + width + 1
            istartoffset = max(0, -istart)
            iendoffset = max(0, iend - spks.shape[0])
            traversals[ii, istartoffset : width * 2 + 1 - iendoffset] = spks[istart + istartoffset : iend - iendoffset, cellidx]
            pred_travs[ii, istartoffset : width * 2 + 1 - iendoffset] = spks_prediction[istart + istartoffset : iend - iendoffset, cellidx]

        if fill_nan:
            traversals[np.isnan(traversals)] = 0.0
            pred_travs[np.isnan(pred_travs)] = 0.0

        return traversals, pred_travs

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


def _catdata(list_of_dicts, key):
    """helper function for concatenating data from a list of dictionaries"""
    data, ises, ienv = helpers.named_transpose(
        [
            helpers.named_transpose(
                [(env_data, ises, ienv) for ises, ses_data in enumerate(each_dict[key]) for ienv, env_data in enumerate(ses_data)],
                map_func=np.array,
            )
            for each_dict in list_of_dicts
        ]
    )
    return data, ises, ienv


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
                "idx_rel",
                "map_corr",
                "map_frob_norm",
                "map_var",
                "pf_gauss_width",
                "pf_gauss_amp",
                "pf_gauss_r2",
                "pf_mean",
                "pf_var",
                "pf_cv",
                "pf_max",
                "pf_tdot_mean",
                "pf_tdot_std",
                "pf_tdot_cv",
                "pf_tcorr_mean",
                "pf_tcorr_std",
                "svca_shared",
                "svca_total",
                "svca_shared_prediction",
                "svca_total_prediction",
                "rank_pf_prediction",
                "cv_by_env_trial",
                "cv_by_env_trial_rdm",
                "cv_by_env_trial_cvrdm",
                "svc_shared_position",
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
        map_corr = []
        map_frob_norm = []
        map_var = []
        rel_mse = []
        rel_cor = []
        idx_rel = []
        pf_gauss_width = []
        pf_gauss_amp = []
        pf_gauss_r2 = []
        pf_mean = []
        pf_var = []
        pf_norm = []
        pf_cv = []
        pf_max = []
        pf_tdot_mean = []
        pf_tdot_std = []
        pf_tdot_cv = []
        pf_tcorr_mean = []
        pf_tcorr_std = []
        kernels = []
        cv_kernels = []
        rank_pf_prediction = []
        for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
            # get reliable cells (for each environment) and spkmaps for each environment (with all cells)
            c_idx_reliable = v.get_reliable(envnum=None, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs)
            c_spkmaps = v.prepare_spkmaps(envnum=None, smooth=args.smooth, cutoffs=args.cutoffs, maxcutoffs=args.maxcutoffs, reliable=False)
            c_rel_spkmaps = [spkmap[cir] for spkmap, cir in zip(c_spkmaps, c_idx_reliable)]

            # record index of reliable cells (for later analyses)
            idx_rel.append(c_idx_reliable)

            # get reliable values for each environment
            c_mse, c_cor = v.get_reliability_values(envnum=None, with_test=False)

            # get place field for each cell
            c_placefields = [np.nanmean(spkmap, axis=1) for spkmap in c_spkmaps]

            # get place field gaussian fit
            amp, _, width, r2 = helpers.named_transpose([helpers.fit_gaussians(cpf, x=v.distcenters) for cpf in c_placefields])
            pf_gauss_width.append(width)
            pf_gauss_amp.append(amp)
            pf_gauss_r2.append(r2)

            # define a train and test index for each environment
            c_train_idx, c_test_idx = helpers.named_transpose([helpers.cvFoldSplit(np.arange(spkmap.shape[1]), 2) for spkmap in c_spkmaps])
            train_maps = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, c_train_idx)]
            test_maps = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, c_test_idx)]
            map_corr.append([helpers.vectorCorrelation(train.reshape(-1), test.reshape(-1)) for train, test in zip(train_maps, test_maps)])
            map_frob_norm.append([np.linalg.norm(cpf, ord="fro") for cpf in c_placefields])
            map_var.append([np.nanvar(cpf.reshape(-1)) for cpf in c_placefields])

            # make place field a unit vector
            nan_norm = lambda x: np.nansum(x**2, axis=1, keepdims=True) ** 0.5
            c_unitpf = [placefield / nan_norm(placefield) for placefield in c_placefields]

            # add each to list
            allcell_maps.append(c_spkmaps)
            relcell_maps.append(c_rel_spkmaps)
            rel_mse.append(c_mse)
            rel_cor.append(c_cor)
            pf_var.append([np.nanvar(placefield, axis=1) for placefield in c_placefields])
            pf_norm.append([np.nansum(placefield**2, axis=1) for placefield in c_placefields])
            pf_max.append([np.nanmax(placefield, axis=1) for placefield in c_placefields])

            # compute spatial kernel matrices
            train_idx, test_idx = helpers.named_transpose([helpers.cvFoldSplit(np.arange(spkmap.shape[1]), 2) for spkmap in c_spkmaps])
            c_pf_train = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, train_idx)]
            c_pf_test = [np.nanmean(spkmap[:, tidx], axis=1) for spkmap, tidx in zip(c_spkmaps, test_idx)]
            c_pf_train_centered = [pf - np.nanmean(pf, axis=0) for pf in c_pf_train]
            c_pf_test_centered = [pf - np.nanmean(pf, axis=0) for pf in c_pf_test]

            kernels.append([np.cov(pf.T) for pf in c_placefields])
            cv_kernels.append([cpftrain.T @ cpftest / (cpftrain.shape[0] - 1) for cpftrain, cpftest in zip(c_pf_train_centered, c_pf_test_centered)])

            # get other place field statistics
            c_pf_mean = [fs.nanmean(placefield, axis=1) for placefield in c_placefields]
            c_pf_cv = [fs.nanstd(placefield, axis=1) / fs.nanmean(placefield, axis=1) for placefield in c_placefields]
            c_pf_amplitude = [fs.nansum(np.expand_dims(placefield, 1) * spkmap, axis=2) for placefield, spkmap in zip(c_unitpf, c_spkmaps)]
            c_pf_tdot_mean = [np.nanmean(amplitude, axis=1) for amplitude in c_pf_amplitude]
            c_pf_tdot_std = [np.nanstd(amplitude, axis=1) for amplitude in c_pf_amplitude]
            c_pf_tdot_cv = [fs.nanstd(amplitude, axis=1) / fs.nanmean(amplitude, axis=1) for amplitude in c_pf_amplitude]

            pf_mean.append(c_pf_mean)
            pf_cv.append(c_pf_cv)
            pf_tdot_mean.append(c_pf_tdot_mean)
            pf_tdot_std.append(c_pf_tdot_std)
            pf_tdot_cv.append(c_pf_tdot_cv)

            # get trial by trial correlation with the mean place field and the trial by trial place field
            c_pf_tcorr = [
                helpers.vectorCorrelation(spkmap, np.repeat(np.expand_dims(placefield, 1), spkmap.shape[1], 1), axis=2, ignore_nan=True)
                for spkmap, placefield in zip(c_spkmaps, c_placefields)
            ]

            pf_tcorr_mean.append([np.nanmean(tcorr, axis=1) for tcorr in c_pf_tcorr])
            pf_tcorr_std.append([np.nanstd(tcorr, axis=1) for tcorr in c_pf_tcorr])

            # Measure rank of pf prediction (as a lookup table)
            c_frame_position, _ = v.get_frame_behavior(use_average=True)
            c_idx_nan = np.all(np.isnan(c_frame_position), axis=0)
            c_frame_position = np.floor(c_frame_position[:, ~c_idx_nan])
            max_pos_bin = max([sm.shape[1] for sm in c_spkmaps])
            if np.any(c_frame_position > max_pos_bin):
                raise ValueError("frame position is greater than spkmap size")
            # fix off by one error (this is just a binning artifact, it's okay to do this)
            # also the last spkmap bin is usually a nan anyway...
            c_frame_position[c_frame_position == max_pos_bin] = max_pos_bin - 1
            c_frame_pos_by_env = [np.unique(cfp[~np.isnan(cfp)]) for cfp in c_frame_position]
            all_lookup_values = np.concatenate([spkmap[:, cfpbe.astype(int)] for spkmap, cfpbe in zip(c_spkmaps, c_frame_pos_by_env)], axis=1)
            idx_nan = np.any(np.isnan(all_lookup_values), axis=0)
            all_lookup_values = all_lookup_values[:, ~idx_nan]
            rank_pf_prediction.append(np.linalg.matrix_rank(all_lookup_values))

        cv_by_env_trial = []
        cv_by_env_trial_rdm = []
        cv_by_env_trial_cvrdm = []
        svc_shared_position = []
        for v in tqdm(vss, leave=False, desc="doing special cvPCA and SVCA analyses on trial position data"):
            c_spkmaps_full = v.get_spkmap(average=False, smooth=args.smooth, trials="full")
            c_spkmaps_full_avg = [np.nanmean(c, axis=1) for c in c_spkmaps_full]

            # Do cvPCA comparison between full spkmap with poisson noise across fake trials and on the actual trials
            c_spkmaps_full = [c[:, np.random.permutation(c.shape[1])] for c in c_spkmaps_full]  # randomize trials for easy splitting
            c_spkmaps_full_train = [c[:, : int(c.shape[1] / 2)] for c in c_spkmaps_full]
            c_spkmaps_full_test = [c[:, int(c.shape[1] / 2) : int(c.shape[1] / 2) * 2] for c in c_spkmaps_full]

            # Find positions with nans
            idx_nans = [
                np.isnan(ctr).any(axis=(0, 1)) | np.isnan(cte).any(axis=(0, 1)) for ctr, cte in zip(c_spkmaps_full_train, c_spkmaps_full_test)
            ]

            # Filter nans
            c_spkmaps_full_train = [ctr[:, :, ~idx_nan] for ctr, idx_nan in zip(c_spkmaps_full_train, idx_nans)]
            c_spkmaps_full_test = [cte[:, :, ~idx_nan] for cte, idx_nan in zip(c_spkmaps_full_test, idx_nans)]
            c_spkmaps_full_avg = [c[:, ~idx_nan] for c, idx_nan in zip(c_spkmaps_full_avg, idx_nans)]
            c_spkmaps_full = [c[:, :, ~idx_nan] for c, idx_nan in zip(c_spkmaps_full, idx_nans)]

            # Generate random samples from poisson distribution with means set by average and number of trials equivalent to measured
            c_spkmaps_full_train_rdm = [
                np.moveaxis(np.random.poisson(np.max(c, 0), [ctr.shape[1], *c.shape]), 0, 1)
                for c, ctr in zip(c_spkmaps_full_avg, c_spkmaps_full_train)
            ]
            c_spkmaps_full_test_rdm = [
                np.moveaxis(np.random.poisson(np.max(c, 0), [cte.shape[1], *c.shape]), 0, 1)
                for c, cte in zip(c_spkmaps_full_avg, c_spkmaps_full_test)
            ]

            # Get average of these particular train/test split
            c_spkmaps_full_train_avg = [np.nanmean(c, axis=1) for c in c_spkmaps_full_train]
            c_spkmaps_full_test_avg = [np.nanmean(c, axis=1) for c in c_spkmaps_full_test]

            # Generate cross-validated samples from averages on train/test
            c_spkmaps_full_train_avg_rdm = [
                np.moveaxis(np.random.poisson(np.max(c, 0), [ctr.shape[1], *c.shape]), 0, 1)
                for c, ctr in zip(c_spkmaps_full_train_avg, c_spkmaps_full_train)
            ]
            c_spkmaps_full_test_avg_rdm = [
                np.moveaxis(np.random.poisson(np.max(c, 0), [cte.shape[1], *c.shape]), 0, 1)
                for c, cte in zip(c_spkmaps_full_test_avg, c_spkmaps_full_test)
            ]

            # Flatten along positions and trials
            c_spkmaps_full_train_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_train_rdm]
            c_spkmaps_full_test_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_test_rdm]
            c_spkmaps_full_train = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_train]
            c_spkmaps_full_test = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_test]
            c_spkmaps_full_train_avg_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_train_avg_rdm]
            c_spkmaps_full_test_avg_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_test_avg_rdm]

            # perform cvpca on full spkmap with poisson noise
            s_rdm = [
                np.nanmean(helpers.shuff_cvPCA(csftr.T, csfte.T, nshuff=5, cvmethod=helpers.cvPCA_paper_neurons), axis=0)
                for csftr, csfte in zip(c_spkmaps_full_train_rdm, c_spkmaps_full_test_rdm)
            ]
            s_trial = [
                np.nanmean(helpers.shuff_cvPCA(csftr.T, csfte.T, nshuff=5, cvmethod=helpers.cvPCA_paper_neurons), axis=0)
                for csftr, csfte in zip(c_spkmaps_full_train, c_spkmaps_full_test)
            ]
            s_cvrdm = [
                np.nanmean(helpers.shuff_cvPCA(csftr.T, csfte.T, nshuff=5, cvmethod=helpers.cvPCA_paper_neurons), axis=0)
                for csftr, csfte in zip(c_spkmaps_full_train_avg_rdm, c_spkmaps_full_test_avg_rdm)
            ]

            # Also do SVCA on the full spkmap across trials
            c_spkmaps_full_rs = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full]

            # Create population
            time_split_prms = dict(
                num_groups=2,
                chunks_per_group=-5,
                num_buffer=1,
            )
            npops = [Population(c, time_split_prms=time_split_prms) for c in c_spkmaps_full_rs]

            # Split population
            train_source, train_target = helpers.named_transpose(
                [npop.get_split_data(0, center=False, scale=True, scale_type="preserve") for npop in npops]
            )
            test_source, test_target = helpers.named_transpose(
                [npop.get_split_data(1, center=False, scale=True, scale_type="preserve") for npop in npops]
            )

            # Fit SVCA on position averaged data across trials...
            svca_position = [SVCA().fit(ts, tt) for ts, tt in zip(train_source, train_target)]
            svc_shared_position = [sv.score(ts, tt)[0].numpy() for sv, ts, tt in zip(svca_position, test_source, test_target)]

            cv_by_env_trial.append(s_trial)
            cv_by_env_trial_rdm.append(s_rdm)
            cv_by_env_trial_cvrdm.append(s_cvrdm)
            svc_shared_position.append(svc_shared_position)

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
            idx_use_roi = v.get_plane_idx(keep_planes=[1, 2, 3, 4])
            idx_rois.append(idx_use_roi)
        ospks = [v.vrexp.loadone("mpci.roiActivityDeconvolvedOasis")[:, idx] for v, idx in zip(vss, idx_rois)]

        # get spkmaps of all cells / just reliable cells
        svca_shared = []
        svca_total = []
        for ospk in tqdm(ospks, leave=False, desc="doing SVCA"):
            c_shared_var, c_tot_cov_space_var = helpers.split_and_svca(ospk.T, verbose=False)[:2]
            svca_shared.append(c_shared_var)
            svca_total.append(c_tot_cov_space_var)

        # measure the "dimensionality" of the place field lookup by the dimensionality of the spkmaps...
        svca_shared_prediction = []
        svca_total_prediction = []
        svca_shared_prediction_cv = []
        svca_total_prediction_cv = []
        for ii, v in enumerate(tqdm(vss, leave=True, desc="creating place field lookup")):
            c_spks = v.prepare_spks()
            c_spkmaps = v.get_spkmap(average=True, smooth=args.smooth)
            c_frame_position, c_idx_valid = v.get_frame_behavior(use_average=True)
            c_pf_prediction = v.generate_spks_prediction(c_spks, c_spkmaps, c_frame_position, c_idx_valid, background_value=0.0)
            idx_nan = np.isnan(c_pf_prediction).any(axis=1)
            c_pf_prediction[idx_nan] = 0.0

            time_split_prms = dict(
                num_groups=2,
                chunks_per_group=-2,
                num_buffer=2,
            )
            npop = Population(c_pf_prediction.T, time_split_prms=time_split_prms)
            train_source_pred, train_target_pred = npop.get_split_data(0, center=True, scale=False)
            test_source_pred, test_target_pred = npop.get_split_data(1, center=True, scale=False)

            svca = SVCA(num_components=temp_files["rank_pf_prediction"][ii], truncated=True).fit(train_source_pred, train_target_pred)
            pf_shared_var, pf_total_var = svca.score(test_source_pred, test_target_pred, normalize=True)
            svca_shared_prediction.append(pf_shared_var)
            svca_total_prediction.append(pf_total_var)

            # Then also do a similar thing but with cross-validated data
            # (This isn't perfect per se, but is perfectly well cross-validated)
            # Ideally I'd probably pick train/test timepoints, then build a spkmap from those samples specifically
            # Instead, I'm just building two spike maps from train/test trials, and then applying them to the
            # train/test timepoints for the SVCA analysis. Not "perfect" but cross-validated in terms of the spikemaps.
            c_spkmaps_train = v.get_spkmap(average=True, smooth=args.smooth, trials="train")
            c_spkmaps_test = v.get_spkmap(average=True, smooth=args.smooth, trials="test")
            c_frame_position, c_idx_valid = v.get_frame_behavior(use_average=True)
            c_pf_pred_train = v.generate_spks_prediction(c_spks, c_spkmaps_train, c_frame_position, c_idx_valid, background_value=0.0)
            c_pf_pred_test = v.generate_spks_prediction(c_spks, c_spkmaps_test, c_frame_position, c_idx_valid, background_value=0.0)
            idx_nan = np.isnan(c_pf_pred_train).any(axis=1) | np.isnan(c_pf_pred_test).any(axis=1)
            c_pf_pred_train[idx_nan] = 0.0
            c_pf_pred_test[idx_nan] = 0.0

            time_split_prms = dict(
                num_groups=2,
                chunks_per_group=-2,
                num_buffer=2,
            )
            npop = Population(c_pf_pred_train.T, time_split_prms=time_split_prms)
            train_source_pred = npop.apply_split(c_pf_pred_train.T, 0, center=True, scale=False)[npop.cell_split_indices[0]]
            train_target_pred = npop.apply_split(c_pf_pred_train.T, 0, center=True, scale=False)[npop.cell_split_indices[1]]
            test_source_pred = npop.apply_split(c_pf_pred_test.T, 1, center=True, scale=False)[npop.cell_split_indices[0]]
            test_target_pred = npop.apply_split(c_pf_pred_test.T, 1, center=True, scale=False)[npop.cell_split_indices[1]]

            svca_cv = SVCA(num_components=temp_files["rank_pf_prediction"][ii], truncated=True).fit(train_source_pred, train_target_pred)
            pf_shared_var_cv, pf_total_var_cv = svca_cv.score(test_source_pred, test_target_pred, normalize=True)
            svca_shared_prediction_cv.append(pf_shared_var_cv)
            svca_total_prediction_cv.append(pf_total_var_cv)

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
            "cv_by_env_trial": cv_by_env_trial,
            "cv_by_env_trial_rdm": cv_by_env_trial_rdm,
            "cv_by_env_trial_cvrdm": cv_by_env_trial_cvrdm,
            "kernels": kernels,
            "cv_kernels": cv_kernels,
            "map_corr": map_corr,
            "map_frob_norm": map_frob_norm,
            "map_var": map_var,
            "rel_mse": rel_mse,
            "rel_cor": rel_cor,
            "idx_rel": idx_rel,
            "pf_gauss_width": pf_gauss_width,
            "pf_gauss_amp": pf_gauss_amp,
            "pf_gauss_r2": pf_gauss_r2,
            "pf_mean": pf_mean,
            "pf_var": pf_var,
            "pf_cv": pf_cv,
            "pf_max": pf_max,
            "pf_tdot_mean": pf_tdot_mean,
            "pf_tdot_std": pf_tdot_std,
            "pf_tdot_cv": pf_tdot_cv,
            "pf_tcorr_mean": pf_tcorr_mean,
            "pf_tcorr_std": pf_tcorr_std,
            "svca_shared": svca_shared,
            "svca_total": svca_total,
            "svca_shared_prediction": svca_shared_prediction,
            "svca_total_prediction": svca_total_prediction,
            "svca_shared_prediction_cv": svca_shared_prediction_cv,
            "svca_total_prediction_cv": svca_total_prediction_cv,
            "svca_shared_position": svc_shared_position,
            "rank_pf_prediction": rank_pf_prediction,
        }
        if save_as_temp:
            pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")

    else:
        print("Successfully loaded temporary data for variance structure analysis.")

    # return spectra data
    return temp_files


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
    # pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")


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


def generate_example_pfvars(pcm, spectra_data, ises, envnum, num_cells=20, with_show=True, with_save=False):
    """some example plots of different cells and their place field statistics"""
    if ises < 0:
        ises = len(pcm.pcss) + ises

    envidx = pcm.pcss[ises].envnum_to_idx(envnum)[0]
    pfvars = [
        "rel_mse",
        "rel_cor",
        "pf_mean",
        "pf_var",
        "pf_norm",
        "pf_cv",
        "pf_max",
        "pf_tdot_mean",
        "pf_tdot_std",
        "pf_tdot_cv",
        "pf_tcorr_mean",
        "pf_tcorr_std",
    ]
    num_vars = len(pfvars)
    get_vpos = lambda ii: np.linspace(0, 1, num_vars + 4)[num_vars - ii]  # extra position for "legend"

    # get normalizing values for each variable
    all_pfvars = {var: np.concatenate([envvar for sesvar in spectra_data[var] for envvar in sesvar]) for var in pfvars}
    pfvars_mean = {var: np.nanmean(all_pfvars[var]) for var in pfvars}
    pfvars_std = {var: np.nanstd(all_pfvars[var]) for var in pfvars}

    cmap = mpl.colormaps["hot"]
    cmap.set_bad("black")
    spkmap = pcm.pcss[ises].get_spkmap(envnum=envnum, average=False, smooth=0.1)[0]
    vmin = 0
    vmax = np.nanmax(np.nanmean(spkmap, axis=1))
    icells = np.random.choice(spkmap.shape[0], num_cells // 2, replace=False)  # get some random cells
    idx_pf_var_high = np.argsort(spectra_data["pf_var"][ises][envidx])[-num_cells * 4 :]
    icells = np.concatenate([icells, np.random.choice(idx_pf_var_high, num_cells - (num_cells // 2), replace=False)])  # get some more random cells
    for ic in icells:
        # A figure with a heatmap in the top left, a small line plot below it, and text to the right
        fig = plt.figure(layout="constrained")
        gs = fig.add_gridspec(nrows=4, ncols=3, left=0.05, right=0.95, hspace=0.1, wspace=0.05)
        ax_heatmap = fig.add_subplot(gs[:-1, :-1])
        ax_curve = fig.add_subplot(gs[-1, :-1])
        ax_text = fig.add_subplot(gs[:, -1])
        helpers.clear_axis(ax_text)

        fig.suptitle(f"Session: {pcm.pcss[ises].vrexp.sessionPrint()} - Cell: {ic} - Env: {envnum}")

        # plot heatmap of spikemap
        extent = [pcm.pcss[ises].distedges[0], pcm.pcss[ises].distedges[-1], 0, spkmap.shape[1]]
        ax_heatmap.imshow(spkmap[ic], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=extent, interpolation="none")
        ax_heatmap.set_xlabel("Virtual Position (cm)")
        ax_heatmap.set_ylabel("Trials")

        # plot place field average across trials
        ax_curve.plot(pcm.pcss[ises].distcenters, np.nanmean(spkmap[ic], axis=0), color="k", linewidth=2)
        ax_curve.set_xlabel("Virtual Position (cm)")
        ax_curve.set_ylabel("dec. [Ca]")
        ax_curve.set_ylim(0, vmax)

        # plot text of place field statistics
        for ii, var in enumerate(pfvars):
            c_value = spectra_data[var][ises][envidx][ic]
            c_norm_value = (c_value - pfvars_mean[var]) / pfvars_std[var]
            ax_text.text(0.5, get_vpos(ii), f"{var}: {c_value:.3f}/{c_norm_value:.3f}", ha="center", va="center")
        ax_text.text(0.5, get_vpos(-2), f"variable: z-scored value", ha="center", va="center", fontstyle="italic")

        if with_show:
            plt.show()

        if with_save:
            folder = Path(f"example_ROIs_pfvars_env{envnum}")
            special_name = f"ses{ises}_env{envnum}_cell{ic}"
            pcm.saveFigure(fig.number, pcm.track.mouse_name, folder / special_name)


def compare_reliability_measures(pcm, spectra_data, ises, envnum, with_show=True, with_save=False):
    if ises < 0:
        ises = len(pcm.pcss) + ises
    ymin = -6
    envidx = pcm.pcss[ises].envnum_to_idx(envnum)[0]
    c_relmse = spectra_data["rel_mse"][ises][envidx]
    c_relcor = spectra_data["rel_cor"][ises][envidx]
    idx_keep = c_relmse > ymin
    fraction_keep = np.sum(idx_keep) / len(c_relmse)
    c_relmse = c_relmse[idx_keep]
    c_relcor = c_relcor[idx_keep]
    grid = sns.jointplot(x=c_relcor, y=c_relmse, color=("k", 0.1), size=8, edgecolor=None)
    grid.set_axis_labels("Rel-Corr", f"Rel-MSE ({100*fraction_keep:.2f}%>{ymin})", fontsize=12)
    grid.figure.tight_layout()
    plt.legend([], [], frameon=False)

    if with_show:
        plt.show()

    if with_save:
        folder = Path(f"reliability_comparison_env{envnum}")
        special_name = f"ses{ises}_env{envnum}_relcomparison"
        pcm.saveFigure(plt.gcf().number, pcm.track.mouse_name, folder / special_name)


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
            ax[0, i].set_xlabel("Dimension")
            ax[1, i].set_xlabel("Dimension")

        # make inset colorbar for session id
        iax = ax[0, i].inset_axes([0.6, 0.85, 0.29, 0.07])
        iax.xaxis.set_ticks_position("bottom")
        cmap_norm = mpl.colors.Normalize(vmin=0, vmax=num_sessions - 1)
        m = mpl.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
        fig.colorbar(m, cax=iax, orientation="horizontal", label="Session ID" + ("\n(relative for env)" if not color_by_session else ""))

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


def plot_spectral_data_posterversion(
    pcm,
    spectra_data,
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    environments = list(spectra_data["envstats"].keys())
    environments = [e for e in environments if e > 0]
    num_envs = len(environments)
    sessions_per_env = [len(spectra_data["envstats"][env]) for env in environments]

    # sort environments by number of sessions (decreasing)
    env_order = [x for _, x in sorted(zip(sessions_per_env, environments), reverse=True)]

    cmap = mpl.colormaps["winter"].resampled(num_sessions)
    plt.rcParams.update({"font.size": 24})
    max_dim = 0
    min_var = np.inf
    max_var = 0
    do_ylog = False

    # total_variance = np.full((num_sessions, num_envs), np.nan)
    total_variance = [np.full(num_sessions, np.nan) for _ in range(num_envs)]

    width_ratios = [1] * num_envs + [0.1] + [1]
    fig, ax = plt.subplots(1, num_envs + 2, figsize=(16, 5), width_ratios=width_ratios, layout="constrained")
    for i in range(num_envs):
        c_env = env_order[i]
        for j in range(num_sessions):
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]
                cdata = copy(spectra_data["cv_by_env_all"][j][eidx])
                cdata_dim = np.where(cdata < 0)[0][0]
                cdata[cdata_dim:] = np.nan
                max_dim = max(max_dim, cdata_dim)
                min_var = min(min_var, np.nanmin(cdata))
                max_var = max(max_var, np.nanmax(cdata))
                ax[i].plot(range(1, len(cdata) + 1), cdata, color=cmap(j))
                total_variance[i][j] = np.nansum(cdata)

        ax[i].set_xlabel("Dimension")
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        if do_ylog:
            ax[i].set_yscale("log")

        if i == 0:
            ax[i].set_ylabel("Variance (au)")

        if i == num_envs - 1:
            # make inset colorbar for session id
            # iax = ax[i].inset_axes([-0.5, 0.65, 0.8, 0.07])
            # iax = fig.add_axes([0.05, 0.25, 0.25, 0.05])
            # iax.xaxis.set_ticks_position("bottom")
            cmap_norm = mpl.colors.Normalize(vmin=0, vmax=num_sessions - 1)
            m = mpl.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
            cb = fig.colorbar(m, cax=ax[-2], ticks=[0, num_sessions - 1], orientation="vertical")
            cb.ax.set_ylabel("Session ID", ha="center", labelpad=-90)

    max_total_variance = np.nanmax(np.concatenate(total_variance))
    # total_variance = total_variance / np.nanmax(np.concatenate(total_variance))
    total_variance = [tv / max_total_variance for tv in total_variance]
    env_cols = "krb"
    for i in range(num_envs):
        ax[-1].plot(range(num_sessions), total_variance[i], color=env_cols[i], marker=".", markersize=12)
    # for j in range(num_sessions):
    #     ax[-2].plot(range(num_envs), total_variance[j], color=cmap(j), marker=".", markersize=12)
    ax[-1].set_xlabel("Session #")
    ax[-1].set_ylabel("Relative Total Var.")
    ax[-1].set_xlim(-0.5, num_sessions - 0.5)
    ax[-1].set_ylim(0, 1.1)
    ax[-1].set_xticks([0, num_sessions - 1])
    ax[-1].set_yticks([0, 1])
    ax[-1].spines["top"].set_visible(False)
    ax[-1].spines["right"].set_visible(False)

    yticks = ax[0].get_yticks()
    ax[0].set_yticks(ticks=yticks, labels=[""] * len(yticks))

    for i in range(num_envs):
        ax[i].set_xlim(0, max_dim - 1)
        if do_ylog:
            ax[i].set_ylim(min_var * 0.95, max_var * 1.05)
        else:
            ax[i].set_ylim(0, max_var * 1.1)
        ax[i].text((max_dim - 1) / 2, max_var, f"Env {i}", ha="center", va="top", fontsize=24)

    for i in range(1, num_envs):
        ax[i].spines["left"].set_visible(False)
        ax[i].set_yticks([])
        ax[i].minorticks_off()

    if with_show:
        plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcm.saveDirectory("example_plots")
        save_path = save_directory / f"increase_familiarity_flip_{pcm.track.mouse_name}"
        helpers.save_figure(fig, save_path)

    if with_save:
        special_name = "by_session"
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "cv_spectra_" + special_name + "_posterversion")


def plot_pfstat_data_posterversion(
    pcm,
    spectra_data,
    metrics=["pf_norm", "pf_tcorr_mean"],
    reductions=["mean", "mean"],
    fancy_names=["PF Amplitude", "PF Consistency"],
    with_show=True,
    with_save=False,
):
    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    environments = list(spectra_data["envstats"].keys())
    environments = [e for e in environments if e > 0]
    num_envs = len(environments)
    sessions_per_env = [len(spectra_data["envstats"][env]) for env in environments]

    # sort environments by number of sessions (decreasing)
    env_order = [x for _, x in sorted(zip(sessions_per_env, environments), reverse=True)]

    cmap = mpl.colormaps["winter"].resampled(num_sessions)
    plt.rcParams.update({"font.size": 24})
    figdim = 5

    reduced_variables = [np.full((num_sessions, num_envs), np.nan) for _ in metrics]

    for imetric, metric in enumerate(metrics):
        for i in range(num_envs):
            c_env = env_order[i]
            for j in range(num_sessions):
                if j in spectra_data["envstats"][c_env]:
                    eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]
                    cdata = copy(spectra_data[metric][j][eidx])
                    if reductions[imetric] == "mean":
                        c_red_data = np.nanmean(cdata)
                    elif reductions[imetric] == "std":
                        c_red_data = np.nanstd(cdata)
                    else:
                        raise ValueError("reduction must be 'mean' or 'std'")
                    reduced_variables[imetric][j, i] = c_red_data

    fig, ax = plt.subplots(1, len(metrics), figsize=(0.765 * len(metrics) * figdim, figdim), layout="constrained")
    reduced_variables = [reduced_variable / np.nanmax(reduced_variable) for reduced_variable in reduced_variables]
    for imetric in range(len(metrics)):
        for j in range(num_sessions):
            ax[imetric].plot(range(num_envs), reduced_variables[imetric][j], color=cmap(j), marker=".", markersize=12)
        ax[imetric].set_xlabel("Env #")
        ax[imetric].set_ylabel(f"Relative {metrics[imetric] if fancy_names[imetric] is None else fancy_names[imetric]}")
        ax[imetric].set_xlim(-0.5, num_envs - 0.5)
        ax[imetric].set_ylim(0)
        ax[imetric].set_xticks(range(num_envs))
        ax[imetric].set_yticks([0, 1])
        ax[imetric].spines["top"].set_visible(False)
        ax[imetric].spines["right"].set_visible(False)

    # make inset colorbar for session id
    iax = ax[-1].inset_axes([0.1, 0.15, 0.8, 0.03])
    iax.xaxis.set_ticks_position("bottom")
    cmap_norm = mpl.colors.Normalize(vmin=0, vmax=num_sessions - 1)
    m = mpl.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
    cb = fig.colorbar(m, cax=iax, ticks=[0, num_sessions - 1], orientation="horizontal")
    cb.ax.set_xlabel("Session ID", ha="center", labelpad=-20)

    if with_show:
        plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcm.saveDirectory("example_plots")
        metric_names = "_".join(metrics)
        save_path = save_directory / f"increase_familiarity_{pcm.track.mouse_name}_{metric_names}"
        helpers.save_figure(fig, save_path)

    if with_save:
        special_name = "by_session"
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "cv_spectra_" + special_name + "_posterversion")

    # By session on x axis
    reduced_variables = [[np.full(num_sessions, np.nan) for _ in range(num_envs)] for _ in metrics]

    for imetric, metric in enumerate(metrics):
        for i in range(num_envs):
            c_env = env_order[i]
            for j in range(num_sessions):
                if j in spectra_data["envstats"][c_env]:
                    eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]
                    cdata = copy(spectra_data[metric][j][eidx])
                    if reductions[imetric] == "mean":
                        c_red_data = np.nanmean(cdata)
                    elif reductions[imetric] == "std":
                        c_red_data = np.nanstd(cdata)
                    else:
                        raise ValueError("reduction must be 'mean' or 'std'")
                    reduced_variables[imetric][i][j] = c_red_data

    env_cols = "krb"
    fig, ax = plt.subplots(1, len(metrics), figsize=(0.765 * len(metrics) * figdim, figdim), layout="constrained")
    reduced_variables = [reduced_variable / np.nanmax(reduced_variable) for reduced_variable in reduced_variables]
    for imetric in range(len(metrics)):
        for i in range(num_envs):
            ax[imetric].plot(range(num_sessions), reduced_variables[imetric][i], color=env_cols[i], marker=".", markersize=12)
        ax[imetric].set_xlabel("Session #")
        ax[imetric].set_ylabel(f"Relative {metrics[imetric] if fancy_names[imetric] is None else fancy_names[imetric]}")
        ax[imetric].set_xlim(-0.5, num_sessions - 0.5)
        ax[imetric].set_ylim(0)
        ax[imetric].set_xticks([0, num_sessions - 1])
        ax[imetric].set_yticks([0, 1])
        ax[imetric].spines["top"].set_visible(False)
        ax[imetric].spines["right"].set_visible(False)

    if with_show:
        plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcm.saveDirectory("example_plots")
        metric_names = "_".join(metrics)
        save_path = save_directory / f"increase_familiarity_flip_{pcm.track.mouse_name}_{metric_names}"
        helpers.save_figure(fig, save_path)

    if with_save:
        special_name = "by_session"
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "cv_spectra_" + special_name + "_flip_posterversion")


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


def get_total_variance(spectra_data):
    """Helper method for getting the total stimulus-related variance from spectra data in the standard format"""
    single_stim_variance = []
    across_stim_variance = []
    for c_by_env, c_across in zip(spectra_data["cv_by_env_all"], spectra_data["cv_across_all"]):
        c_single_stim_variance = np.sum(np.stack(c_by_env), axis=1)
        c_across_stim_variance = np.sum(c_across)
        single_stim_variance.append(c_single_stim_variance)
        across_stim_variance.append(c_across_stim_variance)

    data = dict(
        single_stim_variance=single_stim_variance,
        across_stim_variance=across_stim_variance,
    )
    return data


def get_exp_fits(spectra_data):
    """Helper method for getting the exponential fits from spectra data in the standard format"""
    num_sessions = len(spectra_data["names"])

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

    fit_data = dict(
        single_amplitude=single_amplitude,
        single_decay=single_decay,
        single_r2=single_r2,
        across_amplitude=across_amplitude,
        across_decay=across_decay,
        across_r2=across_r2,
    )

    return fit_data


def get_pf_summary(pcm, spectra_data, mean=True, reliable=False):
    """Helper method for getting the summarized place field variables in the spectra data in the standard format"""

    # names of place-field related variables to compare with exponential fit data
    relvars = ["rel_mse", "rel_cor"]  # these are always used
    # these need either "all" or "rel" appended to the front
    pfvars = [
        "pf_mean",
        "pf_var",
        "pf_norm",
        "pf_cv",
        "pf_max",
        "pf_tdot_mean",
        "pf_tdot_std",
        "pf_tdot_cv",
        "pf_tcorr_mean",
        "pf_tcorr_std",
    ]

    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    # summary function
    sum_func = np.nanmean if mean else np.nanstd

    allvars = relvars + pfvars
    var_dict = {cvar: [] for cvar in allvars}

    # get all the variables and append them with the usual structure
    for cvar in allvars:
        # go through each variable
        for j in range(num_sessions):
            # make a list for the variable summary across each environment in this session
            cc = []
            for i in range(num_envs):
                # check if this session has data for this environment
                c_env = pcm.environments[i]
                if j in spectra_data["envstats"][c_env]:
                    # if it does, then summarize the current variable, check nans, and add it to the session list
                    eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]
                    cdata = sum_func(spectra_data[cvar][j][eidx])
                    if np.all(np.isnan(spectra_data[cvar][j][eidx])):
                        print(f"Warning: {cvar} is all nan for {pcm.track.mouse_name} {c_env} {j}")
                    cc.append(cdata)
            # append this sessions data to vardict
            var_dict[cvar].append(cc)

    return var_dict


def compare_exp_fits(pcm, spectra_data, amplitude=True, mean=True, color_by_session=True, with_show=True, with_save=False):
    """
    First make exponential fits of the eigenspectra, then use place field and other properties to predict the fit parameters
    """
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    fit_data = get_exp_fits(spectra_data)
    pfv_data = get_pf_summary(pcm, spectra_data, mean=mean, reliable=False)

    # names of place-field related variables to compare with exponential fit data
    use_vars = [
        "rel_cor",
        "pf_mean",
        "pf_var",
        "pf_norm",
        "pf_cv",
        "pf_max",
        "pf_tdot_mean",
        "pf_tdot_std",
        "pf_tdot_cv",
        "pf_tcorr_mean",
        "pf_tcorr_std",
    ]
    num_vars = len(use_vars)

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
    single = fit_data["single_amplitude"] if amplitude else fit_data["single_decay"]

    # make the plot
    figdim = 1.5
    fig, ax = plt.subplots(num_envs, num_vars, figsize=(num_vars * figdim, num_envs * figdim), layout="constrained", sharex="col", sharey=True)
    for i in range(num_envs):
        c_env = pcm.environments[i]
        for j in range(num_sessions):
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]
                for ipf, c_var in enumerate(use_vars):
                    ax[i, ipf].scatter(pfv_data[c_var][j][eidx], single[j][eidx], color=get_color(c_env, j), marker=".")
                    if ipf == 0:
                        ax[i, ipf].set_ylabel(f"Environment {c_env}\n{'Amplitude' if amplitude else 'Decay'}")
                    else:
                        ax[i, ipf].set_ylabel("Amplitude" if amplitude else "Decay")
                    if i == 0:
                        ax[i, ipf].set_title(c_var)
                    ax[i, ipf].set_xlabel(c_var)
                    ax[i, ipf].set_yscale("linear")
                    if c_var == "rel_mse":
                        ax[i, ipf].set_xlim(-4, 1)

    ax[0, 0].set_ylim(bottom=0)

    if with_show:
        plt.show()

    if with_save:
        special_name = "amplitude" if amplitude else "decay"
        special_name = special_name + ("_mean" if mean else "_std")
        special_name = special_name + ("_by_session" if color_by_session else "_by_relative_session")
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "comparison_pfvars_eigenspectra_" + special_name)


def plot_spatial_kernels(pcm, spectra_data, cv=False, rewzone=True, with_show=True, with_save=False):
    """
    plot spatial kernels for variance structure analysis
    """

    # make plots of spectra data
    num_sessions = len(spectra_data["names"])
    num_envs = len(spectra_data["envstats"])

    cmap = mpl.colormaps["inferno"]

    kernel_key = "cv_kernels" if cv else "kernels"
    kernel_opposite = "kernels" if cv else "cv_kernels"

    figdim = 2
    fig, ax = plt.subplots(num_envs, num_sessions, figsize=(num_sessions * figdim, num_envs * figdim), layout="constrained", sharex=True, sharey=True)
    for i in range(num_envs):
        c_env = pcm.environments[i]
        made_ylabel = False
        for j in range(num_sessions):
            rewpos, rewhw = helpers.environmentRewardZone(pcm.pcss[j].vrexp)
            if j in spectra_data["envstats"][c_env]:
                eidx = pcm.pcss[j].envnum_to_idx(c_env)[0]

                rewloc = rewpos[eidx] - rewhw[eidx]

                cdata = spectra_data[kernel_key][j][eidx]
                codata = spectra_data[kernel_opposite][j][eidx]
                vmin, vmax = np.nanpercentile(cdata, 2), np.nanpercentile(cdata, 98)
                vmino, vmaxo = np.nanpercentile(codata, 2), np.nanpercentile(codata, 98)
                vmin, vmax = min(vmin, vmino), max(vmax, vmaxo)

                cim = ax[i, j].imshow(cdata, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")

                # plot vertical & horizontal lines at reward zone
                if rewzone:
                    ax[i, j].axvline(rewloc, color="w", linestyle="--", alpha=0.5)
                    ax[i, j].axhline(rewloc, color="w", linestyle="--", alpha=0.5)
                    ax[i, j].text(0.99, 0.99, "-- reward zone", color="w", fontsize=6, ha="right", va="top", transform=ax[i, j].transAxes)
                ax[i, j].set_title(f"Session {j}")
                if not made_ylabel:
                    ax[i, j].set_ylabel(f"Environment {c_env}\n\nSpatial Position")
                    made_ylabel = True
                if i == num_envs - 1:
                    ax[i, j].set_xlabel("Spatial Position")

                # make inset colorbar
                axins = ax[i, j].inset_axes([0.1, 0.15, 0.3, 0.05])
                cb = fig.colorbar(cim, cax=axins, orientation="horizontal")
                cb.set_ticks([vmin, vmax])
                cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"], color="w", fontsize=8)

            else:
                # clear axis
                ax[i, j].axis("off")

    if with_show:
        plt.show()

    if with_save:
        special_name = "cv" if cv else "noncv"
        pcm.saveFigure(fig.number, pcm.track.mouse_name, "kernels_" + special_name)


# =================================== code for comparing spectral data across mice =================================== #
def compare_spectral_averages(summary_dicts, return_extras=False):
    """
    get average spectral data for each mouse (specifically the average eigenspectra for single / all environments)

    summary dicts is a list of dictionaries containing full spectra data for each mouse to be compared
    returns a list of lists, where the outer list corresponds to each mouse (e.g. each summary_dict)
    and the inner list contains the eigenspectra for each session / environment
    """
    # get average eigenspectra for each mouse
    single_env = []
    across_env = []

    if return_extras:
        single_env_trial = []
        single_env_trial_rdm = []
        single_env_trial_cvrdm = []

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

        if return_extras:
            all_single_env_trial = []
            all_single_env_trial_rdm = []
            all_single_env_trial_cvrdm = []
            for cc in summary_dict["cv_by_env_trial"]:
                for c in cc:
                    all_single_env_trial.append(c)
            for cc in summary_dict["cv_by_env_trial_rdm"]:
                for c in cc:
                    all_single_env_trial_rdm.append(c)
            for cc in summary_dict["cv_by_env_trial_cvrdm"]:
                for c in cc:
                    all_single_env_trial_cvrdm.append(c)

            single_env_trial.append(all_single_env_trial)
            single_env_trial_rdm.append(all_single_env_trial_rdm)
            single_env_trial_cvrdm.append(all_single_env_trial_cvrdm)

    if return_extras:
        return single_env, across_env, single_env_trial, single_env_trial_rdm, single_env_trial_cvrdm

    return single_env, across_env


def compare_cvpca_to_cvfourier(summary_dicts):
    """
    get average across environment cvpca and svca for each mouse

    summary dicts is a list of dictionaries containing full spectra data for each mouse to be compared
    returns a list of lists, where the outer list corresponds to each mouse (e.g. each summary_dict)
    and the inner list contains the eigenspectra for each session / environment
    """
    # get average eigenspectra for each mouse
    cvpca = []
    cvfourier = []
    for summary_dict in summary_dicts:
        c_cvpca = []
        c_cvfourier = []
        # go through each session's across environment eigenspectra
        for c in [c for cc in summary_dict["cv_by_env_all"] for c in cc]:
            c_cvpca.append(c)

        for c in [c for cc in summary_dict["cvf_by_env_cov_all"] for c in cc]:
            c_cvfourier.append(c)

        # add them all to master list
        cvpca.append(c_cvpca)
        cvfourier.append(c_cvfourier)

    return cvpca, cvfourier


def compare_svca_to_cvpca(summary_dicts):
    """
    get average across environment cvpca and svca for each mouse

    summary dicts is a list of dictionaries containing full spectra data for each mouse to be compared
    returns a list of lists, where the outer list corresponds to each mouse (e.g. each summary_dict)
    and the inner list contains the eigenspectra for each session / environment
    """
    # get average eigenspectra for each mouse
    cvpca = []
    svca = []
    svca_pred = []
    svca_pred_cv = []
    rank_pf_pred = []
    for summary_dict in summary_dicts:
        c_cvpca = []
        c_svca = []
        c_svca_pred = []
        c_svca_pred_cv = []
        c_rank_pf_pred = []
        c_cvfft = []
        # go through each session's across environment eigenspectra
        for c in summary_dict["cv_across_all"]:
            c_cvpca.append(c)

        # go through each session's svca results
        for c in summary_dict["svca_shared"]:
            c_svca.append(c)

        # go through each session's svca of place field predictions
        for c in summary_dict["svca_shared_prediction"]:
            c_svca_pred.append(c)

        # go through each session's svca of place field predictions that are cross-validated
        for c in summary_dict["svca_shared_prediction_cv"]:
            c_svca_pred_cv.append(c)

        # go through each session's rank pf prediction
        for c in summary_dict["rank_pf_prediction"]:
            c_rank_pf_pred.append(c)

        # add them all to master list
        cvpca.append(c_cvpca)
        svca.append(c_svca)
        svca_pred.append(c_svca_pred)
        svca_pred_cv.append(c_svca_pred_cv)
        rank_pf_pred.append(c_rank_pf_pred)

    return cvpca, svca, svca_pred, svca_pred_cv, rank_pf_pred


def compare_value_by_environment(pcms, summary_dicts, value_name, reduction="sum", relative_session=False, first_offset=0):
    if type(reduction) != str:
        if len(reduction) == 2 and type(reduction[0]) == str and callable(reduction[1]):
            reduce_func = reduction[1]
            reduction = reduction[0]
            assert reduce_func(np.arange(3)).shape == (), "reduction function must return a scalar!"
    elif type(reduction) == str:
        reduce_funcs = {
            "sum": np.sum,
            "mean": np.mean,
            "std": np.std,
            "nansum": np.nansum,
            "nanmean": np.nanmean,
            "nanstd": np.nanstd,
        }
        if reduction not in reduce_funcs:
            raise ValueError(f"reduction method {reduction} not recognized")
        reduce_func = reduce_funcs[reduction]
    else:
        raise ValueError("reduction must be a string or a tuple with a string and a callable")

    value = []
    session_offset = []
    for pcm, spectra_data in zip(pcms, summary_dicts):
        mouse_name = pcm.track.mouse_name
        env_order = [env for env in helpers.get_env_order(mousedb, mouse_name) if env != -1]  # ignore environment=-1 for this analysis

        if len(env_order) == 0:
            print("No valid environments for mouse", mouse_name)
            continue

        c_vals = []
        c_offsets = []
        for env in env_order:
            c_val = np.zeros(len(spectra_data["envstats"][env]))
            for ii, ises in enumerate(spectra_data["envstats"][env]):
                cenvidx = pcm.pcss[ises].envnum_to_idx(env)[0]
                c_val[ii] = reduce_func(spectra_data[value_name][ises][cenvidx])
            c_vals.append(c_val)
            c_offsets.append(spectra_data["envstats"][env])

        value.append(c_vals)
        session_offset.append(c_offsets)

    max_envs = max([len(c) for c in value])
    max_sessions = max([len(cc) for c in value for cc in c])

    if relative_session:
        data = np.full((len(value), max_envs, max_sessions), np.nan)
        for ii, tv in enumerate(value):
            for jj, ev in enumerate(tv):
                data[ii, jj, : len(ev)] = ev
    else:
        data = np.full((len(value), max_envs, max_sessions + first_offset), np.nan)
        for ii, (tv, so) in enumerate(zip(value, session_offset)):
            for jj, (ev, eo) in enumerate(zip(tv, so)):
                c_eo = np.array(eo) + (first_offset if jj > 0 else 0)
                data[ii, jj, c_eo] = ev

    return data


def plot_value_comparison(
    pcms,
    summary_dicts,
    value_name,
    reduction="sum",
    relative_value=False,
    relative_session=False,
    first_offset=0,
    poster2024=False,
    with_show=True,
    with_save=False,
    fancy_name=None,
):
    value = compare_value_by_environment(
        pcms,
        summary_dicts,
        value_name,
        reduction=reduction,
        relative_session=relative_session,
        first_offset=first_offset,
    )

    if type(reduction) != str and len(reduction) == 2 and type(reduction[0]) == str and callable(reduction[1]):
        reduction = reduction[0]

    if value.shape[1] < 6:
        cols = "krbgcmy"
    else:
        cmap = mpl.colormaps["tab10"].resampled(value.shape[1])
        cols = [cmap(i) for i in range(value.shape[1])]

    if relative_value:
        first_notnan = np.argmax(~np.isnan(value[:, 0]), axis=1)
        value = value / value[np.arange(value.shape[0]), np.zeros(value.shape[0], dtype=int), first_notnan].reshape(-1, 1, 1)

    average_value = np.nanmean(value, axis=0)
    se_value = np.nanstd(value, axis=0)  # / np.sqrt(np.sum(~np.isnan(value), axis=0))

    if poster2024:
        plt.rcParams.update({"font.size": 24})

    figdim = 5
    fig, ax = plt.subplots(1, 1, figsize=(figdim, figdim), layout="constrained")
    for ii, (av, sv) in enumerate(zip(average_value, se_value)):
        ax.plot(range(len(av)), av, color=cols[ii], label=f"Env {ii}")
        ax.fill_between(range(len(av)), av - sv, av + sv, color=cols[ii], alpha=0.3, edgecolor=None)
    ax.set_xlabel(("Relative " if relative_session else "") + "Session #")
    ax.set_ylabel(reduction.title() + " " + ("Relative " if relative_value else "") + (value_name if fancy_name is None else fancy_name))
    ylims = ax.get_ylim()
    max_ytick = np.ceil(ylims[1] * 10) / 10
    max_ytick = 4 ** np.floor(np.log2(ylims[1]) / np.log2(4))
    ylim_max = max_ytick + 1
    ax.set_ylim(0, ylim_max)  # * 1.25)
    ax.set_yticks(np.linspace(0, max_ytick, 5))
    cols = "krb"
    offset = ylim_max * 0.1
    for i, c in enumerate(cols):
        ax.text(0, max_ytick - i * offset, f"Env {i}", color=c)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if with_show:
        plt.show()

    if with_save and poster2024:
        save_directory = pcms[0].saveDirectory("example_plots")
        save_path = save_directory / f"{value_name}_comparison_across_mice"
        helpers.save_figure(fig, save_path)

    if with_save and not poster2024:
        special_name = f"{value_name}_{reduction}"
        special_name += "_rv" if relative_value else ""
        special_name += "_relative" if relative_session else ""
        pcms[0].saveFigure(fig.number, "comparisons", "value_comparison_" + special_name)


def plot_value_to_value_comparison(
    pcms,
    summary_dicts,
    first_name,
    second_name,
    first_reduction="mean",
    second_reduction="mean",
    relative_value=False,
    relative_session=False,
    first_offset=0,
    with_show=True,
    with_save=False,
):
    first_value = compare_value_by_environment(
        pcms, summary_dicts, first_name, reduction=first_reduction, relative_session=relative_session, first_offset=first_offset
    )
    second_value = compare_value_by_environment(
        pcms, summary_dicts, second_name, reduction=second_reduction, relative_session=relative_session, first_offset=first_offset
    )

    def _process_reduction_name(reduction):
        if type(reduction) != str and len(reduction) == 2 and type(reduction[0]) == str and callable(reduction[1]):
            return reduction[0]
        return reduction

    fr_name = _process_reduction_name(first_reduction)
    sr_name = _process_reduction_name(second_reduction)

    msg = f"shape of first value ({first_value.shape}) must match shape of second value ({second_value.shape})"
    assert first_value.shape == second_value.shape, msg

    if relative_value:

        def _normalize(value):
            first_notnan = np.argmax(~np.isnan(value[:, 0]), axis=1)
            return value / value[np.arange(value.shape[0]), np.zeros(value.shape[0], dtype=int), first_notnan].reshape(-1, 1, 1)

        first_value = _normalize(first_value)
        second_value = _normalize(second_value)

    color_by = ["Mouse", "Environment", ("Relative " if relative_session else "") + "Session"]
    cmaps = ["Dark2", "tab10", "cool"]

    figdim = 3
    fig, ax = plt.subplots(1, 3, figsize=(3 * figdim, figdim), layout="constrained", sharex=True, sharey=True)
    for ii, cb in enumerate(color_by):
        cmap = mpl.colormaps[cmaps[ii]].resampled(first_value.shape[ii])
        c_fv = np.swapaxes(np.copy(first_value), 0, ii)
        c_sv = np.swapaxes(np.copy(second_value), 0, ii)
        for jj, (fv, sv) in enumerate(zip(c_fv, c_sv)):
            ax[ii].scatter(fv, sv, s=20, color=(cmap(jj), 0.5), edgecolor="k", linewidth=0.1, label=f"{cb} {jj}")
        ax[ii].set_xlabel(fr_name + " " + first_name + ("_rv" if relative_value else ""))
        if ii == 0:
            ax[ii].set_ylabel(sr_name + " " + second_name)
        iax = ax[ii].inset_axes([0.05, 0.85, 0.35, 0.08])
        iax.xaxis.set_ticks_position("bottom")
        norm = mpl.colors.Normalize(vmin=-0.5, vmax=first_value.shape[ii] - 0.5)
        m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(m, cax=iax, orientation="horizontal")
        ax[ii].set_title(f"Color: {cb}")

    if with_show:
        plt.show()

    if with_save:
        special_name = f"{first_name}_{fr_name}_" + f"{second_name}_{sr_name}"
        special_name += "_rv" if relative_value else ""
        special_name += "_relative" if relative_session else ""
        pcms[0].saveFigure(fig.number, "comparisons", "value_value_cmp_" + special_name)


def plot_total_variance_comparison(pcms, summary_dicts, relative_session=False, first_offset=0, with_show=True, with_save=False):
    plot_value_comparison(
        pcms,
        summary_dicts,
        "cv_by_env_all",
        reduction="sum",
        relative_value=True,
        relative_session=relative_session,
        first_offset=first_offset,
        with_show=with_show,
        with_save=with_save,
        fancy_name="Variance",
    )


def plot_spectral_averages_comparison(
    pcms,
    single_env,
    across_env,
    single_env_trial,
    single_env_trial_rdm,
    single_env_trial_cvrdm,
    do_xlog=False,
    do_ylog=False,
    ylog_min=1e-3,
    with_show=True,
    with_save=False,
):
    """
    if across_num set to a number, will only include sessions with that many environments included (None means use all)
    """
    # Show the trial based versions...
    show_extras = False

    # if not using a y-log axis, then set the minimum to -inf to not change any data
    if not do_ylog:
        ylog_min = -np.inf

    num_envs = []
    for pcm in pcms:
        num_envs.append([])
        for pcss in pcm.pcss:
            num_envs[-1].append(len(pcss.environments))

    # create processing method
    def _process(data):
        """internal function for processing a set of eigenspectra"""
        data = np.stack(data)
        # normalize each row so they sum to 1
        data = data / np.sum(data, axis=1, keepdims=True)
        # remove any values below the minimum for log scaling
        if isinstance(ylog_min, float):
            # take the average across rows (across sessions / environments)
            data = np.mean(data, axis=0)
            data[data < ylog_min] = np.nan
        else:
            # get dimension of each row
            dims = _dimension(data)
            # replace everything after the dimension with nans
            for i, d in enumerate(dims):
                data[i, d + 1 :] = np.nan
            data = np.nanmean(data, axis=0)
        # return processed data
        return data

    def _dimension(data):
        """internal function for measuring the dimensionality of cross-validated eigenspectra"""
        data = np.stack(data)
        # find first point where the value in data is less than 0
        return np.argmax(data <= 0, axis=1) - 1

    poster2024 = True
    num_mice = len(pcms)
    mouse_names = [pcm.track.mouse_name for pcm in pcms]
    mouse_names = helpers.short_mouse_names(mouse_names)

    if poster2024:
        colors = ["k" for _ in range(num_mice)]
    else:
        cmap = mpl.colormaps["turbo"].resampled(num_mice)
        colors = [cmap(i) for i in range(num_mice)]

    if show_extras:
        colors = ["k" for _ in range(num_mice)]
        colors_trial = ["b" for _ in range(num_mice)]
        colors_trial_rdm = ["r" for _ in range(num_mice)]
        colors_trial_cvrdm = ["g" for _ in range(num_mice)]

    plt.rcParams.update({"font.size": 24})

    figdim = 6.5
    # note: easy to readd second row for cumulative variance
    fig, ax = plt.subplots(1, 3, figsize=(2 * figdim, 1 * figdim), width_ratios=[1, 1, 0.5], layout="constrained")
    ax = np.array(ax).reshape(1, 3)
    for imouse, (mouse_name, c_single_env, c_across_env, num_env) in enumerate(zip(mouse_names, single_env, across_env, num_envs)):
        c_double_env = [c_across_env[i] for i, n in enumerate(num_env) if n == 2]
        c_single_data = _process(c_single_env)
        c_double_data = _process(c_double_env)
        c_num_envs = np.unique(num_env)
        c_dims = []
        for c_num in c_num_envs:
            c_dims.append(_dimension([c_across_env[i] for i, n in enumerate(num_env) if n == c_num]).mean())
        label = ("Each Mouse" if imouse == 0 else None) if poster2024 else mouse_name
        ax[0, 0].plot(range(1, len(c_single_data) + 1), c_single_data, color=colors[imouse], label=label)
        ax[0, 1].plot(range(1, len(c_double_data) + 1), c_double_data, color=colors[imouse], label=label)
        ax[0, 2].plot(c_num_envs, c_dims, color=colors[imouse], marker=".", markersize=16, label=label)

    if show_extras:
        for imouse, (mouse_name, c_trial, c_trial_rdm, c_trial_cvrdm) in enumerate(
            zip(mouse_names, single_env_trial, single_env_trial_rdm, single_env_trial_cvrdm)
        ):
            c_trial_data = _process(c_trial)
            c_trial_rdm_data = _process(c_trial_rdm)
            c_trial_cvrdm_data = _process(c_trial_cvrdm)
            label_trial = "Trials" if imouse == 0 else None
            label_trial_rdm = "Trials-RDM" if imouse == 0 else None
            label_trial_cvrdm = "Trials-CVRDM" if imouse == 0 else None
            ax[0, 0].plot(range(1, len(c_trial_data) + 1), c_trial_data, color=colors_trial[imouse], label=label_trial)
            ax[0, 0].plot(range(1, len(c_trial_rdm_data) + 1), c_trial_rdm_data, color=colors_trial_rdm[imouse], label=label_trial_rdm)
            ax[0, 0].plot(range(1, len(c_trial_cvrdm_data) + 1), c_trial_cvrdm_data, color=colors_trial_cvrdm[imouse], label=label_trial_cvrdm)

    ax[0, 0].set_xlabel(f"Dimension ({'log' if do_xlog else 'linear'})")
    ax[0, 1].set_xlabel(f"Dimension ({'log' if do_xlog else 'linear'})")
    ax[0, 2].set_xlabel("# Environments", loc="right")
    # ax[1, 0].set_xlabel(f"Dimension ({'log' if do_xlog else 'linear'})")
    # ax[1, 1].set_xlabel(f"Dimension ({'log' if do_xlog else 'linear'})")
    ax[0, 0].set_ylabel(f"Relative Variance ({'log' if do_ylog else 'linear'})")
    ax[0, 2].set_ylabel("Dimensionality")
    # ax[1, 0].set_ylabel("Cumulative Variance")
    ax[0, 0].set_title("N=1 Environment")
    ax[0, 1].set_title("N=2 Environments")
    ax[0, 0].legend(loc="upper right")

    if do_xlog:
        for aa in ax:
            for a in aa:
                a.set_xscale("log")

    xlims = [ax[0, 0].get_xlim(), ax[0, 1].get_xlim()]
    xlim = (min([x[0] for x in xlims]), max([x[1] for x in xlims]))
    ax[0, 0].set_xlim(xlim)
    ax[0, 1].set_xlim(xlim)

    ax[0, 2].set_xlim(0.5, max([max(c_num_envs) for c_num_envs in num_envs]) + 0.5)
    ax[0, 2].set_ylim(0)

    if do_ylog:
        for aa in ax:
            for a in aa:
                a.set_yscale("log")

        ax[0, 2].set_yscale("linear")

    ax[0, 0].spines["top"].set_visible(False)
    ax[0, 0].spines["right"].set_visible(False)
    ax[0, 1].set_yticks([], labels=None)
    ax[0, 1].minorticks_off()
    ax[0, 1].spines["left"].set_visible(False)
    ax[0, 1].spines["right"].set_visible(False)
    ax[0, 1].spines["top"].set_visible(False)
    ylims = [ax[0, 0].get_ylim(), ax[0, 1].get_ylim()]
    ylim = (min([y[0] for y in ylims]), max([y[1] for y in ylims]))
    ax[0, 0].set_ylim(ylim)
    ax[0, 1].set_ylim(ylim)

    ax[0, 2].spines["top"].set_visible(False)
    ax[0, 2].spines["right"].set_visible(False)
    ax[0, 2].spines["bottom"].set_visible(False)

    ylims = ax[0, 0].get_ylim()
    text_ypos = ylims[0] + (ylims[1] - ylims[0]) * 0.00005
    ax[0, 0].text(1, text_ypos, "dim := \nlast before negatives")
    ax[0, 1].text(np.mean(xlim), ylims[0] + (ylims[1] - ylims[0]) * 0.45, "exponential scaling", ha="center", va="center")

    if with_show:
        pass
        plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcms[0].saveDirectory("comparisons")
        special_name = "logx_" if do_xlog else "linx_"
        special_name = special_name + ("logy" if do_ylog else "liny")
        save_path = save_directory / ("cv_spectral_average_comparison_poster2024_" + special_name)
        helpers.save_figure(fig, save_path)

    if with_save:
        special_name = "logx_" if do_xlog else "linx_"
        special_name = special_name + ("logy" if do_ylog else "liny")
        pcms[0].saveFigure(fig.number, "comparisons", "cv_spectral_average_comparison_" + special_name)

    # Overlapping plot method:
    cols = ["mediumorchid", "black"]

    figdim = 6.5
    # note: easy to readd second row for cumulative variance
    fig, ax = plt.subplots(1, 2, figsize=(2 * figdim, 1 * figdim), width_ratios=[2, 1], layout="constrained")
    ax = np.array(ax).reshape(1, 2)
    for imouse, (mouse_name, c_single_env, c_across_env, num_env) in enumerate(zip(mouse_names, single_env, across_env, num_envs)):
        c_double_env = [c_across_env[i] for i, n in enumerate(num_env) if n == 2]
        c_single_data = _process(c_single_env)
        c_double_data = _process(c_double_env)
        c_num_envs = np.unique(num_env)
        c_dims = []
        for c_num in c_num_envs:
            c_dims.append(_dimension([c_across_env[i] for i, n in enumerate(num_env) if n == c_num]).mean())
        label1 = ("N=1 Env" if imouse == 0 else None) if poster2024 else mouse_name
        label2 = ("N=2 Env" if imouse == 0 else None) if poster2024 else mouse_name
        label = ("Each Mouse" if imouse == 0 else None) if poster2024 else mouse_name
        ax[0, 0].plot(range(1, len(c_single_data) + 1), c_single_data, color=cols[0], label=label1)
        ax[0, 0].plot(range(1, len(c_double_data) + 1), c_double_data, color=cols[1], label=label2)
        ax[0, 1].plot(c_num_envs, c_dims, color=colors[imouse], marker=".", markersize=16, label=label)

    ax[0, 0].set_xlabel(f"Dimension ({'log' if do_xlog else 'linear'})")
    ax[0, 1].set_xlabel("# Environments", loc="right")
    ax[0, 0].set_ylabel(f"Relative Variance ({'log' if do_ylog else 'linear'})")
    ax[0, 1].set_ylabel("Dimensionality")
    ax[0, 0].legend(loc="upper right")
    xlim = ax[0, 0].get_xlim()

    ax[0, 1].set_xlim(0.5, max([max(c_num_envs) for c_num_envs in num_envs]) + 0.5)
    ax[0, 1].set_ylim(0)
    ax[0, 0].set_yscale("log")

    ax[0, 0].spines["top"].set_visible(False)
    ax[0, 0].spines["right"].set_visible(False)
    ax[0, 1].spines["top"].set_visible(False)
    ax[0, 1].spines["right"].set_visible(False)
    ax[0, 1].spines["bottom"].set_visible(False)

    ylims = ax[0, 0].get_ylim()
    text_ypos = ylims[0] + (ylims[1] - ylims[0]) * 0.00005
    ax[0, 0].text(1, text_ypos, "dim := \nlast before negatives")

    if with_show:
        pass
        plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcms[0].saveDirectory("comparisons")
        special_name = "logx_" if do_xlog else "linx_"
        special_name = special_name + ("logy" if do_ylog else "liny")
        save_path = save_directory / ("cv_spectral_average_overlapping_poster2024_" + special_name)
        helpers.save_figure(fig, save_path)


def plot_svca_vs_cvpca(
    pcms,
    summary_dicts,
    include_cvpca=True,
    include_pfpred=True,
    use_partratio=False,
    normalize=True,
    do_ylog=True,
    ylog_min=1e-6,
    with_show=True,
    with_save=False,
):
    cvpca, svca, svca_pred, svca_pred_cv, rank_pf_pred = compare_svca_to_cvpca(summary_dicts)

    # if not using a y-log axis, then set the minimum to -inf to not change any data
    if not do_ylog:
        ylog_min = -np.inf

    num_envs = []
    for pcm in pcms:
        num_envs.append([])
        for pcss in pcm.pcss:
            num_envs[-1].append(len(pcss.environments))

    # create processing method
    def _process(data):
        """internal function for processing a set of eigenspectra"""
        # pad data with nans
        max_dims = max([len(d) for d in data])
        data = [np.append(d, np.full(max_dims - len(d), np.nan)) for d in data]
        # stack data
        data = np.stack(data)
        if normalize:
            # normalize each row so they sum to 1
            data = data / np.nansum(data, axis=1, keepdims=True)
        if isinstance(ylog_min, float):
            # take the average across rows (across sessions / environments)
            data = np.mean(data, axis=0)
            data[data < ylog_min] = np.nan
        else:
            # get dimension of each row
            dims = _dimension(data)
            # replace everything after the dimension with nans
            for i, d in enumerate(dims):
                data[i, d + 1 :] = np.nan
            data = np.nanmean(data, axis=0)
        # return processed data
        return data

    def _dimension(data):
        """internal function for measuring the dimensionality of cross-validated eigenspectra"""
        max_dims = max([len(d) for d in data])
        data = [np.append(d, np.full(max_dims - len(d), np.nan)) for d in data]
        data = np.stack(data)
        # find first point where the value in data is less than 0
        return np.argmax(data <= 0, axis=1) - 1

    def _partratio(data):
        """internal function for measuring the partition ratio of cross-validated eigenspectra
        using the participation ratio, which is probably smarter and less susceptible to noise"""
        data = [np.array(d) for d in data]
        positive_data = [np.maximum(d, 0) for d in data]
        squared_sum = [np.sum(d) ** 2 for d in positive_data]
        sum_squared = [np.sum(d**2) for d in positive_data]
        pr = [s / ss for s, ss in zip(squared_sum, sum_squared)]
        return np.array(pr)

    poster2024 = True
    num_mice = len(pcms)
    mouse_names = [pcm.track.mouse_name for pcm in pcms]
    mouse_names = helpers.short_mouse_names(mouse_names)

    if poster2024:
        colors = ["k" for _ in range(num_mice)]
    else:
        cmap = mpl.colormaps["turbo"].resampled(num_mice)
        colors = [cmap(i) for i in range(num_mice)]

    plt.rcParams.update({"font.size": 24})

    figdim = 6.5
    fig, ax = plt.subplots(1, 2, figsize=(10.5, figdim), width_ratios=[1, 0.4], layout="constrained")
    for imouse, (mouse_name, c_svca, c_cvpca, c_svca_pred, c_svca_pred_cv, c_rank_pred, num_env) in enumerate(
        zip(mouse_names, svca, cvpca, svca_pred, svca_pred_cv, rank_pf_pred, num_envs)
    ):
        c_svca_data = _process(c_svca)
        # c_svca_pred_data = _process(c_svca_pred)
        if include_pfpred:
            c_svca_pred_cv_data = _process(c_svca_pred_cv)
            label_pred_cv = "PF-Pred" if imouse == 0 else None
        c_label = mouse_name  # + (" (svca)" if include_cvpca and imouse == 0 else "")
        label = "Full" if imouse == 0 else None
        ax[0].plot(range(1, len(c_svca_data) + 1), c_svca_data, color="k", label=label)
        if include_pfpred:
            ax[0].plot(range(1, len(c_svca_pred_cv_data) + 1), c_svca_pred_cv_data, color="b", linestyle="-", label=label_pred_cv)
            # ax[0].plot(range(1, len(c_svca_pred_data) + 1), c_svca_pred_data, color="r", linestyle="-", label=label_pred)
        if include_cvpca:
            c_cvpca_data = _process(c_cvpca)
            c_label = "cvpca" if imouse == (num_mice - 1) else None
            ax[0].plot(range(1, len(c_cvpca_data) + 1), c_cvpca_data, color="r", linestyle="-", label=c_label)

        c_num_envs = np.unique(num_env)
        c_dims = []
        s_dims = []
        spred_dims = []
        spredcv_dims = []
        dim_func = _dimension if not use_partratio else _partratio
        for c_num in c_num_envs:
            c_dims.append(dim_func([c_cvpca[i] for i, n in enumerate(num_env) if n == c_num]).mean())
            s_dims.append(dim_func([c_svca[i] for i, n in enumerate(num_env) if n == c_num]).mean())
            spred_dims.append(dim_func([c_svca_pred[i] for i, n in enumerate(num_env) if n == c_num]).mean())
            spredcv_dims.append(dim_func([c_svca_pred_cv[i] for i, n in enumerate(num_env) if n == c_num]).mean())

        svca_label = "Time" if imouse == 0 else None
        if include_cvpca:
            cvpca_label = "Pos" if imouse == 0 else None
        if include_pfpred:
            pf_pred_label = "PF Pred" if imouse == 0 else None
            pf_pred_cv_label = "PF Pred CV" if imouse == 0 else None
        ax[1].plot(c_num_envs, s_dims, color="k", linestyle="-", marker=".", markersize=12, label=svca_label)
        if include_cvpca:
            ax[1].plot(c_num_envs, c_dims, color="r", linestyle="-", marker=".", markersize=12, label=cvpca_label)
        if include_pfpred:
            # ax[1].plot(c_num_envs, spred_dims, color="r", linestyle="-", marker=".", markersize=12, label=pf_pred_label)
            ax[1].plot(c_num_envs, spredcv_dims, color="b", linestyle="-", marker=".", markersize=12, label=pf_pred_cv_label)

    ax[0].set_xlabel("SVC-Time Dimension (log)")
    ax[0].set_ylabel(f"Relative Variance ({'log' if do_ylog else 'linear'})")
    ax[1].set_xlabel("# Environments", loc="right")
    ax[1].set_ylabel("Dimensionality" + (" (log)" if not use_partratio and do_ylog else ""))
    ax[0].legend(loc="upper right", fontsize=20)
    # ax[1].legend(loc="center", fontsize=20)
    ax[0].set_xscale("log")
    ax[1].set_xlim(0.5, max([max(c_num_envs) for c_num_envs in num_envs]) + 0.5)
    if use_partratio:
        ax[1].set_ylim(0, 70)
    else:
        ax[1].set_ylim(2, 9e3)
    if do_ylog:
        ax[0].set_yscale("log")
        if not use_partratio:
            ax[1].set_yscale("log")

    if poster2024:
        pass
        # ax[0].text(1, 4e-7, "100x higher dim. than\nspatial representations", ha="left", va="bottom")

    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)

    # xlims = ax[0].get_xlim()
    # ax[0].set_xlim(xlims[0], xlims[1] * 3)

    if with_show:
        plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcms[0].saveDirectory("comparisons")
        special_name = "_withcvpca" if include_cvpca else ""
        special_name = special_name + ("_withpfpred" if include_pfpred else "")
        special_name = special_name + ("logy" if do_ylog else "liny")
        special_name = special_name + ("_partratio" if use_partratio else "")
        save_path = save_directory / ("svca_comparison" + special_name)
        helpers.save_figure(fig, save_path)

    if with_save:
        special_name = "_vs_cvpca" if include_cvpca else ""
        special_name += "_norm" if normalize else ""
        special_name += "_logy" if do_ylog else "_liny"
        pcms[0].saveFigure(fig.number, "comparisons", "svca" + special_name)


def plot_cvpca_vs_fourier(
    pcms,
    summary_dicts,
    include_cvpca=True,
    include_pfpred=True,
    use_partratio=False,
    normalize=True,
    do_xlog=True,
    do_ylog=True,
    ylog_min=1e-6,
    with_show=True,
    with_save=False,
):
    cvpca, cvfourier = compare_cvpca_to_cvfourier(summary_dicts)

    cvpca = [np.stack(c) for c in cvpca]
    num_dims = [c.shape[1] for c in cvpca]
    cvfourier = [np.mean(np.stack(c), axis=1)[:, :nd] for c, nd in zip(cvfourier, num_dims)]

    # if not using a y-log axis, then set the minimum to -inf to not change any data
    if not do_ylog:
        ylog_min = -np.inf

    # create processing method
    def _process(data):
        """internal function for processing a set of eigenspectra"""
        # normalize each spectrum
        if normalize:
            data = data / np.nansum(data, axis=1, keepdims=True)
        # average across sessions / environments
        data = np.mean(data, axis=0)
        # remove any values below the minimum for log scaling
        data[data < ylog_min] = np.nan
        return data

    num_mice = len(pcms)
    mouse_names = [pcm.track.mouse_name for pcm in pcms]
    mouse_names = helpers.short_mouse_names(mouse_names)

    plt.rcParams.update({"font.size": 24})

    figdim = 5
    fig, ax = plt.subplots(1, 1, figsize=(figdim, figdim), layout="constrained")
    ax = np.reshape(ax, 1)
    for imouse, (mouse_name, c_cvpca, c_fourier) in enumerate(zip(mouse_names, cvpca, cvfourier)):
        c_cvpca = _process(c_cvpca)
        c_fourier = _process(c_fourier)
        cv_label = "cvPCA" if imouse == 0 else None
        fr_label = "cvFourier" if imouse == 0 else None
        ax[0].plot(range(1, len(c_cvpca) + 1), c_cvpca, color="k", label=cv_label)
        ax[0].plot(range(1, len(c_fourier) + 1), c_fourier, color="b", label=fr_label)

    ax[0].set_xlabel("Dimension" + (" (log)" if do_ylog else " (linear)"))
    ax[0].set_ylabel(f"{'Rel. ' if normalize else ''}Var. ({'log' if do_ylog else 'linear'})")
    ax[0].legend(loc="upper right", fontsize=20)
    if do_xlog:
        ax[0].set_xscale("log")
    if do_ylog:
        ax[0].set_yscale("log")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    if with_show:
        plt.show()

    if with_save:
        save_directory = pcms[0].saveDirectory("comparisons")
        special_name = ""
        special_name += "_norm" if normalize else ""
        special_name += "_logx" if do_xlog else "_linx"
        special_name += "_logy" if do_ylog else "_liny"
        save_path = save_directory / ("cvpca_vs_fourier" + special_name)
        helpers.save_figure(fig, save_path)


def plot_all_exponential_fits(pcms, spectra_data, relative_session=True, with_show=True, with_save=False):
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

    single_session_id = np.concatenate(
        [
            ises * np.ones(len(sdenv)) / (len(sd["cv_by_env_all"]) - 1 if relative_session else 1)
            for sd in spectra_data
            for ises, sdenv in enumerate(sd["cv_by_env_all"])
        ]
    )
    across_session_id = np.concatenate([np.arange(len(aenv)) / (len(aenv) - 1 if relative_session else 1) for aenv in across_env])
    max_session = 1 if relative_session else max([len(aenv) for aenv in across_env])

    mouse_cmap = mpl.colormaps["Dark2"].resampled(len(single_env) - 1)
    r2_cmap = mpl.colormaps["plasma"]
    session_cmap = mpl.colormaps["cool"]

    figdim = 3
    alpha = 0.7
    s = 25

    fig, ax = plt.subplots(2, 3, figsize=(3 * figdim, 2 * figdim), layout="constrained", sharex="col", sharey="col")
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
    ax[0, 2].scatter(
        np.concatenate(single_decay),
        np.concatenate(single_amp),
        s=s,
        c=single_session_id,
        cmap=session_cmap,
        vmin=0,
        vmax=max_session,
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
    ax[1, 2].scatter(
        np.concatenate(across_decay),
        np.concatenate(across_amp),
        s=s,
        c=across_session_id,
        cmap=session_cmap,
        vmin=0,
        vmax=max_session,
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
    ax[0, 1].set_title("Single Env Spectra")
    ax[1, 0].set_title("Across Env Spectra")
    ax[1, 1].set_title("Across Env Spectra")
    ax[1, 2].set_title("Across Env Spectra")

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

    iax = ax[0, 2].inset_axes([0.6, 0.85, 0.29, 0.07])
    iax.xaxis.set_ticks_position("bottom")
    norm = mpl.colors.Normalize(vmin=0, vmax=max_session)
    m = mpl.cm.ScalarMappable(cmap=session_cmap, norm=norm)
    fig.colorbar(m, cax=iax, orientation="horizontal", label=f"{'Relative' if relative_session else 'Absolute'} Session ID")

    if with_show:
        plt.show()

    if with_save:
        special_name = "relative_session" if relative_session else "absolute_session"
        pcms[0].saveFigure(fig.number, "comparisons", "exponential_fit_results_" + special_name)


def predict_exp_fits_across_mice(pcms, spectra_data, amplitude=True, with_show=True, with_save=False):
    """
    predict exponential fits of eigenspectra from pfvars across mice
    """
    exponential_fits = [get_exp_fits(sd) for sd in spectra_data]
    pfv_data_means = [get_pf_summary(pcm, sd, mean=True, reliable=False) for pcm, sd in zip(pcms, spectra_data)]
    pfv_data_stds = [get_pf_summary(pcm, sd, mean=False, reliable=False) for pcm, sd in zip(pcms, spectra_data)]

    ignore_vars = ["rel_mse"]
    pfv_names = [k for k in pfv_data_means[0].keys() if k not in ignore_vars]
    fit_target = "single_amplitude" if amplitude else "single_decay"

    # get all the data in a format that can be used for comparison (within each mouse)
    fit_data, fit_session, fit_env = _catdata(exponential_fits, fit_target)

    # concatenated target data for all mice
    all_fit_data = np.concatenate(fit_data)
    all_fit_mouse = np.concatenate([i * np.ones(len(f)) for i, f in enumerate(fit_data)])
    all_fit_session = np.concatenate(fit_session)
    all_fit_env = np.concatenate(fit_env)

    target_norm = np.mean(all_fit_data), np.std(all_fit_data)

    # dictionary of all the concatenated data
    pfv_all = {}
    for k in pfv_names:
        pfv_all[k + "_mean"] = _catdata(pfv_data_means, k)[0]
        pfv_all[k + "_std"] = _catdata(pfv_data_stds, k)[0]

    pfv_data_norms = {k: (np.mean(np.concatenate(v)), np.std(np.concatenate(v))) for k, v in pfv_all.items()}

    # measure fit coefficient and r2 for each pfvar
    def _get_fit(data, target, data_norm=(0, 1), target_norm=(0, 1)):
        """internal function for getting the fit coefficient and r2 for a given target"""
        norm_data = [(d - data_norm[0]) / data_norm[1] for d in data]
        norm_targets = [(t - target_norm[0]) / target_norm[1] for t in target]
        fits = [LinearRegression().fit(d.reshape(-1, 1), t) for d, t in zip(norm_data, norm_targets)]
        preds = [f.predict(d.reshape(-1, 1)) for f, d in zip(fits, norm_data)]
        coefs = [f.coef_[0] for f in fits]
        r2s = [r2_score(t, p) for t, p in zip(norm_targets, preds)]
        return np.array(coefs), np.array(r2s)

    coefs, r2s = helpers.named_transpose(
        [
            _get_fit(value, fit_data, data_norm=data_norm, target_norm=target_norm)
            for value, data_norm in zip(pfv_all.values(), pfv_data_norms.values())
        ]
    )

    poster2024 = True
    if poster2024:
        figdim = 5
        xscale = 3 / 5
        yscale = 1
        keep = ["pf_norm_mean", "pf_tcorr_mean_mean"]
        fancy_names = ["Amp", "Cons."]
        idx_to_keep = [list(pfv_all.keys()).index(k) for k in keep]
        coefs_keep = [coefs[itk] for itk in idx_to_keep]
        r2s_keep = [r2s[itk] for itk in idx_to_keep]
        zipped = zip(keep, coefs_keep, r2s_keep)
        show_coefficient = False
        colors = ["k" for _ in keep]
        plt.rcParams.update({"font.size": 24})
    else:
        figdim = 3
        xscale = 4
        yscale = 2
        keep = list(pfv_all.keys())
        fancy_names = keep
        zipped = zip(keep, coefs, r2s)
        show_coefficient = True
        cmap = mpl.colormaps["turbo"].resampled(len(pfv_names))
        cmap_idx = {k: i for i, k in enumerate(pfv_names)}
        colors = [cmap(cmap_idx[k]) for k in keep]

    fig, ax = plt.subplots(2 if show_coefficient else 1, 1, figsize=(xscale * figdim, yscale * figdim), layout="constrained", sharex=True)
    ax = np.reshape(ax, -1)
    for i, (k, c, r) in enumerate(zipped):
        name = "_".join(k.split("_")[:-1])
        if show_coefficient:
            if poster2024:
                xvals = i * np.ones_like(c)
            else:
                xvals = 0.1 * helpers.beeswarm(c) + i
            ax[0].plot(xvals, c, label=k, color=colors[i], marker=".", markersize=16, linestyle="none")
        if poster2024:
            xvals = i * np.ones_like(r)
        else:
            xvals = 0.1 * helpers.beeswarm(r) + i
        ax[-1].plot(xvals, r, label=k, color=colors[i], marker=".", markersize=16, linestyle="none")
    if poster2024:
        if show_coefficient:
            stack_coefs = np.stack(coefs_keep)
            ax[0].plot(range(len(keep)), stack_coefs, color="k", linestyle="-")
        stack_r2s = np.stack(r2s_keep)
        ax[-1].plot(range(len(keep)), stack_r2s, color="k", linestyle="-")
    ax[-1].set_xlim(-0.3, len(keep) - 0.7)
    ax[-1].set_ylim(0, 1)
    ax[-1].set_xticks(range(len(keep)))
    ax[-1].set_xticklabels(fancy_names, rotation=0, ha="center")
    if show_coefficient:
        ax[0].set_ylabel("Fit Coefficient")
    ax[-1].set_ylabel(r"$R^2$", labelpad=-15)
    ax[-1].set_yticks([0, 1])
    ax[-1].set_xlabel("Place Field")

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    if with_show:
        plt.show()

    if with_save and poster2024:
        save_directory = pcms[0].saveDirectory("example_plots")
        save_path = save_directory / f"r2_norm_tcorr_{'expamp' if amplitude else 'expdecay'}"
        helpers.save_figure(fig, save_path)

    if with_save:
        pcms[0].saveFigure(fig.number, "comparisons", "exponential_fit_predictions")


def predict_total_variance_across_mice(pcms, spectra_data, with_show=True, with_save=False):
    """
    predict total variance in eigenspectra from pfvars across mice
    """
    total_variance = [get_total_variance(sd) for sd in spectra_data]
    pfv_data_means = [get_pf_summary(pcm, sd, mean=True, reliable=False) for pcm, sd in zip(pcms, spectra_data)]
    pfv_data_stds = [get_pf_summary(pcm, sd, mean=False, reliable=False) for pcm, sd in zip(pcms, spectra_data)]

    ignore_vars = ["rel_mse"]
    pfv_names = [k for k in pfv_data_means[0].keys() if k not in ignore_vars]
    fit_target = "single_stim_variance"

    # get all the data in a format that can be used for comparison (within each mouse)
    fit_data, fit_session, fit_env = _catdata(total_variance, fit_target)

    # concatenated target data for all mice
    all_fit_data = np.concatenate(fit_data)
    all_fit_mouse = np.concatenate([i * np.ones(len(f)) for i, f in enumerate(fit_data)])
    all_fit_session = np.concatenate(fit_session)
    all_fit_env = np.concatenate(fit_env)

    target_norm = np.mean(all_fit_data), np.std(all_fit_data)

    # dictionary of all the concatenated data
    pfv_all = {}
    for k in pfv_names:
        pfv_all[k + "_mean"] = _catdata(pfv_data_means, k)[0]
        pfv_all[k + "_std"] = _catdata(pfv_data_stds, k)[0]

    pfv_data_norms = {k: (np.mean(np.concatenate(v)), np.std(np.concatenate(v))) for k, v in pfv_all.items()}

    # measure fit coefficient and r2 for each pfvar
    def _get_fit(data, target, data_norm=(0, 1), target_norm=(0, 1)):
        """internal function for getting the fit coefficient and r2 for a given target"""
        norm_data = [(d - data_norm[0]) / data_norm[1] for d in data]
        norm_targets = [(t - target_norm[0]) / target_norm[1] for t in target]
        fits = [LinearRegression().fit(d.reshape(-1, 1), t) for d, t in zip(norm_data, norm_targets)]
        preds = [f.predict(d.reshape(-1, 1)) for f, d in zip(fits, norm_data)]
        coefs = [f.coef_[0] for f in fits]
        r2s = [r2_score(t, p) for t, p in zip(norm_targets, preds)]
        return np.array(coefs), np.array(r2s)

    coefs, r2s = helpers.named_transpose(
        [
            _get_fit(value, fit_data, data_norm=data_norm, target_norm=target_norm)
            for value, data_norm in zip(pfv_all.values(), pfv_data_norms.values())
        ]
    )

    cmap = mpl.colormaps["turbo"].resampled(len(pfv_names))
    cmap_idx = {k: i for i, k in enumerate(pfv_names)}
    figdim = 3
    fig, ax = plt.subplots(2, 1, figsize=(4 * figdim, 2 * figdim), layout="constrained", sharex=True)
    for i, (k, c, r) in enumerate(zip(pfv_all.keys(), coefs, r2s)):
        name = "_".join(k.split("_")[:-1])
        ax[0].scatter(i + 0.1 * helpers.beeswarm(c), c, label=k, color=cmap(cmap_idx[name]), s=15, alpha=0.7)
        ax[1].scatter(i + 0.1 * helpers.beeswarm(r), r, label=k, color=cmap(cmap_idx[name]), s=15, alpha=0.7)
    ax[1].set_xticks(range(len(pfv_all.keys())))
    ax[1].set_xticklabels(pfv_all.keys(), rotation=45, ha="right")
    ax[0].set_ylabel("Fit Coefficient")
    ax[1].set_ylabel("R**2")

    if with_show:
        plt.show()

    if with_save:
        pcms[0].saveFigure(fig.number, "comparisons", "total_variance_predictions")
