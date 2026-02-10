from copy import copy
from tqdm import tqdm
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .. import helpers
from .. import database
from .. import fileManagement as fm
from .standardAnalysis import standardAnalysis
from .. import faststats as fs

sessiondb = database.vrDatabase("vrSessions")


# ---- decorators for pcss class methods ----
def prepare_data(func):
    """decorator to load data if not already autoloaded"""

    @wraps(func)
    def wrapper(pcss_instance, *args, **kwargs):
        if not pcss_instance.dataloaded:
            pcss_instance.load_data()

        # return function with loaded data
        return func(pcss_instance, *args, **kwargs)

    # return decorated function
    return wrapper


def save_directory(name=""):
    dirName = fm.analysisPath() / "placeCellAnalysis" / name
    if not (dirName.is_dir()):
        dirName.mkdir(parents=True)
    return dirName


def red_reliability(
    cutoffs=(0.4, 0.7),
    ises=None,
    ipcss=None,
    include_manual=True,
    use_s2p=False,
    s2p_cutoff=0.65,
    **kwConditions,
):
    """
    Method for returning a list of reliability indices and red cell assignments

    If ises (session iterable) and ipcss (placeCellSingleSession iterable) are not
    passed it will create them using the kwConditions

    If use_s2p=True, ignores the standard red cell detection indexing and determines
    red cell identity by whether the s2p red cell method exceeds s2p_cutoff!

    include_manual determines if red cell indices include manual annotations
    kwConditions are passed to vrAnalysis/database/getTable via sessiondb.iterSessions()
    It automatically assumes that imaging=True and vrRegistration=True
    """

    # generate session and analysis iterables if not provided
    remake_ipcss = ipcss is None  # if it is None, we have to remake it
    if ises is None:
        ises = sessiondb.iterSessions(imaging=True, vrRegistration=True, **kwConditions)
        remake_ipcss = True  # if we remade the session iterable, remake pcss iterable even if provided

    if remake_ipcss:
        ipcss = []
        for ses in tqdm(ises):
            ipcss.append(placeCellSingleSession(ses))

    # initialize lists for storing the data
    ses_name = []
    ses_per_mouse = []
    red_idx = []
    env_nums = []
    relmse = []
    relcor = []

    miceInSessions = sessiondb.miceInSessions(ises)
    mouseCounter = dict(zip(miceInSessions, [0] * len(miceInSessions)))

    # iterate through requested sessions and store data
    for ses, pcss in zip(ises, ipcss):
        c_use_rois = pcss.idxUseROI  # boolean array of ROIs within target planes
        ses_name.append(str(ses))
        ses_per_mouse.append(mouseCounter[ses.mouseName])
        if use_s2p:
            # get red cell indices for this session (within target planes) using s2p output only
            red_idx.append(ses.loadone("mpciROIs.redS2P")[c_use_rois] >= s2p_cutoff)
        else:
            # get red cell indices for this session (within target planes) using standard red cell indices
            red_idx.append(ses.getRedIdx(include_manual=include_manual)[c_use_rois])
        env_nums.append(pcss.environments)
        relmse.append(pcss.relmse)
        relcor.append(pcss.relcor)

        mouseCounter[ses.mouseName] += 1

    # organize everything by environment
    environments = np.unique(np.concatenate(env_nums))
    inenv_per_mouse = [[-1 for _ in range(len(environments))] for _ in range(len(ises))]
    # inenv_per_mouse = -np.ones((len(environments), len(ises)))
    mouseEnvironmentCounter = dict(
        zip(
            miceInSessions,
            [[0 for _ in range(len(environments))] for _ in range(len(miceInSessions))],
        )
    )
    for ii, (en, rm, rc) in enumerate(zip(env_nums, relmse, relcor)):
        env_match = [np.where(environments == e)[0] for e in en]
        assert all(
            [len(e) == 1 for e in env_match]
        ), f"In session {ses_name[ii]}, environments have an error (environments:{en}), (all environments:{environments})"

        # global environment index
        idx = [e[0] for e in env_match]

        for i in idx:
            # if mouse experienced environment, then indicate how many times the mouse has seen it before
            inenv_per_mouse[ii][i] = mouseEnvironmentCounter[ises[ii].mouseName][i]
            mouseEnvironmentCounter[ises[ii].mouseName][i] += 1  # and update for next visit

        # remake reliability arrays indexing by global environment
        c_num_rois = rm.shape[1]
        relmse[ii] = np.full((len(environments), c_num_rois), np.nan)
        relcor[ii] = np.full((len(environments), c_num_rois), np.nan)

        for j, i in enumerate(idx):
            relmse[ii][i] = rm[j]
            relcor[ii][i] = rc[j]

    # methods for organizing by mouse
    def getMouseElements(var, idx):
        return [v for v, i in zip(var, idx) if i]

    def getEnvironmentSorted(inenv):
        first_use = np.argmax(inenv >= 0, axis=0).astype(float)
        used = np.any(inenv >= 0, axis=0)
        first_use[~used] = np.inf
        env_order = np.argsort(first_use)[used]
        env_sorted = sorted([i for i, u in enumerate(used) if u], key=lambda x: env_order[x])
        return env_sorted

    # now organize by mouse
    miceInSession = sorted(sessiondb.miceInSessions(ises))
    ctl_reliable = [[] for _ in range(len(miceInSession))]
    red_reliable = [[] for _ in range(len(miceInSession))]
    env_sort = [[] for _ in range(len(miceInSession))]

    for ii, mouse in enumerate(miceInSession):
        idx_mouse = [mouse in sn for sn in ses_name]
        c_inenv = np.stack(getMouseElements(inenv_per_mouse, idx_mouse))
        c_red_idx = getMouseElements(red_idx, idx_mouse)
        c_relmse = getMouseElements(relmse, idx_mouse)
        c_relcor = getMouseElements(relcor, idx_mouse)
        c_envsort = getEnvironmentSorted(c_inenv)

        c_ctl_reliable = np.full(c_inenv.shape, np.nan)
        c_red_reliable = np.full(c_inenv.shape, np.nan)
        for idxses in range(c_inenv.shape[0]):
            for idxenv in range(c_inenv.shape[1]):
                if not c_inenv[idxses, idxenv] >= 0:
                    continue
                pass_mse = c_relmse[idxses][idxenv] > cutoffs[0]
                pass_cor = c_relcor[idxses][idxenv] > cutoffs[1]
                c_ctl_reliable[idxses, idxenv] = 100 * np.sum((pass_mse & pass_cor) & ~c_red_idx[idxses]) / np.sum(~c_red_idx[idxses])
                c_red_reliable[idxses, idxenv] = 100 * np.sum((pass_mse & pass_cor) & c_red_idx[idxses]) / np.sum(c_red_idx[idxses])

        env_sort[ii] = c_envsort
        ctl_reliable[ii] = c_ctl_reliable
        red_reliable[ii] = c_red_reliable

    # return data
    return miceInSession, env_sort, ctl_reliable, red_reliable


def plot_reliable_difference(
    cutoffs=(0.4, 0.7),
    withSave=False,
    withShow=True,
    ises=None,
    ipcss=None,
    include_manual=True,
    use_s2p=False,
    s2p_cutoff=0.65,
    **kwConditions,
):
    """plot difference in reliability for red and control cells across environments for all mice"""

    # get measurements of red reliability
    arguments = {
        "cutoffs": cutoffs,
        "ises": ises,
        "ipcss": ipcss,
        "include_manual": include_manual,
        "use_s2p": use_s2p,
        "s2p_cutoff": s2p_cutoff,
    }
    miceInSession, env_sort, ctl_reliable, red_reliable = red_reliability(**arguments, **kwConditions)

    numMice = len(miceInSession)
    numEnvs = ctl_reliable[0].shape[1]

    if numEnvs == 3:
        cmap = lambda x: ["k", "r", "b"][x]
    else:
        cmap = mpl.colormaps["brg"].resampled(numEnvs)

    labelSize = 18
    slabelSize = 12

    plt.close("all")
    fig, ax = plt.subplots(1, numMice, figsize=(4 * numMice, 4), layout="constrained")
    for ii, (mouse, esort, crel, rrel) in enumerate(zip(miceInSession, env_sort, ctl_reliable, red_reliable)):
        c_num_sessions = crel.shape[0]
        for idxorder, idxenv in enumerate(esort):
            ax[ii].plot(
                range(c_num_sessions),
                rrel[:, idxenv] - crel[:, idxenv],
                color=cmap(idxorder),
                lw=1.5,
                marker=".",
                markersize=14,
                label=f"env {idxorder}",
            )

        maxAbsDiff = np.nanmax(np.abs(rrel - crel)) * 1.1
        ax[ii].axhline(color="k", linestyle="--", lw=0.5)
        ax[ii].set_ylim(-maxAbsDiff, maxAbsDiff)

        ax[ii].tick_params(axis="x", labelsize=slabelSize)
        ax[ii].tick_params(axis="y", labelsize=slabelSize)
        ax[ii].set_xlabel("Session #", fontsize=labelSize)
        if ii == 0:
            ax[ii].set_ylabel("Red - Control Reliability (%)", fontsize=labelSize)
            ax[ii].legend(loc="lower left", fontsize=slabelSize)
        ax[ii].set_title(mouse, fontsize=labelSize)

    # Save figure if requested
    if withSave:
        print(f"Saving a plot of difference in reliable fraction of cells (all mice all environments)")
        append_string = "_wS2P" if use_s2p else ""
        plt.savefig(save_directory() / ("difference_reliable_fraction" + append_string))

    # Show figure if requested
    plt.show() if withShow else plt.close()


def plot_reliable_fraction(
    cutoffs=(0.4, 0.7),
    withSave=False,
    withShow=True,
    ises=None,
    ipcss=None,
    include_manual=True,
    use_s2p=False,
    s2p_cutoff=0.65,
    **kwConditions,
):
    """plot difference in reliability for red and control cells across environments for all mice"""

    # get measurements of red reliability
    arguments = {
        "cutoffs": cutoffs,
        "ises": ises,
        "ipcss": ipcss,
        "include_manual": include_manual,
        "use_s2p": use_s2p,
        "s2p_cutoff": s2p_cutoff,
    }
    miceInSession, env_sort, ctl_reliable, red_reliable = red_reliability(**arguments, **kwConditions)

    numMice = len(miceInSession)
    numEnvs = ctl_reliable[0].shape[1]

    if numEnvs == 3:
        cmap = lambda x: ["k", "r", "b"][x]
    else:
        cmap = mpl.colormaps["brg"].resampled(numEnvs)

    labelSize = 16

    plt.close("all")
    fig, ax = plt.subplots(numEnvs, numMice, figsize=(4 * numMice, 4 * numEnvs), layout="constrained")
    if numEnvs == 1:
        ax = np.reshape(ax, (1, -1))

    for ii, (mouse, esort, crel, rrel) in enumerate(zip(miceInSession, env_sort, ctl_reliable, red_reliable)):
        c_num_sessions = crel.shape[0]
        for idxorder, idxenv in enumerate(esort):
            ax[idxorder, ii].plot(
                range(c_num_sessions),
                crel[:, idxenv],
                color="k",
                lw=1.5,
                marker=".",
                markersize=14,
                label="control",
            )
            ax[idxorder, ii].plot(
                range(c_num_sessions),
                rrel[:, idxenv],
                color="r",
                lw=1.5,
                marker=".",
                markersize=14,
                label="red",
            )
            ax[idxorder, ii].set_xlim(-0.5, c_num_sessions - 0.5)
            ax[idxorder, ii].set_ylim(0, 1.1 * np.max((np.nanmax(crel), np.nanmax(rrel))))

            ax[idxorder, ii].set_xlabel("Session #", fontsize=labelSize)

            if ii == 0:
                ax[idxorder, ii].set_ylabel(f"Reliable % - Env {idxorder}", fontsize=labelSize)

            ax[idxorder, ii].legend(loc="lower left")

            if idxorder == 0:
                ax[idxorder, ii].set_title(mouse, fontsize=labelSize)

    # Save figure if requested
    if withSave:
        print(f"Saving a plot of reliable fraction of cells (all mice all environments)")
        append_string = "_wS2P" if use_s2p else ""
        plt.savefig(save_directory() / ("reliable_fraction" + append_string))

    # Show figure if requested
    plt.show() if withShow else plt.close()


class placeCellSingleSession(standardAnalysis):
    """
    Performs basic place cell (and behavioral) analysis on single sessions.

    Takes as required input a vrexp object. Optional inputs define parameters of analysis,
    including which activity to run measurement on (could be deconvolvedOasis, or neuropilF, for example).

    Standard usage:
    ---------------
    == I just started this file, will populate standard usage later! ==
    """

    def __init__(
        self,
        vrexp,
        onefile="mpci.roiActivityDeconvolvedOasis",
        autoload=True,
        keep_planes=[1, 2, 3, 4],
        distStep=1,
        speedThreshold=5,
        numcv=2,
        standardizeSpks=True,
        smoothWidth=1,
        smoothWidth_reliability=10,
        full_trial_flexibility=3,
        use_all_rois=False,
    ):
        self.name = "placeCellSingleSession"
        self.onefile = onefile
        self.vrexp = vrexp
        self.distStep = distStep
        self.speedThreshold = speedThreshold or 0.0
        self.numcv = numcv
        self.standardizeSpks = standardizeSpks
        self.smoothWidth = smoothWidth
        self.smoothWidth_reliability = smoothWidth_reliability
        self.full_trial_flexibility = full_trial_flexibility
        self.keep_planes = keep_planes if keep_planes is not None else [i for i in range(len(vrexp.value["roiPerPlane"]))]
        self.use_all_rois = use_all_rois

        # automatically load data
        self.dataloaded = False
        self.load_fast_data()
        if autoload:
            self.load_data()

    def envnum_to_idx(self, envnum, validate=True):
        """
        convert list of environment numbers to indices of environment within this session
        e.g. if session has environments [1,3,4], and environment 3 is requested, turn it into index 1
        """
        envnum = helpers.check_iterable(envnum)
        envidx = [np.where(self.environments == ev)[0][0] if ev in self.environments else np.nan for ev in envnum]
        if validate:
            assert all([~np.isnan(ei) for ei in envidx]), f"requested environment(s) not in session, contains={self.environments}, requested={envnum}"
        return envidx

    def get_plane_idx(self, keep_planes=None):
        """simple method for getting index to ROIs in plane"""
        if keep_planes is not None:
            self.keep_planes = keep_planes

        # get idx of rois within keep planes
        stackPosition = self.vrexp.loadone("mpciROIs.stackPosition")
        roiPlaneIdx = stackPosition[:, 2].astype(np.int32)  # plane index

        # figure out which ROIs are in the target planes
        self.idxUseROI = np.any(np.stack([roiPlaneIdx == pidx for pidx in self.keep_planes]), axis=0)
        self.roiPlaneIdx = roiPlaneIdx[self.idxUseROI]
        self.numROIs = self.vrexp.getNumROIs(self.keep_planes)
        return self.idxUseROI

    def load_fast_data(self):
        # get environment data
        self.trial_envnum = self.vrexp.loadone("trials.environmentIndex")
        self.environments = np.unique(self.trial_envnum)
        self.numEnvironments = len(self.environments)
        # get behavioral data
        self.load_behavioral_data()
        # get index to ROIs
        # TODO: confirm that this can be removed from fast_data()!: self.get_plane_idx()

    def clear_data(self):
        """method for clearing data to free up memory and/or resetting variables"""
        attrs_to_delete = [
            "idxUseROI",
            "numROIs",
            "occmap",
            "speedmap",
            "rawspkmap",
            "sample_counts",
            "spkmap",
            "distedges",
            "distcenters",
            "numTrials",
            "boolFullTrials",
            "idxFullTrials",
            "idxFullTrialEachEnv",
            "train_idx",
            "test_idx",
            "relmse",
            "relcor",
            "test_relmse",
            "test_relcor",
        ]

        for attr in attrs_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)

        self.dataloaded = False

    def load_behavioral_data(self, distStep=None, speedThreshold=None, full_trial_flexibility=None):
        """load standard behavioral data for basic place cell analysis"""
        # update analysis parameters if requested
        if distStep is not None:
            self.distStep = distStep
        if speedThreshold is not None:
            self.speedThreshold = speedThreshold
        if full_trial_flexibility is not None:
            self.full_trial_flexibility = full_trial_flexibility

        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {
            "distStep": self.distStep,
            "speedThreshold": self.speedThreshold,
            "speedSmoothing": self.smoothWidth,
            "get_spkmap": False,
        }
        self.occmap, self.speedmap, _, _, self.sample_counts, self.distedges = helpers.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)
        self.distcenters = helpers.edge2center(self.distedges)

        self.numTrials = self.occmap.shape[0]

        # find out which trials the mouse explored the whole environment
        if self.full_trial_flexibility is None:
            # if full trial flexiblity is None, then they need to have visited every bin
            idx_to_required_bins = np.arange(self.occmap.shape[1])
        else:
            start_idx = np.where(self.distedges >= self.full_trial_flexibility)[0][0]
            end_idx = np.where(self.distedges <= self.distedges[-1] - self.full_trial_flexibility)[0][-1]
            idx_to_required_bins = np.arange(start_idx, end_idx)

        self.boolFullTrials = np.all(~np.isnan(self.occmap[:, idx_to_required_bins]), axis=1)
        self.idxFullTrials = np.where(self.boolFullTrials)[0]
        self.idxFullTrialEachEnv = [np.where(self.boolFullTrials & (self.trial_envnum == env))[0] for env in self.environments]

    def load_data(
        self,
        onefile=None,
        distStep=None,
        speedThreshold=None,
        numcv=None,
        keep_planes=None,
        with_test=False,
        full_trial_flexibility=None,
        new_split=True,
        keep_buffer=False,
    ):
        """load standard data for basic place cell analysis"""
        # update onefile if using a different measure of activity
        if onefile is not None:
            self.onefile = onefile

        # update analysis parameters if requested
        if distStep is not None:
            self.distStep = distStep
        if speedThreshold is not None:
            self.speedThreshold = speedThreshold
        if numcv is not None:
            self.numcv = numcv
        if keep_planes is not None:
            self.keep_planes = keep_planes
        if full_trial_flexibility is not None:
            self.full_trial_flexibility = full_trial_flexibility

        _ = self.get_plane_idx(keep_planes=self.keep_planes)

        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {
            "distStep": self.distStep,
            "onefile": self.onefile,
            "speedThreshold": self.speedThreshold,
            "standardizeSpks": self.standardizeSpks,
            "idxROIs": self.idxUseROI if not self.use_all_rois else None,
            "speedSmoothing": self.smoothWidth,
        }
        self.occmap, self.speedmap, _, self.rawspkmap, self.sample_counts, self.distedges = helpers.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)

        self.distcenters = helpers.edge2center(self.distedges)

        self.numTrials = self.occmap.shape[0]

        self.boolFullTrials, self.idxFullTrials, self.idxFullTrialEachEnv = self._return_trial_indices(
            self.occmap, self.distedges, self.full_trial_flexibility
        )

        # report that data has been loaded
        self.dataloaded = True

        # measure reliability
        self.measure_reliability(new_split=new_split, with_test=with_test)

        if not keep_buffer:
            self.vrexp.clearBuffer()

    def _return_trial_indices(self, occmap, distedges, full_trial_flexibility=None):
        """helper for determining with trials the mouse explored the whole environment"""
        if full_trial_flexibility is not None:
            self.full_trial_flexibility = full_trial_flexibility

        assert occmap.shape[0] == self.vrexp.value["numTrials"], "occmap doesn't have the same number of trials as the session object indicates!"

        # find out which trials the mouse explored the whole environment
        if self.full_trial_flexibility is None:
            # if full trial flexiblity is None, then they need to have visited every bin
            idx_to_required_bins = np.arange(self.occmap.shape[1])
        else:
            start_idx = np.where(distedges >= self.full_trial_flexibility)[0][0]
            end_idx = np.where(distedges <= distedges[-1] - self.full_trial_flexibility)[0][-1]
            idx_to_required_bins = np.arange(start_idx, end_idx)

        boolFullTrials = np.all(~np.isnan(occmap[:, idx_to_required_bins]), axis=1)
        idxFullTrials = np.where(boolFullTrials)[0]
        idxFullTrialEachEnv = [np.where(boolFullTrials & (self.trial_envnum == env))[0] for env in self.environments]
        return boolFullTrials, idxFullTrials, idxFullTrialEachEnv

    def prepare_spks(self, onefile="mpci.roiActivityDeconvolvedOasis", standardize=True):
        """get spks (imaging frames x neurons) for the session, only using neurons in standard planes"""
        spks = self.vrexp.loadone(onefile)[:, self.idxUseROI]
        if standardize:
            if "deconvolved" in onefile:
                # If using deconvolved traces, should have zero baseline
                spks = spks / fs.std(spks, axis=0, keepdims=True)

            else:
                # If using fluorescence traces, should have non-zero baseline
                idx_zeros = fs.std(spks, axis=0) == 0
                spks = fs.median_zscore(spks, axis=0)
                spks[:, idx_zeros] = 0

        return spks

    @prepare_data
    def load_spkmap(
        self,
        envnum=None,
        activity_type="mpci.roiActivityDeconvolvedOasis",
        average=True,
        trials="full",
        pop_nan=True,
        new_split=False,
        split_params={},
        return_params=False,
    ):
        """
        method for loading spkmaps from a list of environments
        similar to get_spkmap (see it's docstring) but load spkmaps from session objects
        """
        # use all environments if none requested
        if envnum is None:
            envnum = np.copy(self.environments)

        # convert envnum into iterable index to environment in session
        envnum = helpers.check_iterable(envnum)  # make sure envnum is iterable
        envidx = self.envnum_to_idx(envnum)  # convert environment numbers to indices

        # load spkmaps
        if return_params:
            env_spkmap, env_params = self.vrexp.load_spkmaps(activity_type=activity_type, envnum=envnum, return_params=True)
        else:
            env_spkmap = self.vrexp.load_spkmaps(activity_type=activity_type, envnum=envnum)

        # get requested trials
        if (trials == "train" or trials == "test") and new_split:
            self.define_train_test_split(**split_params)

        if trials == "train":
            idx_keep_trials = [np.isin(self.idxFullTrialEachEnv[ei], self.train_idx[ei]) for ei in envidx]
        elif trials == "test":
            idx_keep_trials = [np.isin(self.idxFullTrialEachEnv[ei], self.test_idx[ei]) for ei in envidx]
        elif trials == "full":
            idx_keep_trials = [np.ones(len(self.idxFullTrialEachEnv[ei]), dtype=bool) for ei in envidx]
        else:
            raise ValueError(f"Didn't recognize trials option (received '{trials}', expected 'full', 'train', or 'test')")

        env_spkmap = [esm[:, ikt] for esm, ikt in zip(env_spkmap, idx_keep_trials)]

        # remove positions with nans in any spkmap if requested
        if pop_nan:
            marginal_axis = 0 if average else (0, 1)
            nan_positions = np.any(np.stack([np.any(np.isnan(esm), axis=marginal_axis) for esm in env_spkmap]), axis=0)
            env_spkmap = [esm[..., ~nan_positions] for esm in env_spkmap]

        # return spkmap
        if return_params:
            return env_spkmap, env_params
        else:
            return env_spkmap

    @prepare_data
    def get_spkmap(
        self,
        envnum=None,
        average=True,
        smooth=None,
        trials="full",
        pop_nan=True,
        new_split=False,
        split_params={},
        rawspkmap=None,
        occmap=None,
    ):
        """
        method for getting a spkmap from a list of environments

        can average across trials before smoothing & dividing (if requested)
        can smooth across spatial positions if requested (smooth is None for no smoothing
        and a number corresponding to the gaussian filter width in cm for smoothing)

        can subselect trials, either "full", "train", or "test", and reset the train/test split with
        "new_split" if requested (using split_params for how to divide trials into train/test)

        how it works:
        will get the occmap and spkmap for each environment (in a list), using the requested trials
        will sum across trials if requested
        will smooth across positions if requested
        then will divide the spkmap by occupancy map to get a rate map in each position

        transposes output to have shape (num_ROIs, num_trials, num_spatial_bins)
        or (num_ROIs, num_spatial_bins) if average=True

        if pop_nan is set to True, will figure out which positions have nans in any spkmap
        and will remove those positions from all spkmaps

        if rawspkmap isn't provided, will use the "rawspkmap" attribute of self
        if it is provided, will use that instead

        if occmap isn't provided, will use the "occmap" attribute of self
        if it is provided, will use that instead
        """
        # use all environments if none requested
        if envnum is None:
            envnum = np.copy(self.environments)

        # convert envnum into iterable index to environment in session
        envnum = helpers.check_iterable(envnum)  # make sure envnum is iterable
        envidx = self.envnum_to_idx(envnum)  # convert environment numbers to indices

        # pick the raw spkmap to use
        if occmap is None:
            occmap = self.occmap
            if rawspkmap is not None:
                # check if trials and positions match
                # don't check if numROIs match because the rawspkmap might be for something different
                assert rawspkmap.shape[0] == occmap.shape[0], "number of trials isn't equal"
                assert rawspkmap.shape[1] == occmap.shape[1], "number of spatial bins isn't equal"
            else:
                rawspkmap = self.rawspkmap
        else:
            assert occmap.shape[0] == self.occmap.shape[0], "provided occmap and self.occmap trial numbers don't match"
            if rawspkmap is not None:
                assert rawspkmap.shape[0] == occmap.shape[0], "rawspkmap and occmap trial numbers don't match"
                assert rawspkmap.shape[1] == occmap.shape[1], "rawspkmap and occmap spatial bins don't match"
            else:
                rawspkmap = self.rawspkmap

        # get occupancy and rawspkmap from requested trials (or across environments)
        if isinstance(trials, np.ndarray):
            # Convert trial array to list of lists by environment
            env_of_trial = self.trial_envnum[trials]
            trials_by_env = [trials[env_of_trial == env] for env in self.environments]
            env_occmap = [occmap[trials_by_env[ei]] for ei in envidx]
            env_spkmap = [rawspkmap[trials_by_env[ei]] for ei in envidx]

        elif trials == "full":
            env_occmap = [occmap[self.idxFullTrialEachEnv[ei]] for ei in envidx]
            env_spkmap = [rawspkmap[self.idxFullTrialEachEnv[ei]] for ei in envidx]

        elif trials == "train":
            if new_split:
                self.define_train_test_split(**split_params)
            env_occmap = [occmap[self.train_idx[ei]] for ei in envidx]
            env_spkmap = [rawspkmap[self.train_idx[ei]] for ei in envidx]

        elif trials == "test":
            if new_split:
                self.define_train_test_split(**split_params)
            env_occmap = [occmap[self.test_idx[ei]] for ei in envidx]
            env_spkmap = [rawspkmap[self.test_idx[ei]] for ei in envidx]

        else:
            raise ValueError(f"Didn't recognize trials option (received '{trials}', expected 'full', 'train', or 'test')")

        # get spkmaps for each environment
        env_spkmap = [self._make_spkmap(maps=(eom, esm), average=average, smooth=smooth) for eom, esm in zip(env_occmap, env_spkmap)]

        # remove positions with nans in any spkmap if requested
        if pop_nan:
            marginal_axis = 0 if average else (0, 1)
            nan_positions = np.any(np.stack([np.all(np.isnan(esm), axis=marginal_axis) for esm in env_spkmap]), axis=0)
            env_spkmap = [esm[..., ~nan_positions] for esm in env_spkmap]

        # return spkmap
        return env_spkmap

    @prepare_data
    def _make_spkmap(self, maps=None, average=False, smooth=None):
        """
        central method for doing averaging, smoothing, correcting, and transposing for spkmaps
        will use self.occmap and self.rawspkmap if None provided
        """
        if maps is None:
            occmap, spkmap = self.occmap, self.rawspkmap
        else:
            occmap, spkmap = maps
            assert occmap.shape[0] == spkmap.shape[0], "number of trials isn't equal"
            assert occmap.shape[1] == spkmap.shape[1], "number of spatial bins isn't equal"

        # average (sum, because divide happens later) across trials if requested
        if average:
            occmap = fs.nansum(occmap, axis=0, keepdims=True)
            spkmap = fs.nansum(spkmap, axis=0, keepdims=True)

        # do smoothing across spatial positions if requested
        if smooth is not None:
            # if smoothing, nans will get confusing so we need to reset nans with 0s then reset them
            occ_idxnan = np.isnan(occmap)
            occmap[occ_idxnan] = 0
            spkmap[occ_idxnan] = 0

            # do smoothing
            kk = helpers.getGaussKernel(self.distcenters, smooth)
            occmap = helpers.convolveToeplitz(occmap, kk, axis=1)
            spkmap = helpers.convolveToeplitz(spkmap, kk, axis=1)

            # reset nans
            occmap[occ_idxnan] = np.nan
            spkmap[occ_idxnan] = np.nan

        # correct spkmap by occupancy
        spkmap = helpers.correctMap(occmap, spkmap)

        # reshape to (numROIs, numTrials, numPositions)
        spkmap = spkmap.transpose(2, 0, 1)

        # squeeze out trials dimension if averaging
        if average:
            spkmap = spkmap.squeeze()

        return spkmap

    def define_train_test_split(self, total_folds=3, train_folds=2):
        """
        method for creating a train/test split

        how it works:
        will divide the trials (for each environment, using idxFullTrialEachEnv) into
        N=total_folds groups. Then, will take M=train_folds of these groups and concatenate
        to make the "train_idx", a list of trials used for "training" and the rest into
        "test_idx", a list of trials used for "testing".

        In practice, I usually use total_folds=3 and train_folds=2 because I measure reliability
        using the train_idx which requires it's own train/test split, then make snake plots
        with the test_idx.
        """
        assert train_folds < total_folds, "train_folds can't be >= total_folds"
        foldIdx = [helpers.cvFoldSplit(idxTrialEnv, total_folds) for idxTrialEnv in self.idxFullTrialEachEnv]
        self.train_folds = train_folds
        self.total_folds = total_folds
        # each of these is a list of training/testing trials for each environment
        self.train_idx = [np.concatenate(fidx[:train_folds]) for fidx in foldIdx]
        self.test_idx = [np.concatenate(fidx[train_folds:]) for fidx in foldIdx]

    @prepare_data
    def measure_reliability(self, new_split=True, with_test=False, smoothWidth=-1, total_folds=3, train_folds=2, rawspkmap=None, return_only=False):
        """method for measuring reliability in each environment"""
        if smoothWidth == -1:
            smoothWidth = self.smoothWidth_reliability

        # create a train/test split
        if new_split:
            self.define_train_test_split(total_folds=total_folds, train_folds=train_folds)

        # measure reliability of spiking (in two ways)
        spkmap = self.get_spkmap(average=False, smooth=smoothWidth, trials="train", rawspkmap=rawspkmap)
        relmse, relcor = helpers.named_transpose([helpers.measureReliability(smap, numcv=self.numcv) for smap in spkmap])
        relloo = [helpers.reliability_loo(smap) for smap in spkmap]

        # If not returning values only, save them to the object
        if not return_only:
            self.relmse, self.relcor, self.relloo = np.stack(relmse), np.stack(relcor), np.stack(relloo)

        if with_test:
            # measure on test trials
            spkmap = self.get_spkmap(average=False, smooth=smoothWidth, trials="test", rawspkmap=rawspkmap)
            test_relmse, test_relcor = helpers.named_transpose([helpers.measureReliability(smap, numcv=self.numcv) for smap in spkmap])
            test_relloo = [helpers.reliability_loo(smap) for smap in spkmap]
            if not return_only:
                self.test_relmse, self.test_relcor, self.test_relloo = np.stack(test_relmse), np.stack(test_relcor), np.stack(test_relloo)
        else:
            # Alert the user that the training data was (re)calculated without testing
            if not return_only:
                self.test_relmse, self.test_relcor, self.test_relloo = None, None, None

        if return_only:
            reliability = dict(
                relmse=np.stack(relmse),
                relcor=np.stack(relcor),
                relloo=np.stack(relloo),
            )
            if with_test:
                reliability.update(
                    dict(
                        test_relmse=np.stack(test_relmse),
                        test_relcor=np.stack(test_relcor),
                        test_relloo=np.stack(test_relloo),
                    )
                )
            return reliability

    @prepare_data
    def get_reliability_values(self, envnum=None, with_test=False, rawspkmap=None):
        """support for getting reliability values from requested or all environments"""
        if envnum is None:
            envnum = copy(self.environments)  # default environment is all of them
        envnum = helpers.check_iterable(envnum)  # make sure it's an iterable
        envidx = self.envnum_to_idx(envnum)  # convert environment numbers to indices

        # If rawspkmap isn't provided, use the reliability values stored in the object
        # (these are calculated with the standard self.onefile set at initialization)
        if rawspkmap is None:
            relmse = self.relmse
            relcor = self.relcor
            relloo = self.relloo
            if with_test:
                test_relmse = self.test_relmse
                test_relcor = self.test_relcor
                test_relloo = self.test_relloo
        else:
            # if rawspkmap is provided, calculate reliability values using this spkmap
            reliability_output = self.measure_reliability(return_only=True, with_test=with_test, rawspkmap=rawspkmap)
            relmse = reliability_output["relmse"]
            relcor = reliability_output["relcor"]
            relloo = reliability_output["relloo"]
            if with_test:
                test_relmse = reliability_output["test_relmse"]
                test_relcor = reliability_output["test_relcor"]
                test_relloo = reliability_output["test_relloo"]

        mse = [relmse[ii] for ii in envidx]
        cor = [relcor[ii] for ii in envidx]
        loo = [relloo[ii] for ii in envidx]

        # if not with_test trials, just return mse/cor on train trials
        if not with_test:
            return mse, cor, loo

        # if with_test, get these too and return them all
        msetest = [test_relmse[ii] for ii in envidx]
        cortest = [test_relcor[ii] for ii in envidx]
        lootest = [test_relloo[ii] for ii in envidx]
        return mse, cor, loo, msetest, cortest, lootest

    @prepare_data
    def get_reliable(self, envnum=None, cutoffs=None, maxcutoffs=None, rawspkmap=None):
        """central method for getting reliable cells from list of environments (by environment index)"""
        if envnum is None:
            envnum = copy(self.environments)  # default environment is all of them
        envnum = helpers.check_iterable(envnum)  # make sure it's an iterable
        envidx = self.envnum_to_idx(envnum)  # convert environment numbers to indices
        cutoffs = (-np.inf, -np.inf) if cutoffs is None else cutoffs
        maxcutoffs = (np.inf, np.inf) if maxcutoffs is None else maxcutoffs

        relmse, relcor = self.get_reliability_values(with_test=False, rawspkmap=rawspkmap)
        idx_reliable = [
            (relmse[ei] >= cutoffs[0]) & (relcor[ei] >= cutoffs[1]) & (relmse[ei] <= maxcutoffs[0]) & (relcor[ei] <= maxcutoffs[1]) for ei in envidx
        ]
        return idx_reliable

    def get_place_field(self, spkmap, method="max", force_with_negative=False):
        """
        get sorting index and place field location for spkmap

        If spkmap has shape: (numROIs, numTrials, numPositions) will average across trials
        If spkmap has shape: (numROIs, numPositions) will use as is

        Returns
        -------
        pfloc : np.ndarray
            place field location for each ROI
        pfidx : np.ndarray
            sorting index for each ROI
        """
        assert method == "com" or method == "max", f"invalid method ({method}), must be either 'com' or 'max'"

        # Get ROI x Position profile of activity for each ROI as a function of position if trials included in spkmap
        if spkmap.ndim == 3:
            spkmap = fs.nanmean(spkmap, axis=1)

        # if method is 'com' (=center of mass), use weighted mean to get place field location
        if method == "com":
            # note that this can generate buggy behavior if spkmap isn't based on mostly positive signals!
            if not force_with_negative and np.any(spkmap < 0):
                raise ValueError("cannot use center of mass method when spkmap data is negative")
            nonnegative_map = np.maximum(spkmap, 0)
            pfloc = fs.nansum(nonnegative_map * self.distcenters.reshape(1, -1), axis=1) / fs.nansum(nonnegative_map, axis=1)

        # if method is 'max' (=maximum rate), use maximum to get place field location
        if method == "max":
            pfloc = np.nanargmax(spkmap, axis=1)

        # Then sort...
        pfidx = np.argsort(pfloc)

        return pfloc, pfidx

    def get_roicat_latents(self):
        """get latents describing ROIs from ROINet"""
        latent_path = self.vrexp.roicatPath() / "roinet_latents.npy"
        if not latent_path.exists():
            raise FileNotFoundError(f"could not find latents file at {latent_path}")
        return np.load(latent_path)

    def get_roicat_master_embeddings(self):
        """get embeddings of ROIs latents from ROINet (from the master UMAP of all ROIs)"""
        embeddings_path = self.vrexp.roicatPath() / "master_umap_embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"could not find latents file at {embeddings_path}")
        return np.load(embeddings_path)

    @prepare_data
    def make_snake(self, envnum=None, reliable=True, cutoffs=(0.4, 0.7), maxcutoffs=None, method="max", rawspkmap=None):
        """make snake data from train and test sessions, for particular environment if requested"""
        # default environment is all of them
        if envnum is None:
            envnum = copy(self.environments)

        # envnum must be an iterable
        envnum = helpers.check_iterable(envnum)

        # get spkmaps for requested environments
        train_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="train", rawspkmap=rawspkmap)
        test_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="test", rawspkmap=rawspkmap)

        # filter by reliable ROIs if requested
        if reliable:
            # get idx of reliable ROIs for each environment
            idx_reliable = self.get_reliable(envnum, cutoffs=cutoffs, maxcutoffs=maxcutoffs)
            train_profile = [tp[ir] for tp, ir in zip(train_profile, idx_reliable)]
            test_profile = [tp[ir] for tp, ir in zip(test_profile, idx_reliable)]

        # get place field indices
        train_pfidx = [self.get_place_field(train_prof, method=method)[1] for train_prof in train_profile]

        # make train and test snakes by sorting and squeezing out trials
        train_snake = [train_prof[tpi] for train_prof, tpi in zip(train_profile, train_pfidx)]
        test_snake = [test_prof[tpi] for test_prof, tpi in zip(test_profile, train_pfidx)]

        # :)
        return train_snake, test_snake

    @prepare_data
    def make_remap_data(self, reliable=True, cutoffs=(0.4, 0.7), maxcutoffs=None, method="max", rawspkmap=None):
        """make snake data across environments with remapping indices (for N environments, an NxN grid of snakes and indices)"""
        # get index to each environment for this session
        envnum = helpers.check_iterable(copy(self.environments))
        num_envs = len(envnum)

        # get train/test spkmap profile for each environment (average across trials)
        train_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="train", rawspkmap=rawspkmap)
        test_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="test", rawspkmap=rawspkmap)
        full_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="full", rawspkmap=rawspkmap)

        # filter by reliable ROIs if requested
        if reliable:
            idx_reliable = self.get_reliable(envnum, cutoffs=cutoffs, maxcutoffs=maxcutoffs)
        else:
            idx_reliable = [np.ones(tp.shape[0], dtype=bool) for tp in train_profile]

        # get sorting index for each environment (including only reliable cells if requested)
        train_pfidx = [self.get_place_field(tp[ir], method=method)[1] for tp, ir in zip(train_profile, idx_reliable)]

        # make c-v snake plots across all environment combinations
        snake_plots = []
        for ii in range(num_envs):
            # sort by environment ii (on test trials if ii==jj and full trials if ii!=jj)
            c_plots = []
            for jj in range(num_envs):
                if ii == jj:
                    # snake of test trials for env @ii, sorted by pf on train trials in env @ii, filtered by reliable on train @ii if requested
                    c_snake = test_profile[ii][idx_reliable[ii]][train_pfidx[ii]]
                    c_plots.append(c_snake)
                else:
                    # snake of full trials for env @jj, sorted by pf on train trials in env @ii, filtered by reliable on @ii if requested
                    c_snake = full_profile[jj][idx_reliable[ii]][train_pfidx[ii]]
                    c_plots.append(c_snake)

            # add row to snake_plots
            snake_plots.append(c_plots)

        # :)
        return snake_plots

    @prepare_data
    def plot_snake(
        self,
        envnum=None,
        reliable=True,
        cutoffs=(0.4, 0.7),
        maxcutoffs=None,
        method="max",
        normalize=0,
        rewzone=True,
        interpolation="none",
        withShow=True,
        withSave=False,
    ):
        """method for plotting cross-validated snake plot"""
        # default environment is all of them
        if envnum is None:
            envnum = copy(self.environments)

        # envnum must be an iterable
        envnum = helpers.check_iterable(envnum)
        assert all([e in self.environments for e in envnum]), "envnums must be valid environment numbers within self.environments"

        # get number of environments
        numEnv = len(envnum)

        # make snakes and prepare plotting data
        train_snake, test_snake = self.make_snake(
            envnum=envnum,
            reliable=reliable,
            cutoffs=cutoffs,
            maxcutoffs=maxcutoffs,
            method=method,
        )
        extent = [[self.distedges[0], self.distedges[-1], 0, ts.shape[0]] for ts in train_snake]
        if normalize > 0:
            vmin, vmax = -np.abs(normalize), np.abs(normalize)
        elif normalize < 0:
            maxrois = np.concatenate([np.nanmax(np.abs(ts), axis=1) for ts in train_snake] + [np.nanmax(np.abs(ts), axis=1) for ts in test_snake])
            vmin, vmax = -np.percentile(maxrois, -normalize), np.percentile(maxrois, -normalize)
        else:
            magnitude = np.nanmax(np.abs(np.concatenate((np.concatenate(train_snake), np.concatenate(test_snake)))))
            vmin, vmax = -magnitude, magnitude

        cb_ticks = np.linspace(np.fix(vmin), np.fix(vmax), int(min(11, np.fix(vmax) - np.fix(vmin) + 1)))
        labelSize = 14
        cb_unit = r"$\sigma$" if self.standardizeSpks else "au"
        cb_label = f"Activity ({cb_unit})"

        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = helpers.environmentRewardZone(self.vrexp)
            rewPos = [rewPos[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rewHalfwidth = [rewHalfwidth[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rect_train = [
                mpl.patches.Rectangle((rp - rhw, 0), rhw * 2, ts.shape[0], edgecolor="none", facecolor="k", alpha=0.2)
                for rp, rhw, ts in zip(rewPos, rewHalfwidth, train_snake)
            ]
            rect_test = [
                mpl.patches.Rectangle((rp - rhw, 0), rhw * 2, ts.shape[0], edgecolor="none", facecolor="k", alpha=0.2)
                for rp, rhw, ts in zip(rewPos, rewHalfwidth, train_snake)
            ]

        plt.close("all")
        cmap = mpl.colormaps["bwr"]

        fig, ax = plt.subplots(numEnv, 3, width_ratios=[10, 10, 1], figsize=(9, 3 * numEnv), layout="constrained")

        if numEnv == 1:
            ax = np.reshape(ax, (1, -1))

        for idx, env in enumerate(envnum):
            cim = ax[idx, 0].imshow(
                train_snake[idx],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent[idx],
                aspect="auto",
                interpolation=interpolation,
            )
            if idx == numEnv - 1:
                ax[idx, 0].set_xlabel("Position (cm)", fontsize=labelSize)
            ax[idx, 0].set_ylabel(f"Env:{env}, ROIs", fontsize=labelSize)
            if idx == 0:
                ax[idx, 0].set_title("Train Trials", fontsize=labelSize)

            ax[idx, 1].imshow(
                test_snake[idx],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent[idx],
                aspect="auto",
                interpolation=interpolation,
            )
            if idx == numEnv - 1:
                ax[idx, 1].set_xlabel("Position (cm)", fontsize=labelSize)
            ax[idx, 1].set_ylabel("ROIs", fontsize=labelSize)
            if idx == 0:
                ax[idx, 1].set_title("Test Trials", fontsize=labelSize)

            if rewzone:
                ax[idx, 0].add_patch(rect_train[idx])
                ax[idx, 1].add_patch(rect_test[idx])

            fig.colorbar(cim, ticks=cb_ticks, orientation="vertical", cax=ax[idx, 2])
            ax[idx, 2].set_ylabel(cb_label, fontsize=labelSize)

        if withSave:
            if len(envnum) == len(self.environments):
                self.saveFigure(fig.number, f"snake_plot")
            else:
                print("If envnum is less than all the environments, you can't save with this program!")

        # Show figure if requested
        plt.show() if withShow else plt.close()

    def plot_remap_snakes(
        self,
        reliable=True,
        cutoffs=(0.4, 0.7),
        method="max",
        normalize=0,
        rewzone=True,
        interpolation="none",
        force_single_env=False,
        symmetric=True,
        withLabels=True,
        withShow=True,
        withSave=False,
    ):
        """method for plotting cross-validated snake plot"""
        # plotting remap snakes always uses all environments
        envnum = helpers.check_iterable(copy(self.environments))  # always use all environments (as an iterable)
        numEnv = len(envnum)

        if numEnv == 1 and not (force_single_env):
            print(f"Session {self.vrexp.sessionPrint()} only uses 1 environment, not plotting remap snakes")
            return None

        # make snakes
        snake_remap = self.make_remap_data(reliable=reliable, cutoffs=cutoffs, method=method)

        # prepare plotting data
        extent = lambda ii, jj: [
            self.distedges[0],
            self.distedges[-1],
            0,
            snake_remap[ii][jj].shape[0],
        ]
        if normalize > 0:
            vmin, vmax = -np.abs(normalize), np.abs(normalize)
        elif normalize < 0:
            maxrois = np.concatenate([np.concatenate([np.nanmax(np.abs(srp), axis=1) for srp in s_remap]) for s_remap in snake_remap])
            vmin, vmax = -np.percentile(maxrois, -normalize), np.percentile(maxrois, -normalize)
        else:
            magnitude = np.nanmax(np.abs(np.vstack([np.concatenate(srp) for srp in snake_remap])))
            vmin, vmax = -magnitude, magnitude

        if not symmetric:
            vmin = 0

        cb_ticks = np.linspace(np.fix(vmin), np.fix(vmax), int(min(11, np.fix(vmax) - np.fix(vmin) + 1)))
        labelSize = 14
        cb_unit = r"$\sigma$" if self.standardizeSpks else "au"
        cb_label = f"Activity ({cb_unit})"
        cols = "krb"

        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = helpers.environmentRewardZone(self.vrexp)
            rewPos = [rewPos[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rewHalfwidth = [rewHalfwidth[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rect = lambda ii, jj: mpl.patches.Rectangle(
                (rewPos[jj] - rewHalfwidth[jj], 0),
                rewHalfwidth[jj] * 2,
                snake_remap[ii][jj].shape[0],
                edgecolor="none",
                facecolor="k" if symmetric else cols[jj],
                alpha=0.2,
            )

        plt.close("all")
        cmap = mpl.colormaps["bwr"] if symmetric else mpl.colormaps["gray_r"]

        fig_dim = 3
        width_ratios = [*[fig_dim for _ in range(numEnv)], fig_dim / 10]

        if not (withLabels):
            width_ratios = width_ratios[:-1]
        fig, ax = plt.subplots(
            numEnv,
            numEnv + 1 * withLabels,
            width_ratios=width_ratios,
            figsize=(sum(width_ratios), fig_dim * numEnv),
            layout="constrained",
        )

        if numEnv == 1:
            ax = np.reshape(ax, (1, -1))

        for ii in range(numEnv):
            for jj in range(numEnv):
                # make image
                aim = ax[ii, jj].imshow(
                    snake_remap[ii][jj],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent(ii, jj),
                    aspect="auto",
                    interpolation=interpolation,
                )

                # label images
                if ii == numEnv - 1:
                    ax[ii, jj].set_xlabel("Position (cm)", fontsize=labelSize)

                ax[ii, jj].set_ylabel(f"ROIs @Env:{envnum[ii]}", fontsize=labelSize)
                if ii == 0:
                    ax[ii, jj].set_title(f"Test @Env:{envnum[jj]}", fontsize=labelSize)

                if rewzone:
                    ax[ii, jj].add_patch(rect(ii, jj))

            if withLabels:
                fig.colorbar(aim, ticks=cb_ticks, orientation="vertical", cax=ax[ii, numEnv])
                ax[ii, numEnv].set_ylabel(cb_label, fontsize=labelSize)

        if not (withLabels):
            for ii in range(numEnv):
                for jj in range(numEnv):
                    ax[ii, jj].set_xticks([])
                    ax[ii, jj].set_yticks([])
                    ax[ii, jj].xaxis.set_tick_params(labelbottom=False)
                    ax[ii, jj].yaxis.set_tick_params(labelleft=False)
                    # ax[ii, jj].set_xlabel(None)
                    # ax[ii, jj].set_ylabel(None)
                    # ax[ii, jj].set_title(None)

        if withSave:
            name = f"remap_snake_plot"
            if not (withLabels):
                name = name + "_nolabel"
            self.saveFigure(fig.number, name)

        # Show figure if requested
        plt.show() if withShow else plt.close()
