from copy import copy
from tqdm import tqdm
from functools import wraps
import faststats as fs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .. import functions
from .. import helpers
from .. import database
from .. import fileManagement as fm
from .standardAnalysis import standardAnalysis

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
        ), f"In session {sessionName[ii]}, environments have an error (environments:{en}), (all environments:{environments})"

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
    ):
        self.name = "placeCellSingleSession"
        self.onefile = onefile
        self.vrexp = vrexp
        self.distStep = distStep
        self.speedThreshold = speedThreshold
        self.numcv = numcv
        self.standardizeSpks = standardizeSpks
        self.smoothWidth = smoothWidth
        self.keep_planes = keep_planes if keep_planes is not None else [i for i in range(len(vrexp.value["roiPerPlane"]))]

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

    def load_fast_data(self):
        # get environment data
        self.trial_envnum = self.vrexp.loadone("trials.environmentIndex")
        self.environments = np.unique(self.trial_envnum)
        self.numEnvironments = len(self.environments)
        # get behavioral data
        self.load_behavioral_data()
        # get index to ROIs
        self.get_plane_idx()

    def clear_data(self):
        """method for clearing data to free up memory and/or resetting variables"""
        attrs_to_delete = [
            "idxUseROI",
            "numROIs",
            "occmap",
            "speedmap",
            "rawspkmap",
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

        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {
            "distStep": self.distStep,
            "speedThreshold": self.speedThreshold,
        }
        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {
            "distStep": self.distStep,
            "speedThreshold": self.speedThreshold,
            "speedSmoothing": self.smoothWidth,
            "get_spkmap": False,
        }
        self.occmap, self.speedmap, _, _, self.distedges = functions.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)
        self.distcenters = helpers.edge2center(self.distedges)

        self.numTrials = self.occmap.shape[0]

        # find out which trials the mouse explored the whole environment
        if full_trial_flexibility is None:
            # if full trial flexiblity is None, then they need to have visited every bin
            idx_to_required_bins = np.arange(self.occmap.shape[1])
        else:
            start_idx = np.where(self.distedges >= full_trial_flexibility)[0][0]
            end_idx = np.where(self.distedges <= self.distedges[-1] - full_trial_flexibility)[0][-1]
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
        full_trial_flexibility=3,
        new_split=True,
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

        self.get_plane_idx(keep_planes=self.keep_planes)

        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {
            "distStep": self.distStep,
            "onefile": self.onefile,
            "speedThreshold": self.speedThreshold,
            "standardizeSpks": self.standardizeSpks,
            "idxROIs": self.idxUseROI,
            "speedSmoothing": self.smoothWidth,
        }
        self.occmap, self.speedmap, _, self.rawspkmap, self.distedges = functions.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)

        self.distcenters = helpers.edge2center(self.distedges)

        self.numTrials = self.occmap.shape[0]

        # find out which trials the mouse explored the whole environment
        if full_trial_flexibility is None:
            # if full trial flexiblity is None, then they need to have visited every bin
            idx_to_required_bins = np.arange(self.occmap.shape[1])
        else:
            start_idx = np.where(self.distedges >= full_trial_flexibility)[0][0]
            end_idx = np.where(self.distedges <= self.distedges[-1] - full_trial_flexibility)[0][-1]
            idx_to_required_bins = np.arange(start_idx, end_idx)

        self.boolFullTrials = np.all(~np.isnan(self.occmap[:, idx_to_required_bins]), axis=1)
        self.idxFullTrials = np.where(self.boolFullTrials)[0]
        self.idxFullTrialEachEnv = [np.where(self.boolFullTrials & (self.trial_envnum == env))[0] for env in self.environments]

        # report that data has been loaded
        self.dataloaded = True

        # measure reliability
        self.measure_reliability(new_split=new_split, with_test=with_test)

    @prepare_data
    def get_spkmap(self, envnum=None, average=True, smooth=None, trials="full", new_split=False, split_params={}):
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
        """
        # use all environments if none requested
        if envnum is None:
            envnum = np.copy(self.environments)

        # convert envnum into iterable index to environment in session
        envnum = helpers.check_iterable(envnum)  # make sure envnum is iterable
        envidx = self.envnum_to_idx(envnum)  # convert environment numbers to indices

        # get occupancy and rawspkmap from requested trials (or across environments)
        if trials == "full":
            env_occmap = [self.occmap[self.idxFullTrialEachEnv[ei]] for ei in envidx]
            env_spkmap = [self.rawspkmap[self.idxFullTrialEachEnv[ei]] for ei in envidx]

        elif trials == "train":
            if new_split:
                self.define_train_test_split(**split_params)
            env_occmap = [self.occmap[self.train_idx[ei]] for ei in envidx]
            env_spkmap = [self.rawspkmap[self.train_idx[ei]] for ei in envidx]

        elif trials == "test":
            if new_split:
                self.define_train_test_split(**split_params)
            env_occmap = [self.occmap[self.test_idx[ei]] for ei in envidx]
            env_spkmap = [self.rawspkmap[self.test_idx[ei]] for ei in envidx]

        else:
            raise ValueError(f"Didn't recognize trials option (received '{trials}', expected 'full', 'train', or 'test')")

        # get spkmaps for each environment
        env_spkmap = [self._make_spkmap(maps=(eom, esm), average=average, smooth=smooth) for eom, esm in zip(env_occmap, env_spkmap)]

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
            occmap = fs.sum(occmap, axis=0, keepdims=True)
            spkmap = fs.sum(spkmap, axis=0, keepdims=True)

        # do smoothing across spatial positions if requested
        if smooth is not None:
            # if smoothing, nans will get confusing so we need to reset nans with 0s then reset them
            occ_idxnan = np.isnan(occmap)
            spk_idxnan = np.isnan(spkmap)
            occmap[occ_idxnan] = 0
            spkmap[spk_idxnan] = 0

            # do smoothing
            kk = helpers.getGaussKernel(self.distcenters, smooth)
            occmap = helpers.convolveToeplitz(occmap, kk, axis=1)
            spkmap = helpers.convolveToeplitz(spkmap, kk, axis=1)

            # reset nans
            occmap[occ_idxnan] = np.nan
            spkmap[spk_idxnan] = np.nan

        # correct spkmap by occupancy
        spkmap = functions.correctMap(occmap, spkmap)

        # reshape to (numROIs, numTrials, numPositions)
        spkmap = spkmap.transpose(2, 0, 1)

        # squeeze out trials dimension if averaging
        if average:
            spkmap = spkmap.squeeze()

        # return spkmap
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
    def measure_reliability(self, new_split=True, with_test=False, smoothWidth=-1, total_folds=3, train_folds=2):
        """method for measuring reliability in each environment"""
        if smoothWidth == -1:
            smoothWidth = self.smoothWidth

        # create a train/test split
        if new_split:
            self.define_train_test_split(total_folds=total_folds, train_folds=train_folds)

        # measure reliability of spiking (in two ways)
        spkmap = self.get_spkmap(average=False, smooth=smoothWidth, trials="train")
        relmse, relcor = helpers.named_transpose([functions.measureReliability(smap, numcv=self.numcv) for smap in spkmap])
        self.relmse, self.relcor = np.stack(relmse), np.stack(relcor)

        if with_test:
            # measure on test trials
            spkmap = self.get_spkmap(average=False, smooth=smoothWidth, trials="test")
            relmse, relcor = helpers.named_transpose([functions.measureReliability(smap, numcv=self.numcv) for smap in spkmap])
            self.test_relmse, self.test_relcor = np.stack(relmse), np.stack(relcor)
        else:
            # Alert the user that the training data was recalculated without testing
            self.test_relmse, self.test_relcor = None, None

    @prepare_data
    def get_reliability_values(self, envnum=None, with_test=False):
        """support for getting reliability values from requested or all environments"""
        if envnum is None:
            envnum = copy(self.environments)  # default environment is all of them
        envnum = helpers.check_iterable(envnum)  # make sure it's an iterable
        envidx = self.envnum_to_idx(envnum)  # convert environment numbers to indices
        mse = [self.relmse[ii] for ii in envidx]
        cor = [self.relcor[ii] for ii in envidx]

        # if not with_test trials, just return mse/cor on train trials
        if not with_test:
            return mse, cor

        # if with_test, get these too and return them all
        msetest = [self.test_relmse[ii] for ii in envidx]
        cortest = [self.test_relcor[ii] for ii in envidx]
        return mse, cor, msetest, cortest

    @prepare_data
    def get_reliable(self, envnum=None, cutoffs=None, maxcutoffs=None):
        """central method for getting reliable cells from list of environments (by environment index)"""
        if envnum is None:
            envnum = copy(self.environments)  # default environment is all of them
        envnum = helpers.check_iterable(envnum)  # make sure it's an iterable
        envidx = self.envnum_to_idx(envnum)  # convert environment numbers to indices
        cutoffs = (-np.inf, -np.inf) if cutoffs is None else cutoffs
        maxcutoffs = (np.inf, np.inf) if maxcutoffs is None else maxcutoffs
        idx_reliable = [
            (self.relmse[ei] >= cutoffs[0])
            & (self.relcor[ei] >= cutoffs[1])
            & (self.relmse[ei] <= maxcutoffs[0])
            & (self.relcor[ei] <= maxcutoffs[1])
            for ei in envidx
        ]
        return idx_reliable

    def get_place_field(self, spkmap, method="max", force_with_negative=False):
        """
        get sorting index and place field location for spkmap

        If spkmap has shape: (numROIs, numTrials, numPositions) will average across trials
        If spkmap has shape: (numROIs, numPositions) will use as is
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

    @prepare_data
    def make_snake(self, envnum=None, reliable=True, cutoffs=(0.4, 0.7), maxcutoffs=None, method="max"):
        """make snake data from train and test sessions, for particular environment if requested"""
        # default environment is all of them
        if envnum is None:
            envnum = copy(self.environments)

        # envnum must be an iterable
        envnum = helpers.check_iterable(envnum)

        # get spkmaps for requested environments
        train_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="train")
        test_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="test")

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
    def make_remap_data(self, reliable=True, cutoffs=(0.4, 0.7), maxcutoffs=None, method="max"):
        """make snake data across environments with remapping indices (for N environments, an NxN grid of snakes and indices)"""
        # get index to each environment for this session
        envnum = helpers.check_iterable(copy(self.environments))
        num_envs = len(envnum)

        # get train/test spkmap profile for each environment (average across trials)
        train_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="train")
        test_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="test")
        full_profile = self.get_spkmap(envnum, average=True, smooth=self.smoothWidth, trials="full")

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
        labelSize = 20
        cb_unit = r"$\sigma$" if self.standardizeSpks else "au"
        cb_label = f"Activity ({cb_unit})"

        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = functions.environmentRewardZone(self.vrexp)
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

        cb_ticks = np.linspace(np.fix(vmin), np.fix(vmax), int(min(11, np.fix(vmax) - np.fix(vmin) + 1)))
        labelSize = 20
        cb_unit = r"$\sigma$" if self.standardizeSpks else "au"
        cb_label = f"Activity ({cb_unit})"

        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = functions.environmentRewardZone(self.vrexp)
            rewPos = [rewPos[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rewHalfwidth = [rewHalfwidth[np.where(self.environments == ev)[0][0]] for ev in envnum]
            rect = lambda ii, jj: mpl.patches.Rectangle(
                (rewPos[jj] - rewHalfwidth[jj], 0),
                rewHalfwidth[jj] * 2,
                snake_remap[ii][jj].shape[0],
                edgecolor="none",
                facecolor="k",
                alpha=0.2,
            )

        plt.close("all")
        cmap = mpl.colormaps["bwr"]

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
                    ax[ii, jj].set_xlabel(None)
                    ax[ii, jj].set_ylabel(None)
                    ax[ii, jj].set_title(None)

        if withSave:
            name = f"remap_snake_plot"
            if not (withLabels):
                name = name + "_nolabel"
            self.saveFigure(fig.number, name)

        # Show figure if requested
        plt.show() if withShow else plt.close()
