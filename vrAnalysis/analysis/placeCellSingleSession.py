import time
from copy import copy
from tqdm import tqdm
import numpy as np
import numba as nb
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

from .. import session
from .. import functions
from .. import helpers
from .. import database
from .. import fileManagement as fm
from .standardAnalysis import standardAnalysis

vrdb = database.vrDatabase()

def save_directory(name=''):
    dirName = fm.analysisPath() / 'placeCellAnalysis' / name
    if not(dirName.is_dir()): dirName.mkdir(parents=True)
    return dirName

def red_reliability(cutoffs=(0.5, 0.8), ises=None, ipcss=None, include_manual=True, use_s2p=False, s2p_cutoff=0.65, **kwConditions):
    """
    Method for returning a list of reliability indices and red cell assignments

    If ises (session iterable) and ipcss (placeCellSingleSession iterable) are not
    passed it will create them using the kwConditions

    If use_s2p=True, ignores the standard red cell detection indexing and determines
    red cell identity by whether the s2p red cell method exceeds s2p_cutoff!
    
    include_manual determines if red cell indices include manual annotations
    kwConditions are passed to vrAnalysis/database/getTable via vrdb.iterSessions()
    It automatically assumes that imaging=True and vrRegistration=True
    """

    # generate session and analysis iterables if not provided
    remake_ipcss = ipcss is None # if it is None, we have to remake it
    if ises is None:
        ises = vrdb.iterSessions(imaging=True, vrRegistration=True, **kwConditions)
        remake_ipcss = True # if we remade the session iterable, remake pcss iterable even if provided

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

    miceInSessions = vrdb.miceInSessions(ises)
    mouseCounter = dict(zip(miceInSessions, [0]*len(miceInSessions))) 
    
    # iterate through requested sessions and store data
    for ses, pcss in zip(ises, ipcss):
        c_use_rois = pcss.idxUseROI # boolean array of ROIs within target planes
        ses_name.append(str(ses))
        ses_per_mouse.append(mouseCounter[ses.mouseName])
        if use_s2p:
            # get red cell indices for this session (within target planes) using s2p output only
            red_idx.append(ses.loadone('mpciROIs.redS2P')[c_use_rois] >= s2p_cutoff)
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
    mouseEnvironmentCounter = dict(zip(miceInSessions, [[0 for _ in range(len(environments))] for _ in range(len(miceInSessions))]))
    for ii, (en, rm, rc) in enumerate(zip(env_nums, relmse, relcor)):
        env_match = [np.where(environments==e)[0] for e in en]
        assert all([len(e)==1 for e in env_match]), f"In session {sessionName[ii]}, environments have an error (environments:{en}), (all environments:{environments})"

        # global environment index
        idx = [e[0] for e in env_match]

        for i in idx:
            # if mouse experienced environment, then indicate how many times the mouse has seen it before
            inenv_per_mouse[ii][i] = mouseEnvironmentCounter[ises[ii].mouseName][i]
            mouseEnvironmentCounter[ises[ii].mouseName][i] += 1 # and update for next visit

        # remake reliability arrays indexing by global environment
        c_num_rois = rm.shape[1] 
        relmse[ii] = np.full((len(environments), c_num_rois), np.nan)
        relcor[ii] = np.full((len(environments), c_num_rois), np.nan)

        for j, i in enumerate(idx):
            relmse[ii][i] = rm[j]
            relcor[ii][i] = rc[j]

    # methods for organizing by mouse
    def getMouseElements(var, idx):
        return [v for v,i in zip(var,idx) if i]
    
    def getEnvironmentSorted(inenv):
        first_use = np.argmax(inenv>=0, axis=0).astype(float)
        used = np.any(inenv>=0, axis=0)
        first_use[~used] = np.inf
        env_order = np.argsort(first_use)[used]
        env_sorted = sorted([i for i,u in enumerate(used) if u], key=lambda x: env_order[x])
        return env_sorted
        
    # now organize by mouse
    miceInSession = sorted(vrdb.miceInSessions(ises))
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
                if not c_inenv[idxses,idxenv] >= 0:
                    continue
                pass_mse = c_relmse[idxses][idxenv] > cutoffs[0]
                pass_cor = c_relcor[idxses][idxenv] > cutoffs[1]
                c_ctl_reliable[idxses, idxenv] = 100*np.sum((pass_mse & pass_cor) & ~c_red_idx[idxses]) / np.sum(~c_red_idx[idxses])
                c_red_reliable[idxses, idxenv] = 100*np.sum((pass_mse & pass_cor) & c_red_idx[idxses]) / np.sum(c_red_idx[idxses])

        env_sort[ii] = c_envsort
        ctl_reliable[ii] = c_ctl_reliable
        red_reliable[ii] = c_red_reliable
        
    # return data
    return miceInSession, env_sort, ctl_reliable, red_reliable 
    

def plot_reliable_difference(cutoffs=(0.5, 0.8), withSave=False, withShow=True, ises=None, ipcss=None, include_manual=True, use_s2p=False, s2p_cutoff=0.65, **kwConditions):
    """plot difference in reliability for red and control cells across environments for all mice"""

    # get measurements of red reliability
    arguments = {'cutoffs':cutoffs, 'ises':ises, 'ipcss':ipcss, 'include_manual':include_manual, 'use_s2p':use_s2p, 's2p_cutoff':s2p_cutoff}
    miceInSession, env_sort, ctl_reliable, red_reliable = red_reliability(**arguments, **kwConditions)
    
    numMice = len(miceInSession)
    numEnvs = ctl_reliable[0].shape[1]

    if numEnvs==3:
        cmap = lambda x: ['k','r','b'][x] 
    else:
        cmap = mpl.colormaps['brg'].resampled(numEnvs)
    
    labelSize = 18
    slabelSize = 12
    
    plt.close('all')
    fig, ax = plt.subplots(1,numMice,figsize=(4*numMice, 4), layout='constrained')
    for ii, (mouse, esort, crel, rrel) in enumerate(zip(miceInSession, env_sort, ctl_reliable, red_reliable)):
        c_num_sessions = crel.shape[0]
        for idxorder, idxenv in enumerate(esort):
            ax[ii].plot(range(c_num_sessions), rrel[:,idxenv]-crel[:,idxenv], color=cmap(idxorder), lw=1.5, marker='.', markersize=14, label=f"env {idxorder}")

        maxAbsDiff = np.nanmax(np.abs(rrel-crel))*1.1
        ax[ii].axhline(color='k', linestyle='--', lw=0.5)
        ax[ii].set_ylim(-maxAbsDiff, maxAbsDiff)

        ax[ii].tick_params(axis='x', labelsize=slabelSize)
        ax[ii].tick_params(axis='y', labelsize=slabelSize)
        ax[ii].set_xlabel('Session #', fontsize=labelSize)
        if ii==0:
            ax[ii].set_ylabel('Red - Control Reliability (%)', fontsize=labelSize)
            ax[ii].legend(loc='lower left', fontsize=slabelSize)
        ax[ii].set_title(mouse, fontsize=labelSize)

    # Save figure if requested
    if withSave: 
        print(f"Saving a plot of difference in reliable fraction of cells (all mice all environments)")
        append_string = '_wS2P' if use_s2p else ''
        plt.savefig(save_directory() / ('difference_reliable_fraction'+append_string))
    
    # Show figure if requested
    plt.show() if withShow else plt.close()


def plot_reliable_fraction(cutoffs=(0.5, 0.8), withSave=False, withShow=True, ises=None, ipcss=None, include_manual=True, use_s2p=False, s2p_cutoff=0.65, **kwConditions):
    """plot difference in reliability for red and control cells across environments for all mice"""

    # get measurements of red reliability
    arguments = {'cutoffs':cutoffs, 'ises':ises, 'ipcss':ipcss, 'include_manual':include_manual, 'use_s2p':use_s2p, 's2p_cutoff':s2p_cutoff}
    miceInSession, env_sort, ctl_reliable, red_reliable = red_reliability(**arguments, **kwConditions)
    
    numMice = len(miceInSession)
    numEnvs = ctl_reliable[0].shape[1]

    if numEnvs==3:
        cmap = lambda x: ['k','r','b'][x] 
    else:
        cmap = mpl.colormaps['brg'].resampled(numEnvs)

    labelSize = 16
    
    plt.close('all')
    fig, ax = plt.subplots(numEnvs,numMice,figsize=(4*numMice, 4*numEnvs), layout='constrained')
    if numEnvs==1: ax = np.reshape(ax, (1, -1))
        
    for ii, (mouse, esort, crel, rrel) in enumerate(zip(miceInSession, env_sort, ctl_reliable, red_reliable)):
        c_num_sessions = crel.shape[0]
        for idxorder, idxenv in enumerate(esort):
            ax[idxorder, ii].plot(range(c_num_sessions), crel[:,idxenv], color='k', lw=1.5, marker='.', markersize=14, label="control")
            ax[idxorder, ii].plot(range(c_num_sessions), rrel[:,idxenv], color='r', lw=1.5, marker='.', markersize=14, label="red")
            ax[idxorder, ii].set_xlim(-0.5, c_num_sessions-0.5)
            ax[idxorder, ii].set_ylim(0, 1.1*np.max((np.nanmax(crel), np.nanmax(rrel))))
            
            ax[idxorder, ii].set_xlabel('Session #', fontsize=labelSize)
            
            if ii==0:
                ax[idxorder, ii].set_ylabel(f'Reliable % - Env {idxorder}', fontsize=labelSize)
            
            ax[idxorder, ii].legend(loc='lower left')
            
            if idxorder==0:
                ax[idxorder, ii].set_title(mouse, fontsize=labelSize)

    # Save figure if requested
    if withSave: 
        print(f"Saving a plot of reliable fraction of cells (all mice all environments)")
        append_string = '_wS2P' if use_s2p else ''
        plt.savefig(save_directory() / ('reliable_fraction'+append_string))
    
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
    def __init__(self, vrexp, onefile='mpci.roiActivityDeconvolvedOasis', autoload=True, keepPlanes=[1,2,3,4], distStep=(1,5,2), speedThreshold=5, numcv=2, standardizeSpks=True):
        self.name = 'placeCellSingleSession'
        self.onefile = onefile
        self.vrexp = vrexp
        self.distStep = distStep
        self.speedThreshold = speedThreshold
        self.numcv = numcv
        self.standardizeSpks = standardizeSpks
        self.keepPlanes = keepPlanes if keepPlanes is not None else [i for i in range(len(vrexp.value['roiPerPlane']))]
        
        # automatically load data
        self.dataloaded = False
        self.load_fast_data()
        if autoload: self.load_data()

    def envnum_to_idx(self, envnum):
        """
        convert list of environment numbers to indices of environment within this session
        e.g. if session has environments [1,3,4], and environment 3 is requested, turn it into index 1
        """
        envnum = helpers.check_iterable(envnum)
        return [np.where(self.environments==ev)[0][0] if ev in self.environments else np.nan for ev in envnum]

    def load_fast_data(self):
        # get environment data
        self.trial_envnum = self.vrexp.loadone('trials.environmentIndex')
        self.environments = np.unique(self.trial_envnum)
        self.numEnvironments = len(self.environments)
        
    def load_data(self, onefile=None, distStep=None, speedThreshold=None, numcv=None, keepPlanes=None):
        """load standard data for basic place cell analysis"""
        # update onefile if using a different measure of activity
        if onefile is not None: self.onefile = onefile

        # update analysis parameters if requested
        if distStep is not None: self.distStep = distStep
        if speedThreshold is not None: self.speedThreshold = speedThreshold
        if numcv is not None: self.numcv = numcv
        if keepPlanes is not None: self.keepPlanes = keepPlanes
        
        # get idx of rois within keep planes
        stackPosition = self.vrexp.loadone('mpciROIs.stackPosition')
        roiPlaneIdx = stackPosition[:,2].astype(np.int32) # plane index
        
        # figure out which ROIs are in the target planes
        self.idxUseROI = np.any(np.stack([roiPlaneIdx==pidx for pidx in self.keepPlanes]),axis=0)
        self.numROIs = self.vrexp.getNumROIs(self.keepPlanes)
        
        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        kwargs = {'distStep':self.distStep, 'onefile':self.onefile, 'speedThreshold':self.speedThreshold, 'standardizeSpks':self.standardizeSpks}
        self.omap, self.smap, _, self.spkmap, self.distedges = functions.getBehaviorAndSpikeMaps(self.vrexp, **kwargs)
        self.spkmap = self.spkmap[self.idxUseROI]
        
        self.distcenters = helpers.edge2center(self.distedges)
        
        self.numTrials = self.omap.shape[0]
        
        # find out which trials the mouse explored the whole environment
        self.boolFullTrials = np.all(~np.isnan(self.omap),axis=1) 
        self.idxFullTrials = np.where(self.boolFullTrials)[0]
        self.idxFullTrialEachEnv = [np.where(self.boolFullTrials & (self.trial_envnum==env))[0] for env in self.environments]
        
        # report that data has been loaded
        self.dataloaded = True

        # measure reliability 
        self.measure_reliability()
        
    def clear_data(self):
        """method for clearing data to free up memory"""
        del self.idxUseROI
        del self.numROIs
        del self.omap
        del self.smap
        del self.spkmap
        del self.distedges
        del self.distcenters
        del self.numTrials
        del self.boolFullTrials
        del self.idxFullTrials
        del self.idxFullTrialEachEnv
        del self.train_idx
        del self.test_idx
        del self.relmse
        del self.relcor
        del self.test_relmse
        del self.test_relcor
        self.dataloaded = False
        
    def measure_reliability(self, with_test=False):
        """method for measuring reliability in each environment"""
        if not(self.dataloaded): self.load_data()
        
        foldIdx = [helpers.cvFoldSplit(idxTrialEnv, 3) for idxTrialEnv in self.idxFullTrialEachEnv]
        self.train_idx = [np.concatenate(fidx[:2]) for fidx in foldIdx]
        self.test_idx = [fidx[2] for fidx in foldIdx]
        
        # measure reliability of spiking (in two ways)
        relmse, relcor = zip(*[functions.measureReliability(self.spkmap[:,tidx], numcv=self.numcv) for tidx in self.train_idx])
        self.relmse, self.relcor = np.stack(relmse), np.stack(relcor)
        
        if with_test:
            # measure on test trials
            relmse, relcor = zip(*[functions.measureReliability(self.spkmap[:,tidx], numcv=self.numcv) for tidx in self.test_idx])
            self.test_relmse, self.test_relcor = np.stack(relmse), np.stack(relcor)
        else:
            # Alert the user that the training data was recalculated without testing
            self.test_relmse = None

    def get_reliable(self, envnum=None, cutoffs=None):
        """central method for getting reliable cells from list of environments (by environment index)"""
        if not(self.dataloaded): self.load_data()
        if envnum is None: envnum = copy(self.environments) # default environment is all of them
        envnum = helpers.check_iterable(envnum) # make sure it's an iterable
        envidx = self.envnum_to_idx(envnum) # convert environment numbers to indices
        cutoffs = (-np.inf, -np.inf) if cutoffs is None else cutoffs
        idx_reliable = [(self.relmse[ii] >= cutoffs[0]) & (self.relcor[ii] >= cutoffs[1]) for ii in envidx]
        return idx_reliable
        
    def get_place_field(self, roi_idx=None, trial_idx=None, method='max'):
        """get sorting index based on spikemap, roi index, and trial index"""
        if not(self.dataloaded): self.load_data()
        
        assert method=='com' or method=='max', f"invalid method ({method}), must be either 'com' or 'max'"
        if roi_idx is None: roi_idx = np.ones(numROIs, dtype=bool)            
        if trial_idx is None: trial_idx = np.ones(numTrials, dtype=bool)
            
        # Get ROI x Position profile of activity for each ROI as a function of position
        meanProfile = np.mean(self.spkmap[roi_idx][:,trial_idx], axis=1) 

        # if method is 'com' (=center of mass), use weighted mean to get place field location
        if method=='com':
            # note that this can generate buggy behavior if spkmap isn't based on mostly positive signals!
            
            nonnegativeProfile = np.maximum(meanProfile, 0) 
            pfloc = np.sum(nonnegativeProfile * self.distcenters.reshape(1,-1), axis=1) / np.sum(nonnegativeProfile, axis=1)

        # if method is 'max' (=maximum rate), use maximum to get place field location
        if method=='max':
            pfloc = np.argmax(meanProfile, axis=1)

        # Then sort...
        pfidx = np.argsort(pfloc)

        return pfloc, pfidx
    
    def make_snake(self, envnum=None, with_reliable=True, cutoffs=(0.5, 0.8), method='max'):
        """make snake data from train and test sessions, for particular environment if requested"""
        if not(self.dataloaded): self.load_data()
        
        # default environment is all of them
        if envnum is None: envnum = copy(self.environments)

        # envnum must be an iterable
        envnum = helpers.check_iterable(envnum)

        # convert environment numbers to indices
        envidx = self.envnum_to_idx(envnum)
        
        # get specific trial indices for given environment(s)
        ctrain_idx = [self.train_idx[ii] for ii in envidx]
        ctest_idx = [self.test_idx[ii] for ii in envidx]
        
        # get roi indices to use
        self.idx_in_snake = self.get_reliable(envnum, cutoffs=cutoffs if with_reliable else None)
        
        # get pf sort indices
        train_pfidx = [self.get_place_field(roi_idx=idxroi, trial_idx=idxenvtrain, method=method)[1] for idxroi, idxenvtrain in zip(self.idx_in_snake, ctrain_idx)]
        
        # get spkmap of only reliable ROIs in each environment
        spkmap = [self.spkmap[idxroi] for idxroi in self.idx_in_snake] 
        
        # average (ROI x position) activity for each environment in training and testing trials
        trainProfile = [np.mean(sm[:, idxenvtrain], axis=1) for sm, idxenvtrain in zip(spkmap, ctrain_idx)] 
        testProfile = [np.mean(sm[:, idxenvtest], axis=1) for sm, idxenvtest in zip(spkmap, ctest_idx)]

        # average activity for each environment sorted by order on training trials
        train_snake = [trainProf[pfidx] for trainProf, pfidx in zip(trainProfile, train_pfidx)]
        test_snake = [testProf[pfidx] for testProf, pfidx in zip(testProfile, train_pfidx)]

        # :)
        return train_snake, test_snake

    def make_remap_data(self, with_reliable=True, cutoffs=(0.5, 0.8), method='max'):
        """make snake data across environments with remapping indices (for N environments, an NxN grid of snakes and indices)"""
        if not(self.dataloaded): self.load_data()
        
        envnum = helpers.check_iterable(copy(self.environments)) # always use all environments (as an iterable)
        envidx = self.envnum_to_idx(envnum)
        
        # get roi indices to use
        self.idx_in_snake = self.get_reliable(envnum, cutoffs=cutoffs if with_reliable else None)

        # get pf sort indices
        train_pfidx = [self.get_place_field(roi_idx=idxroi, trial_idx=idxenvtrain, method=method)[1] for idxroi, idxenvtrain in zip(self.idx_in_snake, self.train_idx)]

        # get spkmap of only reliable ROIs in each environment
        spkmap = [self.spkmap[idxroi] for idxroi in self.idx_in_snake] 
        
        # make c-v snake plots across all environment combinations
        snake_plots = []
        for ii, env in enumerate(envidx):
            # choose ROIs and sort by environment ii (on test trials if ii==jj and full trials if ii!=jj)
            c_plots = []
            for jj, env in enumerate(envidx):
                if ii==jj:
                    # snake of reliable in env @ii, on test trials for env @jj, sorted by pf on train trials in env @ii
                    c_snake = np.mean(spkmap[ii][:,self.test_idx[jj]], axis=1)[train_pfidx[ii]]
                    c_plots.append(c_snake)
                else:
                    # snake of reliable in env @ii, on full trials for env @jj, sorted by pf on train trials in env @ii
                    c_snake = np.mean(spkmap[ii][:,self.idxFullTrialEachEnv[jj]], axis=1)[train_pfidx[ii]]
                    c_plots.append(c_snake)
            snake_plots.append(c_plots)

        # :)
        return snake_plots
    
    def plot_snake(self, envnum=None, with_reliable=True, cutoffs=(0.5, 0.8), method='max', normalize=0, rewzone=True, interpolation='none', withShow=True, withSave=False):
        """method for plotting cross-validated snake plot"""
        if not(self.dataloaded): self.load_data()
        
        # default environment is all of them
        if envnum is None: envnum = copy(self.environments)

        # envnum must be an iterable
        envnum = helpers.check_iterable(envnum)
        assert all([e in self.environments for e in envnum]), "envnums must be valid environment numbers within self.environments"

        # get number of environments
        numEnv = len(envnum)

        # make snakes and prepare plotting data
        train_snake, test_snake = self.make_snake(envnum=envnum, with_reliable=with_reliable, cutoffs=cutoffs, method=method)
        extent = [[self.distedges[0], self.distedges[-1], 0, ts.shape[0]] for ts in train_snake]
        if normalize > 0:
            vmin, vmax = -np.abs(normalize), np.abs(normalize)
        elif normalize < 0:
            maxrois = np.concatenate([np.max(np.abs(ts), axis=1) for ts in train_snake]+[np.max(np.abs(ts), axis=1) for ts in test_snake])
            vmin, vmax = -np.percentile(maxrois, -normalize), np.percentile(maxrois, -normalize)
        else:
            magnitude = np.max(np.abs(np.concatenate((np.concatenate(train_snake),np.concatenate(test_snake)))))
            vmin, vmax = -magnitude, magnitude

        cb_ticks = np.linspace(np.fix(vmin), np.fix(vmax), int(min(11, np.fix(vmax)-np.fix(vmin)+1)))
        labelSize = 20
        cb_unit = r'$\sigma$' if self.standardizeSpks else 'au'
        cb_label = f"Activity ({cb_unit})"
        
        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = functions.environmentRewardZone(self.vrexp)
            rewPos = [rewPos[np.where(self.environments==ev)[0][0]] for ev in envnum]
            rewHalfwidth = [rewHalfwidth[np.where(self.environments==ev)[0][0]] for ev in envnum]
            rect_train = [mpl.patches.Rectangle((rp-rhw, 0), rhw*2, ts.shape[0], edgecolor='none', facecolor='k', alpha=0.2) for rp, rhw, ts in zip(rewPos, rewHalfwidth, train_snake)]
            rect_test = [mpl.patches.Rectangle((rp-rhw, 0), rhw*2, ts.shape[0], edgecolor='none', facecolor='k', alpha=0.2) for rp, rhw, ts in zip(rewPos, rewHalfwidth, train_snake)]
            
        plt.close('all')
        cmap = mpl.colormaps['bwr']
        
        fig, ax = plt.subplots(numEnv, 3, width_ratios=[10,10,1], figsize=(9,3*numEnv), layout='constrained')

        if numEnv==1: ax = np.reshape(ax, (1, -1))
            
        for idx, env in enumerate(envnum):
            cim = ax[idx, 0].imshow(train_snake[idx], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent[idx], aspect='auto', interpolation=interpolation)
            if idx==numEnv-1:
                ax[idx,0].set_xlabel('Position (cm)', fontsize=labelSize)
            ax[idx,0].set_ylabel(f'Env:{env}, ROIs', fontsize=labelSize)
            if idx==0:
                ax[idx,0].set_title('Train Trials', fontsize=labelSize)

            ax[idx,1].imshow(test_snake[idx], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent[idx], aspect='auto', interpolation=interpolation)
            if idx==numEnv-1:
                ax[idx,1].set_xlabel('Position (cm)', fontsize=labelSize)
            ax[idx,1].set_ylabel('ROIs', fontsize=labelSize)
            if idx==0:
                ax[idx,1].set_title('Test Trials', fontsize=labelSize)

            if rewzone:
                ax[idx,0].add_patch(rect_train[idx])
                ax[idx,1].add_patch(rect_test[idx])

            fig.colorbar(cim, ticks=cb_ticks, orientation='vertical', cax=ax[idx, 2])
            ax[idx, 2].set_ylabel(cb_label, fontsize=labelSize)

        if withSave: 
            if len(envnum)==len(self.environments):
                self.saveFigure(fig.number, f'snake_plot')
            else:
                print("If envnum is less than all the environments, you can't save with this program!")
        
        # Show figure if requested
        plt.show() if withShow else plt.close()

    
    def plot_remap_snakes(self, with_reliable=True, cutoffs=(0.5, 0.8), method='max', normalize=0, rewzone=True, interpolation='none', force_single_env=False, withLabels=True, withShow=True, withSave=False):
        """method for plotting cross-validated snake plot"""
        # plotting remap snakes always uses all environments
        envnum = helpers.check_iterable(copy(self.environments)) # always use all environments (as an iterable)
        envidx = self.envnum_to_idx(envnum)
        numEnv = len(envnum)

        if numEnv==1 and not(force_single_env):
            print(f"Session {self.vrexp.sessionPrint()} only uses 1 environment, not plotting remap snakes")
            return None

        # make snakes
        snake_remap = self.make_remap_data(with_reliable=with_reliable, cutoffs=cutoffs, method=method)

        # prepare plotting data
        extent = lambda ii, jj : [self.distedges[0], self.distedges[-1], 0, snake_remap[ii][jj].shape[0]]
        if normalize > 0:
            vmin, vmax = -np.abs(normalize), np.abs(normalize)
        elif normalize < 0:
            maxrois = np.concatenate([np.concatenate([np.max(np.abs(srp), axis=1) for srp in s_remap]) for s_remap in snake_remap])
            vmin, vmax = -np.percentile(maxrois, -normalize), np.percentile(maxrois, -normalize)
        else:
            magnitude = np.max(np.abs(np.vstack([np.concatenate(srp) for srp in snake_remap])))
            vmin, vmax = -magnitude, magnitude

        cb_ticks = np.linspace(np.fix(vmin), np.fix(vmax), int(min(11, np.fix(vmax)-np.fix(vmin)+1)))
        labelSize = 20
        cb_unit = r'$\sigma$' if self.standardizeSpks else 'au'
        cb_label = f"Activity ({cb_unit})"
        
        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = functions.environmentRewardZone(self.vrexp)
            rewPos = [rewPos[np.where(self.environments==ev)[0][0]] for ev in envnum]
            rewHalfwidth = [rewHalfwidth[np.where(self.environments==ev)[0][0]] for ev in envnum]
            rect = lambda ii, jj : mpl.patches.Rectangle((rewPos[jj]-rewHalfwidth[jj], 0), rewHalfwidth[jj]*2, snake_remap[ii][jj].shape[0], edgecolor='none', facecolor='k', alpha=0.2)
            
        plt.close('all')
        cmap = mpl.colormaps['bwr']

        fig_dim = 3
        width_ratios = [*[fig_dim for _ in range(numEnv)], fig_dim/10]

        if not(withLabels):
            width_ratios = width_ratios[:-1]
        fig, ax = plt.subplots(numEnv, numEnv+1*withLabels, width_ratios=width_ratios, figsize=(sum(width_ratios), fig_dim*numEnv), layout='constrained')

        if numEnv==1: ax = np.reshape(ax, (1,-1))
            
        for ii in range(numEnv):
            for jj in range(numEnv):
                # make image
                aim = ax[ii, jj].imshow(snake_remap[ii][jj], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent(ii,jj), aspect='auto', interpolation=interpolation)

                # label images
                if ii==numEnv-1:
                    ax[ii, jj].set_xlabel('Position (cm)', fontsize=labelSize)

                ax[ii, jj].set_ylabel(f"ROIs @Env:{envnum[ii]}", fontsize=labelSize)
                if ii==0:
                    ax[ii, jj].set_title(f"Test @Env:{envnum[jj]}", fontsize=labelSize)

                if rewzone:
                    ax[ii, jj].add_patch(rect(ii, jj))

            if withLabels:
                fig.colorbar(aim, ticks=cb_ticks, orientation='vertical', cax=ax[ii, numEnv])
                ax[ii, numEnv].set_ylabel(cb_label, fontsize=labelSize)

        if not(withLabels):
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
            name = f'remap_snake_plot'
            if not(withLabels): name = name + '_nolabel'
            self.saveFigure(fig.number, name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()
        









