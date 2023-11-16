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

sessiondb = database.vrDatabase('vrSessions')

def analysisDirectory():
    """
    return the directory to analysis data
    
    in your `fileManagement` file, hard-code whatever path you want to save analysis data and figures on.
    this method will just use that path.
    """
    return fm.analysisPath()
    
def saveDirectory(typename):
    """
    defines an analysis specific save directory
    
    it's always inside the `analysisDirectory()`, but then will add a folder
    for the particular analysis type (i.e. self.name) and then adds an additional
    folder for the specific analysis you are doing (name)
    """
    # Define and create target directory
    dirName = analysisDirectory() / typename
    if not(dirName.is_dir()): dirName.mkdir(parents=True)
    return dirName

def saveFigure(figNumber, typename, name):
    """
    save a figure currently open in matplotlib
    
    attempts to save matplotlib figure(figNumber) in the save directory with a particular name
    """
    print(f"{typename} is saving a {name} figure!")
    plt.figure(figNumber)
    plt.savefig(saveDirectory(typename) /name)
    

def getEnvironmentSorted(inenv, ge=0):
        first_use = np.argmax(inenv>=ge, axis=0).astype(float)
        used = np.any(inenv>=ge, axis=0)
        first_use[~used] = np.inf
        env_order = np.argsort(first_use)[used]
        env_sorted = sorted([i for i,u in enumerate(used) if u], key=lambda x: env_order[x])
        return env_sorted
    
def sessionStats(ises=None, keepPlanes=[1,2,3,4], include_manual=True, use_s2p=False, s2p_cutoff=0.65, **kwConditions):
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
    if ises is None:
        ises = sessiondb.iterSessions(imaging=True, vrRegistration=True, **kwConditions)
    
    # initialize lists for storing the data
    ses_name = []
    ses_per_mouse = []
    red_idx = []
    env_nums = []

    miceInSessions = sessiondb.miceInSessions(ises)
    mouseCounter = dict(zip(miceInSessions, [0]*len(miceInSessions))) 
    
    # iterate through requested sessions and store data
    for ses in ises:
        c_use_rois = ses.idxToPlanes(keepPlanes=keepPlanes) # boolean array of ROIs within target planes
        ses_name.append(str(ses))
        ses_per_mouse.append(mouseCounter[ses.mouseName])
        if use_s2p:
            # get red cell indices for this session (within target planes) using s2p output only
            red_idx.append(ses.loadone('mpciROIs.redS2P')[c_use_rois] >= s2p_cutoff)
        else:
            # get red cell indices for this session (within target planes) using standard red cell indices
            red_idx.append(ses.getRedIdx(include_manual=include_manual)[c_use_rois]) 
        env_nums.append(np.unique(ses.loadone('trials.environmentIndex')))
        mouseCounter[ses.mouseName] += 1

    # organize everything by environment
    environments = np.unique(np.concatenate(env_nums)) 
    inenv_per_mouse = [[-1 for _ in range(len(environments))] for _ in range(len(ises))]
    # inenv_per_mouse = -np.ones((len(environments), len(ises)))
    mouseEnvironmentCounter = dict(zip(miceInSessions, [[0 for _ in range(len(environments))] for _ in range(len(miceInSessions))]))
    for ii, en in enumerate(env_nums):
        env_match = [np.where(environments==e)[0] for e in en]
        assert all([len(e)==1 for e in env_match]), f"In session {sessionName[ii]}, environments have an error (environments:{en}), (all environments:{environments})"

        # global environment index
        idx = [e[0] for e in env_match]

        for i in idx:
            # if mouse experienced environment, then indicate how many times the mouse has seen it before
            inenv_per_mouse[ii][i] = mouseEnvironmentCounter[ises[ii].mouseName][i]
            mouseEnvironmentCounter[ises[ii].mouseName][i] += 1 # and update for next visit

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
    miceInSession = sorted(sessiondb.miceInSessions(ises))
    
    total_cell_count = [[] for _ in range(len(miceInSession))]
    red_cell_count = [[] for _ in range(len(miceInSessions))]
    env_counter = [[] for _ in range(len(miceInSession))]
    trial_counter = [[] for _ in range(len(miceInSession))]
    
    for ii, mouse in enumerate(miceInSession):
        idx_mouse = [mouse in sn for sn in ses_name]
        c_inenv = np.stack(getMouseElements(inenv_per_mouse, idx_mouse))
        c_red_idx = getMouseElements(red_idx, idx_mouse)
        c_envsort = getEnvironmentSorted(c_inenv)

        # this is silly but I'm hacking now
        c_ses = getMouseElements(ises, idx_mouse)
        trial_counter[ii] = copy(c_inenv)
        for cs, tc in zip(c_ses, trial_counter[ii]):
            for ie, envidx in enumerate(environments):
                tc[ie] = np.sum(cs.loadone('trials.environmentIndex')==envidx)

        total_cell_count[ii] = [len(cri) for cri in c_red_idx]
        red_cell_count[ii] = [np.sum(cri) for cri in c_red_idx]
        env_counter[ii] = c_inenv
        
    return miceInSession, env_counter, trial_counter, total_cell_count, red_cell_count


def plot_environmentStats(ises=None, keepPlanes=[1,2,3,4], include_manual=True, use_s2p=False, s2p_cutoff=0.65, **kwConditions):
    miceInSession, env_counter, trial_counter, total_cell_count, red_cell_count = sessionStats(ises=ises,
                                                                                               keepPlanes=keepPlanes,
                                                                                               include_manual=include_manual,
                                                                                               use_s2p=use_s2p,
                                                                                               s2p_cutoff=s2p_cutoff,
                                                                                               **kwConditions)
    
    numMice = len(miceInSession)
    numEnvs = trial_counter[0].shape[1]
    maxSessions = [tc.shape[0] for tc in trial_counter]

    envcolor = 'krb'
    labelSize = 18
    slabelSize = 14
    lw = 1.5
    mrk = 6

    figdim = 4
    
    plt.close('all')
    fig, ax = plt.subplots(1,numMice,figsize=((figdim*1.4)*numMice, figdim), layout='constrained')
    for imouse, (mname, mses, tcount) in enumerate(zip(miceInSession, maxSessions, trial_counter)):
        c_esort = getEnvironmentSorted(tcount, ge=1)
        cum_trials = np.cumsum(np.concatenate((np.zeros((mses,1)), tcount[:,c_esort]),axis=1),axis=1)

        xdata = range(mses)
        for ii, ienv in reversed(list(enumerate(c_esort))):
            ybottom = cum_trials[:, ii]
            ytop = cum_trials[:, ii+1]
            if imouse==0:
                ax[imouse].fill_between(xdata, ybottom, ytop, color=envcolor[ienv], label=f"Env {ienv+1}")
            else:
                ax[imouse].fill_between(xdata, ybottom, ytop, color=envcolor[ienv])

        ax[imouse].set_xlim(0, mses-1)
        ax[imouse].set_ylim(0)
        ax[imouse].tick_params(axis='x', labelsize=slabelSize)
        ax[imouse].tick_params(axis='y', labelsize=slabelSize)
        ax[imouse].set_xlabel('Session #', fontsize=labelSize)
        ax[imouse].set_ylabel('# Trials / Env', fontsize=labelSize)
        ax[imouse].set_title(f"{mname}", fontsize=labelSize)
        
        if imouse==0:
            ax[imouse].legend(loc='upper left', fontsize=slabelSize)

    saveFigure(fig.number, 'sessionStats', 'trial_counts')
    plt.show()
    

def plot_roiCountStats(ises=None, keepPlanes=[1,2,3,4], include_manual=True, use_s2p=False, s2p_cutoff=0.65, **kwConditions):
    miceInSession, env_counter, trial_counter, total_cell_count, red_cell_count = sessionStats(ises=ises,
                                                                                               keepPlanes=keepPlanes,
                                                                                               include_manual=include_manual,
                                                                                               use_s2p=use_s2p,
                                                                                               s2p_cutoff=s2p_cutoff,
                                                                                               **kwConditions)

    numMice = len(miceInSession)
    maxSessions = [tc.shape[0] for tc in trial_counter]

    mousecolor = 'bgcmr'
    labelSize = 18
    slabelSize = 14
    lw = 1.5
    mrk = 6
    figdim = 4
    
    plt.close('all')
    fig, ax = plt.subplots(1,2,figsize=(3*figdim, figdim), layout='constrained')
    for imouse, (mname, mses, tcells, rcells) in enumerate(zip(miceInSession, maxSessions, total_cell_count, red_cell_count)):
        ax[0].plot(range(mses), tcells, color=mousecolor[imouse], lw=lw, marker='o', markersize=mrk)
        ax[0].tick_params(axis='x', labelsize=slabelSize)
        ax[0].tick_params(axis='y', labelsize=slabelSize)
        ax[0].set_xlabel('Session #', fontsize=labelSize)
        ax[0].set_ylabel('# ROIs', fontsize=labelSize)
        ax[0].set_title('Total ROIs Per Session', fontsize=labelSize)
                        
        ax[1].plot(range(mses), rcells, color=mousecolor[imouse], lw=lw, marker='o', markersize=mrk, label=f"{mname}")
        ax[1].tick_params(axis='x', labelsize=slabelSize)
        ax[1].tick_params(axis='y', labelsize=slabelSize)
        ax[1].set_xlabel('Session #', fontsize=labelSize)
        ax[1].set_ylabel('# ROIs', fontsize=labelSize)
        ax[1].set_title('Red ROIs Per Session', fontsize=labelSize)
        ax[1].legend(loc='lower right', fontsize=slabelSize-4)

    ax[0].set_ylim(0)
    ax[1].set_ylim(0)

    saveFigure(fig.number, 'sessionStats', 'roi_counts')
    plt.show()
    
    
    