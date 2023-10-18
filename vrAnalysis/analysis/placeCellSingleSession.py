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
from .standardAnalysis import standardAnalysis

def redReliability(ignoreScratched=True, **kwConditions):
    """Method for returning a list of 
    
class placeCellSingleSession(standardAnalysis):
    """
    Performs basic place cell (and behavioral) analysis on single sessions.
    
    Takes as required input a vrexp object. Optional inputs define parameters of analysis, 
    including which activity to run measurement on (could be deconvolvedOasis, or neuropilF, for example).
    
    Standard usage:
    ---------------
    == I just started this file, will populate standard usage later! ==
    """
    def __init__(self, vrexp, onefile='mpci.roiActivityDeconvolvedOasis', autoload=True, keepPlanes=[1,2,3,4], speedThreshold=5, numcv=2, standardizeSpks=True, doSmoothing=0):
        self.name = 'placeCellSingleSession'
        self.onefile = onefile
        self.vrexp = vrexp
        self.speedThreshold = speedThreshold
        self.numcv = numcv
        self.standardizeSpks = standardizeSpks
        self.doSmoothing = doSmoothing
        self.keepPlanes = keepPlanes if keepPlanes is not None else [i for i in range(len(vrexp.value['roiPerPlane']))]
        
        # automatically load data
        self.dataloaded = False
        if autoload: self.load_data()
    
    def load_data(self, onefile=None, speedThreshold=None, numcv=None, keepPlanes=None):
        """load standard data for basic place cell analysis"""
        # update onefile if using a different measure of activity
        if onefile is not None: self.onefile = onefile

        # update analysis parameters if requested
        if speedThreshold is not None: self.speedThreshold = speedThreshold
        if numcv is not None: self.numcv = numcv
        if keepPlanes is not None: self.keepPlanes = keepPlanes

        # get environment data
        self.trial_envnum = self.vrexp.loadone('trials.environmentIndex')
        self.environments = np.unique(self.trial_envnum)
        self.numEnvironments = len(self.environments)
        
        # get idx of rois within keep planes
        stackPosition = self.vrexp.loadone('mpciROIs.stackPosition')
        roiPlaneIdx = stackPosition[:,2].astype(np.int32) # plane index
        
        # figure out which ROIs are in the target planes
        self.idxUseROI = np.any(np.stack([roiPlaneIdx==pidx for pidx in self.keepPlanes]),axis=0)
        self.numROIs = self.vrexp.getNumROIs(self.keepPlanes)
        
        # measure smoothed occupancy map and speed maps, along with the distance bins used to create them
        self.omap, self.smap, self.lickmap, self.distedges = functions.getBehaviorMaps(self.vrexp, speedThreshold=self.speedThreshold) 
        self.distcenters = helpers.edge2center(self.distedges)
        
        self.numTrials = self.omap.shape[0]
        
        # convert behavioral data into timeframe of spiking data
        self.frameTrialIdx, self.framePosition, self.frameSpeed = self.vrexp.getFrameBehavior() 

        # produce the spkmap (activity vs position)
        self.spkmap = functions.getSpikeMap(self.vrexp, self.frameTrialIdx, self.framePosition, self.frameSpeed, self.distedges, self.omap,
                                            speedThreshold=self.speedThreshold, standardizeSpks=self.standardizeSpks, doSmoothing=self.doSmoothing)[self.idxUseROI]
        
        # find out which trials the mouse explored the whole environment
        self.boolFullTrials = np.all(~np.isnan(self.omap),axis=1) 
        self.idxFullTrials = np.where(self.boolFullTrials)[0]
        self.idxFullTrialEachEnv = [np.where(self.boolFullTrials & (self.trial_envnum==env))[0] for env in self.environments]
        
        # measure reliability 
        self.measure_reliability()
        
        # report that data has been loaded
        self.dataloaded = True

    def measure_reliability(self, with_test=False):
        """method for measuring reliability and spatial information"""
        foldIdx = [helpers.cvFoldSplit(idxTrialEnv, 3) for idxTrialEnv in self.idxFullTrialEachEnv]
        self.train_idx = [np.concatenate(fidx[:2]) for fidx in foldIdx]
        self.test_idx = [fidx[2] for fidx in foldIdx]
        
        # measure reliability of spiking (in two ways)
        relmse, relcor = zip(*[functions.measureReliability(self.spkmap[:,tidx], numcv=self.numcv) for tidx in self.train_idx])
        self.relmse, self.relcor = np.stack(relmse), np.stack(relcor)
        
        # make a nonnegative spike map if it wasn't made already
        if self.standardizeSpks:
            spkmap = functions.getSpikeMap(self.vrexp, self.frameTrialIdx, self.framePosition, self.frameSpeed, self.distedges, self.omap,
                                           speedThreshold=self.speedThreshold, standardizeSpks=False, doSmoothing=self.doSmoothing)[self.idxUseROI]
        else:
            spkmap = self.spkmap

        # measure spatial info of spiking
        self.spInfo = np.stack([functions.measureSpatialInformation(self.omap[tidx], spkmap[:,tidx]) for tidx in self.train_idx]) 
        
        if with_test:
            # measure on test trials
            relmse, relcor = zip(*[functions.measureReliability(self.spkmap[:,tidx], numcv=self.numcv) for tidx in self.test_idx])
            self.test_relmse, self.test_relcor = np.stack(relmse), np.stack(relcor)
            self.test_spInfo = np.stack([functions.measureSpatialInformation(self.omap[tidx], spkmap[:,tidx]) for tidx in self.test_idx]) 
        else:
            # Alert the user that the training data was recalculated without testing
            self.test_relmse = None
            self.test_spInfo = None

    def get_place_field(self, roi_idx=None, trial_idx=None, method='com'):
        """get sorting index based on spikemap, roi index, and trial index"""
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
    
    def make_snake(self, envnum=None, with_reliable=True, cutoffs=(0.5, 0.8), method='com'):
        """make snake data from train and test sessions, for particular environment if requested"""
        # default environment is all of them
        if envnum is None: envnum = copy(self.environments)

        # envnum must be an iterable
        if not(helpers.checkIterable(envnum)): envnum = [envnum]

        # convert environment numbers to indices
        envidx = [np.where(self.environments==ev)[0][0] for ev in envnum]
        numEnv = len(envnum)
        
        # get specific trial indices for given environment(s)
        ctrain_idx = [self.train_idx[ii] for ii in envidx]
        ctest_idx = [self.test_idx[ii] for ii in envidx]
        
        # get roi indices to use
        if with_reliable:
            self.idx_in_snake = [(self.relmse[ii] >= cutoffs[0]) & (self.relcor[ii] >= cutoffs[1]) for ii in envidx]
        else:
            self.idx_in_snake = [np.ones(self.numROIs, dtype=bool) for ii in envidx]
        
        # get pf sort indices
        train_pfidx = [self.get_place_field(roi_idx=idxroi, trial_idx=idxenvtrain, method=method)[1] for idxroi, idxenvtrain in zip(self.idx_in_snake, ctrain_idx)]
        test_pfidx = [self.get_place_field(roi_idx=idxroi, trial_idx=idxenvtest, method=method)[1] for idxroi, idxenvtest in zip(self.idx_in_snake, ctest_idx)]

        # get snakes
        spkmap = [self.spkmap[idxroi] for idxroi in self.idx_in_snake]
        trainProfile = [np.mean(sm[:, idxenvtrain], axis=1) for sm, idxenvtrain in zip(spkmap, ctrain_idx)]
        testProfile = [np.mean(sm[:, idxenvtest], axis=1) for sm, idxenvtest in zip(spkmap, ctest_idx)]
        train_snake = [trainProf[pfidx] for trainProf, pfidx in zip(trainProfile, train_pfidx)]
        test_snake = [testProf[pfidx] for testProf, pfidx in zip(testProfile, test_pfidx)]
        
        return train_snake, test_snake

    def plot_snake(self, envnum=None, with_reliable=True, cutoffs=(0.5, 0.8), method='com', normalize=0, rewzone=True):
        """method for plotting cross-validated snake plot"""
        # default environment is all of them
        if envnum is None: envnum = copy(self.environments)

        # envnum must be an iterable
        if not(helpers.checkIterable(envnum)): envnum = [envnum]
        assert all([e in self.environments for e in envnum]), "envnums must be valid environment numbers within self.environments"

        # get number of environments
        numEnv = len(envnum)

        # make snakes and prepare plotting data
        train_snake, test_snake = self.make_snake(envnum=envnum, with_reliable=with_reliable, cutoffs=cutoffs, method=method)
        extent = [[self.distedges[0], self.distedges[-1], 0, ts.shape[0]] for ts in train_snake]
        if normalize:
            vmin, vmax = -np.abs(normalize), np.abs(normalize)
        else:
            magnitude = np.max(np.abs(np.concatenate((np.concatenate(train_snake),np.concatenate(test_snake)))))
            vmin, vmax = -magnitude, magnitude

        cb_ticks = np.linspace(np.fix(vmin), np.fix(vmax), int(min(11, np.fix(vmax)-np.fix(vmin)+1)))
        
        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rewPos, rewHalfwidth = functions.environmentRewardZone(self.vrexp)
            rewPos = [rewPos[np.where(self.environments==ev)[0][0]] for ev in envnum]
            rewHalfwidth = [rewHalfwidth[np.where(self.environments==ev)[0][0]] for ev in envnum]
            rect_train = [mpl.patches.Rectangle((rp, 0), rhw*2, ts.shape[0], edgecolor='none', facecolor='k', alpha=0.2) for rp, rhw, ts in zip(rewPos, rewHalfwidth, train_snake)]
            rect_test = [mpl.patches.Rectangle((rp, 0), rhw*2, ts.shape[0], edgecolor='none', facecolor='k', alpha=0.2) for rp, rhw, ts in zip(rewPos, rewHalfwidth, train_snake)]
            
        plt.close('all')
        cmap = mpl.colormaps['bwr']
        
        fig, ax = plt.subplots(numEnv, 3, width_ratios=[10,10,1], figsize=(9,3*numEnv), layout='constrained')

        if numEnv==1: ax = np.reshape(ax, (1, -1))
            
        for idx, env in enumerate(envnum):
            cim = ax[idx, 0].imshow(train_snake[idx], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent[idx], aspect='auto')
            if idx==numEnv-1:
                ax[idx,0].set_xlabel('Virtual Position (cm)')
            ax[idx,0].set_ylabel(f'Env:{env}, ROIs')
            if idx==0:
                ax[idx,0].set_title('Train Trials')

            ax[idx,1].imshow(test_snake[idx], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent[idx], aspect='auto')
            if idx==numEnv-1:
                ax[idx,1].set_xlabel('Virtual Position (cm)')
            ax[idx,1].set_ylabel('ROIs')
            if idx==0:
                ax[idx,1].set_title('Test Trials')

            if rewzone:
                ax[idx,0].add_patch(rect_train[idx])
                ax[idx,1].add_patch(rect_test[idx])

            fig.colorbar(cim, ticks=cb_ticks, orientation='vertical', cax=ax[idx, 2])
            ax[idx, 2].set_ylabel('Activity')
            
        plt.show()
        
        return fig, ax










