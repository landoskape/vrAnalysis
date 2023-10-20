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
from . import placeCellSingleSession

vrdb = database.vrDatabase()


class placeCellMultiSession(standardAnalysis):
    """
    Performs basic place cell (and behavioral) analysis on multiple sessions.
    
    Takes as required input a tracker object. Optional inputs define parameters of analysis, 
    including which activity to run measurement on (could be deconvolvedOasis, or neuropilF, for example).
    
    Standard usage:
    ---------------
    == I just started this file, will populate standard usage later! ==
    """
    def __init__(self, track, onefile='mpci.roiActivityDeconvolvedOasis', autoload=True, keepPlanes=[1,2,3,4], distStep=(1,5,2), speedThreshold=5, numcv=2, standardizeSpks=True):
        self.name = 'placeCellMultiSession'
        self.onefile = onefile
        self.track = track
        self.autoload = autoload
        self.distStep = distStep
        self.speedThreshold = speedThreshold
        self.numcv = numcv
        self.standardizeSpks = standardizeSpks
        self.keepPlanes = keepPlanes if keepPlanes is not None else [i for i in range(len(vrexp.value['roiPerPlane']))]
        
        self.create_pcss()

    def envnum_to_idx(self, envnum):
        """
        convert list of environment numbers to indices of environment within this session
        e.g. if session has environments [1,3,4], and environment 3 is requested, turn it into index 1
        """
        envnum = helpers.check_iterable(envnum)
        return [np.where(self.environments==ev)[0][0] for ev in envnum]

    def make_pcss_arguments(self):
        pcss_arguments = {
            'onefile':self.onefile, 
            'autoload':self.autoload,
            'keepPlanes':self.keepPlanes,
            'distStep':self.distStep,
            'speedThreshold':self.speedThreshold,
            'numcv':self.numcv,
            'standardizeSpks':self.standardizeSpks
        }
        return pcss_arguments
            
    def create_pcss(self, autoload=None, onefile=None, distStep=None, speedThreshold=None, numcv=None, keepPlanes=None):
        """load standard data for basic place cell analysis"""
        # update onefile if using a different measure of activity
        if onefile is not None: self.onefile = onefile

        # update analysis parameters if requested
        if autoload is not None: self.autoload = autoload
        if distStep is not None: self.distStep = distStep
        if speedThreshold is not None: self.speedThreshold = speedThreshold
        if numcv is not None: self.numcv = numcv
        if keepPlanes is not None: self.keepPlanes = keepPlanes

        # create place cell single session objects for each session
        pcss_arguments = self.make_pcss_arguments()
        self.pcss = [placeCellSingleSession(ses, **pcss_arguments) for ses in self.track.sessions]
        self.pcss_loaded = [False for _ in range(len(self.pcss))]
        
    def load_pcss_data(self, idx_ses=None):
        self.idx_ses, self.num_ses = self.track.get_idx_session(idx_ses=idx_ses)
        idx_to_load = [idx for idx in self.idx_ses if not(self.pcss_loaded[idx])]
        if len(idx_to_load)>0:
            for sesidx in tqdm(idx_to_load):
                self.pcss[sesidx].load_data()
                self.pcss_loaded[sesidx] = True
        
    def clear_pcss_data(self):
        for pcss, ploaded in zip(self.pcss, self.pcss_loaded): 
            pcss.clear_data()
            ploaded = False

    def get_from_pcss(self, attribute, idx):
        return [getattr(self.pcss[i], attribute) for i in idx]
    
    def make_snake_data(self, envnum, idx_ses=None, sortby=None, cutoffs=(0.5, 0.8), method='max'):
        self.idx_ses, self.num_ses = self.track.get_idx_session(idx_ses=idx_ses)
        if sortby is None:
            sortby = self.idx_ses[0]
        else:
            assert sortby in self.idx_ses, f"sortby session ({sortby}) is not in requested sessions ({self.idx_ses})"

        # idx of session to sort by
        idx_sortby = {val: idx for idx, val in enumerate(self.idx_ses)}[sortby]
        
        self.load_pcss_data(idx_ses=self.idx_ses)
        self.idx_tracked = self.track.get_tracked_idx(idx_ses=self.idx_ses, keepPlanes=self.keepPlanes)
        
        envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in self.idx_ses]
        
        spkmaps = self.get_from_pcss('spkmap', self.idx_ses)
        idx_reliable = [self.pcss[i].get_reliable(cutoffs=cutoffs)[ei] for i, ei in zip(self.idx_ses, envidx)]
        idx_train = [self.pcss[i].train_idx[ei] for i, ei in zip(self.idx_ses, envidx)]
        idx_test = [self.pcss[i].test_idx[ei] for i, ei in zip(self.idx_ses, envidx)]
        idx_full = [self.pcss[i].idxFullTrialEachEnv[ei] for i, ei in zip(self.idx_ses, envidx)]
        
        track_spkmaps = [spkmap[idx_track] for spkmap, idx_track in zip(spkmaps, self.idx_tracked)]
        track_idx_reliable = [idx_rel[idx_track] for idx_rel, idx_track in zip(idx_reliable, self.idx_tracked)]

        track_pfidx = self.pcss[sortby].get_place_field(roi_idx=self.idx_tracked[idx_sortby][track_idx_reliable[idx_sortby]], trial_idx=idx_train[idx_sortby], method=method)[1]
        
        snake_data = []
        for ii, idx_ses in enumerate(self.idx_ses):
            if idx_ses == sortby:
                # use test trials
                c_data = np.mean(track_spkmaps[ii][track_idx_reliable[idx_sortby]][track_pfidx][:,idx_test[ii]], axis=1)
                snake_data.append(c_data)
            else:
                c_data = np.mean(track_spkmaps[ii][track_idx_reliable[idx_sortby]][track_pfidx][:,idx_full[ii]], axis=1)
                snake_data.append(c_data)
                
        return snake_data, sortby
        
    def plot_snake(self, envnum, idx_ses=None, sortby=None, cutoffs=(0.5, 0.8), method='max', normalize=0, rewzone=True, interpolation='none', withShow=True, withSave=False):
        """method for plotting cross-validated snake plot"""
        self.idx_ses, self.num_ses = self.track.get_idx_session(idx_ses=idx_ses)
        snake_data, sortby = self.make_snake_data(envnum, idx_ses=idx_ses, sortby=sortby, cutoffs=cutoffs, method=method)
        distedges = self.pcss[idx_ses[0]].distedges
        envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in self.idx_ses]
        assert all([ei>=0 for ei in envidx]), "you requested some sessions that don't have the requested environment!"
        
        extent = [distedges[0], distedges[-1], 0, snake_data[0].shape[0]]
        if normalize:
            vmin, vmax = -np.abs(normalize), np.abs(normalize)
        else:
            magnitude = np.max(np.abs(np.concatenate(snake_data, axis=1)))
            vmin, vmax = -magnitude, magnitude

        cb_ticks = np.linspace(np.fix(vmin), np.fix(vmax), int(min(11, np.fix(vmax)-np.fix(vmin)+1)))
        labelSize = 12
        cb_unit = r'$\sigma$' if self.standardizeSpks else 'au'
        cb_label = f"Activity ({cb_unit})"
        
        # load reward zone information
        if rewzone:
            # get reward zone start and stop, and filter to requested environments
            rew_zone_data = [functions.environmentRewardZone(self.pcss[i].vrexp) for i in self.idx_ses]
            rewPos, rewHW = zip(*rew_zone_data)
            rewPos = list(set([rp[ei] for rp, ei in zip(rewPos, envidx)]))
            rewHW = list(set([rh[ei] for rh, ei in zip(rewHW, envidx)]))
            assert len(rewPos)==1, f"reward positions are not consistent: ({rewPos})"
            assert len(rewHW)==1, f"reward halfwidths are not consistent: ({rewHW})"
            rect = [mpl.patches.Rectangle((rewPos[0]-rewHW[0], 0), rewHW[0]*2, sd.shape[0], edgecolor='none', facecolor='k', alpha=0.2) for sd in snake_data]
            
        plt.close('all')
        cmap = mpl.colormaps['bwr']

        fig_dim = 4
        width_ratios = [*[fig_dim for _ in range(self.num_ses)], fig_dim/10]
        fig, ax = plt.subplots(1, self.num_ses+1, width_ratios=width_ratios, figsize=(sum(width_ratios),fig_dim), layout='constrained')

        for idx, ses in enumerate(self.idx_ses):
            cim = ax[idx].imshow(snake_data[idx], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', interpolation=interpolation)
            ax[idx].set_xlabel('Virtual Position (cm)', fontsize=labelSize)
            if idx==0:
                ax[idx].set_ylabel('ROIs', fontsize=labelSize)

            title = f"{self.track.sessions[ses].sessionPrint()}"
            if ses==sortby: title=title+' - sorted here!'
            ax[idx].set_title(title, fontsize=labelSize)

            if rewzone:
                ax[idx].add_patch(rect[idx])

        fig.colorbar(cim, ticks=cb_ticks, orientation='vertical', cax=ax[self.num_ses])
        ax[self.num_ses].set_ylabel(cb_label, fontsize=labelSize)

        # if withSave: 
        #     if len(envnum)==len(self.environments):
        #         self.saveFigure(fig.number, f'snake_plot')
        #     else:
        #         print("If envnum is less than all the environments, you can't save with this program!")
        
        # Show figure if requested
        plt.show() if withShow else plt.close()
    
        
        
        
        
            
    



