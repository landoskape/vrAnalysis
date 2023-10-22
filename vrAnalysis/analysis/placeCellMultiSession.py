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
from .standardAnalysis import multipleAnalysis
from . import placeCellSingleSession

vrdb = database.vrDatabase()

class placeCellMultiSession(multipleAnalysis):
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

    def get_place_field(self, snake, method='max'):
        """get sorting index based on spikemap, roi index, and trial index"""
        assert method=='com' or method=='max', f"invalid method ({method}), must be either 'com' or 'max'"
        
        # if method is 'com' (=center of mass), use weighted mean to get place field location
        if method=='com':
            # note that this can generate buggy behavior if spkmap isn't based on mostly positive signals!
            distcenters = list(set([tuple(self.pcss[i].distcenters) for i in self.idx_ses]))
            assert len(distcenters)==1, "more than 1 distcenter array found for the requested sessions!"
            distcenters = np.array(distcenters[0])
            nonnegativeProfile = np.maximum(snake, 0) 
            pfloc = np.sum(nonnegativeProfile * distcenters.reshape(1,-1), axis=1) / np.sum(nonnegativeProfile, axis=1)

        # if method is 'max' (=maximum rate), use maximum to get place field location
        if method=='max':
            pfloc = np.argmax(snake, axis=1)

        # Then sort...
        pfidx = np.argsort(pfloc)

        return pfloc, pfidx
        
    def make_snake_data(self, envnum, idx_ses=None, sortby=None, cutoffs=(0.5, 0.8), method='max', include_red=True):
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
        assert all([~np.isnan(ei) for ei in envidx]), "requested environment not in all sessions"
        
        spkmaps = self.get_from_pcss('spkmap', self.idx_ses)
        idx_reliable = [self.pcss[i].get_reliable(cutoffs=cutoffs)[ei] for i, ei in zip(self.idx_ses, envidx)]
        idx_red = [self.pcss[i].vrexp.getRedIdx(keepPlanes=self.keepPlanes) for i in self.idx_ses]
        idx_train = [self.pcss[i].train_idx[ei] for i, ei in zip(self.idx_ses, envidx)]
        idx_test = [self.pcss[i].test_idx[ei] for i, ei in zip(self.idx_ses, envidx)]
        idx_full = [self.pcss[i].idxFullTrialEachEnv[ei] for i, ei in zip(self.idx_ses, envidx)]
        
        track_spkmaps = [spkmap[idx_track] for spkmap, idx_track in zip(spkmaps, self.idx_tracked)]
        track_idx_reliable = [idx_rel[idx_track] for idx_rel, idx_track in zip(idx_reliable, self.idx_tracked)]
        track_idx_red = [i_red[idx_track] for i_red, idx_track in zip(idx_red, self.idx_tracked)]
        
        track_pfidx = self.pcss[sortby].get_place_field(roi_idx=self.idx_tracked[idx_sortby][track_idx_reliable[idx_sortby]], trial_idx=idx_train[idx_sortby], method=method)[1]
        
        snake_data = []
        idx_red_data = [ti_red[track_idx_reliable[idx_sortby]][track_pfidx] for ti_red in track_idx_red]
        for ii, idx_ses in enumerate(self.idx_ses):
            if idx_ses == sortby:
                # use test trials
                c_data = np.mean(track_spkmaps[ii][track_idx_reliable[idx_sortby]][track_pfidx][:,idx_test[ii]], axis=1)
                snake_data.append(c_data)
            else:
                c_data = np.mean(track_spkmaps[ii][track_idx_reliable[idx_sortby]][track_pfidx][:,idx_full[ii]], axis=1)
                snake_data.append(c_data)

            #c_red = track_idx_red[ii][track_idx_reliable[idx_sortby]]
                
        return snake_data, sortby, idx_red_data


    def measure_pfplasticity(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), method='max', absval=True, split_red=False):
        """method for getting change in place field plasticity as a function of sessions apart for tracked cells"""
        self.idx_ses, self.num_ses = self.track.get_idx_session(idx_ses=idx_ses)
        snake_data = [self.make_snake_data(envnum, idx_ses=idx_ses, sortby=sortby, cutoffs=cutoffs, method=method)
                      for sortby in idx_ses]
        snake_data, _, idx_red = map(list, zip(*snake_data))

        numROIs = [snake_data[ii][0].shape[0] for ii in range(self.num_ses)] # different # of ROIs for each sortby list
        pfloc = [np.zeros((self.num_ses, nr)) for nr in numROIs]
        for isort in range(self.num_ses):
            for iplot in range(self.num_ses):
                pfloc[isort][iplot] = self.get_place_field(snake_data[isort][iplot], method=method)[0]

        session_offsets = np.arange(1,self.num_ses)
        pf_differences = [[] for _ in session_offsets]
        if split_red: 
            pf_diff_red = [[] for _ in session_offsets]
        else:
            pf_diff_red = None

        transform = lambda x: np.abs(x) if absval else x
        
        for pf, ired in zip(pfloc, idx_red):
            for iso, offset in enumerate(session_offsets):
                for idx in range(0, self.num_ses - offset):
                    if split_red:
                        c_pf_diff = transform(pf[idx+1][~ired[0]] - pf[idx][~ired[0]])
                        c_pf_diff_red = transform(pf[idx+1][ired[0]] - pf[idx][ired[0]])
                        pf_differences[iso] = pf_differences[iso] + list(c_pf_diff[~np.isnan(c_pf_diff)])
                        pf_diff_red[iso] = pf_diff_red[iso] + list(c_pf_diff_red[~np.isnan(c_pf_diff_red)])
                    else:
                        c_pf_diff = transform(pf[idx+1] - pf[idx])
                        pf_differences[iso] = pf_differences[iso] + list(c_pf_diff[~np.isnan(c_pf_diff)])
        
        return snake_data, pf_differences, pf_diff_red
        
        
    def plot_snake(self, envnum, idx_ses=None, sortby=None, cutoffs=(0.5, 0.8), method='max', normalize=0, rewzone=True, interpolation='none', withShow=True, withSave=False):
        """method for plotting cross-validated snake plot"""
        self.idx_ses, self.num_ses = self.track.get_idx_session(idx_ses=idx_ses)
        snake_data, sortby, idx_red = self.make_snake_data(envnum, idx_ses=idx_ses, sortby=sortby, cutoffs=cutoffs, method=method)
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
            multby = -1 if ses==sortby else 1
            cim = ax[idx].imshow(multby*snake_data[idx], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', interpolation=interpolation)
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

        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            save_name = f"tracked_snake_env{envnum}_ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()


    def plot_pfplasticity(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), method='max', absval=True, split_red=False, withShow=True, withSave=False):
        snake_data, pf_differences, pf_diff_red = self.measure_pfplasticity(envnum, 
                                                                            idx_ses=idx_ses, 
                                                                            cutoffs=cutoffs,
                                                                            method=method, 
                                                                            absval=absval,
                                                                            split_red=split_red)

        ses_offsets = range(1, len(pf_differences)+1)
        num_offsets = len(ses_offsets)
        
        labelSize = 18
        lw = 1.5
        numBins = 20
        figdim = 3

        max_diff = max([np.max(np.abs(pfd)) for pfd in pf_differences])
        if split_red:
            max_diff_red = max([np.max(np.abs(pfd)) for pfd in pf_diff_red])
            max_diff = max([max_diff, max_diff_red])

        numBins = 11 if method=='max' else 21
        if absval:
            bins = np.linspace(0, max_diff, numBins)
        else:
            bins = np.linspace(-max_diff, max_diff, numBins)
        centers = helpers.edge2center(bins)
        barwidth = bins[1] - bins[0]

        fig, ax = plt.subplots(1, num_offsets, figsize=(figdim*num_offsets, figdim), layout='constrained')
        for ioff, offset in enumerate(ses_offsets):
            cdata = np.histogram(pf_differences[ioff], bins=bins)[0]
            cdata = 100 * cdata / np.sum(cdata)
            ax[ioff].bar(centers, cdata, color='k', width=barwidth) 
            if split_red:
                rdata = np.histogram(pf_diff_red[ioff], bins=bins)[0]
                rdata = 100 * rdata / np.sum(rdata)
                ax[ioff].bar(centers, rdata, color='r', width=barwidth, alpha=0.5) 

                # plot p-value
                rs = sp.stats.ranksums(pf_differences[ioff], pf_diff_red[ioff])
                ytextpos = max([max(rdata), max(cdata)])*0.95
                ptext = f"p={rs.pvalue:0.4f}"
                ax[ioff].text(centers[-1], ytextpos, ptext, horizontalalignment='right', verticalalignment='center')
                
                # plot N's
                nctl_textpos = ytextpos/9*8
                nred_textpos = ytextpos/9*7
                nctltext = f"N(ctl)={len(pf_differences[ioff])}"
                nredtext = f"N(red)={len(pf_diff_red[ioff])}"
                ax[ioff].text(centers[-1], nctl_textpos, nctltext, horizontalalignment='right', verticalalignment='center')
                ax[ioff].text(centers[-1], nred_textpos, nredtext, horizontalalignment='right', verticalalignment='center')
                
            ax[ioff].set_xlabel(fr"$\Delta$PF {method}", fontsize=labelSize)
            if ioff==0:
                ax[ioff].set_ylabel("% Counts", fontsize=labelSize)
            ax[ioff].set_title(fr"$\Delta$Ses={offset}", fontsize=labelSize)

        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            redname = 'wred_' if split_red else ''
            save_name = f"pfplasticity_env{envnum}_{redname}ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()

        
        
        
        
        
        
            
    



