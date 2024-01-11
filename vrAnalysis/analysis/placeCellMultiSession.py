from copy import copy
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from .. import functions
from .. import helpers
from .standardAnalysis import multipleAnalysis
from . import placeCellSingleSession

class placeCellMultiSession(multipleAnalysis):
    """
    Performs basic place cell (and behavioral) analysis on multiple sessions.
    
    Takes as required input a tracker object. Optional inputs define parameters of analysis, 
    including which activity to run measurement on (could be deconvolvedOasis, or neuropilF, for example).
    
    Standard usage:
    ---------------
    == I just started this file, will populate standard usage later! ==
    """
    def __init__(self, track, onefile='mpci.roiActivityDeconvolvedOasis', autoload=False, keep_planes=[1,2,3,4], distStep=(1,5,2), speedThreshold=5, numcv=2, standardizeSpks=True):
        self.name = 'placeCellMultiSession'
        self.onefile = onefile
        self.track = track
        self.autoload = autoload
        self.distStep = distStep
        self.speedThreshold = speedThreshold
        self.numcv = numcv
        self.standardizeSpks = standardizeSpks
        self.keep_planes = keep_planes if keep_planes is not None else [i for i in range(len(track.sessions[0].value['roiPerPlane']))]
        
        self.create_pcss()

    def idx_ses_with_env(self, envnum):
        return [ii for ii, pcss in enumerate(self.pcss) if envnum in pcss.environments]
        
    def make_pcss_arguments(self):
        pcss_arguments = {
            'onefile':self.onefile, 
            'autoload':self.autoload,
            'keep_planes':self.keep_planes,
            'distStep':self.distStep,
            'speedThreshold':self.speedThreshold,
            'numcv':self.numcv,
            'standardizeSpks':self.standardizeSpks
        }
        return pcss_arguments
            
    def create_pcss(self, autoload=None, onefile=None, distStep=None, speedThreshold=None, numcv=None, keep_planes=None):
        """load standard data for basic place cell analysis"""
        # update onefile if using a different measure of activity
        if onefile is not None: self.onefile = onefile

        # update analysis parameters if requested
        if autoload is not None: self.autoload = autoload
        if distStep is not None: self.distStep = distStep
        if speedThreshold is not None: self.speedThreshold = speedThreshold
        if numcv is not None: self.numcv = numcv
        if keep_planes is not None: self.keep_planes = keep_planes

        # create place cell single session objects for each session
        pcss_arguments = self.make_pcss_arguments()
        self.pcss = [placeCellSingleSession(ses, **pcss_arguments) for ses in self.track.sessions]
        self.pcss_loaded = [self.autoload for _ in range(len(self.pcss))]
        self.environments = np.unique(np.concatenate([pcss.environments for pcss in self.pcss]))
        
    def load_pcss_data(self, idx_ses=None, **kwargs):
        self.idx_ses = self.track.get_idx_session(idx_ses=idx_ses)
        self.num_ses = len(self.idx_ses)
        idx_to_load = [idx for idx in self.idx_ses if not(self.pcss_loaded[idx])]
        if len(idx_to_load)>0:
            for sesidx in tqdm(idx_to_load):
                self.pcss[sesidx].load_data(**kwargs)
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
    
    def get_spkmaps(self, envnum, trials=None, tracked=True, idx_ses=None, pf_method='max', by_plane=False):
        """
        method for retrieving spkmap from a particular environment across sessions

        also will retrieve red index and reliability values

        if tracked=True, will filter spkmaps by whether ROIs were tracked and sort
        each sessions spkmap to be aligned by track index across sessions

        if trials=None, will return the full spkmap (#ROIs, #Trials, #SpatialBins).
        if trials='train', 'test', or 'full', then will return the average across
        the requested trials.

        pf_method determines how to measure the place field location-- can either be
        'max' for the location at peak value or 'com' for a center of mass measurement
        """
        # define idx_ses if not provided (use all sessions for requested environment)
        if idx_ses is None:
            idx_ses = self.idx_ses_with_env(envnum)

        # check that requested environment is in all requested sessions
        envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in idx_ses]
        assert all([~np.isnan(ei) for ei in envidx]), "requested environment not in all requested sessions"
        
        # load all data now
        self.load_pcss_data(idx_ses=idx_ses)
        
        # get track index if requested
        if tracked:
            idx_tracked = self.track.get_tracked_idx(idx_ses=idx_ses, keep_planes=self.keep_planes)
        
        # get spkmaps (#ROI, #Trials, #SpatialBins)
        spkmaps = self.get_from_pcss('spkmap', idx_ses)
        
        # retrieve red index for ROIs 
        idx_red = [self.pcss[i].vrexp.getRedIdx(keep_planes=self.keep_planes) for i in idx_ses]
        
        # retrieve requested trial indices (or all trial indices for requested environment)
        if trials=='train':
            # get train trials
            idx_trials = [self.pcss[i].train_idx[ei] for i, ei in zip(idx_ses, envidx)]
        elif trials=='test':
            # get test trials 
            idx_trials = [self.pcss[i].test_idx[ei] for i, ei in zip(idx_ses, envidx)]
        else:
            # Otherwise use all trials in requested environment
            idx_trials = [self.pcss[i].idxFullTrialEachEnv[ei] for i, ei in zip(idx_ses, envidx)]

        # get reliability values for ROIs
        relmse, relcor = map(list, zip(*[self.pcss[i].get_reliability_values(envnum=envnum) for i in idx_ses]))

        # get_reliability_values returns a list of relmse/relcor values for each environment
        # since only envnum (len(envnum)==1) was requested, get the 0th index for each relmse & relcor
        relmse = list(map(lambda x: x[0], relmse)) 
        relcor = list(map(lambda x: x[0], relcor))

        # get place field for ROIs
        pfloc, pfidx = map(list, zip(*[self.pcss[ises].get_place_field(trial_idx=idx_trials[ii], method=pf_method) for ii, ises in enumerate(idx_ses)]))
        
        # if using tracked only, then filter and sort by tracking index
        if tracked:
            spkmaps = [spkmap[idx_track] for spkmap, idx_track in zip(spkmaps, idx_tracked)]
            idx_red = [i_red[idx_track] for i_red, idx_track in zip(idx_red, idx_tracked)]
            relmse = [mse[idx_track] for mse, idx_track in zip(relmse, idx_tracked)]
            relcor = [cor[idx_track] for cor, idx_track in zip(relcor, idx_tracked)]
            pfloc = [pfl[idx_track] for pfl, idx_track in zip(pfloc, idx_tracked)]
            pfidx = [pfi[idx_track] for pfi, idx_track in zip(pfidx, idx_tracked)]

        # always filter spkmaps by trials for requested environment (either train/test/all)
        spkmaps = [spkmap[:, trials] for spkmap, trials in zip(spkmaps, idx_trials)]
        
        # if trial average requested, average over trials
        if trials:
            spkmaps = list(map(lambda smap: np.mean(smap, axis=1), spkmaps))
        
        # if by_plane=True, then split each dataset up into lists of the data for each plane
        if by_plane:
            spkmaps = self.track.split_by_plane(spkmaps, dim=0, tracked=tracked, idx_ses=idx_ses, keep_planes=self.keep_planes)
            idx_red = self.track.split_by_plane(idx_red, dim=0, tracked=tracked, idx_ses=idx_ses, keep_planes=self.keep_planes)
            relmse = self.track.split_by_plane(relmse, dim=0, tracked=tracked, idx_ses=idx_ses, keep_planes=self.keep_planes)
            relcor = self.track.split_by_plane(relcor, dim=0, tracked=tracked, idx_ses=idx_ses, keep_planes=self.keep_planes)
            pfloc = self.track.split_by_plane(pfloc, dim=0, tracked=tracked, idx_ses=idx_ses, keep_planes=self.keep_planes)
            pfidx = self.track.split_by_plane(pfidx, dim=0, tracked=tracked, idx_ses=idx_ses, keep_planes=self.keep_planes)

        # return data
        return spkmaps, relmse, relcor, pfloc, pfidx, idx_red
    
        
    def make_rel_data(self, envnum, idx_ses=None, sortby=None):
        """
        This method returns a comparison of reliability on source and target sessions.

        When source==target, compares reliability on train and test trials.

        Returns
        -------
        mse & cor: 
            both contain reliability values (for mse/cor metrics)
            each is a tuple of reliability values on sortby session and target session, containing ROIs tracked across the pair of sessions
        sortby:
            index of source session (aka sortby), from which reliability values in the first element of each tuple come from
        idx_ses:
            index of target sessions, from which reliability values in second element of each tuple come from (includes sortby session)
        """

        if idx_ses is None:
            idx_ses = self.idx_ses_with_env(envnum)
        
        if sortby is None: 
            sortby = idx_ses[0]
        else:
            assert sortby in idx_ses, f"sortby session ({sortby}) is not in requested sessions ({idx_ses})"
        
        idx_sortby = {val: idx for idx, val in enumerate(idx_ses)}[sortby] # get idx of sortby session from idx_ses
        
        # get idx of requested environment for each session
        envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in idx_ses]
        in_session = [~np.isnan(ei) for ei in envidx]
        assert all(in_session), f"requested environment only in following sessions: {[idx for idx, inses in zip(idx_ses, in_session) if inses]}"
        self.load_pcss_data(idx_ses=idx_ses, with_test=True) # required for reliability values -- include test for comparison of reliability within sortby session
        for i in idx_ses:
            if not hasattr(self.pcss[i], 'test_relmse') or self.pcss[i].test_relmse is None:
                self.pcss[i].measure_reliability(with_test=True)
            
        # handle tracking 
        idx_tracked_target, idx_tracked_sortby = map(list, zip(*[self.track.get_tracked_idx(idx_ses=[i, sortby], keep_planes=self.keep_planes) for i in idx_ses]))

        # get reliability values for all the cells - it's a tuple of relmse / relcor for each pcss
        relmse, relcor, relmse_test, relcor_test = map(list, zip(*[self.pcss[i].get_reliability_values(envnum=envnum, with_test=True) for i in idx_ses]))

        # for consistency, the relmse value is a list, no matter how many envnum's there are. Since we only requested
        # one envnum, the first value of the list is the array of reliability values for each session
        relmse, relcor = list(map(lambda x: x[0], relmse)), list(map(lambda x: x[0], relcor))      
        relmse_test, relcor_test = list(map(lambda x: x[0], relmse_test)), list(map(lambda x: x[0], relcor_test))      
        
        # get index of red cells and filter by tracked
        idx_red = [self.pcss[i].vrexp.getRedIdx(keep_planes=self.keep_planes) for i in self.idx_ses]
        
        # get tracked reliability arrays (for each tracked (not sortby session), tuple of reliability on sortby / target for tracked across this pair of sessions)
        mse = []
        cor = []
        red = []
        for ii, ises in enumerate(idx_ses):
            if ises!=sortby:
                # compare source to target
                cmse = (relmse[idx_sortby][idx_tracked_sortby[ii]], relmse[ii][idx_tracked_target[ii]])
                ccor = (relcor[idx_sortby][idx_tracked_sortby[ii]], relcor[ii][idx_tracked_target[ii]])
            else:
                # compare source to test trials in source
                cmse = (relmse[idx_sortby][idx_tracked_sortby[ii]], relmse_test[ii][idx_tracked_target[ii]])
                ccor = (relcor[idx_sortby][idx_tracked_sortby[ii]], relcor_test[ii][idx_tracked_target[ii]])
            cred = (idx_red[idx_sortby][idx_tracked_sortby[ii]], idx_red[ii][idx_tracked_target[ii]])
            mse.append(cmse)
            cor.append(ccor)
            red.append(cred)

        # return paired reliability arrays along with sortby session and all target session indices
        return mse, cor, red, sortby, idx_ses


    def make_skew_data(self, envnum, idx_ses=None, sortby=None, cutoffs=(0.2, 0.5), maxcutoffs=None):
        if idx_ses is None:
            if envnum is None:
                # use all sessions if not requesting ROIs based on reliability in certain environment
                self.idx_ses = [ii for ii in range(len(self.pcss))]
            else:
                # otherwise use all sessions with environment
                self.idx_ses = self.idx_ses_with_env(envnum)

        # check sortby and get idx to sortby session
        if sortby is None:
            sortby = self.idx_ses[-1]
        else:
            assert sortby in self.idx_ses, f"sortby session ({sortby}) is not in requested sessions ({self.idx_ses})"

        idx_sortby = {val: idx for idx, val in enumerate(self.idx_ses)}[sortby]

        # handle tracking
        idx_tracked_target, idx_tracked_sortby = map(list, zip(*[self.track.get_tracked_idx(idx_ses=[i, sortby], keep_planes=self.keep_planes) for i in self.idx_ses]))

        # handle environment request
        if envnum is not None:
            envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in self.idx_ses]
            in_session = [~np.isnan(ei) for ei in envidx]
            assert all(in_session), f"requested environment only in following sessions: {[idx for idx, inses in zip(self.idx_ses, in_session) if inses]}"
            self.load_pcss_data(idx_ses=self.idx_ses)

        # get spiking data (in time!)
        idx_to_planes = [self.pcss[i].vrexp.idxToPlanes(keep_planes=self.keep_planes) for i in self.idx_ses]    
        spkdata = [self.pcss[i].vrexp.loadone(self.onefile).T[pi,:] for i, pi in zip(self.idx_ses, idx_to_planes)]

        # handle reliability (if environment requested)
        if envnum is not None:
            idx_reliable = [self.pcss[i].get_reliable(cutoffs=cutoffs, maxcutoffs=maxcutoffs)[ei] for i, ei in zip(self.idx_ses, envidx)]
        else:
            idx_reliable = [np.full(sd.shape[0], True) for sd in spkdata] # if no environment requested, use all ROIs

        # get red idx
        idx_red = [self.pcss[i].vrexp.getRedIdx(keep_planes=self.keep_planes) for i in self.idx_ses]

        # filter by tracked
        spkdata = [sd[idx_tracked] for sd, idx_tracked in zip(spkdata, idx_tracked_target)]
        idx_reliable = [ir[idx_tracked] for ir, idx_tracked in zip(idx_reliable, idx_tracked_target)]
        idx_reliable_sortby = [ir[idx_tracked] for ir, idx_tracked in zip(idx_reliable, idx_tracked_sortby)]
        idx_red = [ir[idx_tracked] for ir, idx_tracked in zip(idx_red, idx_tracked_target)]
        
        # filter by reliability
        spkdata = [sd[irel] for sd, irel in zip(spkdata, idx_reliable_sortby)]
        idx_red = [ir[irel] for ir, irel in zip(idx_red, idx_reliable_sortby)]

        # get skew for each ROI
        skew = [sp.stats.skew(sd, axis=1) for sd in spkdata]

        return skew, idx_red

    def make_snake_data(self, envnum, idx_ses=None, sortby=None, cutoffs=(0.5, 0.8), maxcutoffs=None, method='max', include_red=True):
        self.idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses

        if sortby is None:
            sortby = self.idx_ses[-1]
        else:
            assert sortby in self.idx_ses, f"sortby session ({sortby}) is not in requested sessions ({self.idx_ses})"

        # idx of session to sort by
        idx_sortby = {val: idx for idx, val in enumerate(self.idx_ses)}[sortby]
        
        self.load_pcss_data(idx_ses=self.idx_ses)
        self.idx_tracked = self.track.get_tracked_idx(idx_ses=self.idx_ses, keep_planes=self.keep_planes)
        
        envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in self.idx_ses]
        in_session = [~np.isnan(ei) for ei in envidx]
        assert all(in_session), f"requested environment only in following sessions: {[idx for idx, inses in zip(self.idx_ses, in_session) if inses]}"
        
        spkmaps = self.get_from_pcss('spkmap', self.idx_ses)
        idx_reliable = [self.pcss[i].get_reliable(cutoffs=cutoffs, maxcutoffs=maxcutoffs)[ei] for i, ei in zip(self.idx_ses, envidx)]
        idx_red = [self.pcss[i].vrexp.getRedIdx(keep_planes=self.keep_planes) for i in self.idx_ses]
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


    def make_paired_snake(self, envnum, target, sortby, cutoffs=(0.5, 0.8), maxcutoffs=None, method='max', both_reliable=False):
        """similar to above "make_snake_data" but only contains the data from two sessions, a sortby session and a target session"""
        # idx of session to sort by
        idx_target = 0
        idx_sortby = 1
        idx_ses = [target, sortby]
        
        self.load_pcss_data(idx_ses=idx_ses)
        self.idx_tracked = self.track.get_tracked_idx(idx_ses=idx_ses, keep_planes=self.keep_planes)
        
        envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in idx_ses]
        assert all([~np.isnan(ei) for ei in envidx]), "requested environment not in all sessions"
        
        spkmaps = self.get_from_pcss('spkmap', idx_ses)
        idx_reliable = [self.pcss[i].get_reliable(cutoffs=cutoffs, maxcutoffs=maxcutoffs)[ei] for i, ei in zip(idx_ses, envidx)]
        idx_red = [self.pcss[i].vrexp.getRedIdx(keep_planes=self.keep_planes) for i in idx_ses]
        idx_train = [self.pcss[i].train_idx[ei] for i, ei in zip(idx_ses, envidx)]
        idx_test = [self.pcss[i].test_idx[ei] for i, ei in zip(idx_ses, envidx)]
        idx_full = [self.pcss[i].idxFullTrialEachEnv[ei] for i, ei in zip(idx_ses, envidx)]

        relmse, relcor = map(list, zip(*[self.pcss[i].get_reliability_values(envnum=envnum) for i in idx_ses]))
        relmse = list(map(lambda x: x[0], relmse))
        relcor = list(map(lambda x: x[0], relcor))
        
        #print(f"{target}/{sortby} -- {type(relmse)}, {len(relmse)}, -- {type(relcor)}, {len(relcor)}")
        #for mse in relmse:
        #    print(type(mse), len(mse), mse.shape)
        
        track_spkmaps = [spkmap[idx_track] for spkmap, idx_track in zip(spkmaps, self.idx_tracked)]
        track_idx_reliable = [idx_rel[idx_track] for idx_rel, idx_track in zip(idx_reliable, self.idx_tracked)]
        track_idx_red = [i_red[idx_track] for i_red, idx_track in zip(idx_red, self.idx_tracked)]
        track_relmse = [mse[idx_track] for mse, idx_track in zip(relmse, self.idx_tracked)]
        track_relcor = [cor[idx_track] for cor, idx_track in zip(relcor, self.idx_tracked)]
        
        # use reliable on "sortby" sessions
        keep_idx_reliable = track_idx_reliable[idx_sortby]
        if both_reliable: 
            # if requesting reliable on both sessions, then add reliability on target session
            keep_idx_reliable &= track_idx_reliable[idx_target]
            
        track_pfidx = self.pcss[sortby].get_place_field(roi_idx=self.idx_tracked[idx_sortby][track_idx_reliable[idx_sortby]], trial_idx=idx_train[idx_sortby], method=method)[1]

        idx_red_data = [ti_red[track_idx_reliable[idx_sortby]][track_pfidx] for ti_red in track_idx_red]
        target_snake = np.mean(track_spkmaps[idx_target][track_idx_reliable[idx_sortby]][track_pfidx][:,idx_test[idx_target]], axis=1)
        sortby_snake = np.mean(track_spkmaps[idx_sortby][track_idx_reliable[idx_sortby]][track_pfidx][:,idx_full[idx_sortby]], axis=1)
        target_mse = track_relmse[idx_target][keep_idx_reliable][track_pfidx] # rel-mse in target, filtered by whether reliable on sortby sessions, sorted by place field (just like snakes)
        target_cor = track_relcor[idx_target][keep_idx_reliable][track_pfidx] # rel-cor in target, filtered by whether reliable on sortby sessions, sorted by place field (just like snakes)
        sortby_mse = track_relmse[idx_sortby][keep_idx_reliable][track_pfidx] # rel-mse in sortby, filtered by whether reliable on sortby sessions, sorted by place field (just like snakes)
        sortby_cor = track_relcor[idx_sortby][keep_idx_reliable][track_pfidx] # rel-cor in sortby, filtered by whether reliable on sortby sessions, sorted by place field (just like snakes)
        return [target_snake, sortby_snake], idx_red_data, [target_mse, sortby_mse], [target_cor, sortby_cor]
        

    def measure_pfreliability(self, envnum, idx_ses=None, cutoffs=None, maxcutoffs=None, method='max'):
        """method for getting change in place field plasticity as a function of sessions apart for tracked cells"""
        store_idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        store_num_ses = len(store_idx_ses)
        self.load_pcss_data(idx_ses=store_idx_ses)
        target_snake = []
        sortby_snake = []
        target_red = []
        sortby_red = []
        target_relmse = []
        sortby_relmse = []
        target_relcor = []
        sortby_relcor = []
        for sortby in store_idx_ses:
            c_target_snake = []
            c_sortby_snake = []
            c_target_ired = []
            c_sortby_ired = []
            c_target_relmse = []
            c_sortby_relmse = []
            c_target_relcor = []
            c_sortby_relcor = []
            for target in store_idx_ses:
                cdata, cired, crelmse, crelcor = self.make_paired_snake(envnum, target, sortby, cutoffs=cutoffs, maxcutoffs=maxcutoffs, method=method)
                c_target_snake.append(cdata[0])
                c_sortby_snake.append(cdata[1])
                c_target_ired.append(cired[0])
                c_sortby_ired.append(cired[1])
                c_target_relmse.append(crelmse[0])
                c_sortby_relmse.append(crelmse[1])
                c_target_relcor.append(crelcor[0])
                c_sortby_relcor.append(crelcor[1])
            target_snake.append(c_target_snake)
            sortby_snake.append(c_sortby_snake)
            target_red.append(c_target_ired)
            sortby_red.append(c_sortby_ired)
            target_relmse.append(c_target_relmse)
            sortby_relmse.append(c_sortby_relmse)
            target_relcor.append(c_target_relcor)
            sortby_relcor.append(c_sortby_relcor)

        self.idx_ses, self.num_ses = store_idx_ses, store_num_ses
        return target_relmse, target_relcor, target_snake, target_red, sortby_red
    
    def measure_rel_plasticity(self, envnum, idx_ses=None, cutoffs=None, maxcutoffs=None, method='max'):
        """method for getting change in reliability as a function of sessions apart for tracked cells"""
        if cutoffs is None: cutoffs = [-np.inf, -np.inf]
        if maxcutoffs is None: maxcutoffs = [np.inf, np.inf]
        store_idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        store_num_ses = len(store_idx_ses)
        self.load_pcss_data(idx_ses=store_idx_ses)
        target_red = []
        sortby_red = []
        target_relmse = []
        target_relcor = []
        for sortby in store_idx_ses:
            c_target_ired = []
            c_sortby_ired = []
            c_target_relmse = []
            c_target_relcor = []
            for target in store_idx_ses:
                _, cired, crelmse, crelcor = self.make_paired_snake(envnum, target, sortby, cutoffs=cutoffs, maxcutoffs=maxcutoffs, method=method)
                crelmse = ((crelmse[0] > cutoffs[0]) & (crelmse[0] < maxcutoffs[0]))
                crelcor = ((crelcor[0] > cutoffs[1]) & (crelcor[0] < maxcutoffs[1]))
                c_target_ired.append(cired[0])
                c_sortby_ired.append(cired[1])
                c_target_relmse.append(crelmse)
                c_target_relcor.append(crelcor)
            target_red.append(c_target_ired)
            sortby_red.append(c_sortby_ired)
            target_relmse.append(c_target_relmse)
            target_relcor.append(c_target_relcor)
            
        self.idx_ses, self.num_ses = store_idx_ses, store_num_ses
        return target_relmse, target_relcor, target_red, sortby_red
    
    def measure_pfplasticity(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), both_reliable=False):
        """method for getting change in place field plasticity as a function of sessions apart for tracked cells"""
        store_idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        store_num_ses = len(store_idx_ses)
        self.load_pcss_data(idx_ses=store_idx_ses)
        target_snake = []
        sortby_snake = []
        target_red = []
        sortby_red = []
        for sortby in store_idx_ses:
            c_target_snake = []
            c_sortby_snake = []
            c_target_ired = []
            c_sortby_ired = []
            for target in store_idx_ses:
                cdata, cired, _, _ = self.make_paired_snake(envnum, target, sortby, cutoffs=cutoffs, both_reliable=both_reliable)
                c_target_snake.append(cdata[0])
                c_sortby_snake.append(cdata[1])
                c_target_ired.append(cired[0])
                c_sortby_ired.append(cired[1])
            target_snake.append(c_target_snake)
            sortby_snake.append(c_sortby_snake)
            target_red.append(c_target_ired)
            sortby_red.append(c_sortby_ired)

        # I need a grid of histograms comparing the snakes across sessions
        r2 = []
        pc = []
        r2_stat = []
        pc_stat = []
        for isort, (snakes_target, snakes_sortby, red_target, red_sortby) in enumerate(zip(target_snake, sortby_snake, target_red, sortby_red)):
            c_r2 = []
            c_pc = []
            c_r2_stat = []
            c_pc_stat = []
            for itarget, (snake_target, snake_sortby, r_target, r_sortby) in enumerate(zip(snakes_target, snakes_sortby, red_target, red_sortby)):
                assert snake_target.shape[0] == snake_sortby.shape[0], "oops"
                # red is only if red in both target and snake session
                c_idx_red = r_target & r_sortby
                # get R-squared
                dv_target = np.max(snake_target, axis=1, keepdims=True)
                dv_sortby = np.max(snake_sortby, axis=1, keepdims=True)
                st = snake_target / (dv_target + 1*(dv_target==0))
                ss = snake_sortby / (dv_sortby + 1*(dv_sortby==0))
                cc_r2 = helpers.vectorRSquared(ss, st, axis=1)
                cc_r2[cc_r2==-np.inf] = np.nan
                # also get correlation
                cc_pc = helpers.vectorCorrelation(ss, st, axis=1)
                # then add results
                c_r2.append(cc_r2)
                c_pc.append(cc_pc)
                # now do stats (just ranksum) 
                c_r2_stat.append(sp.stats.ranksums(cc_r2[~c_idx_red], cc_r2[c_idx_red]))
                c_pc_stat.append(sp.stats.ranksums(cc_pc[~c_idx_red], cc_pc[c_idx_red]))
            # keep all results
            r2.append(c_r2)
            pc.append(c_pc)
            r2_stat.append(c_r2_stat)
            pc_stat.append(c_pc_stat)

        self.idx_ses, self.num_ses = store_idx_ses, store_num_ses
        return r2, pc, r2_stat, pc_stat, target_red, sortby_red

        """
        The code commented below is how I used to do it, but I'm updating to make a grid
        so need to rewrite everything anyway. Also, I'm switching to the R2 method...
        
        # numROIs = [snake_data[ii][0].shape[0] for ii in range(self.num_ses)] # different # of ROIs for each sortby list
        # pfloc = [np.zeros((self.num_ses, nr)) for nr in numROIs]
        # for isort in range(self.num_ses):
        #     for iplot in range(self.num_ses):
        #         pfloc[isort][iplot] = self.get_place_field(snake_data[isort][iplot], method=method)[0]

        # session_offsets = np.arange(1,self.num_ses)
        # pf_differences = [[] for _ in session_offsets]
        # if split_red: 
        #     pf_diff_red = [[] for _ in session_offsets]
        # else:
        #     pf_diff_red = None

        # transform = lambda x: np.abs(x) if absval else x
        
        # for pf, ired in zip(pfloc, idx_red):
        #     for iso, offset in enumerate(session_offsets):
        #         for idx in range(0, self.num_ses - offset):
        #             if split_red:
        #                 c_pf_diff = transform(pf[idx+1][~ired[0]] - pf[idx][~ired[0]])
        #                 c_pf_diff_red = transform(pf[idx+1][ired[0]] - pf[idx][ired[0]])
        #                 pf_differences[iso] = pf_differences[iso] + list(c_pf_diff[~np.isnan(c_pf_diff)])
        #                 pf_diff_red[iso] = pf_diff_red[iso] + list(c_pf_diff_red[~np.isnan(c_pf_diff_red)])
        #             else:
        #                 c_pf_diff = transform(pf[idx+1] - pf[idx])
        #                 pf_differences[iso] = pf_differences[iso] + list(c_pf_diff[~np.isnan(c_pf_diff)])
        # return snake_data, pf_differences, pf_diff_red
        """


    def plot_snake(self, envnum, idx_ses=None, sortby=None, cutoffs=(0.5, 0.8), maxcutoffs=None, sort_by_red=False, method='max', normalize=0, rewzone=True, interpolation='none', withShow=True, withSave=False):
        """method for plotting cross-validated snake plot"""
        self.idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        self.num_ses = len(self.idx_ses)
        snake_data, sortby, idx_red = self.make_snake_data(envnum, idx_ses=idx_ses, sortby=sortby, cutoffs=cutoffs, maxcutoffs=maxcutoffs, method=method)
        distedges = self.pcss[self.idx_ses[0]].distedges
        envidx = [self.pcss[i].envnum_to_idx(envnum)[0] for i in self.idx_ses]
        assert all([ei>=0 for ei in envidx]), "you requested some sessions that don't have the requested environment!"
        
        extent = [distedges[0], distedges[-1], 0, snake_data[0].shape[0]]
        
        if normalize > 0:
            vmin, vmax = -np.abs(normalize), np.abs(normalize)
        elif normalize < 0:
            maxrois = np.concatenate([np.max(np.abs(sd), axis=1) for sd in snake_data])
            vmin, vmax = -np.percentile(maxrois, -normalize), np.percentile(maxrois, -normalize)
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

        if sort_by_red:
            idx_sortby = {val: idx for idx, val in enumerate(self.idx_ses)}[sortby]
            for ii in range(self.num_ses):
                snake_data[ii] = np.concatenate((snake_data[ii][~idx_red[idx_sortby]], snake_data[ii][idx_red[idx_sortby]]), axis=0)
                
        for idx, ses in enumerate(self.idx_ses):
            multby = -1 if ses==sortby else 1
            cim = ax[idx].imshow(multby*snake_data[idx], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', interpolation=interpolation)
            if sort_by_red:
                ax[idx].plot([distedges[1],distedges[1]],[0,np.sum(idx_red[idx_sortby])-1], c='r', lw=5)
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
            sesidx = '_'.join([str(i) for i in self.idx_ses])
            save_name = f"tracked_snake_env{envnum}_ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()

        return snake_data, sortby, idx_red


    def plot_rel_comparison(self, envnum, idx_ses=None, sortby=None, rel_method='pc', withShow=True, withSave=False):
        assert isinstance(rel_method, str) and (rel_method.lower()=='pc' or rel_method.lower()=='r2'), "rel_method must be 'r2' or 'pc'"

        if idx_ses is None:
            idx_ses = self.idx_ses_with_env(envnum)
        
        if sortby is None:
            sortby = idx_ses[0]
        else:
            assert isinstance(sortby, int) and sortby in idx_ses, "sortby must be in idx_ses"

        # Collect data
        mse, cor, red, source, target = self.make_rel_data(envnum, idx_ses=idx_ses, sortby=sortby)
        rel = mse if rel_method=='r2' else cor
        red_idx = [rd[0] | rd[1] for rd in red] # red if either red in source or target

        idx_to_ses = {val:idx for idx, val in enumerate(idx_ses)}
        num_rows = 4
        num_cols = len(idx_ses)+1
        
        figdim = 2
        labelSize = 12
        rel_name = 'R^2' if rel_method=='r2' else 'PC'
        num_bins = 8
        cmap = sns.light_palette("k", as_cmap=True)
        rcmap = sns.light_palette("r", as_cmap=True)
        width_ratios = [*[figdim]*len(idx_ses), figdim/5]

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(figdim*num_cols, figdim*num_rows), width_ratios=width_ratios, layout='constrained')
        # fig.subplots_adjust(wspace=0.02)
        
        ims = []
        for crel, cred, it in zip(rel, red_idx, target):
            cidx = idx_to_ses[it]
            if source==it: 
                for spine in ('left', 'right', 'bottom', 'top'):
                    for row in range(num_rows):
                        getattr(ax[row,cidx].spines, spine).set(color='b', linewidth=3)
            
            # first measure 2d histograms and get bin edges
            H_ctl, xe, ye = np.histogram2d(crel[0][~cred], crel[1][~cred], bins=num_bins, density=False)
            H_red, _, _ = np.histogram2d(crel[0][cred], crel[1][cred], bins=[xe, ye], density=False)
            H_ctl = 100 * H_ctl / np.sum(H_ctl) # normalize to percentage in each bin (not density by area!!)
            H_red = 100 * H_red / np.sum(H_red) 
            H_diff = (H_red - H_ctl).T # transpose for visualization (down is x...)

            sns.histplot(x=crel[0][~cred], y=crel[1][~cred], bins=[xe, ye], thresh=0, cmap=cmap,  ax=ax[0, cidx])
            #sns.kdeplot(x=crel[0][~cred], y=crel[1][~cred], color=('k', 0.5), fill=False, levels=8, thresh=0.0, ax=ax[0, cidx], legend=False)
            
            sns.histplot(x=crel[0][cred], y=crel[1][cred], bins=[xe, ye], thresh=0, cmap=rcmap,  ax=ax[1, cidx])
            #sns.kdeplot(x=crel[0][cred], y=crel[1][cred], color=('r', 0.5), fill=False, levels=8, thresh=0.0, ax=ax[1, cidx], legend=False)
            #sns.scatterplot(x=crel[0][cred], y=crel[1][cred], color=('r', 0.8), s=8, ax=ax[1, cidx], legend=False, lw=0.5, edgecolors='k')
            
            sns.histplot(x=crel[0][~cred], y=crel[1][~cred], bins=[xe, ye], thresh=0, cmap=cmap,  ax=ax[2, cidx])
            #sns.kdeplot(x=crel[0][~cred], y=crel[1][~cred], color=('k', 0.5), fill=False, levels=8, thresh=0.0, ax=ax[2, cidx], legend=False)
            sns.scatterplot(x=crel[0][cred], y=crel[1][cred], color=('r', 0.8), s=8, ax=ax[2, cidx], legend=False, lw=0.5, edgecolors='k')
            #sns.kdeplot(x=crel[0][cred], y=crel[1][cred], color=('r', 0.5), fill=False, levels=8, thresh=0.0, ax=ax[2, cidx], legend=False)
                        
            extent = [xe.min(), xe.max(), ye.min(), ye.max()]
            cim = ax[3, cidx].imshow(H_diff, extent=extent, interpolation=None, aspect='auto', origin='lower', cmap=sns.color_palette("icefire", as_cmap=True))
            ims.append(cim)

            for row in range(num_rows):
                ax[row,cidx].set_xlim(-0.5, 1)
                ax[row,cidx].set_ylim(-0.5, 1)

            ax[-1, cidx].set_xlabel("src", fontsize=labelSize)
            if cidx==0:
                ax[0, cidx].set_ylabel("tgt (ctl)", fontsize=labelSize)
                ax[1, cidx].set_ylabel("tgt (red)", fontsize=labelSize)
                ax[2, cidx].set_ylabel("tgt (both)", fontsize=labelSize)
                ax[3, cidx].set_ylabel("$\Delta$ Red-Ctl", fontsize=labelSize)
            ax[0, cidx].set_title(f"S{sortby}>T{it}", fontsize=labelSize)

        # Control appearance
        for row in range(0, num_rows):
            for col in range(0, num_cols):
                if col < num_cols-1:
                    ax[row, col].set_box_aspect(1)
                    # ax[row, col].set_xlim(-0.5, 1)
                    # ax[row, col].set_ylim(-0.5, 1)
                if row==num_rows-1:
                    ax[row, col].set_xticks([0, 0.5, 1])
                if col==0:
                    ax[row, col].set_yticks([0, 0.5, 1])
                if row<num_rows-1 and col<num_cols-1:
                    ax[row, col].set_xticks([])
                if col>0 and col<num_cols-1:
                    ax[row, col].set_yticks([])
                if col==num_cols-1 and row<num_rows-1:
                    ax[row, col].set_axis_off()
        
        # align color axis of difference maps and create colorbar
        vminmax = np.max(np.abs(np.concatenate([im.get_clim() for im in ims])))
        for im in ims:
            im.set(clim=(-vminmax,vminmax))

        plt.colorbar(ims[0], cax=ax[3,-1], location='right')
        ax[3,-1].set_ylabel(r'% difference')

        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            save_name = f"rel_comparison_{envnum}_sortby{sortby}_{rel_method}_ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()



    def hist_pfplasticity(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), present='r2', method='max', split_red=False, withShow=True, withSave=False):
        idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        r2, pc, r2_stat, pc_stat, target_red, sortby_red = self.measure_pfplasticity(envnum, idx_ses=idx_ses, cutoffs=cutoffs)
        num_ses = len(idx_ses)
        
        labelSize = 18
        lw = 1.5
        numBins = 20
        figdim = 1.5

        if present=='r2':
            data = r2
            data_stat = r2_stat
        elif present=='pc':
            data = pc
            data_stat = pc_stat
        else:
            raise ValueError("Only 'r2' or 'pc' allowed as 'present' argument!")

        minval, maxval = np.nanmin([np.nanmin(np.concatenate(d)) for d in data]), np.nanmax([np.nanmax(np.concatenate(d)) for d in data])
        minval = np.max([minval, -5])
        numBins = 11

        bins = np.linspace(minval, maxval, numBins)
        centers = helpers.edge2center(bins)
        barwidth = bins[1] - bins[0]

        fig, ax = plt.subplots(num_ses, num_ses, figsize=(figdim*num_ses, figdim*num_ses), layout='constrained', sharex=True)
        ax = np.reshape(ax, (num_ses, num_ses))
        for ii, (ises, itred, isred) in enumerate(zip(idx_ses, target_red, sortby_red)):
            for jj, (jses, jtred, jsred) in enumerate(zip(idx_ses, itred, isred)):
                if split_red:
                    idx_keep_red = jtred & jsred
                else:
                    idx_keep_red = np.full(jtred.shape, False) # none are red if not separating red cells
                
                ctl_counts = np.histogram(data[ii][jj][~idx_keep_red], bins=bins, density=False)[0]
                ctl_counts = 100*ctl_counts/np.sum(ctl_counts)
                if split_red:
                    # print(ii, jj, "red stats:", f"Sortby red: {np.sum(jsred)}", f"Target red: {np.sum(jtred)}")
                    red_counts = np.histogram(data[ii][jj][idx_keep_red], bins=bins, density=False)[0]
                    red_counts = 100*red_counts/np.sum(red_counts)
                    # get stats on difference between control and red
                    rs_stat, rs_pval = sp.stats.ranksums(data[ii][jj][~idx_keep_red], data[ii][jj][idx_keep_red])
                
                ax[ii,jj].bar(centers, ctl_counts, color='k', alpha=0.5, width=barwidth)
                if split_red:
                    ax[ii,jj].bar(centers, red_counts, color='r', alpha=0.5, width=barwidth)
                
                if split_red:
                    ax[ii,jj].axvline(np.mean(data[ii][jj][~idx_keep_red]), color='k')
                    ax[ii,jj].axvline(np.mean(data[ii][jj][idx_keep_red]), color='r')
        
                    ytextpos = max([max(ctl_counts), max(red_counts)])*0.95
                    ptext = f"p={data_stat[ii][jj][1]:0.4f}"
                    ax[ii,jj].text(centers[0], ytextpos, ptext, horizontalalignment='left', verticalalignment='center', fontsize=8)
                    
                    nctl_textpos = ytextpos/9*8
                    nred_textpos = ytextpos/9*7
                    nctltext = f"N(ctl)={np.sum(~idx_keep_red)}"
                    nredtext = f"N(red)={np.sum(idx_keep_red)}"
                    ax[ii,jj].text(centers[0], nctl_textpos, nctltext, horizontalalignment='left', verticalalignment='center', fontsize=8)
                    ax[ii,jj].text(centers[0], nred_textpos, nredtext, horizontalalignment='left', verticalalignment='center', fontsize=8)

                ax[ii,jj].set_yticklabels([])
                if ii==0:
                    ax[ii,jj].set_title(f"target {jj}")
                if jj==0:
                    ax[ii,jj].set_ylabel(f"source {ii}")
                if ii==(num_ses-1):
                    ax[ii,jj].set_xlabel(f"metric: {present}")
            
                # for iredpc in pc[ii][jj][idx_keep_red]:
                #     ax[ii,jj].axvline(iredpc, color='r')

        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            redname = 'wred_' if split_red else ''
            save_name = f"pfplasticity_env{envnum}_metric{present}_{redname}ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()

    
    def compare_pfplasticity(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), both_reliable=False, reduction='mean', min_mse=-8, withShow=True, withSave=False):
        idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        r2, pc, r2_stat, pc_stat, target_red, sortby_red = self.measure_pfplasticity(envnum, idx_ses=idx_ses, cutoffs=cutoffs, both_reliable=both_reliable)
        num_ses = len(idx_ses)

        # put the difference and pval in a numpy array
        r2_diff = np.full((num_ses, num_ses), np.nan)
        pc_diff = np.full((num_ses, num_ses), np.nan)
        r2_pval = np.full((num_ses, num_ses), np.nan)
        pc_pval = np.full((num_ses, num_ses), np.nan)
        for ii, (itred, isred) in enumerate(zip(target_red, sortby_red)):
            for jj, (jtred, jsred) in enumerate(zip(itred, isred)):
                idx_red = jtred & jsred # only red if red in target and sortby sessions

                # copy r2 and pc (especially r2 since we're updating values)
                c_r2 = copy(r2[ii][jj])
                c_r2[c_r2 < min_mse] = min_mse
                c_r2[np.isnan(c_r2)] = min_mse
                c_pc = copy(pc[ii][jj])
                if np.any(idx_red):
                    if reduction=='mean':
                        r2_diff[ii, jj] = np.mean(c_r2[idx_red]) - np.mean(c_r2[~idx_red])
                        pc_diff[ii, jj] = np.mean(c_pc[idx_red]) - np.mean(c_pc[~idx_red])
                    elif reduction=='median':
                        r2_diff[ii, jj] = np.median(c_r2[idx_red]) - np.median(c_r2[~idx_red])
                        pc_diff[ii, jj] = np.median(c_pc[idx_red]) - np.median(c_pc[~idx_red])
                    else:
                        raise ValueError(f"reduction method not recognized ({reduction}), only 'mean' or 'median' are coded!")
                    # recompute pvals after clipping minimum values
                    r2_pval[ii, jj] = sp.stats.ranksums(c_r2[idx_red], c_r2[~idx_red])[1]
                    pc_pval[ii, jj] = sp.stats.ranksums(c_pc[idx_red], c_pc[~idx_red])[1]

        # 1 if nan -- this means there was no data (usually)
        zz = np.isnan(r2_diff)*1.0
        r2_pval[np.isnan(r2_pval)]=1
        pc_pval[np.isnan(pc_pval)]=1
        r2_diff[np.isnan(r2_diff)]=0
        pc_diff[np.isnan(pc_diff)]=0
        
        # plot parameters
        labelSize = 18
        lw = 1.5
        fig_dim = 4
        extent = [-0.5, num_ses-0.5, -0.5, num_ses-0.5]
        width_ratios = [*[fig_dim for _ in range(2)], fig_dim/10, fig_dim/10, fig_dim/10]
        min_pval = np.min([r2_pval.min(), pc_pval.min(), 1e-3])
        pval_norm = mpl.colors.LogNorm(vmin=min_pval, vmax=1)
        pval_cmap = 'viridis_r' #'magma_r' #'bone_r'
        r2_max_diff = np.nanmax(np.abs(r2_diff))
        r2_diff_norm = mpl.colors.Normalize(vmin=0, vmax=r2_max_diff)
        pc_max_diff = np.nanmax(np.abs(pc_diff))
        pc_diff_norm = mpl.colors.Normalize(vmin=0, vmax=pc_max_diff)
        mrksize_range = (3, 60) # min size, range sizes

        plt.close('all')
        fig, ax = plt.subplots(1, 5, width_ratios=width_ratios, figsize=(sum(width_ratios),fig_dim), layout='constrained', num=1)
        im = ax[0].imshow(r2_pval, origin='upper', cmap=pval_cmap, norm=pval_norm)
        ax[1].imshow(pc_pval, origin='upper', cmap=pval_cmap, norm=pval_norm)

        pc_color = (pc_diff >= 0)*1.0
        r2_color = (r2_diff >= 0)*1.0
        pc_size = pc_diff_norm(np.abs(pc_diff))*mrksize_range[1] + mrksize_range[0]
        r2_size = r2_diff_norm(np.abs(r2_diff))*mrksize_range[1] + mrksize_range[0]

        y,x = np.mgrid[range(num_ses), range(num_ses)] #range(num_ses-1, -1, -1)]
        
        ax[0].scatter(x, y, s=r2_size, c=r2_color, marker='o', cmap='bwr', vmin=0, vmax=1)
        ax[0].scatter(x[zz==1], y[zz==1], s=r2_size[zz==1], c='k', marker='o')
        ax[0].scatter(x, y, s=(r2_pval<0.05)*mrksize_range[0], c='w', marker='o')
        ax[0].set_title('R^2')
        ax[0].set_xlabel('Target Session')
        ax[0].set_ylabel('Source Session')
        
        ax[1].scatter(x, y, s=pc_size, c=pc_color, marker='o', cmap='bwr', vmin=0, vmax=1)
        ax[1].scatter(x[zz==1], y[zz==1], s=pc_size[zz==1], c='k', marker='o')
        ax[1].scatter(x, y, s=(pc_pval<0.05)*mrksize_range[0], c='w', marker='o')
        ax[1].set_xlabel('Target Session')
        ax[1].set_title('Correlation (PC)')
        
        fig.colorbar(im, orientation='vertical', cax=ax[2], ticklocation='left', drawedges=False)
        ax[2].set_title('pval')
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        
        num_marker_legend = 7
        x_leg = np.zeros(num_marker_legend)
        y_leg_r2 = np.linspace(-r2_max_diff, r2_max_diff, num_marker_legend)
        y_leg_pc = np.linspace(-pc_max_diff, pc_max_diff, num_marker_legend)
        ax[3].scatter(x_leg, y_leg_r2, s=r2_diff_norm(np.abs(y_leg_r2))*mrksize_range[1] + mrksize_range[0], c=(y_leg_r2 >= 0)*1.0, marker='o', cmap='bwr', vmin=0, vmax=1)
        ax[3].set_xticks([])
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['bottom'].set_visible(False)
        ax[3].set_xlabel(None)
        ax[3].set_title('R^2')
        
        ax[4].scatter(x_leg, y_leg_pc, s=pc_diff_norm(np.abs(y_leg_pc))*mrksize_range[1] + mrksize_range[0], c=(y_leg_pc >= 0)*1.0, marker='o', cmap='bwr', vmin=0, vmax=1)
        ax[4].set_xticks([])
        ax[4].spines['top'].set_visible(False)
        ax[4].spines['right'].set_visible(False)
        ax[4].spines['bottom'].set_visible(False)
        ax[4].set_xlabel(None)
        ax[4].set_title('PC')
        
        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            both_rel_string = 'bothrel' if both_reliable else 'sourcerel'
            save_name = f"summary_compare_pfplasticity_{reduction}_{both_rel_string}_env{envnum}_ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()

    
    def plot_rel_plasticity(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), maxcutoffs=None, withShow=True, withSave=False):
        idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        target_relmse, target_relcor, target_red, sortby_red = self.measure_rel_plasticity(envnum, idx_ses=idx_ses, cutoffs=cutoffs, maxcutoffs=maxcutoffs)

        # Put fraction reliable in array
        num_ses = len(idx_ses)
        frac_relmse_ctl = np.full((num_ses, num_ses), np.nan)
        frac_relmse_red = np.full((num_ses, num_ses), np.nan)
        frac_relcor_ctl = np.full((num_ses, num_ses), np.nan)
        frac_relcor_red = np.full((num_ses, num_ses), np.nan)
        num_ctl = np.zeros((num_ses, num_ses))
        num_red = np.zeros((num_ses, num_ses))
        
        for ii, (itred, isred) in enumerate(zip(target_red, sortby_red)):
            for jj, (jtred, jsred) in enumerate(zip(itred, isred)):
                idx_red = jtred | jsred # only red if red in target and sortby sessions
                num_ctl[ii,jj] = np.sum(~idx_red)
                num_red[ii,jj] = np.sum(idx_red)

                c_relmse = target_relmse[ii][jj]
                c_relcor = target_relcor[ii][jj]
                frac_relmse_ctl[ii,jj] = np.sum(c_relmse[~idx_red]) / len(c_relmse[~idx_red])
                frac_relcor_ctl[ii,jj] = np.sum(c_relcor[~idx_red]) / len(c_relcor[~idx_red])
                if np.any(idx_red):
                    frac_relmse_red[ii,jj] = np.sum(c_relmse[idx_red]) / len(c_relmse[idx_red])
                    frac_relcor_red[ii,jj] = np.sum(c_relcor[idx_red]) / len(c_relcor[idx_red])
                   
        # 1 if nan -- this means there was no data (usually)
        zz = np.isnan(frac_relmse_red)*1.0
        
        # plot parameters
        labelSize = 18
        nsize = 7
        fig_dim = 4
        width_ratios = [fig_dim, fig_dim, fig_dim/10, fig_dim, fig_dim/10]

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = 'viridis_r'

        cmp_bound_rel = np.max(np.abs(frac_relmse_red - frac_relmse_ctl))
        cmp_bound_cor = np.max(np.abs(frac_relcor_red - frac_relcor_ctl))
        norm_cmp_rel = mpl.colors.Normalize(vmin=-cmp_bound_rel, vmax=cmp_bound_rel)
        norm_cmp_cor = mpl.colors.Normalize(vmin=-cmp_bound_cor, vmax=cmp_bound_cor)
        cmap_cmp = 'bwr'

        plt.close('all')
        fig, ax = plt.subplots(2, 5, width_ratios=width_ratios, figsize=(sum(width_ratios),2*fig_dim), layout='constrained', num=1)
        im_r2 = ax[0,0].imshow(frac_relmse_ctl, origin='upper', cmap=cmap, norm=norm)
        ax[0,1].imshow(frac_relmse_red, origin='upper', cmap=cmap, norm=norm)
        im_r2_cmp = ax[0,3].imshow(frac_relmse_red - frac_relmse_ctl, origin='upper', cmap=cmap_cmp, norm=norm_cmp_rel)

        im_pc = ax[1,0].imshow(frac_relcor_ctl, origin='upper', cmap=cmap, norm=norm)
        ax[1,1].imshow(frac_relcor_red, origin='upper', cmap=cmap, norm=norm)
        im_pc_cmp = ax[1,3].imshow(frac_relcor_red - frac_relcor_ctl, origin='upper', cmap=cmap_cmp, norm=norm_cmp_cor)
        
        # label no data with x's
        y,x = np.mgrid[range(num_ses), range(num_ses)] 
        xs_for_zeros = False
        if xs_for_zeros:
            ax[0,0].scatter(x[zz==1], y[zz==1], s=10, c='k', marker='x')
            ax[0,1].scatter(x[zz==1], y[zz==1], s=10, c='k', marker='x')
            ax[1,0].scatter(x[zz==1], y[zz==1], s=10, c='k', marker='x')
            ax[1,1].scatter(x[zz==1], y[zz==1], s=10, c='k', marker='x')
        else:
            for ii in range(num_ses):
                for jj in range(num_ses):
                    for cax in (ax[0,0], ax[1,0]):
                        cax.text(ii, jj, f"{int(num_ctl[ii,jj])}", horizontalalignment='center', verticalalignment='center', fontsize=nsize)
                    for cax in (ax[0,1], ax[1,1]):
                        cax.text(ii, jj, f"{int(num_red[ii,jj])}", horizontalalignment='center', verticalalignment='center', fontsize=nsize)
        ax[0,0].set_title('frac(R^2) - CTL')
        ax[0,1].set_title('frac(R^2) - RED')
        ax[0,3].set_title('diff(R^2) - (r-c)')
        ax[0,0].set_ylabel('Source Session')
        ax[1,0].set_ylabel('Source Session')
        ax[1,0].set_xlabel('Target Session')
        ax[1,1].set_xlabel('Target Session')
        ax[1,3].set_xlabel('Target Session')
        ax[1,0].set_title('frac(PC) - CTL')
        ax[1,1].set_title('frac(PC) - RED')
        ax[1,3].set_title('diff(PC) - (r-c)')
        
        fig.colorbar(im_r2, orientation='vertical', cax=ax[0,2], ticklocation='right', drawedges=False)
        ax[0,2].set_ylabel('R^2')
        fig.colorbar(im_pc, orientation='vertical', cax=ax[1,2], ticklocation='right', drawedges=False)
        ax[1,2].set_ylabel('PC')

        fig.colorbar(im_r2_cmp, orientation='vertical', cax=ax[0,4], ticklocation='right', drawedges=False)
        ax[0,4].set_ylabel('$\Delta$ R^2')
        fig.colorbar(im_pc_cmp, orientation='vertical', cax=ax[1,4], ticklocation='right', drawedges=False)
        ax[1,4].set_ylabel('$\Delta$ PC')
        
        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            save_name = f"summary_rel_plasticity_env{envnum}_ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()


    def plot_pfplasticity(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), both_reliable=False, reduction='mean', min_mse=-8, withShow=True, withSave=False):
        idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        r2, pc, r2_stat, pc_stat, target_red, sortby_red = self.measure_pfplasticity(envnum, idx_ses=idx_ses, cutoffs=cutoffs, both_reliable=both_reliable)
        num_ses = len(idx_ses)

        # put the difference and pval in a numpy array
        r2_avg_ctl = np.full((num_ses, num_ses), np.nan)
        pc_avg_ctl = np.full((num_ses, num_ses), np.nan)
        r2_avg_red = np.full((num_ses, num_ses), np.nan)
        pc_avg_red = np.full((num_ses, num_ses), np.nan)
        for ii, (itred, isred) in enumerate(zip(target_red, sortby_red)):
            for jj, (jtred, jsred) in enumerate(zip(itred, isred)):
                idx_red = jtred & jsred # only red if red in target and sortby sessions

                # copy r2 and pc (especially r2 since we're updating values)
                c_r2 = copy(r2[ii][jj])
                c_r2[c_r2 < min_mse] = min_mse
                c_r2[np.isnan(c_r2)] = min_mse
                c_pc = copy(pc[ii][jj])
                if np.any(idx_red):
                    if reduction=='mean':
                        r2_avg_ctl[ii, jj] = np.mean(c_r2[~idx_red])
                        pc_avg_ctl[ii, jj] = np.mean(c_pc[~idx_red])
                        r2_avg_red[ii, jj] = np.mean(c_r2[idx_red])
                        pc_avg_red[ii, jj] = np.mean(c_pc[idx_red])
                    elif reduction=='median':
                        r2_avg_ctl[ii, jj] = np.median(c_r2[~idx_red])
                        pc_avg_ctl[ii, jj] = np.median(c_pc[~idx_red])
                        r2_avg_red[ii, jj] = np.median(c_r2[idx_red])
                        pc_avg_red[ii, jj] = np.median(c_pc[idx_red])
                    else:
                        raise ValueError(f"reduction method not recognized ({reduction}), only 'mean' or 'median' are coded!")
                   
        # 1 if nan -- this means there was no data (usually)
        zz = np.isnan(r2_avg_ctl)*1.0
        
        # plot parameters
        labelSize = 18
        lw = 1.5
        fig_dim = 4
        width_ratios = [fig_dim, fig_dim, fig_dim/10]

        r2_min = np.nanmin(np.concatenate((r2_avg_ctl, r2_avg_red)))
        pc_min = np.nanmin(np.concatenate((pc_avg_ctl, pc_avg_red)))
        r2_norm = mpl.colors.Normalize(vmin=r2_min, vmax=1)
        pc_norm = mpl.colors.Normalize(vmin=pc_min, vmax=1)
        cmap = 'viridis_r' #'magma_r' #'bone_r'
        
        plt.close('all')
        fig, ax = plt.subplots(2, 3, width_ratios=width_ratios, figsize=(sum(width_ratios),2*fig_dim), layout='constrained', num=1)
        im_r2 = ax[0,0].imshow(r2_avg_ctl, origin='upper', cmap=cmap, norm=r2_norm)
        ax[0,1].imshow(r2_avg_red, origin='upper', cmap=cmap, norm=r2_norm)
        im_pc = ax[1,0].imshow(pc_avg_ctl, origin='upper', cmap=cmap, norm=pc_norm)
        ax[1,1].imshow(pc_avg_red, origin='upper', cmap=cmap, norm=pc_norm)

        # label no data with x's
        y,x = np.mgrid[range(num_ses), range(num_ses)] 
        ax[0,0].scatter(x[zz==1], y[zz==1], s=10, c='k', marker='x')
        ax[0,0].set_title('R^2 - CTL')
        ax[0,1].set_title('R^2 - RED')
        ax[0,0].set_ylabel('Source Session')
        ax[1,0].set_ylabel('Source Session')
        ax[1,0].set_xlabel('Target Session')
        ax[1,1].set_xlabel('Target Session')
        ax[1,0].set_title('PC - CTL')
        ax[1,1].set_title('PC - RED')
        
        fig.colorbar(im_r2, orientation='vertical', cax=ax[0,2], ticklocation='right', drawedges=False)
        ax[0,2].set_ylabel('R^2')
        fig.colorbar(im_pc, orientation='vertical', cax=ax[1,2], ticklocation='right', drawedges=False)
        ax[1,2].set_ylabel('PC')
        
        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            both_rel_string = 'bothrel' if both_reliable else 'sourcerel'
            save_name = f"summary_pfplasticity_{reduction}_{both_rel_string}_env{envnum}_{reduction}_ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()


    def plot_pfreliability(self, envnum, idx_ses=None, cutoffs=None, method='max', reduction='mean', min_mse=-8, withShow=True, withSave=False):
        idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        num_ses = len(idx_ses)
        target_relmse, target_relcor, target_snake, target_red, sortby_red = self.measure_pfreliability(envnum, idx_ses=idx_ses, cutoffs=cutoffs, method='max')
        
        # put the average reliability in a numpy array
        mse_ctl = np.full((num_ses, num_ses), np.nan)
        mse_red = np.full((num_ses, num_ses), np.nan)
        cor_ctl = np.full((num_ses, num_ses), np.nan)
        cor_red = np.full((num_ses, num_ses), np.nan)
        for ii, (itred, isred) in enumerate(zip(target_red, sortby_red)):
            for jj, (jtred, jsred) in enumerate(zip(itred, isred)):
                idx_red = jtred & jsred # only red if red in target and sortby sessions

                # copy r2 and pc (especially r2 since we're updating values)
                c_mse = copy(target_relmse[ii][jj])
                c_mse[c_mse < min_mse] = min_mse # prevent mse from being too low (it goes from 1 to -inf)
                c_mse[np.isnan(c_mse)] = min_mse # prevent mse from being nan -- this happens if the "denominator" is 0
                c_cor = copy(target_relcor[ii][jj])
                if np.any(idx_red):
                    if reduction=='mean':
                        mse_ctl[ii, jj] = np.mean(c_mse[~idx_red])
                        mse_red[ii, jj] = np.mean(c_mse[idx_red])
                        cor_ctl[ii, jj] = np.mean(c_cor[~idx_red])
                        cor_red[ii, jj] = np.mean(c_cor[idx_red])
                    elif reduction=='median':
                        mse_ctl[ii, jj] = np.median(c_mse[~idx_red])
                        mse_red[ii, jj] = np.median(c_mse[idx_red])
                        cor_ctl[ii, jj] = np.median(c_cor[~idx_red])
                        cor_red[ii, jj] = np.median(c_cor[idx_red])
                    else:
                        raise ValueError(f"reduction method not recognized ({reduction}), only 'mean' or 'median' are coded!")
                   
        # 1 if nan -- this means there was no data (usually)
        zz = np.isnan(mse_ctl)*1.0
        
        # plot parameters
        labelSize = 18
        lw = 1.5
        fig_dim = 4
        width_ratios = [fig_dim, fig_dim, fig_dim/10]

        mse_min = np.nanmin(np.concatenate((mse_ctl, mse_red)))
        cor_min = np.nanmin(np.concatenate((cor_ctl, cor_red)))
        mse_norm = mpl.colors.Normalize(vmin=mse_min, vmax=1)
        cor_norm = mpl.colors.Normalize(vmin=cor_min, vmax=1)
        cmap = 'viridis_r' #'magma_r' #'bone_r'
        
        plt.close('all')
        fig, ax = plt.subplots(2, 3, width_ratios=width_ratios, figsize=(sum(width_ratios),2*fig_dim), layout='constrained', num=1)
        im_r2 = ax[0,0].imshow(mse_ctl, origin='upper', cmap=cmap, norm=mse_norm)
        ax[0,1].imshow(mse_red, origin='upper', cmap=cmap, norm=mse_norm)
        im_pc = ax[1,0].imshow(cor_ctl, origin='upper', cmap=cmap, norm=cor_norm)
        ax[1,1].imshow(cor_red, origin='upper', cmap=cmap, norm=cor_norm)

        # label no data with x's
        y,x = np.mgrid[range(num_ses), range(num_ses)] 
        ax[0,0].scatter(x[zz==1], y[zz==1], s=10, c='k', marker='x')
        ax[0,0].set_title('R^2 - CTL')
        ax[0,1].set_title('R^2 - RED')
        ax[0,0].set_ylabel('Source Session')
        ax[1,0].set_ylabel('Source Session')
        ax[1,0].set_xlabel('Target Session')
        ax[1,1].set_xlabel('Target Session')
        ax[1,0].set_title('PC - CTL')
        ax[1,1].set_title('PC - RED')
        
        fig.colorbar(im_r2, orientation='vertical', cax=ax[0,2], ticklocation='right', drawedges=False)
        ax[0,2].set_ylabel('R^2')
        fig.colorbar(im_pc, orientation='vertical', cax=ax[1,2], ticklocation='right', drawedges=False)
        ax[1,2].set_ylabel('PC')
        
        if withSave: 
            sesidx = '_'.join([str(i) for i in idx_ses])
            save_name = f"summary_pfreliability_env{envnum}_{reduction}_ses_{sesidx}"
            self.saveFigure(fig.number, self.track.mouse_name, save_name)
            
        # Show figure if requested
        plt.show() if withShow else plt.close()
    

    def perform_roicat_comparisons(self, envnum, idx_ses=None, cutoffs=(0.5, 0.8), both_reliable=False):
        """method for determining how well ROICaT tracking similarity works for pairs of cells compared to their place field"""
        # start by prepping the meta data (e.g. which sessions to use)
        store_idx_ses = self.idx_ses_with_env(envnum) if idx_ses is None else idx_ses
        store_num_ses = len(store_idx_ses)

        # load all relevant data now for better user reporting
        self.load_pcss_data(idx_ses=store_idx_ses)

        # load paired snakes from matched ROIs
        target_snake = []
        sortby_snake = []
        target_red = []
        sortby_red = []
        for sortby in store_idx_ses:
            c_target_snake = []
            c_sortby_snake = []
            c_target_ired = []
            c_sortby_ired = []
            for target in store_idx_ses:
                cdata, cired, _, _ = self.make_paired_snake(envnum, target, sortby, cutoffs=cutoffs, both_reliable=both_reliable)
                c_target_snake.append(cdata[0])
                c_sortby_snake.append(cdata[1])
                c_target_ired.append(cired[0])
                c_sortby_ired.append(cired[1])
            target_snake.append(c_target_snake)
            sortby_snake.append(c_sortby_snake)
            target_red.append(c_target_ired)
            sortby_red.append(c_sortby_ired)

        # I need a grid of histograms comparing the snakes across sessions
        r2 = []
        pc = []
        r2_stat = []
        pc_stat = []
        for isort, (snakes_target, snakes_sortby, red_target, red_sortby) in enumerate(zip(target_snake, sortby_snake, target_red, sortby_red)):
            c_r2 = []
            c_pc = []
            c_r2_stat = []
            c_pc_stat = []
            for itarget, (snake_target, snake_sortby, r_target, r_sortby) in enumerate(zip(snakes_target, snakes_sortby, red_target, red_sortby)):
                assert snake_target.shape[0] == snake_sortby.shape[0], "oops"
                # red is only if red in both target and snake session
                c_idx_red = r_target & r_sortby
                # get R-squared
                dv_target = np.max(snake_target, axis=1, keepdims=True)
                dv_sortby = np.max(snake_sortby, axis=1, keepdims=True)
                st = snake_target / (dv_target + 1*(dv_target==0))
                ss = snake_sortby / (dv_sortby + 1*(dv_sortby==0))
                cc_r2 = helpers.vectorRSquared(ss, st, axis=1)
                cc_r2[cc_r2==-np.inf] = np.nan
                # also get correlation
                cc_pc = helpers.vectorCorrelation(ss, st, axis=1)
                # then add results
                c_r2.append(cc_r2)
                c_pc.append(cc_pc)
                # now do stats (just ranksum) 
                c_r2_stat.append(sp.stats.ranksums(cc_r2[~c_idx_red], cc_r2[c_idx_red]))
                c_pc_stat.append(sp.stats.ranksums(cc_pc[~c_idx_red], cc_pc[c_idx_red]))
            # keep all results
            r2.append(c_r2)
            pc.append(c_pc)
            r2_stat.append(c_r2_stat)
            pc_stat.append(c_pc_stat)

        self.idx_ses, self.num_ses = store_idx_ses, store_num_ses
        return r2, pc, r2_stat, pc_stat, target_red, sortby_red

