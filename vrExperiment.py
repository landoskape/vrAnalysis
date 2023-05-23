# inclusions
import numpy as np
import scipy as sp
import numba as nb
import scipy.io as scio
import time
from tqdm import tqdm
from pathlib import Path 
import json
from numpyencoder import NumpyEncoder
import vrFunctions as vrf
import basicFunctions as bf


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------- vrExperiment Management Object -----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class vrExperiment:
    """
    The vrExperiment object is a postprocessing object used to load and buffer data, and store highly used methods for analyzing data in vr experiments.
    It accepts three required arguments as strings - the mouseName, dateString, and session, which it uses to identify the directory of the data (w/ Alyx conventions and a method specified local data path)
    vrExperiment is meant to be called after vrExperimentRegistration is already run. From the output of vrExperimentRegistration, vrExperiment can quickly and efficiently manage a recording session.
    """
    def __init__(self,mouseName,dateString,session):
        # initialize vrExperiment object -- this only works if vrExperimentRegistration has already been run and saved on the same session --
        self.mouseName = mouseName
        self.dateString = dateString
        self.session = session
        
        # Load vrExperimentRegistration data
        self.loadRegisteredExperiment()
        # initialize this upon creation for efficient loading of onedata
        self.loadBuffer = {} 
        
    
    # ------------------------------------------------------- basic meta functions for vrExperiment -----------------------------------------------------------
    # path methods
    def dataPath(self):
        # dataPath returns a path object referring to the directory contain data structured in the Alyx format (./mouseName/dateString/session/*data*)
        # for best usage, change this return line to whatever is appropriate on your machine and keep data in the Alyx format :)
        return Path('C:/Users/andrew/Documents/localData')
    
    def sessionPath(self): return self.dataPath()/self.mouseName/self.dateString/self.session
    def onePath(self): return self.sessionPath()/'oneData'
    def rawDataPath(self): return self.sessionPath()/'rawDataPath'
    def suite2pPath(self): return self.sessionPath()/'suite2p'
    
    def printSavedOne(self):
        return [name.stem for name in self.pathDictionary['onePath'].glob('*.npy')]
    
    def sessionPrint(self): 
        # useful function for generating string of session name for useful feedback to user 
        return f"{self.mouseName}/{self.dateString}/{self.session}"   
    
    def registerValue(self,name,value):
        self.value[name]=value
        
    def loadRegisteredExperiment(self):
        self.opts = json.load(open(self.sessionPath()/'vrExperimentOptions.json'))
        self.preprocessing = json.load(open(self.sessionPath()/'vrExperimentPreprocessing.json'))
        self.value = json.load(open(self.sessionPath()/'vrExperimentValues.json'))    
    
    # ----------------------------------------------------------------- one data handling --------------------------------------------------------------------
    def saveone(self,var,*names):
        fileName = self.oneFilename(*names)
        self.loadBuffer[fileName] = var
        np.save(self.onePath() / fileName, var)

    def loadone(self,*names):
        fileName = self.oneFilename(*names)
        if fileName in self.loadBuffer.keys():
            return self.loadBuffer[fileName]
        else:
            oneVar = np.load(self.onePath() / fileName)
            self.loadBuffer[fileName] = oneVar
            return oneVar

    def oneFilename(self,*names):
        return '.'.join([name.lower() for name in names])+'.npy'
    
    def clearBuffer(self,*names):
        if len(names)==0:
            self.loadBuffer = {}
        else:
            for name in names:
                if name in self.loadBuffer.keys(): del self.loadBuffer[name]
        
    # ------------------------------------------ special loading functions for data not stored directly in one format ------------------------------------------
    def loadfcorr(self,meanAdjusted=True):
        F = self.loadone('neuron.frame.F')
        Fneu = self.loadone('neuron.frame.Fneu')
        meanFneu = np.mean(Fneu,axis=1,keepdims=True) if meanAdjusted else np.zeros((np.sum(self.value['roiPerPlane']),1))
        return  F - self.neuropilCoefficient*(Fneu - meanFneu)
    
    
    # ---------------------------------------- postprocessing functions for translating behavior to imaging time frame -----------------------------------------------------
    def getFrameBehavior(self):
        trialStartSample = self.loadone('trial.startBehaveSample')
        behavePosition = self.loadone('behave.position')
        behaveTrialIdx = self.getBehaveTrialIdx(trialStartSample)

        frameTimeStamps = self.loadone('timeline.timestamps')[self.loadone('frame.timelinesample')]
        samplingRate = 1/np.median(np.diff(frameTimeStamps))
        
        # behave timestamps has higher temporal resolution than frame timestamps, so we need to average over behavioral frames
        idxBehaveToFrame = self.loadone('behave.idx2framesample')
        distBehaveToFrame = self.loadone('behave.dist2framesample')

        framePosition = np.zeros(self.value['numFrames'])
        frameTrialIdx = np.zeros(self.value['numFrames'])
        count = np.zeros(self.value['numFrames'])
        vrf.behaveToFrame(behavePosition,behaveTrialIdx,idxBehaveToFrame,distBehaveToFrame,1/2/samplingRate,framePosition,frameTrialIdx,count)
        framePosition[count==0]=np.nan
        frameTrialIdx[count==0]=np.nan
        assert np.min(frameTrialIdx[count>0])==0 and np.max(frameTrialIdx[count>0])==self.value['numTrials']-1, "frameTrialIdx doesn't have correct number of trials"
        
        # Make sure trial indices are integers -- otherwise (in most cases this means multiple behavioral trials were matched with same neural frame)
        assert np.all([i.is_integer() for i in frameTrialIdx[count>0]]), "some neural frames were associated with multiple trials, non integers found"
        
        # Occasionally some neural frames do not map onto any best behavioral frame (usually for random slow behavioral sample) -- interpolate those samples 
        for trial in range(self.value['numTrials']):
            cTrialIdx = np.where(frameTrialIdx==trial)[0]
            trialSlice = slice(cTrialIdx[0], cTrialIdx[-1]+1)
            withinTrialNan = np.isnan(frameTrialIdx[trialSlice])
            if np.any(withinTrialNan):
                trialTimeStamps = frameTimeStamps[trialSlice]
                trialPosition = framePosition[trialSlice]
                frameTrialIdx[trialSlice]=trial
                trialPosition[withinTrialNan]=np.interp(trialTimeStamps[withinTrialNan], trialTimeStamps[~withinTrialNan], trialPosition[~withinTrialNan])
                framePosition[trialSlice] = trialPosition
        
        # Once position is fully interpolated on each trial, compute speed
        frameSpeed = np.append(np.diff(framePosition) * samplingRate, np.nan)
        
        return frameTrialIdx, framePosition, frameSpeed
    
    
    # -------------------- post-processing of behavioral data in one format -----------------
    def getBehaveTrialIdx(self, trialStartFrame):
        # produces a 1-d numpy array where each element indicates the trial index of the behavioral sample onedata arrays
        nspt = np.array([*np.diff(trialStartFrame), self.value['numBehaveTimestamps']-trialStartFrame[-1]]) # num samples per trial
        return np.concatenate([tidx*np.ones(ns) for (tidx,ns) in enumerate(nspt)]).astype(np.uint64)

    def groupBehaveByTrial(self, data, trialStartFrame):
        # returns a list containing trial-separated behavioral data 
        trialIndex = self.getBehaveTrialIdx(trialStartFrame)
        return [data[trialIndex==tidx] for tidx in range(len(trialStartFrame))]

    

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------- vrExperiment Registration Object ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class vrExperimentRegistration(vrExperiment):
    """
    The vrExperimentRegistration object is a preprocessing object used to preprocess the data, save oneformats, and save vrExperiment parameters.
    It accepts three required arguments as strings - the mouseName, dateString, and session, which it uses to identify the directory of the data (w/ Alyx conventions and a method specified local data path)
    It also accepts optional keyword arguments that define how to preprocess the data and defines other important parameters.
    
    Standard usage is:
    vrReg = vrExperimentRegistration(mouseName, dateString, session, **kwarg)
    vrReg.doPreprocessing()
    vrReg.saveParams()
    """
    def __init__(self,mouseName,dateString,session,**userOpts):
        opts = {} # Options for data management
        # Preprocessing options -- these define what was performed in each experiment and what to preprocess --
        opts['vrBehaviorVersion'] = 1 # 1==standard behavioral output (will make conditional loading systems for alternative versions...)
        opts['facecam'] = False # whether or not face video was performed on this session (note: only set to True when DLC has already been run!)
        opts['imaging'] = True # whether or not imaging was performed on this session (note: only set to True when suite2p has already been run!)
        opts['oasis'] = True # whether or not to rerun oasis on calcium signals (note: only used if imaging is run)
        opts['moveRawData'] = False # whether or not to move raw data files to 'rawData'
        # Imaging options -- these options are standard values for imaging, tau & fs directly affect preprocessing when OASIS is turned on (and deconvolution is recomputed)
        opts['neuropilCoefficient'] = 0.7 # for manual neuropil estimation
        opts['isCellThreshold'] = -1 # only process ROIs exceeding probCell > isCellThreshold
        opts['tau'] = 1.5 # GCaMP time constant
        opts['fs'] = 6 # sampling rate (per volume if multiplane)
        # Other options--
        # --
        assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts) # Update default opts with user requests
        
        # -- initialize vrExperiment parameters --
        self.mouseName = mouseName
        self.dateString = dateString
        self.session = session
        self.opts = opts
        
        # initialize dictionaries to be stored as JSONs and loaded by vrExperiment
        self.preprocessing = [] # initialize this for storing which preprocessing stages were performed
        self.loadBuffer = {} # initialize this upon creation for efficient loading throughout preprocessing
        self.value = {} # initialize this dictionary to save important values (e.g. number of trials)
        
        self.tlFile = self.loadTimelineStructure()
        self.vrFile = self.loadBehaviorStructure()
        
        self.saveParams()
    
    def doPreprocessing(self):
        self.processTimeline()
        self.processBehavior()
        self.processImaging()
        self.processFacecam()
        self.processRedCells()
   
    def saveParams(self):
        # saveParams saves the parameters generated by vrExperimentRegistration, primarily being instructions for vrExperiment to load the data.
        with open(self.sessionPath()/'vrExperimentOptions.json','w') as file:
            json.dump(self.opts, file, ensure_ascii=False)
        with open(self.sessionPath()/'vrExperimentPreprocessing.json','w') as file:
            json.dump(self.preprocessing, file, ensure_ascii=False)
        with open(self.sessionPath()/'vrExperimentValues.json','w') as file:
            json.dump(self.value, file, ensure_ascii=False, cls=NumpyEncoder)
            
    # --------------------------------------------------------------- preprocessing methods ------------------------------------------------------------
    def processTimeline(self):
        # get time stamps, photodiode, trial start and end times, room position, lick times, trial idx, visual data visible
        mpepStartTimes = []
        for (mt,me) in zip(self.tlFile['mpepUDPTimes'],self.tlFile['mpepUDPEvents']):
            if isinstance(me,str):
                if 'TrialStart' in me: mpepStartTimes.append(mt)
        
        mpepStartTimes = np.array(mpepStartTimes)
        timestamps = self.getTimelineVar('timestamps') # load timestamps
        
        # Get rotary position -- (load timeline measurement of rotary encoder, which is a circular position counter, use vrExperiment function to convert to a running measurement of position)
        rotaryEncoder = self.getTimelineVar('rotaryEncoder')
        rotaryPosition = self.convertRotaryEncoderToPosition(rotaryEncoder, self.vrFile['rigInfo'])
        
        # Get Licks (uses an edge counter)
        lickDetector = self.getTimelineVar('lickDetector') # load lick detector copy
        lickSamples = np.where(bf.diffsame(lickDetector)==1)[0].astype(np.uint64) # timeline samples of lick times
        
        # Get Reward Commands (measures voltage of output -- assume it's either low or high)
        rewardCommand = self.getTimelineVar('rewardCommand') # load reward command signal
        rewardCommand = np.round(rewardCommand/np.max(rewardCommand))
        rewardSamples = np.where(bf.diffsame(rewardCommand)>0.5)[0].astype(np.uint64) # timeline samples when reward was delivered
        
        # Now process photodiode signal
        photodiode = self.getTimelineVar('photoDiode') # load lick detector copy
        
        # Remove any slow trends
        pdDetrend = sp.signal.detrend(photodiode)
        pdDetrend = (pdDetrend - pdDetrend.min())/pdDetrend.ptp()
        
        # median filter and take smooth derivative
        hfpd = 10
        refreshRate = 30 # hz
        refreshSamples = int(1./refreshRate/np.mean(np.diff(timestamps)))
        pdMedFilt = sp.ndimage.median_filter(pdDetrend,size=refreshSamples)
        pdDerivative,pdIndex = bf.fivePointDer(pdMedFilt,hfpd,returnIndex=True)
        pdDerivative = sp.stats.zscore(pdDerivative)
        pdDerTime = timestamps[pdIndex]
        
        # find upward and downward peaks, not perfect but in practice close enough
        locUp = sp.signal.find_peaks(pdDerivative,height=1,distance=refreshSamples/2)
        locDn = sp.signal.find_peaks(-pdDerivative,height=1,distance=refreshSamples/2)
        flipTimes = np.concatenate((pdDerTime[locUp[0]],pdDerTime[locDn[0]]))
        flipValue = np.concatenate((np.ones(len(locUp[0])),np.zeros(len(locDn[0]))))
        flipSortIdx = np.argsort(flipTimes)
        flipTimes = flipTimes[flipSortIdx]
        flipValue = flipValue[flipSortIdx]
        
        # Naive Method (just look for flips before and after trialstart/trialend mpep message:
        # A sophisticated message uses the time of the photodiode ramps, but those are really just for safety and rare manual curation...
        firstFlipIndex = np.array([np.where(flipTimes >= mpepStart)[0][0] for mpepStart in mpepStartTimes])
        startTrialIndex = bf.nearestpoint(flipTimes[firstFlipIndex], timestamps)[0] # returns frame index of first photodiode flip in each trial
        
        # Check that first flip is always down -- all of the vrControl code prepares trials in this way
        assert np.all(flipValue[firstFlipIndex]==0), f"In session {self.sessionPrint()}, first flips in trial are not all down!!"
        
        # Check shapes of timeline arrays
        assert timestamps.ndim==1, "timelineTimestamps is not a 1-d array!"
        assert timestamps.shape == rotaryPosition.shape, "timeline timestamps and rotary position arrays do not have the same shape!"
            
        # Save timeline oneData
        self.saveone(timestamps, 'timeline.timestamps')
        self.saveone(rotaryPosition, 'timeline.rotaryposition')
        self.saveone(lickSamples, 'lick.timelinesample')
        self.saveone(rewardSamples, 'reward.timelinesample')
        self.saveone(timestamps[startTrialIndex], 'trial.starttimes')
        self.preprocessing.append('timeline')
                
    
    def processBehavior(self):
        if self.opts['vrBehaviorVersion'] != 1: raise ValueError("Have not coded alternative vrBehavior outputs... ")
        
        expInfo = self.vrFile['expInfo']
        trialInfo = self.vrFile['trialInfo']
        self.registerValue('numTrials',np.sum(trialInfo.trialIdx>0))
        print("Self.value['numTrials'] set by trialInfo.trialIdx>0, but this might not be right. There might be smarter ways to determine which trials are 'good' trials...")
        
        # trialInfo contains sparse matrices of size (maxTrials, maxSamples), where numTrials<maxTrials and numSamples<maxSamples
        nzindex = self.createIndex(self.convertDense(trialInfo.time))
        timeStamps = self.getVRData(self.convertDense(trialInfo.time),nzindex)
        roomPosition = self.getVRData(self.convertDense(trialInfo.roomPosition),nzindex)
        
        # oneData with behave prefix is a (numBehavioralSamples, ) shaped array conveying information about the state of VR  
        numTimeStamps = np.array([len(t) for t in timeStamps]) # list of number of behavioral timestamps in each trial 
        behaveTimeStamps = np.concatenate(timeStamps) # time stamp associated with each behavioral sample
        behavePosition = np.concatenate(roomPosition) # virtual position associated with each behavioral sample
        self.registerValue('numBehaveTimestamps', len(behaveTimeStamps))
        
        # Check shapes and sizes
        assert behaveTimeStamps.ndim==1, "behaveTimeStamps is not a 1-d array!"
        assert behaveTimeStamps.shape == behavePosition.shape, "behave oneData arrays do not have the same shape!"

        # oneData with trial prefix is a (numTrials,) shaped array conveying information about the state on each trial
        trialStartFrame = np.array([0,*np.cumsum(numTimeStamps)[:-1]]).astype(np.int64)
        trialEnvironmentIndex = self.convertDense(trialInfo.vrEnvIdx).astype(np.int16) if 'vrEnvIdx' in trialInfo._fieldnames else -1*np.ones(self.value['numTrials'],dtype=np.int16)
        trialRoomLength = expInfo.roomLength[:self.value['numTrials']]
        trialMovementGain = expInfo.mvmtGain[:self.value['numTrials']]
        trialRewardPosition = self.convertDense(trialInfo.rewardPosition)
        trialRewardTolerance = self.convertDense(trialInfo.rewardTolerance)
        trialRewardAvailability = self.convertDense(trialInfo.rewardAvailable).astype(np.bool_)
        rewardDelivery = self.convertDense(trialInfo.rewardDeliveryFrame).astype(np.int64)-1 # get reward delivery frame (frame within trial) first (will be -1 if no reward delivered)
        # adjust frame count to behave arrays
        trialRewardDelivery = np.array([rewardDelivery + np.sum(numTimeStamps[:trialIdx]) if rewardDelivery>=0 else rewardDelivery for (trialIdx, rewardDelivery) in enumerate(rewardDelivery)]) 
        trialActiveLicking = self.convertDense(trialInfo.activeLicking).astype(np.bool_)
        trialActiveStopping = self.convertDense(trialInfo.activeStopping).astype(np.bool_)
        
        # Check shapes and sizes
        assert trialEnvironmentIndex.ndim==1 and len(trialEnvironmentIndex)==self.value['numTrials'], "trialEnvironmentIndex is not a (numTrials,) shaped array!"
        assert trialStartFrame.shape == trialEnvironmentIndex.shape == trialRoomLength.shape == trialMovementGain.shape == trialRewardPosition.shape == trialRewardTolerance.shape == \
                trialRewardAvailability.shape == trialRewardDelivery.shape == trialActiveLicking.shape == trialActiveStopping.shape, "trial oneData arrays do not have the same shape!"
        
        # oneData with lick prefix is a (numLicks,) shaped array containing information about each lick during VR behavior
        licks = self.getVRData(self.convertDense(trialInfo.lick),nzindex)
        lickFrames = [np.nonzero(licks)[0] for licks in licks]
        lickCounts = np.concatenate([licks[lickFrames] for (licks,lickFrames) in zip(licks,lickFrames)])
        lickTrials = np.concatenate([tidx*np.ones_like(lickFrames) for (tidx,lickFrames) in enumerate(lickFrames)])
        lickFrames = np.concatenate(lickFrames)
        if np.sum(lickCounts)>0:
            lickFramesRepeat = np.concatenate([lf*np.ones(lc,dtype=np.uint8) for (lf,lc) in zip(lickFrames,lickCounts)])
            lickTrialsRepeat = np.concatenate([lt*np.ones(lc,dtype=np.uint8) for (lt,lc) in zip(lickTrials,lickCounts)])
            lickCountsRepeat = np.concatenate([lc*np.ones(lc,dtype=np.uint8) for (lc,lc) in zip(lickCounts,lickCounts)])
            lickBehaveSample = lickFramesRepeat + np.array([np.sum(numTimeStamps[:trialIdx]) for trialIdx in lickTrialsRepeat])

            assert len(lickBehaveSample)==np.sum(lickCounts), "the number of licks counted by vrBehavior is not equal to the length of the lickBehaveSample vector!"
            assert lickBehaveSample.ndim==1, "lickBehaveIndex is not a 1-d array!"
            assert 0<=np.max(lickBehaveSample)<=len(behaveTimeStamps), "lickBehaveSample contains index outside range of possible indices for behaveTimeStamps"
        else:
            # No licks found -- create empty array
            lickBehaveSample = np.array([],dtype=np.uint8)
        
        # Align behavioral timestamp data to timeline -- shift each trials timestamps so that they start at the time of the first photodiode flip (which is reliably detected)
        trialStartOffsets = behaveTimeStamps[trialStartFrame] - self.loadone('trial.starttimes') # get offset
        behaveTimeStamps = np.concatenate([bts - trialStartOffsets[tidx] for (tidx,bts) in enumerate(self.groupBehaveByTrial(behaveTimeStamps,trialStartFrame))])
            
        # Save behave onedata
        self.saveone(behaveTimeStamps, 'behave.timestamps')
        self.saveone(behavePosition,'behave.position')
        
        # Save trial onedata
        self.saveone(trialStartFrame,'trial.startbehavesample')
        self.saveone(trialEnvironmentIndex,'trial.environmentindex')
        self.saveone(trialRoomLength, 'trial.roomlength')
        self.saveone(trialMovementGain, 'trial.movementgain')
        self.saveone(trialRewardPosition, 'trial.rewardposition')
        self.saveone(trialRewardTolerance, 'trial.rewardtolerance')
        self.saveone(trialRewardAvailability, 'trial.rewardavailability')
        self.saveone(trialRewardDelivery, 'trial.rewardbehavesample')
        self.saveone(trialActiveLicking, 'trial.activelicking')
        self.saveone(trialActiveStopping, 'trial.activestopping')
        
        # Save lick onedata
        self.saveone(lickBehaveSample, 'lick.behavesample')
        
        # Confirm that vrBehavior has been processed
        self.preprocessing.append('vrBehavior')
        
    
    def processImaging(self):
        if not self.opts['imaging']:
            print(f"In session {self.sessionPrint()}, imaging setting set to False in opts['imaging']. Skipping image processing.")
            return None
                
        # identifies which planes were processed through suite2p (assume that those are all available planes)
        # identifies which s2p outputs are available from each plane
        self.registerValue('planeNames',[plane.parts[-1] for plane in self.suite2pPath().glob('plane*/')])
        self.registerValue('planeIDs',[int(planeName[5:]) for planeName in self.value['planeNames']])
        npysInPlanes = [[npy.stem for npy in list((self.suite2pPath() / planeName).glob('*.npy'))] for planeName in self.value['planeNames']]
        commonNPYs = list(set.intersection(*[set(npy) for npy in npysInPlanes]))
        unionNPYs = list(set.union(*[set(npy) for npy in npysInPlanes]))
        if set(commonNPYs)<set(unionNPYs):
            print(f"The following npy files are present in some but not all plane folders within session {self.sessionPrint()}: {list(set(unionNPYs) - set(commonNPYs))}")
            print(f"Each plane folder contains the following npy files: {commonNPYs}")
        self.registerValue('available',commonNPYs) # a list of npy files available in each plane folder
        required = ['stat', 'ops', 'F', 'Fneu', 'iscell'] # required variables (anything else is either optional or can be computed independently)
        if not self.opts['oasis']: required.append('spks') # add deconvolved spikes to required variable if we aren't recomputing it here
        for varName in required: assert varName in self.value['available'], f"{self.sessionPrint()} is missing {varName} in at least one suite2p folder!" 
        self.registerValue('roiPerPlane',[iscell.shape[0] for iscell in self.loadS2P('iscell',concatenate=False,checkVariables=False)]) # get number of ROIs in each plane
        self.registerValue('framePerPlane',[F.shape[1] for F in self.loadS2P('F',concatenate=False,checkVariables=False)]) # get number of frames in each plane (might be different!)
        assert np.max(self.value['framePerPlane'])-np.min(self.value['framePerPlane'])<=1, f"The frame count in {self.sessionPrint()} varies by more than 1 frame! ({self.value['framePerPlane']})"
        self.registerValue('numROIs',np.sum(self.value['roiPerPlane'])) # number of ROIs in session
        self.registerValue('numFrames',np.min(self.value['framePerPlane'])) # number of frames to use when retrieving imaging data (might be overwritten to something smaller if timeline handled improperly)
        
        # Get timeline sample corresponding to each imaging volume
        timelineTimestamps = self.loadone('timeline.timestamps')
        changeFrames = np.append(0, np.diff(np.ceil(self.getTimelineVar('neuralFrames')/len(self.value['planeIDs']))))==1
        frameSamples = np.where(changeFrames)[0] # TTLs for each volume (increments by 1 for each plane)
        frame2time = timelineTimestamps[frameSamples] # get timelineTimestamps of each imaging volume

        # Handle mismatch between number of imaging frames saved by scanImage (and propagated through suite2p), and between timeline's measurement of the scanImage frame counter
        if len(frameSamples)!=self.value['numFrames']:
            if len(frameSamples)-1==self.value['numFrames']:
                # If frameSamples had one more frame, just trim it and assume everything is fine. This happens when a new volume was started but not finished, so does not required communication to user.
                frameSamples = frameSamples[:-1]
            elif len(frameSamples)-2==self.value['numFrames']:
                print("frameSamples had 2 more than suite2p output. This happens sometimes. I don't like it. I think it's because scanimage sends a TTL before starting the frame")
                frameSamples = frameSamples[:-2]
            else:
                # If frameSamples has too few frames, it's possible that the scanImage signal to timeline was broken but scanImage still continued normally. 
                numMissing = self.value['numFrames'] - len(frameSamples) # measure number of missing frames
                if numMissing < 0: 
                    # If frameSamples had many more frames, generate an error -- something went wrong that needs manual inspection
                    print(f"In session {self.sessionPrint()}, frameSamples has {len(frameSamples)} elements, but {self.value['numFrames']} frames were reported in suite2p. Cannot resolve.")
                    raise ValueError("Cannot fix mismatches when suite2p data is missing!")
                # It's possible that the scanImage signal to timeline was broken but scanImage still continued normally. 
                if numMissing > 1: print(f"In session {self.sessionPrint()}, more than one frameSamples sample was missing. Consider using tiff timelineTimestamps to reproduce accurately.") 
                print(f"In session {self.sessionPrint()}, frameSamples has {len(frameSamples)} elements, but {self.value['numFrames']} frames were saved by suite2p. Will extend frameSamples using the typical sampling rate and nearestpoint algorithm.")
                # If frame2time difference vector is consistent within 1%, then use mean (which is a little more accurate), otherwise use median
                frame2time = timelineTimestamps[frameSamples]
                medianFramePeriod = np.median(np.diff(frame2time)) # measure median sample period
                consistentFrames = np.all(np.abs(np.log(np.diff(frame2time)/medianFramePeriod)) < np.log(1.01)) # True if all frames take within 1% of median frame period
                if consistentFrames: samplePeriod_f2t = np.mean(np.diff(frame2time))
                else: samplePeriod_f2t = np.median(np.diff(frame2time))
                appendFrames = frame2time[-1] + samplePeriod_f2t*(np.arange(numMissing)+1) # add elements to frame2time, assume sampling rate was perfect 
                frame2time = np.concatenate((frame2time,appendFrames))
                frameSamples = bf.nearestpoint(frame2time, timelineTimestamps)[0]
        
        # average percentage difference between all samples differences and median -- just a useful metric to be saved --
        self.registerValue('samplingDeviationMedPercentError',np.exp(np.mean(np.abs(np.log(np.diff(frame2time)/np.median(np.diff(frame2time)))))))
        self.registerValue('samplingDeviationMaxPercentError',np.exp(np.max(np.abs(np.log(np.diff(frame2time)/np.median(np.diff(frame2time)))))))
        
        # compute translation mapping from behave frames to imaging frames
        idxBehaveToFrame,distBehaveToFrame = bf.nearestpoint(self.loadone('behave.timestamps'), frame2time)

        # recompute deconvolution if requested
        if self.opts['oasis']:
            g = np.exp(-1/self.opts['tau']/self.opts['fs'])
            fcorr = self.getfcorr()
            spks = []
            for fc in tqdm(fcorr):
                spks.append(deconvolve(fc,g=(g,),penalty=1)[1])
            spks = np.stack(spks)
            assert spks.shape == self.loadS2P('spks').shape, f"In session {self.sessionPrint()}, oasis was run and did not produce the same shaped array as suite2p spks..."
        else:
            spks = self.loadS2P('spks')
        
        # save onedata (no assertions needed, loadS2P() handles shape checks and this function already handled any mismatch between frameSamples and suite2p output
        self.saveone(frameSamples, 'frame.timelinesample')
        self.saveone(self.loadS2P('F'), 'neuron.frame.f')
        self.saveone(self.loadS2P('Fneu'), 'neuron.frame.fneu')
        self.saveone(spks, 'neuron.frame.spks')
        if 'redcell' in self.value['available']:
            self.saveone(self.loadS2P('redcell'), 'neuron.redcell')
        self.saveone(self.loadS2P('iscell'), 'neuron.iscell')
        self.saveone(self.getPlaneIdx(), 'neuron.planeidx')
        self.saveone(idxBehaveToFrame.astype(int), 'behave.idx2framesample')
        self.saveone(distBehaveToFrame, 'behave.dist2framesample')
        self.preprocessing.append('imaging')
        
    
    def processFacecam(self):
        print("Facecam preprocessing has not been coded yet!")
    
    
    def processRedCells(self):
        if 'redcell' not in self.value['available']:
            print(f"In session {self.sessionPrint()}, 'redcell' is not an available suite2p output. This code assumes that means there is no data on the red channel")
            return 
        print("Red cell processing has not been coded yet!")
        # Note: the code for this is in the redDeveloper GUI file. I need: 
        # 1. suite2p red cell output
        # 2. dot product between masks and filtered reference image
        # 3. pearson correlation (including surround pixels) between masks and filtered reference image
        # 4. central phase correlation bin (probably using cropped phase correlation estimates for faster code)
        # --
        # Then, I need to produce these maps for many cells in many sessions, label them, and create a GUI for updating them manually...
        # To begin with, I think I'll just use high thresholds and assume they mostly work...
        # -- 
        return None
    
    
    # -------------------------------------- methods for handling timeline data produced by rigbox ------------------------------------------------------------
    def loadTimelineStructure(self): 
        tlFileName = self.sessionPath() / f"{self.dateString}_{self.session}_{self.mouseName}_Timeline.mat" # timeline.mat file name
        return scio.loadmat(tlFileName,simplify_cells=True)['Timeline'] # load matlab structure
    
    def getTimelineVar(self, varName):
        if varName=='timestamps':
            return self.tlFile['rawDAQTimestamps']
        else:
            inputNames = [hwInput['name'] for hwInput in self.tlFile['hw']['inputs']]
            assert varName in inputNames, f"{varName} is not a tlFile in session {self.sessionPrint()}"
            return np.squeeze(self.tlFile['rawDAQData'][:,np.where([inputName==varName for inputName in inputNames])[0]])

    def convertRotaryEncoderToPosition(self, rotaryEncoder, rigInfo):
        # rotary encoder is a counter with a big range that sometimes flips around it's axis
        # first get changes in encoder position, fix any big jumps in value, take the cumulative movement and scale to centimeters 
        rotaryMovement = bf.diffsame(rotaryEncoder)
        idxHighValues = rotaryMovement > 2**(rigInfo['rotaryRange']-1)
        idxLowValues = rotaryMovement < -2**(rigInfo['rotaryRange']-1)
        rotaryMovement[idxHighValues] -= 2**rigInfo['rotaryRange']
        rotaryMovement[idxLowValues] += 2**rigInfo['rotaryRange']
        return rigInfo['rotEncSign']*np.cumsum(rotaryMovement)*(2*np.pi*rigInfo['wheelRadius'])/rigInfo['wheelToVR']

   
    # -------------------------------------- methods for handling vrBehavior data produced by vrControl ------------------------------------------------------------
    def loadBehaviorStructure(self):
        vrFileName = self.sessionPath() / f"{self.dateString}_{self.session}_{self.mouseName}_VRBehavior_trial.mat" # vrBehavior output file name
        vrFile = scio.loadmat(vrFileName,struct_as_record=False,squeeze_me=True)
        if 'rigInfo' not in vrFile.keys():
            print(f"In session: {self.sessionPrint()}, vrFile['rigInfo'] does not exist. Assuming default settings for B2!")
            vrFile['rigInfo'] = {'computerName':'ZINKO','rotEncPos':'left','rotEncSign':-1,'wheelToVR':4000,'wheelRadius':9.75,'rotaryRange':32} # save dictionary with default B2 settings
        return vrFile
    
    def convertDense(self, data):
        return np.array(data[:self.value['numTrials']].todense()).squeeze()

    def createIndex(self, timeStamps):
        # requires timestamps as (numTrials x numSamples) dense numpy array 
        return [np.nonzero(t)[0] for t in timeStamps]

    def getVRData(self, data, nzindex):
        return [d[nz] for (d,nz) in zip(data, nzindex)]
    
    
    # -------------------------------------- methods for handling imaging data produced by suite2p ------------------------------------------------------------
    def loadS2P(self,varName,concatenate=True,checkVariables=True):
        # load S2P variable from suite2p folders 
        assert varName in self.value['available'], f"{varName} is not available in the suite2p folders for {self.sessionPrint()}"
        if varName=='ops': concatenate=False 
        var = []
        for planeIdx,planeName in enumerate(self.value['planeNames']):
            cvar = np.load(self.suite2pPath()/planeName/f"{varName}.npy",allow_pickle=True)
            if checkVariables:
                if self.isRoiVar(cvar): # everything except ops should have the same number of ROIs (in axis=0) 
                    assert cvar.shape[0]==self.value['roiPerPlane'][planeIdx], f"{self.sessionPrint}:{self.planeNames[planeIdx]}:{varName} has a different number of ROIs than registered"
                if self.isFrameVar(cvar): # everything except ops, stat, redcell, and iscell should have the same number of frames
                    assert cvar.shape[1]==self.value['framePerPlane'][planeIdx], f"{self.sessionPrint}:{self.planeNames[planeIdx]}:{varName} has a different number of frames than registered"
            var.append(cvar)
        if concatenate: 
            if self.isFrameVar(var[0]): var = [v[:,:self.value['numFrames']] for v in var] # trim if necesary so each plane has the same number of frames
            var = np.concatenate(var,axis=0)
        return var
    
    def loadfcorr(self,meanAdjusted=True,loadFromOne=True):
        if loadFromOne:
            F = self.loadone('neuron','frame','F')
            Fneu = self.loadone('neuron','frame','Fneu')
        else:
            F = self.loadS2P('F')
            Fneu = self.loadS2P('Fneu')
        meanFneu = np.mean(Fneu,axis=1,keepdims=True) if meanAdjusted else np.zeros((np.sum(self.value['roiPerPlane']),1))
        return  F - self.neuropilCoefficient*(Fneu - meanFneu)
    
    # shorthands -- note that these require some assumptions about suite2p variables to be met
    def isRoiVar(self, var): return var.ndim>0 # useful shorthand for determining if suite2p variable includes one element for each ROI
    def isFrameVar(self, var): return var.ndim>1 and var.shape[1]>2 # useful shorthand for determining if suite2p variable includes a column for each frame
    
    def getPlaneIdx(self):
        # produce np array of plane ID associated with each ROI (assuming that the data e.g. spks will be concatenated across planes) 
        return np.concatenate([np.repeat(planeIDs,roiPerPlane) for (planeIDs,roiPerPlane) in zip(self.value['planeIDs'],self.value['roiPerPlane'])]).astype(np.uint8)
        
    def oneFilename(self, *names):
        return '.'.join([name.lower() for name in names])+'.npy'


# ==============================================================================================================================================================================
