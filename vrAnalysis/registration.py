# inclusions
import time
from tqdm import tqdm
import numpy as np
import scipy as sp
import scipy.io as scio
from . import helpers
from . import session


class defaultRigInfo:
    """this is prepared here in case the RigInfo field was not saved for behavioral data"""
    computerName='ZINKO'
    rotEncPos='left'
    rotEncSign=-1
    wheelToVR=4000
    wheelRadius=9.75
    rotaryRange=32

# ---------------------------------------------------------------------------------------------------
# ------------------------------------- behavior processing methods ---------------------------------
# ---------------------------------------------------------------------------------------------------
def standard_behavior(self):
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
    trialStartOffsets = behaveTimeStamps[trialStartFrame] - self.loadone('trials.startTimes') # get offset
    behaveTimeStamps = np.concatenate([bts - trialStartOffsets[tidx] for (tidx,bts) in enumerate(self.groupBehaveByTrial(behaveTimeStamps,trialStartFrame))])
    
    # Save behave onedata
    self.saveone(behaveTimeStamps, 'positionTracking.times')
    self.saveone(behavePosition, 'positionTracking.position')
    
    # Save trial onedata
    self.saveone(trialStartFrame,'trials.positionTracking')
    self.saveone(trialEnvironmentIndex,'trials.environmentIndex')
    self.saveone(trialRoomLength, 'trials.roomlength')
    self.saveone(trialMovementGain, 'trials.movementGain')
    self.saveone(trialRewardPosition, 'trials.rewardPosition')
    self.saveone(trialRewardTolerance, 'trials.rewardZoneHalfwidth')
    self.saveone(trialRewardAvailability, 'trials.rewardAvailability')
    self.saveone(trialRewardDelivery, 'trials.rewardPositionTracking')
    self.saveone(trialActiveLicking, 'trials.activeLicking')
    self.saveone(trialActiveStopping, 'trials.activeStopping')
    
    # Save lick onedata
    self.saveone(lickBehaveSample, 'licksTracking.positionTracking')

    return self


def cr_hippocannula_behavior(self):
    trialInfo = self.vrFile['TRIAL']
    expInfo = self.vrFile['EXP']
    
    numTrials = trialInfo.info.no
    nonNanSamples = np.sum(~np.isnan(trialInfo.time[:,0]))
    assert numTrials == nonNanSamples, f"# trials {trialInfo.info.no} isn't equal to non-nan first time samples {nonNanSamples}"
    self.registerValue('numTrials',numTrials)
    
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
    trialRoomLength = np.ones(self.value['numTrials']) * expInfo.roomLength
    trialMovementGain = np.ones(self.value['numTrials']) # mvmt gain always one
    trialRewardPosition = self.convertDense(trialInfo.trialRewPos)
    trialRewardTolerance = self.convertDense(expInfo.rewPosTolerance * np.ones(self.value['numTrials']))
    trialRewardAvailability = self.convertDense(trialInfo.trialRewAvailable).astype(np.bool_)
    rewardDelivery = self.convertDense(trialInfo.trialRewDelivery)
    rewardDelivery[np.isnan(rewardDelivery)]=0 # about to be (-1), indicating no reward delivered
    rewardDelivery = rewardDelivery.astype(np.int64)-1 # get reward delivery frame (frame within trial) first (will be -1 if no reward delivered)
    
    # adjust frame count to behave arrays
    trialRewardDelivery = np.array([rewardDelivery + np.sum(numTimeStamps[:trialIdx]) if rewardDelivery>=0 else rewardDelivery for (trialIdx, rewardDelivery) in enumerate(rewardDelivery)]) 
    trialActiveLicking = self.convertDense(trialInfo.trialActiveLicking).astype(np.bool_)
    trialActiveStopping = self.convertDense(trialInfo.trialActiveStopping).astype(np.bool_)
    
    # Check shapes and sizes
    assert trialEnvironmentIndex.ndim==1 and len(trialEnvironmentIndex)==self.value['numTrials'], "trialEnvironmentIndex is not a (numTrials,) shaped array!"
    assert trialStartFrame.shape == trialEnvironmentIndex.shape == trialRoomLength.shape == trialMovementGain.shape == trialRewardPosition.shape == trialRewardTolerance.shape == \
            trialRewardAvailability.shape == trialRewardDelivery.shape == trialActiveLicking.shape == trialActiveStopping.shape, "trial oneData arrays do not have the same shape!"
    
    # oneData with lick prefix is a (numLicks,) shaped array containing information about each lick during VR behavior
    licks = [vrd.astype(np.int16) for vrd in self.getVRData(self.convertDense(trialInfo.lick),nzindex)]
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
    trialStartOffsets = behaveTimeStamps[trialStartFrame] - self.loadone('trials.startTimes') # get offset
    behaveTimeStamps = np.concatenate([bts - trialStartOffsets[tidx] for (tidx,bts) in enumerate(self.groupBehaveByTrial(behaveTimeStamps,trialStartFrame))])
    
    # Save behave onedata 
    self.saveone(behaveTimeStamps, 'positionTracking.times')
    self.saveone(behavePosition, 'positionTracking.position')
    
    # Save trial onedata
    self.saveone(trialStartFrame,'trials.positionTracking')
    self.saveone(trialEnvironmentIndex,'trials.environmentIndex')
    self.saveone(trialRoomLength, 'trials.roomlength')
    self.saveone(trialMovementGain, 'trials.movementGain')
    self.saveone(trialRewardPosition, 'trials.rewardPosition')
    self.saveone(trialRewardTolerance, 'trials.rewardZoneHalfwidth')
    self.saveone(trialRewardAvailability, 'trials.rewardAvailability')
    self.saveone(trialRewardDelivery, 'trials.rewardPositionTracking')
    self.saveone(trialActiveLicking, 'trials.activeLicking')
    self.saveone(trialActiveStopping, 'trials.activeStopping')
    
    # Save lick onedata
    self.saveone(lickBehaveSample, 'licksTracking.positionTracking')

    return self



behavior_processing = {
    1: standard_behavior,
    2: cr_hippocannula_behavior,
}


class vrRegistration(session.vrExperiment):
    """
    The vrRegistration object is a preprocessing object used to preprocess the data, save oneformats, and save vrExperiment parameters.
    It accepts three required arguments as strings - the mouseName, dateString, and session, which it uses to identify the directory of the data (w/ Alyx conventions and a method specified local data path)
    It also accepts optional keyword arguments that define how to preprocess the data and defines other important parameters.
    
    Standard usage is:
    vrReg = vrRegistration(mouseName, dateString, session, **kwarg)
    vrReg.doPreprocessing()
    vrReg.saveParams()
    """
    name = 'vrRegistration'
    def __init__(self,*inputs,**userOpts):
        if len(inputs)==1 and isinstance(inputs[0], vrExperiment):
            # This means we are creating a vrRegistration object from an existing vrExperiment
            self.createObject(*inputs)
            assert userOpts.keys() <= self.opts.keys(), f"userOpts contains the following invalid keys:  {set(userOpts.keys()).difference(opts.keys())}"
            for key,value in userOpts.items():
                if value != self.opts[key]:
                    print(f"In existing vrExperiment, opts['{key}']={self.opts[key]} but you just requested {key}={value}."
                         "This option will not be updated, so if you want to make a change, make sure you do it manually!")
            
        elif len(inputs)==3:
            # Otherwise, we are registering a new vrExperiment, so have to create everything from scratch
            assert all([isinstance(ip,str) for ip in inputs]), "if three inputs are provided, they must be strings indicating the mouseName, date, and session"
            opts = {} # Options for data management
            # Preprocessing options -- these define what was performed in each experiment and what to preprocess --
            opts['vrBehaviorVersion'] = 1 # 1==standard behavioral output (will make conditional loading systems for alternative versions...)
            opts['facecam'] = False # whether or not face video was performed on this session (note: only set to True when DLC has already been run!)
            opts['imaging'] = True # whether or not imaging was performed on this session (note: only set to True when suite2p has already been run!)
            opts['oasis'] = True # whether or not to rerun oasis on calcium signals (note: only used if imaging is run)
            opts['moveRawData'] = False # whether or not to move raw data files to 'rawData'
            opts['redCellProcessing'] = True # whether or not to preprocess redCell features into oneData using the redCellProcessing object (only runs if redcell in self.value['available'])
            opts['clearOne']=False # whether or not to clear previously stored oneData (even if it wouldn't be overwritten by this registration)
            # Imaging options -- these options are standard values for imaging, tau & fs directly affect preprocessing when OASIS is turned on (and deconvolution is recomputed)
            opts['neuropilCoefficient'] = 0.7 # for manual neuropil estimation
            opts['tau'] = 1.5 # GCaMP time constant
            opts['fs'] = 6 # sampling rate (per volume if multiplane)
            # Other options--
            assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
            opts.update(userOpts) # Update default opts with user requests
        
            # -- initialize vrExperiment parameters --
            mouseName, dateString, session = inputs
            self.mouseName = mouseName
            self.dateString = dateString
            self.session = session
            self.opts = opts
            
            if not self.sessionPath().exists():
                raise ValueError(f"Session directory does not exist for {self.sessionPrint()}")

            # initialize dictionaries to be stored as JSONs and loaded by vrExperiment
            self.preprocessing = [] # initialize this for storing which preprocessing stages were performed
            self.loadBuffer = {} # initialize this upon creation for efficient loading throughout preprocessing
            self.value = {} # initialize this dictionary to save important values (e.g. number of trials)

            if not self.onePath().exists(): self.onePath().mkdir(parents=True)
            if not self.rawDataPath().exists(): self.rawDataPath().mkdir(parents=True)
        
        else:
            raise TypeError("input must be either a vrExperiment object or 3 strings indicating the mouseName, date, and session")
        
        
    def doPreprocessing(self):
        if self.opts['clearOne']: self.clearOneData(certainty=True)
        self.processTimeline()
        self.processBehavior()
        self.processImaging()
        self.processRedCells()
        self.processFacecam()
        self.processBehavior2Imaging()
   
    # --------------------------------------------------------------- preprocessing methods ------------------------------------------------------------        
    def processTimeline(self):
        # load these files for raw behavioral & timeline data
        self.loadTimelineStructure()
        self.loadBehaviorStructure()
        
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
        lickSamples = np.where(helpers.diffsame(lickDetector)==1)[0].astype(np.uint64) # timeline samples of lick times
        
        # Get Reward Commands (measures voltage of output -- assume it's either low or high)
        rewardCommand = self.getTimelineVar('rewardCommand') # load reward command signal
        rewardCommand = np.round(rewardCommand/np.max(rewardCommand))
        rewardSamples = np.where(helpers.diffsame(rewardCommand)>0.5)[0].astype(np.uint64) # timeline samples when reward was delivered
        
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
        pdDerivative,pdIndex = helpers.fivePointDer(pdMedFilt,hfpd,returnIndex=True)
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
        startTrialIndex = helpers.nearestpoint(flipTimes[firstFlipIndex], timestamps)[0] # returns frame index of first photodiode flip in each trial
        
        # Check that first flip is always down -- all of the vrControl code prepares trials in this way
        assert np.all(flipValue[firstFlipIndex]==0), f"In session {self.sessionPrint()}, first flips in trial are not all down!!"
        
        # Check shapes of timeline arrays
        assert timestamps.ndim==1, "timelineTimestamps is not a 1-d array!"
        assert timestamps.shape == rotaryPosition.shape, "timeline timestamps and rotary position arrays do not have the same shape!"
        
        # Save timeline oneData
        self.saveone(timestamps, 'wheelPosition.times')
        self.saveone(rotaryPosition, 'wheelPosition.position')
        self.saveone(timestamps[lickSamples], 'licks.times')
        self.saveone(timestamps[rewardSamples], 'rewards.times')
        self.saveone(timestamps[startTrialIndex], 'trials.startTimes')
        self.preprocessing.append('timeline')
                
    
    def processBehavior(self):
        if not(self.opts['vrBehaviorVersion'] in behavior_processing.keys()):
            raise ValueError(f"Have not coded vrBehavior version {self.opts['vrBehaviorVersion']} yet.",
                             "Create a new 'processBehavior(self)' method and add it to the behavior_processing dictionary!")

        # Call behavior processing method
        self = behavior_processing[self.opts['vrBehaviorVersion']](self)
            
        # Confirm that vrBehavior has been processed
        self.preprocessing.append('vrBehavior')
        
    
    def processImaging(self):
        if not self.opts['imaging']:
            print(f"In session {self.sessionPrint()}, imaging setting set to False in opts['imaging']. Skipping image processing.")
            return None
        
        if not self.suite2pPath().exists():
            raise ValueError(f"In session {self.sessionPrint()}, suite2p processing was requested but suite2p directory does not exist.")
                
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
        timelineTimestamps = self.loadone('wheelPosition.times')
        changeFrames = np.append(0, np.diff(np.ceil(self.getTimelineVar('neuralFrames')/len(self.value['planeIDs']))))==1
        frameSamples = np.where(changeFrames)[0] # TTLs for each volume (increments by 1 for each plane)
        frame2time = timelineTimestamps[frameSamples] # get timelineTimestamps of each imaging volume

        # Handle mismatch between number of imaging frames saved by scanImage (and propagated through suite2p), and between timeline's measurement of the scanImage frame counter
        if len(frame2time)!=self.value['numFrames']:
            if len(frame2time)-1==self.value['numFrames']:
                # If frame2time had one more frame, just trim it and assume everything is fine. This happens when a new volume was started but not finished, so does not required communication to user.
                frameSamples = frameSamples[:-1]
                frame2time = frame2time[:-1]
            elif len(frame2time)-2==self.value['numFrames']:
                print("frame2time had 2 more than suite2p output. This happens sometimes. I don't like it. I think it's because scanimage sends a TTL before starting the frame")
                frameSamples = frameSamples[:-2]
                frame2time = frame2time[:-2]
            else:
                # If frameSamples has too few frames, it's possible that the scanImage signal to timeline was broken but scanImage still continued normally. 
                numMissing = self.value['numFrames'] - len(frameSamples) # measure number of missing frames
                if numMissing < 0: 
                    # If frameSamples had many more frames, generate an error -- something went wrong that needs manual inspection
                    print(f"In session {self.sessionPrint()}, frameSamples has {len(frameSamples)} elements, but {self.value['numFrames']} frames were reported in suite2p. Cannot resolve.")
                    raise ValueError("Cannot fix mismatches when suite2p data is missing!")
                # It's possible that the scanImage signal to timeline was broken but scanImage still continued normally. 
                if numMissing > 1: print(f"In session {self.sessionPrint()}, more than one frameSamples sample was missing. Consider using tiff timelineTimestamps to reproduce accurately.") 
                print((f"In session {self.sessionPrint()}, frameSamples has {len(frameSamples)} elements, but {self.value['numFrames']} frames were saved by suite2p. "
                       "Will extend frameSamples using the typical sampling rate and nearestpoint algorithm."
                      ))
                # If frame2time difference vector is consistent within 1%, then use mean (which is a little more accurate), otherwise use median
                frame2time = timelineTimestamps[frameSamples]
                medianFramePeriod = np.median(np.diff(frame2time)) # measure median sample period
                consistentFrames = np.all(np.abs(np.log(np.diff(frame2time)/medianFramePeriod)) < np.log(1.01)) # True if all frames take within 1% of median frame period
                if consistentFrames: samplePeriod_f2t = np.mean(np.diff(frame2time))
                else: samplePeriod_f2t = np.median(np.diff(frame2time))
                appendFrames = frame2time[-1] + samplePeriod_f2t*(np.arange(numMissing)+1) # add elements to frame2time, assume sampling rate was perfect 
                frame2time = np.concatenate((frame2time,appendFrames))
                frameSamples = helpers.nearestpoint(frame2time, timelineTimestamps)[0]
        
        # average percentage difference between all samples differences and median -- just a useful metric to be saved --
        self.registerValue('samplingDeviationMedianPercentError',np.exp(np.mean(np.abs(np.log(np.diff(frame2time)/np.median(np.diff(frame2time)))))))
        self.registerValue('samplingDeviationMaximumPercentError',np.exp(np.max(np.abs(np.log(np.diff(frame2time)/np.median(np.diff(frame2time)))))))
        
        # recompute deconvolution if requested
        spks = self.loadS2P('spks')
        if self.opts['oasis']:
            try:
                from oasis.functions import deconvolve
            except ImportError as error:
                print("Failed to import deconvolve from oasis -- this probably means you only installed the core requirements")
                raise error
            g = np.exp(-1/self.opts['tau']/self.opts['fs'])
            fcorr = self.loadfcorr(loadFromOne=False)
            ospks = []
            print("Performing oasis...")
            for fc in tqdm(fcorr):
                ospks.append(deconvolve(fc, g=(g,), penalty=1)[1])
            ospks = np.stack(ospks)
            assert ospks.shape == self.loadS2P('spks').shape, f"In session {self.sessionPrint()}, oasis was run and did not produce the same shaped array as suite2p spks..."
            
        # save onedata (no assertions needed, loadS2P() handles shape checks and this function already handled any mismatch between frameSamples and suite2p output
        self.saveone(frame2time, 'mpci.times')
        self.saveone(self.loadS2P('F').T, 'mpci.roiActivityF')
        self.saveone(self.loadS2P('Fneu').T, 'mpci.roiNeuropilActivityF')
        self.saveone(spks.T, 'mpci.roiActivityDeconvolved')
        if self.opts['oasis']:
            self.saveone(ospks.T, 'mpci.roiActivityDeconvolvedOasis')
        if 'redcell' in self.value['available']:
            self.saveone(self.loadS2P('redcell')[:,1], 'mpciROIs.redS2P')
        self.saveone(self.loadS2P('iscell'), 'mpciROIs.isCell')
        self.saveone(self.getRoiStackPosition(), 'mpciROIs.stackPosition')
        self.preprocessing.append('imaging')
    
    def processFacecam(self):
        print("Facecam preprocessing has not been coded yet!")
    
    def processBehavior2Imaging(self):
        if not self.opts['imaging']:
            print(f"In session {self.sessionPrint()}, imaging setting set to False in opts['imaging']. Skipping behavior2imaging processing.")
            return None
        # compute translation mapping from behave frames to imaging frames
        idxBehaveToFrame,distBehaveToFrame = helpers.nearestpoint(self.loadone('positionTracking.times'), self.loadone('mpci.times'))
        self.saveone(idxBehaveToFrame.astype(int), 'positionTracking.mpci')
        
    def processRedCells(self):
        if not(self.opts['imaging']) or not(self.opts['redCellProcessing']): return # if not requested, skip function 
        # if imaging was processed and redCellProcessing was requested, then try to preprocess red cell features
        if 'redcell' not in self.value['available']:
            print(f"In session {self.sessionPrint()}, 'redcell' is not an available suite2p output, although 'redCellProcessing' was requested.")
            return 
        
        # create redCell object
        redCell = session.redCellProcessing(self) 
        
        # compute red-features
        dotParameters={'lowcut':12, 'highcut':250, 'order':3, 'fs':512}
        corrParameters={'width':20,'lowcut':12, 'highcut':250, 'order':3, 'fs':512}
        phaseParameters={'width':40,'eps':1e6,'winFunc':'hamming'}
        
        print(f"Computing red cell features for {self.sessionPrint()}... (usually takes 10-20 seconds)")
        dotProduct = redCell.computeDot(planeIdx=None, **dotParameters) # compute dot-product for all ROIs
        corrCoeff = redCell.computeCorr(planeIdx=None, **corrParameters) # compute cross-correlation for all ROIs 
        phaseCorr = redCell.croppedPhaseCorrelation(planeIdx=None, **phaseParameters)[3] # compute central value of phase-correlation for all ROIs
        
        # initialize annotations
        self.saveone(np.full(self.value['numROIs'], False), 'mpciROIs.redCellIdx')
        self.saveone(np.full((2,self.value['numROIs']), False), 'mpciROIs.redCellManualAssignments')
        
        # save oneData
        self.saveone(dotProduct, 'mpciROIs.redDotProduct')
        self.saveone(corrCoeff, 'mpciROIs.redPearson')
        self.saveone(phaseCorr, 'mpciROIs.redPhaseCorrelation')
        self.saveone(np.array(dotParameters), 'parametersRedDotProduct.keyValuePairs')
        self.saveone(np.array(corrParameters), 'parametersRedPearson.keyValuePairs')
        self.saveone(np.array(phaseParameters), 'parametersRedPhaseCorrelation.keyValuePairs')
        
    
    # -------------------------------------- methods for handling timeline data produced by rigbox ------------------------------------------------------------
    def loadTimelineStructure(self): 
        tlFileName = self.sessionPath() / f"{self.dateString}_{self.session}_{self.mouseName}_Timeline.mat" # timeline.mat file name
        self.tlFile = scio.loadmat(tlFileName,simplify_cells=True)['Timeline'] # load matlab structure
    
    def timelineInputs(self, ignoreTimestamps=False):
        if not hasattr(self, 'tlFile'): 
            self.loadTimelineStructure()
        hwInputs = [hwInput['name'] for hwInput in self.tlFile['hw']['inputs']]
        if ignoreTimestamps: return hwInputs
        return ['timestamps', *hwInputs]
        
    def getTimelineVar(self, varName):
        if not hasattr(self, 'tlFile'): 
            self.loadTimelineStructure()
        if varName=='timestamps': 
            return self.tlFile['rawDAQTimestamps']
        else:
            inputNames = self.timelineInputs(ignoreTimestamps=True)
            assert varName in inputNames, f"{varName} is not a tlFile in session {self.sessionPrint()}"
            return np.squeeze(self.tlFile['rawDAQData'][:,np.where([inputName==varName for inputName in inputNames])[0]])

    def convertRotaryEncoderToPosition(self, rotaryEncoder, rigInfo):
        # rotary encoder is a counter with a big range that sometimes flips around it's axis
        # first get changes in encoder position, fix any big jumps in value, take the cumulative movement and scale to centimeters 
        rotaryMovement = helpers.diffsame(rotaryEncoder)
        idxHighValues = rotaryMovement > 2**(rigInfo.rotaryRange-1)
        idxLowValues = rotaryMovement < -2**(rigInfo.rotaryRange-1)
        rotaryMovement[idxHighValues] -= 2**rigInfo.rotaryRange
        rotaryMovement[idxLowValues] += 2**rigInfo.rotaryRange
        return rigInfo.rotEncSign*np.cumsum(rotaryMovement)*(2*np.pi*rigInfo.wheelRadius)/rigInfo.wheelToVR

   
    # -------------------------------------- methods for handling vrBehavior data produced by vrControl ------------------------------------------------------------
    def loadBehaviorStructure(self):
        vrFileName = self.sessionPath() / f"{self.dateString}_{self.session}_{self.mouseName}_VRBehavior_trial.mat" # vrBehavior output file name
        self.vrFile = scio.loadmat(vrFileName,struct_as_record=False,squeeze_me=True)
        if 'rigInfo' not in self.vrFile.keys():
            print(f"In session: {self.sessionPrint()}, vrFile['rigInfo'] does not exist. Assuming default settings for B2! using `defaultRigInfo()`")
            self.vrFile['rigInfo'] = defaultRigInfo()
            #{'computerName':'ZINKO','rotEncPos':'left','rotEncSign':-1,'wheelToVR':4000,'wheelRadius':9.75,'rotaryRange':32} # save dictionary with default B2 settings
        if not(hasattr(self.vrFile['rigInfo'], 'rotaryRange')):
            self.vrFile['rigInfo'].rotaryRange=32
        
    def convertDense(self, data):
        data = data[:self.value['numTrials']]
        if isinstance(data, sp.sparse._csc.csc_matrix):
            data = np.array(data.todense()).squeeze()
        return data
        
    def createIndex(self, timeStamps):
        # requires timestamps as (numTrials x numSamples) dense numpy array 
        if np.any(np.isnan(timeStamps)):
            return [np.where(~np.isnan(t))[0] for t in timeStamps] # in case we have dense timestamps with nans where no data
        else:
            return [np.nonzero(t)[0] for t in timeStamps] # in case we have sparse timestamps with 0s where no data

    def getVRData(self, data, nzindex):
        return [d[nz] for (d,nz) in zip(data, nzindex)]




