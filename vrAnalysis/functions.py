import numpy as np
import scipy as sp
import numba as nb
from . import helpers

# ------------------------------------------------- simple processing functions for behavioral data --------------------------------------------------------
def environmentRewardZone(vrexp):
    """get a list of reward locations for each environment"""
    environmentIndex = vrexp.loadone('trials.environmentIndex')
    environments = np.unique(environmentIndex)
    rewardPosition = vrexp.loadone('trials.rewardPosition')
    rewardZoneHalfwidth = vrexp.loadone('trials.rewardZoneHalfwidth')
    envRewPos = [np.unique(rewardPosition[environmentIndex==env]) for env in environments]
    envRewHW = [np.unique(rewardZoneHalfwidth[environmentIndex==env]) for env in environments]
    assert all([len(erp)==1 for erp in envRewPos]), f"reward position wasn't the same within each environment in session: {vrexp.sessionPrint()}"
    assert all([len(erhw)==1 for erhw in envRewHW]), f"reward zone halfwidth wasn't the same within each environment in session: {vrexp.sessionPrint()}"
    return [erp[0] for erp in envRewPos], [erhw[0] for erhw in envRewHW]
    
# ------------------------------------------------- postprocessing functions for creating spatial maps -----------------------------------------------------
def checkDistStep(distStep):
    """distStep has particular requirements for the functions that use it!"""
    if len(distStep)>1:
        message = "distStep should be a two(or 3) element tuple of ints increasing in size, with an optional third argument describing the standard deviation of the smoothing kernel"
        assert len(distStep)<=3 and distStep[1]>distStep[0] and isinstance(distStep[0],int) and isinstance(distStep[1],int), message
    else:
        if not isinstance(distStep, tuple): 
            distStep = (distStep)
    return distStep

def getBinEdges(vrexp, distStep):
    """get bin edges for virtual position"""
    roomLength = vrexp.loadone('trials.roomLength')
    assert np.unique(roomLength).size==1, f"roomLengths are not all the same in session {vrexp.sessionPrint()}"
    roomLength = roomLength[0]
    numPosition = int(roomLength/distStep[0])
    distedges = np.linspace(0,roomLength,numPosition+1)
    distcenter = helpers.edge2center(distedges)
    return distedges, distcenter, roomLength

def replaceMissingData(data, firstValidBin, lastValidBin, replaceWith=np.nan):
    """switch to nan for any bins that the mouse didn't visit (excluding those in between visited bins)"""
    for trial, (fvb, lvb) in enumerate(zip(firstValidBin, lastValidBin)):
        data[trial,:fvb] = replaceWith
        data[trial,lvb+1:] = replaceWith
    return data
    
def loadBehavioralData(vrexp, distStep, speedThreshold):
    """centralized method for loading behavioral data"""
    distedges, distcenter, roomLength = getBinEdges(vrexp, distStep)
    numPosition = len(distcenter)
    
    trialStartSample = vrexp.loadone('trials.positionTracking')
    behaveTimeStamps = vrexp.loadone('positionTracking.times')
    behavePosition = vrexp.loadone('positionTracking.position')
    lickSamples = vrexp.loadone('licksTracking.positionTracking') # list of behave sample for each lick

    behavePositionBin = np.digitize(behavePosition,distedges)-1 # index of position bin for each sample
    lickPositionBin = behavePositionBin[lickSamples] # index of position bin for each lick
    behaveTrialIdx = vrexp.getBehaveTrialIdx(trialStartSample) # array of trial index for each sample
    lickTrialIdx = behaveTrialIdx[lickSamples] # trial index of each lick
    withinTrialSample = np.append(np.diff(behaveTrialIdx)==0, True) # only true if next sample is from same trial (last sample from each trial == False)
    
    sampleDuration = np.append(np.diff(behaveTimeStamps),0) # time between samples, assume zero time was spent in last sample for each trial
    behaveSpeed = np.append(np.diff(behavePosition)/sampleDuration[:-1], 0) # speed in each sample, assume speed was zero in last sample for each trial
    sampleDurationThresholded = sampleDuration * (behaveSpeed >= speedThreshold) * withinTrialSample # keep sample duration when speed above threshold and sample within trial
    behaveSpeedThresholded = behaveSpeed * (behaveSpeed >= speedThreshold) * withinTrialSample # keep speed when above threshold and sample within trial
    
    # Get high resolution occupancy and speed maps 
    occmap = np.zeros((vrexp.value['numTrials'],numPosition))
    speedmap = np.zeros((vrexp.value['numTrials'],numPosition))
    lickmap = np.zeros((vrexp.value['numTrials'],numPosition))
    count = np.zeros((vrexp.value['numTrials'],numPosition))
    getMaps(sampleDurationThresholded, behaveSpeedThresholded, behaveTrialIdx, behavePositionBin, occmap, speedmap, count)
    getMap(np.ones_like(lickSamples), lickTrialIdx, lickPositionBin, lickmap)

    # Figure out the valid range (outside of this range, set the maps to nan, because their values are not meaningful)
    bpbPerTrial = vrexp.groupBehaveByTrial(behavePositionBin, trialStartSample)
    firstValidBin = [np.min(bpb) for bpb in bpbPerTrial]
    lastValidBin = [np.max(bpb) for bpb in bpbPerTrial]

    return occmap, speedmap, lickmap, firstValidBin, lastValidBin, distcenter, roomLength

def loadSpikeMap(vrexp, distStep=(1,5), onefile='mpci.roiActivityDeconvolved', speedThreshold=0, standardizeSpks=True):
    """centralized method for loading spiking map"""
    distStep = checkDistStep(distStep)
    distedges, distcenter, roomLength = getBinEdges(vrexp, distStep)
    numPosition = len(distcenter)
    
    frameTrialIdx, framePosition, frameSpeed = vrexp.getFrameBehavior()
    framePositionBin = np.where(~np.isnan(framePosition), np.digitize(framePosition, distedges)-1, np.nan)
    idxAboveSpeedThreshold = (frameSpeed >= speedThreshold)
    
    # Now make spike map using frame position
    spks = vrexp.loadone(onefile).T
    spks *= idxAboveSpeedThreshold # set spks to 0 unless above speed threshold
    if standardizeSpks: 
        spks = (spks - np.median(spks,axis=1,keepdims=True)) / np.std(spks,axis=1,keepdims=True)

    # back to frames x ROIs for getSpkMap method
    spks = spks.T 
    
    # Now prepare spkmap
    spkmap = np.zeros((vrexp.value['numTrials'],len(distedges)-1,vrexp.value['numROIs']))
    count = np.zeros((vrexp.value['numTrials'],len(distedges)-1))
    getSpkMap(spks, frameTrialIdx, framePositionBin, spkmap, count, useAverage=False)
    
    return spkmap, count

def getBehaviorAndSpikeMaps(vrexp, distStep=(1,5), onefile='mpci.roiActivityDeconvolved', speedThreshold=0, standardizeSpks=True, doSmoothing=None):
    distStep = checkDistStep(distStep)

    # load key behavioral data (at higher resolution if distStep has multiple values)
    occmap, speedmap, lickmap, firstValidBin, lastValidBin, distcenter, roomLength = loadBehavioralData(vrexp, distStep, speedThreshold)
    numPosition = int(roomLength/distStep[0])

    # load spiking data (at higher resolution if distStep has multiple values)
    spkmap, count = loadSpikeMap(vrexp, distStep=distStep, onefile=onefile, speedThreshold=speedThreshold, standardizeSpks=standardizeSpks)
    
    # now handle conversion from sum of spikes to average
    if len(distStep)==1:
        # no smoothing -- divide spkmap by occmap and set nans where necessary
        correctMap(occmap, spkmap)

        # set bins to nan when mouse didn't visit them
        occmap = replaceMissingData(occmap, firstValidBin, lastValidBin)
        speedmap = replaceMissingData(speedmap , firstValidBin, lastValidBin)
        lickmap = replaceMissingData(lickmap, firstValidBin, lastValidBin)
        spkmap = replaceMissingData(spkmap, firstValidBin, lastValidBin)
        
    else:
        # Create spatial smoothing kernel 
        kk = helpers.getGaussKernel(distcenter, distStep[-1]) # standard deviation = dsFactor if len(distStep)==2, and specified kernel width if len(distStep)==3

        # Smooth maps with convolution (gauss kernel has unit norm so no correction needed)
        occmap = helpers.convolveToeplitz(occmap, kk, axis=1)
        speedmap = helpers.convolveToeplitz(speedmap, kk, axis=1)
        spkmap = helpers.convolveToeplitz(spkmap, kk, axis=1)

        # now correct spkmap by occupancy map after smoothing
        correctMap(occmap, spkmap)
        spkmap = np.transpose(spkmap, (2, 0, 1)) # convert to smart indexing (which is necessary for reshaping)
        numROIs = spkmap.shape[0]
        
        # True if a bin wasn't visited
        didnt_visit = replaceMissingData(np.zeros_like(occmap, dtype=bool), firstValidBin, lastValidBin, replaceWith=True)
        
        # True if every bin in a down-sampled sample wasn't visited
        dsFactor = int(distStep[1]/distStep[0])
        all_didnt_visit = np.all(np.isnan(np.reshape(didnt_visit, (vrexp.value['numTrials'], -1, dsFactor))), axis=2)

        # get downsampled maps
        occmap = np.mean(np.reshape(occmap,(vrexp.value['numTrials'], -1, dsFactor)), axis=2)
        speedmap = np.mean(np.reshape(speedmap,(vrexp.value['numTrials'], -1, dsFactor)), axis=2)
        lickmap = np.sum(np.reshape(lickmap,(vrexp.value['numTrials'], -1, dsFactor)), axis=2) # sum licks for each position (don't take average)
        spkmap = np.mean(np.reshape(spkmap,(numROIs, vrexp.value['numTrials'], -1, dsFactor)), axis=3)
        distedges = np.linspace(0, roomLength, int(numPosition/dsFactor)+1)

        # set nan where mouse didn't visit at all
        occmap[all_didnt_visit] = np.nan
        speedmap[all_didnt_visit] = np.nan
        lickmap[all_didnt_visit] = np.nan
        spkmap[:, all_didnt_visit] = np.nan

    return occmap, speedmap, lickmap, spkmap, distedges
    
def getBehaviorMaps(vrexp, distStep=(1,5), speedThreshold=0):
    # Produce occupancy map and speed map using numba speed ups
    # distStep is a two element tuple of integers - the first defines the spatial resolution of the initial measurement, the second defines the spatial filtering and the downsampling factor
    # First computes the maps with high resolution, then spatially smooths them, then downsamples them
    distStep = checkDistStep(distStep)

    # load key behavioral data 
    occmap, speedmap, lickmap, firstValidBin, lastValidBin, distcenter, roomLength = loadBehavioralData(vrexp, distStep, speedThreshold)
    numPosition = int(roomLength/distStep[0])
    
    if len(distStep)==1:
        # switch to nan for any bins that the mouse didn't visit (excluding those in between visited bins)
        # set bins to nan when mouse didn't visit them
        occmap = replaceMissingData(occmap, firstValidBin, lastValidBin)
        speedmap = replaceMissingData(speedmap , firstValidBin, lastValidBin)
        lickmap = replaceMissingData(lickmap, firstValidBin, lastValidBin)
        
    else:
        # Create spatial smoothing kernel 
        kk = helpers.getGaussKernel(distcenter, distStep[-1]) # standard deviation = dsFactor if len(distStep)==2, and specified kernel width if len(distStep)==3

        # Smooth maps with convolution (gauss kernel has unit norm so no correction needed)
        occmap = helpers.convolveToeplitz(occmap, kk, axis=1)
        speedmap = helpers.convolveToeplitz(speedmap, kk, axis=1)

        # True if a bin wasn't visited
        didnt_visit = replaceMissingData(np.zeros_like(occmap, dtype=bool), firstValidBin, lastValidBin, replaceWith=True)
        
        # True if every bin in a down-sampled sample wasn't visited
        dsFactor = int(distStep[1]/distStep[0])
        all_didnt_visit = np.all(np.isnan(np.reshape(didnt_visit, (vrexp.value['numTrials'], -1, dsFactor))), axis=2)

        # get downsampled maps
        occmap = np.mean(np.reshape(occmap,(vrexp.value['numTrials'],-1,dsFactor)),axis=2)
        speedmap = np.mean(np.reshape(speedmap,(vrexp.value['numTrials'],-1,dsFactor)),axis=2)
        lickmap = np.sum(np.reshape(lickmap,(vrexp.value['numTrials'],-1,dsFactor)),axis=2) # sum licks for each position (don't take average)
        distedges = np.linspace(0,roomLength,int(numPosition/dsFactor)+1)

        # set nan where mouse didn't visit at all
        occmap[all_didnt_visit] = np.nan
        speedmap[all_didnt_visit] = np.nan
        lickmap[all_didnt_visit] = np.nan

    return occmap, speedmap, lickmap, distedges
    
def measureReliability(spkmap, numcv=3, numRepeats=1):
    """
    Function to measure spatial reliability of spiking in the spkmap. 
    spkmap is (numROIs, numTrials, numPositions), trialIdx should be a valid index to the numTrials axis of spkmap.
    cross-validated estimate of reliability by measuring spatial profile on training trials and predicting test trials 
    returns two measures of reliability- one compares prediction of estimate based on training profile or training mean, and one based on correlation between train/test 
    """
    assert not np.any(np.isnan(spkmap)), "spkmap has nans, remove incomplete trials or just measure reliability for valid positions"
    spkmap = spkmap.transpose(1,2,0)
    numTrials,numPosition,numROIs = spkmap.shape
    relmse = np.zeros(numROIs)
    relcor = np.zeros(numROIs)
    for repeat in range(numRepeats):
        foldIdx = helpers.cvFoldSplit(numTrials, numcv)
        for fold in range(numcv):
            cTrainTrial = np.concatenate(foldIdx[:fold]+foldIdx[fold+1:])
            cTestTrial = foldIdx[fold]
            trainProfile = np.mean(spkmap[cTrainTrial],axis=0)
            testProfile = np.mean(spkmap[cTestTrial],axis=0)
            meanTrain = np.mean(trainProfile,axis=0,keepdims=True) # mean across positions for each ROI
            meanTest = np.mean(testProfile,axis=0,keepdims=True)
            numerator = np.sum((testProfile-trainProfile)**2,axis=0)
            denominator = np.sum((testProfile-meanTrain)**2,axis=0)

            # only measure reliability if it has activity
            idxHasActivity = np.any(spkmap!=0, axis=(0,1))
            relmse[idxHasActivity] += (1 - numerator[idxHasActivity]/denominator[idxHasActivity])
            relmse[~idxHasActivity] = np.nan # otherwise set to nan
            relcor += helpers.vectorCorrelation(trainProfile, testProfile) # vectorCorrelation returns 0 if data has 0 standard deviation
    relmse /= (numcv*numRepeats)
    relcor /= (numcv*numRepeats)
    return relmse,relcor

def measureSpatialInformation(omap, spkmap):
    """
    measure the spatial information of spiking for each ROI using the formula in this paper:
    https://proceedings.neurips.cc/paper/1992/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf
    information = sum_x (meanRate(x) * log2(meanRate(x)/meanRate) * prob(x) dx)
    occmap is a (trial x position) occupancy map
    spkmap is an (ROI x trial x position) map of the average firing rate in each spatial bin
    omap and spkmap must be nonegative as they represent probability distributions here
    """
    assert np.all(omap>=0) and np.all(~np.isnan(omap)), "occupancy map must be nonnegative and not have any nans"
    assert np.all(spkmap>=0) and np.all(~np.isnan(spkmap)), "spkmap must be nonnegative and not have any nans"
    probOcc = np.sum(omap,axis=0) / np.sum(omap) # probability of occupying each spatial position (Position,)
    meanSpkPerPos = np.mean(spkmap,axis=1) # mean spiking in each spatial position (ROI x Position)
    meanRoi = np.mean(meanSpkPerPos,axis=1,keepdims=True) # mean spiking for each ROI (ROI,)
    logTerm = meanSpkPerPos / meanRoi # ratio of mean in each position to overall mean 
    validLog = logTerm > 0 # in mutual information, log2(x)=[log2(x) if x>0 else 0]
    logTerm[validLog]=np.log2(logTerm[validLog])
    logTerm[~validLog]=0
    spInfo = np.sum(meanSpkPerPos * logTerm * probOcc, axis=1)
    return spInfo


# ---------------------------------- numba code for speedups ---------------------------------
@nb.njit(parallel=True)
def getMap(valueToSum, trialidx, positionbin, smap):
    # this is the fastest way to get a single summation map
    # -- accepts 1d arrays value, trialidx, positionbin of the same size --
    # -- shape determines the number of trials and position bins (they might not all be represented in trialidx or positionbin, or we could just do np.max()) --
    # -- each value represents some number to be summed as a function of which trial it was in and which positionbin it was in --
    for sample in nb.prange(len(valueToSum)):
        smap[trialidx[sample]][positionbin[sample]] += valueToSum[sample]

@nb.njit(parallel=True)
def getMaps(valueToSum, valueToAverage, trialidx, positionbin, smap, amap, count, correctValueToAverage=True):
    # this is the fastest way to get a summation map and an average map (using summation to correct)
    # -- smap is computed just as smap is computed in the function getMap() --
    # -- amap is first computed as smap, but requires correction because it should be an average (i.e. it is an average signal rather than an accumulating signal) --
    # -- after smap and amap are computed, go through each element and divide to-be-averaged signal (amap[t,p]) by number of samples accumulated if greater than 0 --
    # -- there is a switch for not correcting value to average because sometimes they are computed with a very small spatial resolution and then convolved across spatial position before averaging across time --
    for sample in nb.prange(len(valueToSum)):
        smap[trialidx[sample]][positionbin[sample]] += valueToSum[sample]
        amap[trialidx[sample]][positionbin[sample]] += valueToAverage[sample]
        if correctValueToAverage:
            count[trialidx[sample]][positionbin[sample]] += 1
    if correctValueToAverage:
        for t in nb.prange(amap.shape[0]):
            for p in nb.prange(amap.shape[1]):
                if count[t,p]>0:
                    amap[t,p] /= count[t,p]

@nb.njit(parallel=True)
def correctMap(smap, amap):
    # this is the fastest way to correct a summation map (amap) by time spent (smap) if they were computed separately and the summation map should be averaged across time
    for t in nb.prange(smap.shape[0]):
        for p in nb.prange(smap.shape[1]):
            if smap[t,p]>0:
                amap[t,p] /= smap[t,p]

@nb.njit(parallel=True)
def behaveToFrame(behavePosition,behaveTrialIdx,idxBehaveToFrame,distBehaveToFrame,distanceCutoff,framePosition,frameTrialIdx,count):
    for sample in nb.prange(len(behavePosition)):
        if distBehaveToFrame[sample]<=distanceCutoff:
            framePosition[idxBehaveToFrame[sample]] += behavePosition[sample]
            frameTrialIdx[idxBehaveToFrame[sample]] += behaveTrialIdx[sample]
            count[idxBehaveToFrame[sample]] += 1
    for sample in nb.prange(len(count)):
        if count[sample]>0:
            framePosition[sample] /= count[sample]
            frameTrialIdx[sample] /= count[sample]


@nb.njit(parallel=True)
def timelineToFrame(timelineIndex, timelineVariable, output, count):
    for idx in nb.prange(len(timelineIndex)):
        output[timelineIndex[idx]] += timelineVariable[idx]
        count[timelineIndex[idx]] += 1
    for idx in nb.prange(len(count)):
        if count[idx]>0:
            output[idx] /= count[idx]

@nb.njit(parallel=True)
def getSpkMap(spks, frameTrialIdx, framePositionBin, spkmap, count, useAverage=True):
    for sample in nb.prange(len(frameTrialIdx)):
        if not np.isnan(frameTrialIdx[sample]):
            spkmap[int(frameTrialIdx[sample])][int(framePositionBin[sample])] += spks[sample]
            count[int(frameTrialIdx[sample])][int(framePositionBin[sample])] += 1
    if useAverage:
        for ii in nb.prange(count.shape[0]):
            for jj in nb.prange(count.shape[1]):
                if count[ii][jj] > 0:
                    spkmap[ii][jj] /= count[ii][jj]


                
# ============================================================================================================================================