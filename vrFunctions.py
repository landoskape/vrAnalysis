import numpy as np
import scipy as sp
import numba as nb
import basicFunctions as bf

# ------------------------------------------------- postprocessing functions for creating spatial maps -----------------------------------------------------
def getBehaviorMaps(vrexp, distStep=(1,5), speedThreshold=0):
    # Produce occupancy map and speed map using numba speed ups
    # distStep is a two element tuple of integers - the first defines the spatial resolution of the initial measurement, the second defines the spatial filtering and the downsampling factor
    # First computes the maps with high resolution, then spatially smooths them, then downsamples them
    if len(distStep)>1:
        assert len(distStep)==2 and distStep[1]>distStep[0] and isinstance(distStep[0],int) and isinstance(distStep[1],int), "distStep should be a two element tuple of ints increasing in size"
    else:
        if not isinstance(distStep, tuple): distStep = (distStep)

    roomLength = vrexp.loadone('trial.roomLength')
    assert np.unique(roomLength).size==1, f"roomLengths are not all the same in session {vrexp.sessionPrint()}"
    roomLength = roomLength[0]
    numPosition = int(roomLength/distStep[0])
    distvec = np.linspace(0,roomLength,numPosition+1)
    distcenter = bf.edge2center(distvec)

    trialStartSample = vrexp.loadone('trial.startBehaveSample')
    behaveTimeStamps = vrexp.loadone('behave.timeStamps')
    behavePosition = vrexp.loadone('behave.position')
    lickSamples = vrexp.loadone('lick.behavesample') # list of behave sample for each lick

    behavePositionBin = np.digitize(behavePosition,distvec)-1 # index of position bin for each sample
    lickPositionBin = behavePositionBin[lickSamples] # index of position bin for each lick
    behaveTrialIdx = vrexp.getBehaveTrialIdx(trialStartSample) # array of trial index for each sample
    lickTrialIdx = behaveTrialIdx[lickSamples] # trial index of each lick
    withinTrialSample = np.append(np.diff(behaveTrialIdx)==0, True)
    
    sampleDuration = np.append(np.diff(behaveTimeStamps),0) # time between samples, assume zero time was spent in last sample for each trial
    behaveSpeed = np.append(np.diff(behavePosition)/sampleDuration[:-1], 0)
    sampleDurationThresholded = sampleDuration * (behaveSpeed >= speedThreshold) * withinTrialSample
    behaveSpeedThresholded = behaveSpeed * (behaveSpeed >= speedThreshold) * withinTrialSample
    
    # Get high resolution occupancy and speed maps 
    occmap = np.zeros((vrexp.value['numTrials'],numPosition))
    speedmap = np.zeros((vrexp.value['numTrials'],numPosition))
    lickmap = np.zeros((vrexp.value['numTrials'],numPosition))
    getMaps(sampleDurationThresholded, behaveSpeedThresholded, behaveTrialIdx, behavePositionBin, occmap, speedmap)
    getMap(np.ones_like(lickSamples), lickTrialIdx, lickPositionBin, lickmap)

    # Figure out the valid range (outside of this range, set the maps to nan, because their values are not meaningful)
    bpbPerTrial = vrexp.groupBehaveByTrial(behavePositionBin, trialStartSample)
    firstValidBin = [np.min(bpb) for bpb in bpbPerTrial]
    lastValidBin = [np.max(bpb) for bpb in bpbPerTrial]
    
    if len(distStep)==1:
        # switch to nan for any bins that the mouse didn't visit (excluding those in between visited bins)
        for trial,fvb in enumerate(firstValidBin):
            occmap[trial,:fvb] = np.nan
            speedmap[trial,:fvb] = np.nan
            lickmap[trial,:fvb] = np.nan
        for trial,lvb in enumerate(lastValidBin):
            occmap[trial,lvb+1:] = np.nan
            speedmap[trial,lvb+1:] = np.nan
            lickmap[trial,lvb+1:] = np.nan

    else:
        # Create spatial smoothing kernel 
        kk = bf.getGaussKernel(distcenter, distStep[1])

        # Smooth maps and correct speed map to be an average across time (don't smooth out licks, we're going to sum not average)
        occmap = bf.convolveToeplitz(occmap, kk, axis=1)
        speedmap = bf.convolveToeplitz(speedmap, kk, axis=1)
        correctMap(occmap, speedmap)

        # switch to nan for any bins that the mouse didn't visit (excluding those in between visited bins) -- do this after convolution!
        for trial,fvb in enumerate(firstValidBin):
            occmap[trial,:fvb] = np.nan
            speedmap[trial,:fvb] = np.nan
        for trial,lvb in enumerate(lastValidBin):
            occmap[trial,lvb+1:] = np.nan
            speedmap[trial,lvb+1:] = np.nan            

        dsFactor = int(distStep[1]/distStep[0])
        occmap = np.mean(np.reshape(occmap,(vrexp.value['numTrials'],-1,dsFactor)),axis=2)
        speedmap = np.mean(np.reshape(speedmap,(vrexp.value['numTrials'],-1,dsFactor)),axis=2)
        lickmap = np.sum(np.reshape(lickmap,(vrexp.value['numTrials'],-1,dsFactor)),axis=2) # sum licks for each position (don't take average)
        distvec = np.linspace(0,roomLength,int(numPosition/dsFactor)+1)

    return occmap, speedmap, lickmap, distvec

def getSpikeMap(vrexp, frameTrialIdx, framePosition, frameSpeed, distvec, omap, correctNan=True, speedThreshold=0, useAverage=True, standardizeSpks=True, doSmoothing=None):
    # frameTrialIdx, framePosition, frameSpeed = vrexp.getFrameBehavior()
    framePositionBin = np.where(~np.isnan(framePosition), np.digitize(framePosition, distvec)-1, np.nan)

    # Now make spike map using frame position
    spks = vrexp.loadone('neuron.frame.spks')
    if standardizeSpks:
        spks = (spks - np.median(spks,axis=1,keepdims=True)) / np.std(spks,axis=1,keepdims=True)
    spkmap = np.zeros((vrexp.value['numTrials'],len(distvec)-1,vrexp.value['numROIs']))
    count = np.zeros((vrexp.value['numTrials'],len(distvec)-1))

    getSpkMap(spks.T, frameTrialIdx, framePositionBin, spkmap, count, useAverage=useAverage)
    spkmap = np.transpose(spkmap,(2,0,1)) # convert to smart indices

    if doSmoothing:
        kk = bf.getGaussKernel(bf.edge2center(distvec),doSmoothing)
        idxnan = np.isnan(spkmap)
        spkmap[idxnan]=0
        spkmap = bf.convolveToeplitz(spkmap, kk, mode='same')
        spkmap[idxnan]=np.nan
    
    # correct for nans where there is not positional data in omap
    assert omap.shape==spkmap.shape[1:], "occupancy map and spkmap do not have same number of trials and/or spatial bins"
    spkmap[:,np.isnan(omap)]=np.nan 
    
    return spkmap
    
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
        foldIdx = bf.cvFoldSplit(numTrials, numcv)
        for fold in range(numcv):
            cTrainTrial = np.concatenate(foldIdx[:fold]+foldIdx[fold+1:])
            cTestTrial = foldIdx[fold]
            trainProfile = np.mean(spkmap[cTrainTrial],axis=0)
            testProfile = np.mean(spkmap[cTestTrial],axis=0)
            meanTrain = np.mean(trainProfile,axis=0,keepdims=True) # mean across positions for each ROI
            meanTest = np.mean(testProfile,axis=0,keepdims=True)
            numerator = np.sum((testProfile-trainProfile)**2,axis=0)
            denominator = np.sum((testProfile-meanTrain)**2,axis=0)
            relmse += (1 - numerator/denominator)
            relcor += bf.vectorCorrelation(trainProfile, testProfile)
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
def getMaps(valueToSum, valueToAverage, trialidx, positionbin, smap, amap, correctValueToAverage=True):
    # this is the fastest way to get a summation map and an average map (using summation to correct)
    # -- smap is computed just as smap is computed in the function getMap() --
    # -- amap is first computed as smap, but requires correction for the time spent (i.e. it is an average signal rather than an accumulating signal) --
    # -- after smap and amap are computed, go through each element and divide by to-be-averaged signal (amap[t,p]) by time spent (smap[t,p]) if time spent is greater than 0 --
    # -- there is a switch for not correcting value to average because sometimes they are computed with a very small spatial resolution and then convolved across spatial position before averaging across time --
    for sample in nb.prange(len(valueToSum)):
        smap[trialidx[sample]][positionbin[sample]] += valueToSum[sample]
        amap[trialidx[sample]][positionbin[sample]] += valueToAverage[sample]
    if correctValueToAverage:
        for t in nb.prange(smap.shape[0]):
            for p in nb.prange(smap.shape[1]):
                if smap[t,p]>0:
                    amap[t,p] /= smap[t,p]

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