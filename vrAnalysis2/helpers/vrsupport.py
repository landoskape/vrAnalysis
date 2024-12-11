import numpy as np
import numba as nb
from .. import faststats as fs

from . import edge2center, getGaussKernel, convolveToeplitz, cvFoldSplit, vectorCorrelation


def get_env_order(mousedb, mouse_name):
    env_order = mousedb.getTable(mouseName=mouse_name)["environmentOrder"].item()
    return [int(env) for env in env_order.split(".")]


# ------------------------------------------------- simple processing functions for behavioral data --------------------------------------------------------
def environmentRewardZone(vrexp):
    """get a list of reward locations for each environment"""
    environmentIndex = vrexp.loadone("trials.environmentIndex")
    environments = np.unique(environmentIndex)
    rewardPosition = vrexp.loadone("trials.rewardPosition")
    rewardZoneHalfwidth = vrexp.loadone("trials.rewardZoneHalfwidth")
    envRewPos = [np.unique(rewardPosition[environmentIndex == env]) for env in environments]
    envRewHW = [np.unique(rewardZoneHalfwidth[environmentIndex == env]) for env in environments]
    assert all([len(erp) == 1 for erp in envRewPos]), f"reward position wasn't the same within each environment in session: {vrexp.sessionPrint()}"
    assert all(
        [len(erhw) == 1 for erhw in envRewHW]
    ), f"reward zone halfwidth wasn't the same within each environment in session: {vrexp.sessionPrint()}"
    return [erp[0] for erp in envRewPos], [erhw[0] for erhw in envRewHW]


# ------------------------------------------------- postprocessing functions for creating spatial maps -----------------------------------------------------
def getBinEdges(vrexp, distStep):
    """get bin edges for virtual position"""
    roomLength = vrexp.loadone("trials.roomLength")
    assert np.unique(roomLength).size == 1, f"roomLengths are not all the same in session {vrexp.sessionPrint()}"
    roomLength = roomLength[0]
    numPosition = int(roomLength / distStep)
    distedges = np.linspace(0, roomLength, numPosition + 1)
    distcenter = edge2center(distedges)
    return distedges, distcenter, roomLength


def replaceMissingData(data, firstValidBin, lastValidBin, replaceWith=np.nan):
    """switch to nan for any bins that the mouse didn't visit (excluding those in between visited bins)"""
    for trial, (fvb, lvb) in enumerate(zip(firstValidBin, lastValidBin)):
        data[trial, :fvb] = replaceWith
        data[trial, lvb + 1 :] = replaceWith
    return data


def getBehaviorAndSpikeMaps(
    vrexp,
    distStep=1,
    onefile="mpci.roiActivityDeconvolved",
    speedThreshold=0,
    speedMaxThreshold=np.inf,
    speedSmoothing=1,
    standardizeSpks=True,
    idxROIs=None,
    get_spkmap=True,
):
    """
    This needs documentation badly...

    If onefile is a string, it will be loaded by vrexp.loadone(onefile)
    If onefile is an array, it will be assumed to be the data to make a spkmap with
    - if this is the case, it has to have the same number of time samples of course
    """
    # get edges and centers of position bin
    distedges, distcenter, _ = getBinEdges(vrexp, distStep)
    numPosition = len(distcenter)

    # load behavioral variables
    trialStartSample = vrexp.loadone("trials.positionTracking")
    behaveTimeStamps = vrexp.loadone("positionTracking.times")
    behavePosition = vrexp.loadone("positionTracking.position")
    lickSamples = vrexp.loadone("licksTracking.positionTracking")  # list of behave sample for each lick

    # process behavioral variables
    behavePositionBin = np.digitize(behavePosition, distedges) - 1  # index of position bin for each sample
    lickPositionBin = behavePositionBin[lickSamples]  # index of position bin for each lick
    behaveTrialIdx = vrexp.getBehaveTrialIdx(trialStartSample)  # array of trial index for each sample
    lickTrialIdx = behaveTrialIdx[lickSamples]  # trial index of each lick
    # only true if next sample is from same trial (last sample from each trial == False)
    withinTrialSample = np.append(np.diff(behaveTrialIdx) == 0, True)

    # time between samples, assume zero time was spent in last sample for each trial
    sampleDuration = np.append(np.diff(behaveTimeStamps), 0)
    # speed in each sample, assume speed was zero in last sample for each trial
    behaveSpeed = np.append(np.diff(behavePosition) / sampleDuration[:-1], 0)
    # keep sample duration when speed above threshold and sample within trial
    sampleDuration = sampleDuration * withinTrialSample
    # keep speed when above threshold and sample within trial
    behaveSpeed = behaveSpeed * withinTrialSample

    if get_spkmap:
        # load spiking data and timing of imaging frames
        if isinstance(onefile, str):
            spks = vrexp.loadone(onefile)
            if "deconvolved" in onefile:
                # set negative values to 0 (because they shouldn't be there for deconvolved data)
                spks = np.maximum(spks, 0)
        else:
            spks = np.array(onefile)
            onefile = "provided"
        frameTimeStamps = vrexp.loadone("mpci.times")  # timestamps for each imaging frame
        assert len(frameTimeStamps) == spks.shape[0], "number of imaging frames doesn't match number of samples in spks"
        idxBehaveToFrame = vrexp.loadone("positionTracking.mpci")  # mpci frame index associated with each behavioral frame
        sampling_period = np.median(np.diff(frameTimeStamps))
        distCutoff = sampling_period / 2  # (time) of cutoff for associating imaging frame with behavioral frame
        distBehaveToFrame = frameTimeStamps[idxBehaveToFrame] - behaveTimeStamps

        # filter by requested ROIs if provided
        if idxROIs is not None:
            spks = spks[:, idxROIs]

        # standardize spks to zscore (across time!) (set to 0 where variance is 0)
        if standardizeSpks:
            if "deconvolved" in onefile:
                # If using deconvolved traces, should have zero baseline
                spks = spks / fs.std(spks, axis=0, keepdims=True)

            else:
                # If using fluorescence traces, should have non-zero baseline
                idx_zeros = fs.std(spks, axis=0) == 0
                spks = fs.median_zscore(spks, axis=0)
                spks[:, idx_zeros] = 0

    else:
        # use empty (and small) array for consistent code even when get_spkmap is False
        spks = np.zeros_like(behavePositionBin).reshape(-1, 1)
        frameTimeStamps = vrexp.loadone("mpci.times")  # timestamps for each imaging frame
        idxBehaveToFrame = vrexp.loadone("positionTracking.mpci")  # mpci frame index associated with each behavioral frame
        sampling_period = np.median(np.diff(frameTimeStamps))
        distCutoff = sampling_period / 2  # (time) of cutoff for associating imaging frame with behavioral frame
        distBehaveToFrame = frameTimeStamps[idxBehaveToFrame] - behaveTimeStamps

    # Get high resolution occupancy and speed maps
    occmap = np.zeros((vrexp.value["numTrials"], numPosition))
    speedmap = np.zeros((vrexp.value["numTrials"], numPosition))
    lickmap = np.zeros((vrexp.value["numTrials"], numPosition))
    spkmap = np.zeros((vrexp.value["numTrials"], numPosition, spks.shape[1]))
    count = np.zeros((vrexp.value["numTrials"], numPosition))

    # go through each behavioral frame
    # if speed faster than threshold, keep, otherwise continue
    # if distance to frame lower than threshold, keep, otherwise continue
    # for current trial and position, add sample duration to occupancy map
    # for current trial and position, add speed to speed map
    # for current trial and position, add full list of spikes to spkmap
    # every single time, add 1 to count for that position
    getAllMaps(
        behaveTrialIdx,
        behavePositionBin,
        sampleDuration,
        behaveSpeed,
        speedThreshold,
        speedMaxThreshold,
        spks,
        idxBehaveToFrame,
        distBehaveToFrame,
        distCutoff,
        occmap,
        speedmap,
        spkmap,
        count,
    )
    # also get lick map (which has to be computed differently)
    getMap(np.ones_like(lickSamples), lickTrialIdx, lickPositionBin, lickmap)

    # correct speedmap immediately
    if speedSmoothing is not None:
        kk = getGaussKernel(distcenter, speedSmoothing)
        speedmap = convolveToeplitz(speedmap, kk, axis=1)
        smoothocc = convolveToeplitz(occmap, kk, axis=1)
        speedmap[smoothocc != 0] /= smoothocc[smoothocc != 0]
    else:
        speedmap[occmap != 0] /= occmap[occmap != 0]

    # Figure out the valid range (outside of this range, set the maps to nan, because their values are not meaningful)
    bpbPerTrial = vrexp.groupBehaveByTrial(behavePositionBin, trialStartSample)

    # offsetting by 1 because there is a bug in the vrControl software where the first sample is always set
    # to the minimum position (which is 0), but if there is a built-up buffer in the rotary encoder, the position
    # will jump at the second sample. In general this will always work unless the mice have a truly ridiculous
    # speed at the beginning of the trial...
    firstValidBin = [np.min(bpb[1:] if len(bpb) > 1 else bpb) for bpb in bpbPerTrial]
    lastValidBin = [np.max(bpb) for bpb in bpbPerTrial]

    # set bins to nan when mouse didn't visit them
    occmap = replaceMissingData(occmap, firstValidBin, lastValidBin)
    speedmap = replaceMissingData(speedmap, firstValidBin, lastValidBin)
    lickmap = replaceMissingData(lickmap, firstValidBin, lastValidBin)
    if get_spkmap:
        spkmap = replaceMissingData(spkmap, firstValidBin, lastValidBin)
    else:
        spkmap = None

    return occmap, speedmap, lickmap, spkmap, count, distedges


def measureReliability(spkmap, numcv=3, numRepeats=1, fraction_nan_permitted=0.05):
    """
    Function to measure spatial reliability of spiking in the spkmap.
    spkmap is (numROIs, numTrials, numPositions),
    cross-validated estimate of reliability by measuring spatial profile on training trials and predicting test trials
    returns two measures of reliability- one compares prediction of estimate based on training profile or training mean, and one based on correlation between train/test
    """
    if np.any(np.isnan(spkmap)):
        position_with_nan = np.any(np.isnan(spkmap), axis=(0, 1))
        fraction_nan = np.sum(position_with_nan) / len(position_with_nan)
        if fraction_nan > fraction_nan_permitted:
            raise ValueError("found nans in more positions than fraction permitted. Increase value or choose trials differently!")
        idx_pos = ~position_with_nan
        spkmap = spkmap[:, :, idx_pos]
    spkmap = spkmap.transpose(1, 2, 0)
    numTrials, _, numROIs = spkmap.shape
    relmse = np.zeros(numROIs)
    relcor = np.zeros(numROIs)
    for _ in range(numRepeats):
        foldIdx = cvFoldSplit(numTrials, numcv)
        for fold in range(numcv):
            cTrainTrial = np.concatenate(foldIdx[:fold] + foldIdx[fold + 1 :])
            cTestTrial = foldIdx[fold]
            trainProfile = fs.mean(spkmap[cTrainTrial], axis=0)
            testProfile = fs.mean(spkmap[cTestTrial], axis=0)
            meanTrain = fs.mean(trainProfile, axis=0, keepdims=True)  # mean across positions for each ROI
            numerator = fs.sum((testProfile - trainProfile) ** 2, axis=0)  # bigger when test and train are more different
            denominator = fs.sum((testProfile - meanTrain) ** 2, axis=0)  # bigger when test isn't well fit by the estimate of the average firing rate

            # only measure reliability if it has activity (if denominator is 0, it doesn't have enough activity :) )
            idxHasActivity = np.any(spkmap != 0, axis=(0, 1)) & (denominator != 0)
            relmse[idxHasActivity] += 1 - numerator[idxHasActivity] / denominator[idxHasActivity]
            relmse[~idxHasActivity] = np.nan  # otherwise set to nan
            relcor += vectorCorrelation(trainProfile, testProfile, axis=0)  # vectorCorrelation returns 0 if data has 0 standard deviation
    relmse /= numcv * numRepeats
    relcor /= numcv * numRepeats
    return relmse, relcor


def measureSpatialInformation(occmap, spkmap):
    """
    measure the spatial information of spiking for each ROI using the formula in this paper:
    https://proceedings.neurips.cc/paper/1992/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf
    information = sum_x (meanRate(x) * log2(meanRate(x)/meanRate) * prob(x) dx)
    occmap is a (trial x position) occupancy map
    spkmap is an (ROI x trial x position) map of the average firing rate in each spatial bin
    occmap and spkmap must be nonegative as they represent probability distributions here
    """
    assert np.all(occmap >= 0) and np.all(~np.isnan(occmap)), "occupancy map must be nonnegative and not have any nans"
    assert np.all(spkmap >= 0) and np.all(~np.isnan(spkmap)), "spkmap must be nonnegative and not have any nans"
    probOcc = np.sum(occmap, axis=0) / np.sum(occmap)  # probability of occupying each spatial position (Position,)
    meanSpkPerPos = np.mean(spkmap, axis=1)  # mean spiking in each spatial position (ROI x Position)
    meanRoi = np.mean(meanSpkPerPos, axis=1, keepdims=True)  # mean spiking for each ROI (ROI,)
    logTerm = meanSpkPerPos / meanRoi  # ratio of mean in each position to overall mean
    validLog = logTerm > 0  # in mutual information, log2(x)=[log2(x) if x>0 else 0]
    logTerm[validLog] = np.log2(logTerm[validLog])
    logTerm[~validLog] = 0
    spInfo = np.sum(meanSpkPerPos * logTerm * probOcc, axis=1)
    return spInfo


# ---------------------------------- numba code for speedups ---------------------------------
@nb.njit(parallel=True)
def getAllMaps(
    behaveTrialIdx,
    behavePositionBin,
    sampleDuration,
    behaveSpeed,
    speedThreshold,
    speedMaxThreshold,
    spks,
    idxBehaveToFrame,
    distBehaveToFrame,
    distCutoff,
    occmap,
    speedmap,
    spkmap,
    count,
):
    """
    method for transforming temporal information into positional information to make spatial maps

    behavioral and timing variables are:
    behaveTrialIdx, behavePositionBin, sampleDuration, behaveSpeed, idxBehaveToFrame, distBehaveToFrame
    where these are all (num_behavioral_samples,) and correspond to the relevant value for each sample.
    idxBehaveToFrame refers to which index of spks is closest in time to each index in behavioral samples,
    and distBehaveToFrame refers to the (temporal) "distance" between those two samples

    requires output variables occmap, speedmap, spkmap, and count because it's numba
    these are all (num_trials, num_positions, ...) (where ... indicates the extra dimension for ROIs in the spkmap and spks)

    will accumulate time spent in occmap
    will accumulate weighted speed in speedmap (speed per sample times time_spent in that sample)
    will accumulate weighted spiking in spkmap (spk per sample for each ROI times time_spent in that sample)
    also counts number of temporal samples per spatial bin in "count"
    """
    # For each behavioral sample
    for sample in nb.prange(len(behaveTrialIdx)):
        # If mouse is fast enough and time between behavioral sample and imaging frame is within cutoff,
        if (behaveSpeed[sample] > speedThreshold) and (behaveSpeed[sample] < speedMaxThreshold) and (distBehaveToFrame[sample] < distCutoff):
            # add time spent in that trial/position to occupancy map
            occmap[behaveTrialIdx[sample]][behavePositionBin[sample]] += sampleDuration[sample]
            # and speed in that trial/position to speedmap
            speedmap[behaveTrialIdx[sample]][behavePositionBin[sample]] += behaveSpeed[sample] * sampleDuration[sample]
            # add spikes (usually deconvolved spike rate for each ROI) in that trial/position to spkmap
            spkmap[behaveTrialIdx[sample]][behavePositionBin[sample]] += spks[idxBehaveToFrame[sample]] * sampleDuration[sample]
            # add to count to indicate that samples were collected there
            count[behaveTrialIdx[sample]][behavePositionBin[sample]] += 1


@nb.njit(parallel=True)
def getAverageFramePosition(behavePosition, behaveSpeed, speedThreshold, idxBehaveToFrame, distBehaveToFrame, distCutoff, frame_position, count):
    """
    get the position of each frame by averaging across positions within a sample
    """
    for sample in nb.prange(len(behavePosition)):
        if (distBehaveToFrame[sample] < distCutoff) and (behaveSpeed[sample] > speedThreshold):
            frame_position[idxBehaveToFrame[sample]] += behavePosition[sample]
            count[idxBehaveToFrame[sample]] += 1


@nb.njit(parallel=True)
def getAverageFrameSpeed(behaveSpeed, speedThreshold, idxBehaveToFrame, distBehaveToFrame, distCutoff, frame_speed, count):
    """
    get the speed of each frame by averaging across speeds within a sample
    """
    for sample in nb.prange(len(behaveSpeed)):
        if (distBehaveToFrame[sample] < distCutoff) and (behaveSpeed[sample] > speedThreshold):
            frame_speed[idxBehaveToFrame[sample]] += behaveSpeed[sample]
            count[idxBehaveToFrame[sample]] += 1


@nb.njit(parallel=True)
def getMap(valueToSum, trialidx, positionbin, smap):
    """
    this is the fastest way to get a single summation map
    -- accepts 1d arrays value, trialidx, positionbin of the same size --
    -- shape determines the number of trials and position bins (they might not all be represented in trialidx or positionbin, or we could just do np.max()) --
    -- each value represents some number to be summed as a function of which trial it was in and which positionbin it was in --
    """
    for sample in nb.prange(len(valueToSum)):
        smap[trialidx[sample]][positionbin[sample]] += valueToSum[sample]


def correctMap(smap, amap, raise_error=False):
    """
    divide amap by smap with broadcasting where smap isn't 0 (with some handling of other cases)

    amap: [N, M, ...] "average map" where the values will be divided by relevant value in smap
    smap: [N, M] "summation map" which is used to divide out values in amap

    Why?
    ----
    amap is usually spiking activity or speed, and smap is usually occupancy. To convert temporal recordings
    to spatial maps, I start by summing up the values of speed/spiking in each position, along with summing
    up the time spent in each position. Then, with this method, I divide the summed speed/spiking by the time
    spent, to get an average (weighted) speed or spiking.

    correct amap by smap (where amap[i, j, r] output is set to amap[i, j, r] / smap[i, j] if smap[i, j]>0)

    if raise_error=True, then:
    - if smap[i, j] is 0 and amap[i, j, r]>0 for any r, will generate an error
    - if smap[i, j] is nan and amap[i, j, r] is not nan for any r, will generate an error
    otherwise,
    - sets amap to 0 wherever smap is 0
    - sets amap to nan wherever smap is nan

    function:
    correct a summation map (amap) by time spent (smap) if they were computed separately and the summation map should be averaged across time
    """
    zero_check = smap == 0
    nan_check = np.isnan(smap)
    if raise_error:
        if np.any(amap[zero_check] > 0):
            raise ValueError("amap was greater than 0 where smap was 0 in at least one location")
        if np.any(~np.isnan(amap[nan_check])):
            raise ValueError("amap was not nan where smap was nan in at least one location")
    else:
        amap[zero_check] = 0
        amap[nan_check] = np.nan

    # correct amap by smap and return corrected amap
    _numba_correctMap(smap, amap)
    return amap


@nb.njit(parallel=True)
def _numba_correctMap(smap, amap):
    """
    correct amap by smap (where amap[i, j, r] output is set to amap[i, j, r] / smap[i, j] if smap[i, j]>0)
    """
    for t in nb.prange(smap.shape[0]):
        for p in nb.prange(smap.shape[1]):
            if smap[t, p] > 0:
                amap[t, p] /= smap[t, p]


# ============================================================================================================================================
