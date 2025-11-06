import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .register import B2Registration


# ---------------------------------------------------------------------------------------------------
# ------------------------------------- behavior processing methods ---------------------------------
# ---------------------------------------------------------------------------------------------------
def standard_behavior(b2registration: "B2Registration") -> "B2Registration":
    expInfo = b2registration.vr_file["expInfo"]
    trialInfo = b2registration.vr_file["trialInfo"]
    num_values_per_trial = np.diff(trialInfo.time.tocsr().indptr)
    valid_trials = np.where(num_values_per_trial > 0)[0]
    numTrials = len(valid_trials)
    assert np.array_equal(valid_trials, np.arange(numTrials)), "valid_trials is not a range from 0 to numTrials"
    b2registration.set_value("numTrials", numTrials)

    # trialInfo contains sparse matrices of size (maxTrials, maxSamples), where numTrials<maxTrials and numSamples<maxSamples
    nzindex = b2registration.create_index(b2registration.convert_dense(trialInfo.time))
    timeStamps = b2registration.get_vr_data(b2registration.convert_dense(trialInfo.time), nzindex)
    roomPosition = b2registration.get_vr_data(b2registration.convert_dense(trialInfo.roomPosition), nzindex)

    # oneData with behave prefix is a (numBehavioralSamples, ) shaped array conveying information about the state of VR
    numTimeStamps = np.array([len(t) for t in timeStamps])  # list of number of behavioral timestamps in each trial
    behaveTimeStamps = np.concatenate(timeStamps)  # time stamp associated with each behavioral sample
    behavePosition = np.concatenate(roomPosition)  # virtual position associated with each behavioral sample
    b2registration.set_value("numBehaveTimestamps", len(behaveTimeStamps))

    # Check shapes and sizes
    assert behaveTimeStamps.ndim == 1, "behaveTimeStamps is not a 1-d array!"
    assert behaveTimeStamps.shape == behavePosition.shape, "behave oneData arrays do not have the same shape!"

    # oneData with trial prefix is a (numTrials,) shaped array conveying information about the state on each trial
    trialStartFrame = np.array([0, *np.cumsum(numTimeStamps)[:-1]]).astype(np.int64)
    trialEnvironmentIndex = (
        b2registration.convert_dense(trialInfo.vrEnvIdx).astype(np.int16)
        if "vrEnvIdx" in trialInfo._fieldnames
        else -1 * np.ones(b2registration.get_value("numTrials"), dtype=np.int16)
    )
    trialRoomLength = expInfo.roomLength[: b2registration.get_value("numTrials")]
    trialMovementGain = expInfo.mvmtGain[: b2registration.get_value("numTrials")]
    trialRewardPosition = b2registration.convert_dense(trialInfo.rewardPosition)
    trialRewardTolerance = b2registration.convert_dense(trialInfo.rewardTolerance)
    trialRewardAvailability = b2registration.convert_dense(trialInfo.rewardAvailable).astype(np.bool_)
    rewardDelivery = (
        b2registration.convert_dense(trialInfo.rewardDeliveryFrame).astype(np.int64) - 1
    )  # get reward delivery frame (frame within trial) first (will be -1 if no reward delivered)
    # adjust frame count to behave arrays
    trialRewardDelivery = np.array(
        [
            (rewardDelivery + np.sum(numTimeStamps[:trialIdx]) if rewardDelivery >= 0 else rewardDelivery)
            for (trialIdx, rewardDelivery) in enumerate(rewardDelivery)
        ]
    )
    trialActiveLicking = b2registration.convert_dense(trialInfo.activeLicking).astype(np.bool_)
    trialActiveStopping = b2registration.convert_dense(trialInfo.activeStopping).astype(np.bool_)

    # Check shapes and sizes
    assert trialEnvironmentIndex.ndim == 1 and len(trialEnvironmentIndex) == b2registration.get_value(
        "numTrials"
    ), "trialEnvironmentIndex is not a (numTrials,) shaped array!"
    assert (
        trialStartFrame.shape
        == trialEnvironmentIndex.shape
        == trialRoomLength.shape
        == trialMovementGain.shape
        == trialRewardPosition.shape
        == trialRewardTolerance.shape
        == trialRewardAvailability.shape
        == trialRewardDelivery.shape
        == trialActiveLicking.shape
        == trialActiveStopping.shape
    ), "trial oneData arrays do not have the same shape!"

    # oneData with lick prefix is a (numLicks,) shaped array containing information about each lick during VR behavior
    licks = b2registration.get_vr_data(b2registration.convert_dense(trialInfo.lick), nzindex)
    lickFrames = [np.nonzero(licks)[0] for licks in licks]
    lickCounts = np.concatenate([licks[lickFrames] for (licks, lickFrames) in zip(licks, lickFrames)])
    lickTrials = np.concatenate([tidx * np.ones_like(lickFrames) for (tidx, lickFrames) in enumerate(lickFrames)])
    lickFrames = np.concatenate(lickFrames)
    if np.sum(lickCounts) > 0:
        lickFramesRepeat = np.concatenate([lf * np.ones(lc, dtype=np.uint8) for (lf, lc) in zip(lickFrames, lickCounts)])
        lickTrialsRepeat = np.concatenate([lt * np.ones(lc, dtype=np.uint8) for (lt, lc) in zip(lickTrials, lickCounts)])
        lickCountsRepeat = np.concatenate([lc * np.ones(lc, dtype=np.uint8) for (lc, lc) in zip(lickCounts, lickCounts)])
        lickBehaveSample = lickFramesRepeat + np.array([np.sum(numTimeStamps[:trialIdx]) for trialIdx in lickTrialsRepeat])

        assert len(lickBehaveSample) == np.sum(
            lickCounts
        ), "the number of licks counted by vrBehavior is not equal to the length of the lickBehaveSample vector!"
        assert lickBehaveSample.ndim == 1, "lickBehaveIndex is not a 1-d array!"
        assert (
            0 <= np.max(lickBehaveSample) <= len(behaveTimeStamps)
        ), "lickBehaveSample contains index outside range of possible indices for behaveTimeStamps"
    else:
        # No licks found -- create empty array
        lickBehaveSample = np.array([], dtype=np.uint8)

    # Align behavioral timestamp data to timeline -- shift each trials timestamps so that they start at the time of the first photodiode flip (which is reliably detected)
    trialStartOffsets = behaveTimeStamps[trialStartFrame] - b2registration.loadone("trials.startTimes")  # get offset
    behaveTimeStamps = np.concatenate(
        [bts - trialStartOffsets[tidx] for (tidx, bts) in enumerate(b2registration.group_behave_by_trial(behaveTimeStamps, trialStartFrame))]
    )

    # Save behave onedata
    b2registration.saveone(behaveTimeStamps, "positionTracking.times")
    b2registration.saveone(behavePosition, "positionTracking.position")

    # Save trial onedata
    b2registration.saveone(trialStartFrame, "trials.positionTracking")
    b2registration.saveone(trialEnvironmentIndex, "trials.environmentIndex")
    b2registration.saveone(trialRoomLength, "trials.roomlength")
    b2registration.saveone(trialMovementGain, "trials.movementGain")
    b2registration.saveone(trialRewardPosition, "trials.rewardPosition")
    b2registration.saveone(trialRewardTolerance, "trials.rewardZoneHalfwidth")
    b2registration.saveone(trialRewardAvailability, "trials.rewardAvailability")
    b2registration.saveone(trialRewardDelivery, "trials.rewardPositionTracking")
    b2registration.saveone(trialActiveLicking, "trials.activeLicking")
    b2registration.saveone(trialActiveStopping, "trials.activeStopping")

    # Save lick onedata
    b2registration.saveone(lickBehaveSample, "licksTracking.positionTracking")

    return b2registration


def cr_hippocannula_behavior(b2registration: "B2Registration") -> "B2Registration":
    trialInfo = b2registration.vr_file["TRIAL"]
    expInfo = b2registration.vr_file["EXP"]

    numTrials = trialInfo.info.no
    nonNanSamples = np.sum(~np.isnan(trialInfo.time[:, 0]))
    assert numTrials == nonNanSamples, f"# trials {trialInfo.info.no} isn't equal to non-nan first time samples {nonNanSamples}"
    b2registration.set_value("numTrials", numTrials)

    # trialInfo contains sparse matrices of size (maxTrials, maxSamples), where numTrials<maxTrials and numSamples<maxSamples
    nzindex = b2registration.create_index(b2registration.convert_dense(trialInfo.time))
    timeStamps = b2registration.get_vr_data(b2registration.convert_dense(trialInfo.time), nzindex)
    roomPosition = b2registration.get_vr_data(b2registration.convert_dense(trialInfo.roomPosition), nzindex)

    # oneData with behave prefix is a (numBehavioralSamples, ) shaped array conveying information about the state of VR
    numTimeStamps = np.array([len(t) for t in timeStamps])  # list of number of behavioral timestamps in each trial
    behaveTimeStamps = np.concatenate(timeStamps)  # time stamp associated with each behavioral sample
    behavePosition = np.concatenate(roomPosition)  # virtual position associated with each behavioral sample
    b2registration.set_value("numBehaveTimestamps", len(behaveTimeStamps))

    # Check shapes and sizes
    assert behaveTimeStamps.ndim == 1, "behaveTimeStamps is not a 1-d array!"
    assert behaveTimeStamps.shape == behavePosition.shape, "behave oneData arrays do not have the same shape!"

    # oneData with trial prefix is a (numTrials,) shaped array conveying information about the state on each trial
    trialStartFrame = np.array([0, *np.cumsum(numTimeStamps)[:-1]]).astype(np.int64)
    trialEnvironmentIndex = (
        b2registration.convert_dense(trialInfo.vrEnvIdx).astype(np.int16)
        if "vrEnvIdx" in trialInfo._fieldnames
        else -1 * np.ones(b2registration.get_value("numTrials"), dtype=np.int16)
    )
    trialRoomLength = np.ones(b2registration.get_value("numTrials")) * expInfo.roomLength
    trialMovementGain = np.ones(b2registration.get_value("numTrials"))  # mvmt gain always one
    trialRewardPosition = b2registration.convert_dense(trialInfo.trialRewPos)
    trialRewardTolerance = b2registration.convert_dense(expInfo.rewPosTolerance * np.ones(b2registration.get_value("numTrials")))
    trialRewardAvailability = b2registration.convert_dense(trialInfo.trialRewAvailable).astype(np.bool_)
    rewardDelivery = b2registration.convert_dense(trialInfo.trialRewDelivery)
    rewardDelivery[np.isnan(rewardDelivery)] = 0  # about to be (-1), indicating no reward delivered
    rewardDelivery = rewardDelivery.astype(np.int64) - 1  # get reward delivery frame (frame within trial) first (will be -1 if no reward delivered)

    # adjust frame count to behave arrays
    trialRewardDelivery = np.array(
        [
            (rewardDelivery + np.sum(numTimeStamps[:trialIdx]) if rewardDelivery >= 0 else rewardDelivery)
            for (trialIdx, rewardDelivery) in enumerate(rewardDelivery)
        ]
    )
    trialActiveLicking = b2registration.convert_dense(trialInfo.trialActiveLicking).astype(np.bool_)
    trialActiveStopping = b2registration.convert_dense(trialInfo.trialActiveStopping).astype(np.bool_)

    # Check shapes and sizes
    assert trialEnvironmentIndex.ndim == 1 and len(trialEnvironmentIndex) == b2registration.get_value(
        "numTrials"
    ), "trialEnvironmentIndex is not a (numTrials,) shaped array!"
    assert (
        trialStartFrame.shape
        == trialEnvironmentIndex.shape
        == trialRoomLength.shape
        == trialMovementGain.shape
        == trialRewardPosition.shape
        == trialRewardTolerance.shape
        == trialRewardAvailability.shape
        == trialRewardDelivery.shape
        == trialActiveLicking.shape
        == trialActiveStopping.shape
    ), "trial oneData arrays do not have the same shape!"

    # oneData with lick prefix is a (numLicks,) shaped array containing information about each lick during VR behavior
    licks = [vrd.astype(np.int16) for vrd in b2registration.get_vr_data(b2registration.convert_dense(trialInfo.lick), nzindex)]
    lickFrames = [np.nonzero(licks)[0] for licks in licks]
    lickCounts = np.concatenate([licks[lickFrames] for (licks, lickFrames) in zip(licks, lickFrames)])
    lickTrials = np.concatenate([tidx * np.ones_like(lickFrames) for (tidx, lickFrames) in enumerate(lickFrames)])
    lickFrames = np.concatenate(lickFrames)
    if np.sum(lickCounts) > 0:
        lickFramesRepeat = np.concatenate([lf * np.ones(lc, dtype=np.uint8) for (lf, lc) in zip(lickFrames, lickCounts)])
        lickTrialsRepeat = np.concatenate([lt * np.ones(lc, dtype=np.uint8) for (lt, lc) in zip(lickTrials, lickCounts)])
        lickCountsRepeat = np.concatenate([lc * np.ones(lc, dtype=np.uint8) for (lc, lc) in zip(lickCounts, lickCounts)])
        lickBehaveSample = lickFramesRepeat + np.array([np.sum(numTimeStamps[:trialIdx]) for trialIdx in lickTrialsRepeat])

        assert len(lickBehaveSample) == np.sum(
            lickCounts
        ), "the number of licks counted by vrBehavior is not equal to the length of the lickBehaveSample vector!"
        assert lickBehaveSample.ndim == 1, "lickBehaveIndex is not a 1-d array!"
        assert (
            0 <= np.max(lickBehaveSample) <= len(behaveTimeStamps)
        ), "lickBehaveSample contains index outside range of possible indices for behaveTimeStamps"
    else:
        # No licks found -- create empty array
        lickBehaveSample = np.array([], dtype=np.uint8)

    # Align behavioral timestamp data to timeline -- shift each trials timestamps so that they start at the time of the first photodiode flip (which is reliably detected)
    trialStartOffsets = behaveTimeStamps[trialStartFrame] - b2registration.loadone("trials.startTimes")  # get offset
    behaveTimeStamps = np.concatenate(
        [bts - trialStartOffsets[tidx] for (tidx, bts) in enumerate(b2registration.group_behave_by_trial(behaveTimeStamps, trialStartFrame))]
    )

    # Save behave onedata
    b2registration.saveone(behaveTimeStamps, "positionTracking.times")
    b2registration.saveone(behavePosition, "positionTracking.position")

    # Save trial onedata
    b2registration.saveone(trialStartFrame, "trials.positionTracking")
    b2registration.saveone(trialEnvironmentIndex, "trials.environmentIndex")
    b2registration.saveone(trialRoomLength, "trials.roomlength")
    b2registration.saveone(trialMovementGain, "trials.movementGain")
    b2registration.saveone(trialRewardPosition, "trials.rewardPosition")
    b2registration.saveone(trialRewardTolerance, "trials.rewardZoneHalfwidth")
    b2registration.saveone(trialRewardAvailability, "trials.rewardAvailability")
    b2registration.saveone(trialRewardDelivery, "trials.rewardPositionTracking")
    b2registration.saveone(trialActiveLicking, "trials.activeLicking")
    b2registration.saveone(trialActiveStopping, "trials.activeStopping")

    # Save lick onedata
    b2registration.saveone(lickBehaveSample, "licksTracking.positionTracking")

    return b2registration


"""
BEHAVIOR_PROCESSING: Dictionary of behavior processing functions.

These reflect the different versions of the vrControl software that was used to collect the behavior data.
Because the behavioral data was collected in different ways, we need to process it differently to achieve
the same results structure. 
"""
BEHAVIOR_PROCESSING = {
    1: standard_behavior,
    2: cr_hippocannula_behavior,
}


def register_behavior(b2registration: "B2Registration", behavior_type: int) -> "B2Registration":
    """Register behavior for a given behavior type.

    This is a dispatcher function that calls the appropriate behavior processing function based on the behavior type.

    Parameters:
    ----------
    b2registration: B2Registration object
    behavior_type: int, the behavior type to register

    Returns
    -------
    B2Registration object with behavior registered
    """
    if behavior_type not in BEHAVIOR_PROCESSING.keys():
        raise ValueError(f"Behavior type {behavior_type} not supported. Supported types are: {list(BEHAVIOR_PROCESSING.keys())}.")
    return BEHAVIOR_PROCESSING[behavior_type](b2registration)
