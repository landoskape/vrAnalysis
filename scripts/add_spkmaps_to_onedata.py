from tqdm import tqdm
import numpy as np
from _old_vrAnalysis import helpers
from _old_vrAnalysis import database
from _old_vrAnalysis.analysis import placeCellSingleSession

sessiondb = database.vrDatabase("vrSessions")

# Go through all the sessions with imaging
ises = sessiondb.iterSessions(imaging=True)


# in time and in their spkmaps
def get_fluorescence_data(ses, fluor_type):
    if fluor_type == "raw":
        return ses.loadfcorr().T
    elif fluor_type == "deconvolved":
        return ses.loadone("mpci.roiActivityDeconvolved")
    elif fluor_type == "oasis":
        return ses.loadone("mpci.roiActivityDeconvolvedOasis")
    elif fluor_type == "significant":
        return ses.loadone("mpci.roiSignificantFluorescence", sparse=True).toarray()
    else:
        raise ValueError(f"Didn't recognized fluor_type! Recieved: {fluor_type}")


def get_fluorescence_onename(fluor_type):
    if fluor_type == "raw":
        return "fcorr"
    elif fluor_type == "deconvolved":
        return "mpci.roiActivityDeconvolved"
    elif fluor_type == "oasis":
        return "mpci.roiActivityDeconvolvedOasis"
    elif fluor_type == "significant":
        return "mpci.roiSignificantFluorescence"
    else:
        raise ValueError(f"Didn't recognized fluor_type! Recieved: {fluor_type}")


def get_session_environments(ses):
    trial_envnum = ses.loadone("trials.environmentIndex")
    return np.unique(trial_envnum)


def get_spkmap(ses, fluor_data, params):
    pcss = placeCellSingleSession(ses, autoload=False, keep_planes=params["keep_planes"])
    kwargs = {
        "distStep": params["distStep"],
        "onefile": fluor_data,
        "speedThreshold": params["speedThreshold"],
        "standardizeSpks": params["standardizeSpks"],
        "speedSmoothing": params["speedSmoothing"],
    }
    rawspkmap = helpers.getBehaviorAndSpikeMaps(ses, **kwargs)[3]
    spkmaps = pcss.get_spkmap(average=False, trials="full", pop_nan=False, rawspkmap=rawspkmap, smooth=params["smoothing"])
    return spkmaps


def get_params(pcss):
    params = dict(
        keep_planes=pcss.keep_planes,
        distStep=pcss.distStep,
        speedThreshold=pcss.speedThreshold,
        standardizeSpks=pcss.standardizeSpks,
        speedSmoothing=None,
        smoothing=0.3,
    )
    return params


if __name__ == "__main__":
    keep_planes = [1, 2, 3, 4]
    fluor_types = ["raw", "deconvolved", "oasis", "significant"]

    for ses in tqdm(ises, desc="Processing sessions...", leave=True):
        pcss = placeCellSingleSession(ses, autoload=False, keep_planes=keep_planes)
        params = get_params(pcss)
        params_no_smoothing = params.copy()
        params_no_smoothing.update({"smoothing": None})

        for fluor_type in fluor_types:
            fluor_data = get_fluorescence_data(ses, fluor_type)
            spkmaps = get_spkmap(ses, fluor_data, params)
            fluor_one_name = get_fluorescence_onename(fluor_type)
            ses.save_spkmaps(fluor_one_name, spkmaps, pcss.environments, params=params)

        ses.clearBuffer()
