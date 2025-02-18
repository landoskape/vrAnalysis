import shutil
from tqdm import tqdm
import numpy as np
from vrAnalysis import database
from vrAnalysis2.processors.spkmaps import SpkmapProcessor, SpkmapParams
from vrAnalysis2.sessions import create_b2session

# Go through all the sessions with imaging
sessiondb = database.vrDatabase("vrSessions")
ises = sessiondb.iterSessions(imaging=True)

# Default Params
default_spkmap_params = SpkmapParams(
    dist_step=1.0,
    speed_threshold=1.0,
    speed_max_allowed=np.inf,
    full_trial_flexibility=3.0,
    standardize_spks=True,
    smooth_width=1.0,
    autosave=True,
)


def cache_processed_maps(ises, spks_type, max_gb=1000):
    total_gb = 0
    for ses in tqdm(ises, desc="Processing sessions...", leave=True):
        session = create_b2session(ses.mouseName, ses.dateString, ses.sessionid, spks_type)
        spkmap_processor = SpkmapProcessor(session, params=default_spkmap_params)
        # Use this to get the maps (which will save them if they don't exist with the current params)
        maps = spkmap_processor.maps(clear_one_cache=True)
        total_gb += maps.nbytes() / 1024**3
        print("Finished with session: ", session, "Current GB: ", total_gb)
        if total_gb > max_gb:
            print("Total GB: ", total_gb)
            break


def clear_cache(ises):
    for ses in tqdm(ises, desc="Clearing cache...", leave=True):
        session = create_b2session(ses.mouseName, ses.dateString, ses.sessionid)
        spkmap_processor = SpkmapProcessor(session, params=default_spkmap_params)
        cache_dir = spkmap_processor.cache_directory()
        if cache_dir.exists():
            print("Removing: ", cache_dir)
            print("rmtree commented out, be sure you want to do this!!!!")
            # shutil.rmtree(cache_dir)


if __name__ == "__main__":
    # # # # # clear_cache(ises)
    cache_processed_maps(ises, "significant")
