import shutil
from tqdm import tqdm
import numpy as np
from vrAnalysis import database
from vrAnalysis2.processors.spkmaps import SpkmapProcessor, get_spkmap_params
from vrAnalysis2.sessions import create_b2session

# Go through all the sessions with imaging
sessiondb = database.vrDatabase("vrSessions")
ises = sessiondb.iterSessions(imaging=True)


def cache_processed_maps(ises, spks_type, params_type="default", max_gb=1000):
    params = get_spkmap_params(params_type, updates={"autosave": True})
    total_gb = 0
    progress = tqdm(ises, desc="Processing sessions...", leave=True)
    for ses in progress:
        session = create_b2session(ses.mouseName, ses.dateString, ses.sessionid, dict(spks_type=spks_type))
        spkmap_processor = SpkmapProcessor(session, params=params)
        # Use this to get the maps (which will save them if they don't exist with the current params)
        maps = spkmap_processor.maps(clear_one_cache=True)
        total_gb += maps.nbytes() / 1024**3
        progress.set_postfix(total_gb=total_gb)
        if total_gb > max_gb:
            print("Reached max GB, stopping.")
            break


def clear_cache(ises):
    for ses in tqdm(ises, desc="Clearing cache...", leave=True):
        session = create_b2session(ses.mouseName, ses.dateString, ses.sessionid)
        spkmap_processor = SpkmapProcessor(session)
        cache_dir = spkmap_processor.cache_directory()
        if cache_dir.exists():
            print("Removing: ", cache_dir)
            print("rmtree commented out, be sure you want to do this!!!!")
            # shutil.rmtree(cache_dir)


if __name__ == "__main__":
    # # # # # clear_cache(ises)
    cache_processed_maps(ises, "significant", params_type="default")
    cache_processed_maps(ises, "significant", params_type="smoothed")
