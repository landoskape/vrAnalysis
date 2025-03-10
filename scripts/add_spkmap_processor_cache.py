import shutil
from tqdm import tqdm
import numpy as np
from vrAnalysis2.database import get_database
from vrAnalysis2.processors.spkmaps import SpkmapProcessor, get_spkmap_params
from vrAnalysis2.sessions import create_b2session

# Go through all the sessions with imaging
sessiondb = get_database("vrSessions")
ises = sessiondb.iter_sessions(imaging=True)


def cache_processed_maps(ises, params_type="default", force_recompute: bool = False, max_gb=1000):
    params = get_spkmap_params(params_type, updates={"autosave": True})
    total_gb = 0
    progress = tqdm(ises, desc="Processing sessions...", leave=True)
    for session in progress:
        spkmap_processor = SpkmapProcessor(session, params=params)
        # Use this to get the maps (which will save them if they don't exist with the current params)
        maps = spkmap_processor.get_processed_maps(clear_one_cache=True, force_recompute=force_recompute)
        total_gb += maps.nbytes() / 1024**3
        progress.set_postfix(total_gb=total_gb)
        if total_gb > max_gb:
            print("Reached max GB, stopping.")
            break


def cache_env_maps(ises, params_type="default", force_recompute: bool = False, max_gb=1000):
    params = get_spkmap_params(params_type, updates={"autosave": True})
    total_gb = 0
    progress = tqdm(ises, desc="Processing sessions...", leave=True)
    for session in progress:
        spkmap_processor = SpkmapProcessor(session, params=params)
        # Use this to get the maps (which will save them if they don't exist with the current params)
        maps = spkmap_processor.get_env_maps(use_session_filters=False, force_recompute=force_recompute)
        total_gb += maps.nbytes() / 1024**3
        progress.set_postfix(total_gb=total_gb)
        if total_gb > max_gb:
            print("Reached max GB, stopping.")
            break


def cache_reliability(ises, params_type="default", force_recompute: bool = False):
    params = get_spkmap_params(params_type, updates={"autosave": True})
    progress = tqdm(ises, desc="Processing sessions...", leave=True)
    for session in progress:
        spkmap_processor = SpkmapProcessor(session, params=params)
        # Use this to get the maps (which will save them if they don't exist with the current params)
        _ = spkmap_processor.get_reliability(use_session_filters=False, force_recompute=force_recompute)


def clear_cache(ises):
    for session in tqdm(ises, desc="Clearing cache...", leave=True):
        spkmap_processor = SpkmapProcessor(session)
        cache_dir = spkmap_processor.cache_directory()
        if cache_dir.exists():
            print("Removing: ", cache_dir)
            print("rmtree commented out, be sure you want to do this!!!!")
            # shutil.rmtree(cache_dir)


if __name__ == "__main__":
    # # # # # clear_cache(ises)
    force_recompute = True
    cache_processed_maps(ises, params_type="default", force_recompute=force_recompute)
    cache_processed_maps(ises, params_type="smoothed", force_recompute=force_recompute)
    cache_env_maps(ises, params_type="default", force_recompute=force_recompute)
    cache_env_maps(ises, params_type="smoothed", force_recompute=force_recompute)
    cache_reliability(ises, params_type="default", force_recompute=force_recompute)
    cache_reliability(ises, params_type="smoothed", force_recompute=force_recompute)
