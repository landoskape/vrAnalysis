import shutil
from tqdm import tqdm
from vrAnalysis.database import get_database
from vrAnalysis.processors.spkmaps import SpkmapProcessor, get_spkmap_params
from vrAnalysis.helpers import get_confirmation


def cache_raw_maps(ises, params_type="default", force_recompute: bool = False, max_gb=1000):
    params = get_spkmap_params(params_type, updates={"autosave": True})
    total_gb = 0
    progress = tqdm(ises, desc="Processing sessions...", leave=True)
    for session in progress:
        spkmap_processor = SpkmapProcessor(session, params=params)
        # Use this to get the maps (which will save them if they don't exist with the current params)
        maps = spkmap_processor.get_raw_maps(clear_one_cache=True, force_recompute=force_recompute)
        total_gb += maps.nbytes() / 1024**3
        progress.set_postfix(total_gb=total_gb)
        if total_gb > max_gb:
            print("Reached max GB, stopping.")
            break


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


def cache_reliability(ises, params_type="default", reliability_method="leave_one_out", force_recompute: bool = False):
    params = get_spkmap_params(params_type, updates={"reliability_method": reliability_method, "autosave": True})
    progress = tqdm(ises, desc="Processing sessions...", leave=True)
    for session in progress:
        spkmap_processor = SpkmapProcessor(session, params=params)
        # Use this to get the maps (which will save them if they don't exist with the current params)
        _ = spkmap_processor.get_reliability(use_session_filters=False, force_recompute=force_recompute)


def clear_cache(ises):
    confirmation = get_confirmation("Are you really really sure you want to clear the cache?")
    if confirmation:
        for session in tqdm(ises, desc="Clearing cache...", leave=True):
            spkmap_processor = SpkmapProcessor(session)
            cache_dir = spkmap_processor.cache_directory()
            if cache_dir.exists():
                print("Removing: ", cache_dir)
                shutil.rmtree(cache_dir)


if __name__ == "__main__":
    sessiondb = get_database("vrSessions")
    ises = sessiondb.iter_sessions(imaging=True)

    # For clearing everything that's been cached
    clear_the_whole_cache = False
    if clear_the_whole_cache:
        confirmation = get_confirmation("Are you sure you want to clear the whole cache?")
        if confirmation:
            clear_cache(ises)

    # For computing all the relevant cache
    force_recompute = False
    for spks_type in ["significant", "oasis"]:
        for session in ises:
            session.update_params(spks_type=spks_type)
        cache_raw_maps(ises, params_type="default", force_recompute=force_recompute)
        for params_type in ["default", "smoothed"]:
            cache_processed_maps(ises, params_type=params_type, force_recompute=force_recompute)
            cache_env_maps(ises, params_type=params_type, force_recompute=force_recompute)
            for reliability_method in ["leave_one_out", "mse", "correlation"]:
                cache_reliability(ises, params_type=params_type, reliability_method=reliability_method, force_recompute=force_recompute)
