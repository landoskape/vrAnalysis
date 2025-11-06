from tqdm import tqdm
from argparse import ArgumentParser
from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

import os, sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from _old_vrAnalysis import analysis
from _old_vrAnalysis import helpers
from _old_vrAnalysis import database
from _old_vrAnalysis import tracking
from _old_vrAnalysis import fileManagement as fm


def handle_args():
    parser = ArgumentParser(description="Compute correlations between sessions for tracked cells")
    parser.add_argument("--num_latents", type=int, default=120000, help="Number of latents to use for UMAP (divided across all mice)")
    parser.add_argument(
        "--make_master_mapper",
        type=helpers.argbool,
        default=False,
        help="Whether to create a master mapper from sampled ROIs from each mouse. (Default: False)",
    )
    parser.add_argument(
        "--perform_embeddings",
        type=helpers.argbool,
        default=False,
        help="Whether to use the master umap mapper to embed data from all mice. (Default: False)",
    )
    return parser.parse_args()


def umap_embedding_filename():
    directory = fm.analysisPath() / "roicat"
    if not directory.is_dir():
        directory.mkdir(parents=True)
    return fm.analysisPath() / "roicat" / "master_umap_mapper.npy"


def get_mice():
    mousedb = database.vrDatabase("vrMice")
    tracked_mice = list(mousedb.getTable(tracked=True)["mouseName"])
    ignore_mice = []
    use_mice = [mouse for mouse in tracked_mice if mouse not in ignore_mice]
    return use_mice


def get_random_latents(mouse_name, num_rois):
    track = tracking.tracker(mouse_name)
    pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=[1, 2, 3, 4], speedThreshold=1)
    max_keep_each = num_rois // len(pcm.pcss)
    latents = []
    for pcss in pcm.pcss:
        idx_plane = pcss.get_plane_idx()
        c_latents = pcss.get_roicat_latents()[idx_plane]
        num_keep = min(c_latents.shape[0], max_keep_each)
        idx_keep = np.random.choice(range(c_latents.shape[0]), num_keep, replace=False)
        latents.append(c_latents[idx_keep])
    latents = np.concatenate(latents, axis=0)
    return latents


def save_master_mapper(latents):
    umap = UMAP(n_neighbors=25, n_components=2, n_epochs=400, verbose=True, densmap=False)
    umap.fit(latents)
    fpath = umap_embedding_filename()
    with open(fpath, "wb") as f:
        dump(umap, f)


def embed_mouse_data(mouse_name, umap):
    track = tracking.tracker(mouse_name)
    pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=[1, 2, 3, 4], speedThreshold=1)
    for pcss in pcm.pcss:
        latents = pcss.get_roicat_latents()
        embeddings = umap.transform(latents)
        np.save(pcss.vrexp.roicatPath() / "master_umap_embeddings.npy", embeddings)


if __name__ == "__main__":
    args = handle_args()
    use_mice = get_mice()
    num_mice = len(use_mice)
    max_per_mouse = args.num_latents // num_mice
    if args.make_master_mapper:
        latents = []
        for mouse_name in tqdm(use_mice, desc="Getting latents"):
            latents.append(get_random_latents(use_mice[0], max_per_mouse))
        latents = np.concatenate(latents, axis=0)

        save_master_mapper(latents)

    if args.perform_embeddings:
        with open(umap_embedding_filename(), "rb") as f:
            umap = load(f)

        for mouse_name in tqdm(use_mice, desc="Embedding data"):
            embed_mouse_data(mouse_name, umap)
