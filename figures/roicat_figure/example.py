# add path that contains the vrAnalysis package
import sys
import os
from time import time
from argparse import ArgumentParser
import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

# GUI-related modules
import napari
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QGraphicsProxyWidget,
    QSlider,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QShortcut,
)
from PyQt5.QtGui import QKeySequence

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(mainPath)

from vrAnalysis import tracking
from vrAnalysis.helpers import named_transpose, edge2center, CurrentSelection, SliderSelector
from vrAnalysis import analysis


def handle_inputs():
    parser = ArgumentParser(description="do pcm analyses")
    parser.add_argument("--mouse-name", type=str, help="which mouse to run the example for")
    return parser.parse_args()


def _check_no_duplicates(roi_idx_match, roi_idx_nomatch):
    no_duplicates = []
    for im, inm in zip(roi_idx_match, roi_idx_nomatch):
        c_duplicates = im.reshape(2, -1, 1) == inm.reshape(2, 1, -1)
        no_duplicates.append(~np.any(c_duplicates, axis=0))
    return no_duplicates


def _get_centroids_match(centroids, idx, prms):
    assert len(prms["idx_ses_pairs"]) == len(idx), "idx doesn't match number of session pairs"
    assert len(centroids) == len(prms["idx_ses"]), "centroids doesn't match number of sessions"

    # prepare lookup table for going from the absolute session index to the relative index in centroids
    session_lookup = lambda ises: {val: idx for idx, val in enumerate(prms["idx_ses"])}[ises]

    # prepare list for centroids of matched pairs
    centroids_matched = []
    for ipair, imatch in zip(prms["idx_ses_pairs"], idx):
        i1 = session_lookup(ipair[0])  # idx to first session in pair
        i2 = session_lookup(ipair[1])  # idx to second session in pair
        c1 = centroids[i1][imatch[0]]  # centroids in first session
        c2 = centroids[i2][imatch[1]]  # centroids in second session
        centroids_matched.append(np.stack((c1, c2)))

    return centroids_matched


def _get_plane_of_pair(roistat, idx, prms):
    assert len(idx) == len(prms["idx_ses_pairs"]), "idx and prms['idx_ses_pairs'] don't have same number of elements"

    # prepare lookup table for going from the absolute session index to the relative index in centroids
    session_lookup = lambda ises: {val: idx for idx, val in enumerate(prms["idx_ses"])}[ises]

    # get plane index for each roi from sessions
    roiPlaneIdx = roistat.get_from_pcss("roiPlaneIdx", prms["idx_ses"])

    # prepare list of roi plane index for each pair
    pair_plane = []
    for ipair, ii in zip(prms["idx_ses_pairs"], idx):
        i1 = session_lookup(ipair[0])
        i2 = session_lookup(ipair[1])
        pp1 = roiPlaneIdx[i1][ii[0]]
        pp2 = roiPlaneIdx[i2][ii[1]]
        if not np.all(pp1 == pp2):
            raise ValueError("Planes of pair don't all match")
        pair_plane.append(pp1)

    return pair_plane


def _get_masks(roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch):
    ROIs = roistat.track.get_ROIs(as_coo=False, idx_ses=prms["idx_ses_pairs"][isespair], keep_planes=roistat.keep_planes)
    ROIs = [sp.sparse.vstack(rois, format="csr") for rois in ROIs]  # concatenate across planes

    roi_match = [
        ROIs[0][[pp_roi_match[isespair][0, idxroipair]]],
        ROIs[1][[pp_roi_match[isespair][1, idxroipair]]],
    ]
    roi_nomatch = [
        ROIs[0][[pp_roi_nomatch[isespair][0, idxroipair]]],
        ROIs[1][[pp_roi_nomatch[isespair][1, idxroipair]]],
    ]

    num_pixels = roi_match[0].shape[1]
    hw = int(np.sqrt(num_pixels))
    roi_match = [r.toarray().reshape(hw, hw) for r in roi_match]
    roi_nomatch = [r.toarray().reshape(hw, hw) for r in roi_nomatch]

    return roi_match, roi_nomatch


def _get_spkmaps(roistat, envnum, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch):
    spkmaps = roistat.get_spkmaps(
        envnum,
        trials="full",
        average=False,
        tracked=False,
        idx_ses=prms["idx_ses_pairs"][isespair],
        by_plane=False,
    )[0]
    spkmap_match = [
        spkmaps[0][pp_roi_match[isespair][0, idxroipair]],
        spkmaps[1][pp_roi_match[isespair][1, idxroipair]],
    ]
    spkmap_nomatch = [
        spkmaps[0][pp_roi_nomatch[isespair][0, idxroipair]],
        spkmaps[1][pp_roi_nomatch[isespair][1, idxroipair]],
    ]
    spkmap_match = [sp.stats.zscore(sm, axis=None, nan_policy="omit") for sm in spkmap_match]
    spkmap_nomatch = [sp.stats.zscore(sm, axis=None, nan_policy="omit") for sm in spkmap_nomatch]
    distedges = roistat.pcss[prms["idx_ses_pairs"][isespair][0]].distedges
    return spkmap_match, spkmap_nomatch, distedges


def prepare_pair_example_data(mouse_name):
    """
    Example of matched / not-matched ROIs in aligned FOVs with color-coded place fields
    """
    # Excellent Possibilities (no plane 0):
    # ATL027, sessions 10/11, Match Idx: [1878 1603] NoMatch Idx: [2433 2652]
    # ATL027, sessions 8/9, Match Idx: [11079  8659] NoMatch Idx: [13208 11206]
    # ATL027, sessions 8/9, Match Idx: [7640 5003] NoMatch Idx: [10729  5743]

    # lost possibility:
    # ATL027, sessions 8/9, they were in rough centroid [310, 60-80], the match had a place field > 180cm.

    # Possibilities (no plane 0):
    # ATL027, sessions 9/10, Match Idx: [11828 13054] NoMatch Idx: [9393 9689]
    # ATL027, sessions 9/10, Match Idx: [4984 5500] NoMatch Idx: [5098 7138]
    # ATL027, sessions 9/11, Match Idx: [265 524] NoMatch Idx: [132 891]

    # Prepare main objects for doing analysis
    track = tracking.tracker(mouse_name)
    roistat = analysis.RoicatStats(track)

    # processing methods for preloaded roistat object
    envnum = roistat.env_selector(envmethod="most")

    # get all sessions with chosen environment
    idx_ses = roistat.idx_ses_selector(envnum, sesmethod="all")

    # pick 4 environments, the later the better, but if there's lots of sessions don't do the last one (imaging quality dropped off in last couple sessions usually)
    if len(idx_ses) > 7:
        idx_ses = idx_ses[-6:-2]
    elif len(idx_ses) > 4:
        idx_ses = idx_ses[-4:]

    # report which environment and sessions are being used
    print("env:", envnum, "idx_ses:", idx_ses)

    # get data for all roicat plots
    kwargs = dict(
        sim_name="sConj",
        cutoffs=(0.6, 0.85),
        both_reliable=False,
    )
    _, corr, tracked, pwdist, _, _, pwind, prms = roistat.make_roicat_comparison(envnum, idx_ses=idx_ses, **kwargs)
    centroids = roistat.track.get_centroids(idx_ses=prms["idx_ses"], cat_planes=True, combine=True, keep_planes=roistat.keep_planes)

    # set parameters for possible pairs
    mincorr = 0.6  # minimum correlation for matched pair
    maxcorr = 0.2  # maximum correlation for non-matched pair
    maxdist = 6  # maximum distance for any example pair

    # set parameters for pair of pairs (distance apart etc.)
    maxdistpp = 16  # maximum distance for pair of pairs
    mindistpp = 6  # minimum distance for pair of pairs

    # get possible matches and not-matches
    idx_possible_match = [np.where(t & (c > mincorr) & (pwd < maxdist))[0] for t, c, pwd in zip(tracked, corr, pwdist)]
    idx_possible_nomatch = [np.where(~t & (c < maxcorr) & (pwd < maxdist))[0] for t, c, pwd in zip(tracked, corr, pwdist)]

    # get possible match/nomatch pair indices
    roi_idx_match = [pwi[:, i] for pwi, i in zip(pwind, idx_possible_match)]
    roi_idx_nomatch = [pwi[:, i] for pwi, i in zip(pwind, idx_possible_nomatch)]

    # get average centroid of pairs
    centroids_match = _get_centroids_match(centroids, roi_idx_match, prms)
    centroids_nomatch = _get_centroids_match(centroids, roi_idx_nomatch, prms)
    acentroid_match = [np.mean(c, axis=0) for c in centroids_match]
    acentroid_nomatch = [np.mean(c, axis=0) for c in centroids_nomatch]

    # get distance between pairs of possible match/nomatch for each pair of sessions
    dist_match_nomatch = [sp.spatial.distance.cdist(am, anm) for am, anm in zip(acentroid_match, acentroid_nomatch)]
    good_distance = [(dmn < maxdistpp) & (dmn > mindistpp) for dmn in dist_match_nomatch]

    # check if pairs include any of the same ROIs
    no_duplicates = _check_no_duplicates(roi_idx_match, roi_idx_nomatch)

    # get plane indices for each pair
    plane_pair_match = _get_plane_of_pair(roistat, roi_idx_match, prms)
    plane_pair_nomatch = _get_plane_of_pair(roistat, roi_idx_nomatch, prms)
    same_plane = [ppm.reshape(-1, 1) == ppn.reshape(1, -1) for ppm, ppn in zip(plane_pair_match, plane_pair_nomatch)]

    # possible pair of pairs boolean array
    ipair_match, ipair_nomatch = named_transpose([np.where(gd & nd & smp) for gd, nd, smp in zip(good_distance, no_duplicates, same_plane)])

    # indices to ROI pairs for good pairs of pairs
    pp_roi_match = [rim[:, ipm] for rim, ipm in zip(roi_idx_match, ipair_match)]
    pp_roi_nomatch = [rinm[:, ipnm] for rinm, ipnm in zip(roi_idx_nomatch, ipair_nomatch)]
    pp_centroid_match = [acm[ipm] for acm, ipm in zip(acentroid_match, ipair_match)]
    pp_centroid_nomatch = [acnm[ipnm] for acnm, ipnm in zip(acentroid_nomatch, ipair_nomatch)]

    # get plane indices for each pair
    plane_pair = _get_plane_of_pair(roistat, pp_roi_match, prms)

    # check that match and no-match pairs are in same plane
    assert all([np.all(pp == ppnm) for pp, ppnm in zip(plane_pair, _get_plane_of_pair(roistat, pp_roi_nomatch, prms))])

    return roistat, envnum, plane_pair, pp_roi_match, pp_roi_nomatch, pp_centroid_match, pp_centroid_nomatch, prms


def print_selection(pp_roi_match, pp_roi_nomatch, prms, isespair, idxroipair):
    """simple method for printing output to terminal"""
    print("Session Pair:", prms["idx_ses_pairs"][isespair], "ROI Pair:", idxroipair)
    print(
        "Match Idx:",
        pp_roi_match[isespair][:, idxroipair],
        "NoMatch Idx:",
        pp_roi_nomatch[isespair][:, idxroipair],
    )


# supporting methods
def _get_centroid(pp_centroid_match, pp_centroid_nomatch, isespair, idxroipair):
    return np.mean(np.stack((pp_centroid_match[isespair][idxroipair], pp_centroid_nomatch[isespair][idxroipair])), axis=0)


def _make_data(
    roistat,
    envnum,
    plane_pair,
    pp_roi_match,
    pp_roi_nomatch,
    pp_centroid_match,
    pp_centroid_nomatch,
    prms,
    isespair,
    idxroipair,
    roi_scale,
    zscore_lim,
):
    # Get centroid
    average_centroid = _get_centroid(pp_centroid_match, pp_centroid_nomatch, isespair, idxroipair)

    spkmaps_match, spkmaps_nomatch, distedges = _get_spkmaps(roistat, envnum, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch)
    extents = [[0, sm.shape[0], distedges[0], distedges[-1]] for sm in spkmaps_match]

    # make heatmaps of FOV
    FOVs = [roistat.track.rundata[plane_pair[isespair][idxroipair]]["aligner"]["ims_registered_nonrigid"][i] for i in prms["idx_ses_pairs"][isespair]]
    roi_match, roi_nomatch = _get_masks(roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch)
    normalize = lambda data: [((d - d.min()) / (d.max() - d.min())).astype(np.float32) for d in data]
    FOVs = normalize(FOVs)
    roi_match = normalize(roi_match)
    roi_nomatch = normalize(roi_nomatch)

    fov_plot = [np.tile(fov.reshape(fov.shape[0], fov.shape[1], 1), (1, 1, 3)) for fov in FOVs]
    fov_plot[0][:, :, 0] += roi_match[0] * roi_scale
    fov_plot[1][:, :, 0] += roi_match[1] * roi_scale
    fov_plot[0][:, :, 1] += roi_nomatch[0] * roi_scale
    fov_plot[1][:, :, 2] += roi_nomatch[1] * roi_scale
    fov_plot = normalize(fov_plot)

    # convert spkmaps into images with proper scaling
    norm = mpl.colors.Normalize(vmin=0, vmax=zscore_lim)
    red_colormap = mpl.colormaps["Reds"]
    blue_colormap = mpl.colormaps["Blues"]
    green_colormap = mpl.colormaps["Greens"]
    spkmaps_match = [red_colormap(norm(sm.T)) for sm in spkmaps_match]
    spkmaps_nomatch = [cmap(norm(snm.T)) for cmap, snm in zip([green_colormap, blue_colormap], spkmaps_nomatch)]

    return fov_plot, average_centroid, spkmaps_match, spkmaps_nomatch, extents


def plot_pair_example_figure(
    roistat,
    envnum,
    plane_pair,
    pp_roi_match,
    pp_roi_nomatch,
    pp_centroid_match,
    pp_centroid_nomatch,
    prms,
):
    """this is the script to use to make a nice figure once you've chosen a pair of cells (it's not nice right now, but will be)"""
    # pick random session pair and random index
    isespair = 5
    idxroipair = np.random.choice(pp_roi_match[isespair].shape[1])
    roi_scale = 2.5
    lim = 3  # zscore limit of colormap
    print("Session Pair:", prms["idx_ses_pairs"][isespair], "ROI Pair:", idxroipair)
    print(
        "Match Idx:",
        pp_roi_match[isespair][:, idxroipair],
        "NoMatch Idx:",
        pp_roi_nomatch[isespair][:, idxroipair],
    )

    ccpair = np.mean(
        np.stack((pp_centroid_match[isespair][idxroipair], pp_centroid_nomatch[isespair][idxroipair])),
        axis=0,
    )

    spkmaps_match, spkmaps_nomatch, distedges = _get_spkmaps(roistat, envnum, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch)
    extents = [[0, sm.shape[0], distedges[0], distedges[-1]] for sm in spkmaps_match]

    # make heatmaps of FOV
    FOVs = [roistat.track.rundata[plane_pair[isespair][idxroipair]]["aligner"]["ims_registered_nonrigid"][i] for i in prms["idx_ses_pairs"][isespair]]
    roi_match, roi_nomatch = _get_masks(roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch)
    normalize = lambda data: [((d - d.min()) / (d.max() - d.min())).astype(np.float32) for d in data]
    FOVs = normalize(FOVs)
    roi_match = normalize(roi_match)
    roi_nomatch = normalize(roi_nomatch)

    fov_plot = [np.tile(fov.reshape(fov.shape[0], fov.shape[1], 1), (1, 1, 3)) for fov in FOVs]
    fov_plot[0][:, :, 0] += roi_match[0] * roi_scale
    fov_plot[1][:, :, 0] += roi_match[1] * roi_scale
    fov_plot[0][:, :, 1] += roi_nomatch[0] * roi_scale
    fov_plot[1][:, :, 2] += roi_nomatch[1] * roi_scale
    fov_plot = normalize(fov_plot)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained", sharex=True, sharey=True)
    ax[0].imshow(fov_plot[0])
    ax[1].imshow(fov_plot[1])
    ax[0].set_xlim(ccpair[0] + [-20, 20])
    ax[0].set_ylim(ccpair[1] + [-20, 20])
    ax[0].set_title("Session 1")
    ax[1].set_title("Session 2")
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(6, 6), layout="constrained", sharey=True)
    ax[0, 0].imshow(spkmaps_match[0].T, cmap="Reds", aspect="auto", extent=extents[0], vmin=0, vmax=lim)
    ax[0, 1].imshow(spkmaps_match[1].T, cmap="Reds", aspect="auto", extent=extents[1], vmin=0, vmax=lim)
    ax[1, 0].imshow(spkmaps_nomatch[0].T, cmap="Greens", aspect="auto", extent=extents[0], vmin=0, vmax=lim)
    ax[1, 1].imshow(spkmaps_nomatch[1].T, cmap="Blues", aspect="auto", extent=extents[1], vmin=0, vmax=lim)
    ax[0, 0].set_ylabel("Virtual Position (cm)")
    ax[1, 0].set_ylabel("Virtual Position (cm)")
    ax[1, 0].set_xlabel("Trials")
    ax[1, 1].set_xlabel("Trials")
    ax[0, 0].set_title("Session 1")
    ax[0, 1].set_title("Session 2")
    plt.show()


def plot_pair_example_interactive(
    roistat,
    envnum,
    plane_pair,
    pp_roi_match,
    pp_roi_nomatch,
    pp_centroid_match,
    pp_centroid_nomatch,
    prms,
):

    # Now make figures
    # pick random session pair and random index
    isespair = CurrentSelection(minval=0, maxval=len(plane_pair))
    idxroipair = CurrentSelection(minval=0, maxval=len(plane_pair[isespair()]))
    roi_scale = CurrentSelection(value=2.5, minval=0.0, maxval=100.0)
    zscore_lim = CurrentSelection(value=3, minval=0.1, maxval=100.0)

    # Print Selection
    print_selection(pp_roi_match, pp_roi_nomatch, prms, isespair(), idxroipair())

    # Make Data
    t = time()
    fov_plot, average_centroid, spkmaps_match, spkmaps_nomatch, extents = _make_data(
        roistat,
        envnum,
        plane_pair,
        pp_roi_match,
        pp_roi_nomatch,
        pp_centroid_match,
        pp_centroid_nomatch,
        prms,
        isespair(),
        idxroipair(),
        roi_scale(),
        zscore_lim(),
    )
    print(time() - t)

    # create image items for each FOV
    fov_images = [pg.ImageItem(image=fplot, axisOrder="row-major") for fplot in fov_plot]
    spkmap_match_images = [pg.ImageItem(image=sm_match, axisOrder="row-major") for sm_match in spkmaps_match]
    spkmap_nomatch_images = [pg.ImageItem(image=sm_match, axisOrder="row-major") for sm_match in spkmaps_nomatch]

    # This is the main GUI window, each component of the GUI will be added as a graphics layout in successive rows
    window = pg.GraphicsLayoutWidget(size=(1200, 800))

    # Create graphics layout with viewboxes for the FOV images
    fov_layout = pg.GraphicsLayout()
    window.addItem(fov_layout, row=1, col=0)
    fov_views = [
        fov_layout.addViewBox(row=0, col=ii, enableMouse=True, lockAspect=True, invertY=True, name=f"Session {ises}")
        for ii, ises in enumerate(prms["idx_ses_pairs"][isespair()])
    ]
    for image, view in zip(fov_images, fov_views):
        view.addItem(image)
    fov_views[1].linkView(fov_views[1].XAxis, fov_views[0])
    fov_views[1].linkView(fov_views[1].YAxis, fov_views[0])

    # Create graphics layout with viewboxes for the activity data (match)
    spkmap_match_layout = pg.GraphicsLayout()
    window.addItem(spkmap_match_layout, row=2, col=0)
    spkmap_match_views = [
        spkmap_match_layout.addViewBox(row=0, col=ii, enableMouse=False, lockAspect=False, invertY=True, name=f"Session {ises}")
        for ii, ises in enumerate(prms["idx_ses_pairs"][isespair()])
    ]
    for image, view in zip(spkmap_match_images, spkmap_match_views):
        view.addItem(image)
    spkmap_match_views[1].linkView(spkmap_match_views[1].YAxis, spkmap_match_views[0])

    # Create graphics layout with viewboxes for the activity data (no-match)
    spkmap_nomatch_layout = pg.GraphicsLayout()
    window.addItem(spkmap_nomatch_layout, row=3, col=0)
    spkmap_nomatch_views = [
        spkmap_nomatch_layout.addViewBox(row=0, col=ii, enableMouse=False, lockAspect=False, invertY=True, name=f"Session {ises}")
        for ii, ises in enumerate(prms["idx_ses_pairs"][isespair()])
    ]
    for image, view in zip(spkmap_nomatch_images, spkmap_nomatch_views):
        view.addItem(image)
    spkmap_nomatch_views[1].linkView(spkmap_nomatch_views[1].YAxis, spkmap_nomatch_views[0])

    def _callback(roipair_value):
        # Make Data
        fov_plot, average_centroid, spkmaps_match, spkmaps_nomatch, extents = _make_data(
            roistat,
            envnum,
            plane_pair,
            pp_roi_match,
            pp_roi_nomatch,
            pp_centroid_match,
            pp_centroid_nomatch,
            prms,
            isespair(),
            roipair_value,
            roi_scale(),
            zscore_lim(),
        )
        # create image items for each FOV
        for fi, fplot in zip(fov_images, fov_plot):
            fi.setImage(fplot)
        for si, sm_match in zip(spkmap_match_images, spkmaps_match):
            si.setImage(sm_match)
        for si, sm_nomatch in zip(spkmap_nomatch_images, spkmaps_nomatch):
            si.setImage(sm_nomatch)

    slider = SliderSelector(window, idxroipair, "ROI Pair", callback=_callback, row=4, col=0)  # , shortcut_key="G")

    # show GUI and return window for programmatic interaction
    window.show()
    return window

    # # create image items for each stack
    # imageItems = [pg.ImageItem(image=stacks[stack][0], axisOrder="row-major") for stack in range(numStacks)]
    # if preserveScale:
    #     for imLevel, image in zip(imLevels, imageItems):
    #         image.setLevels(imLevel)

    # # infLines are drawn over the stacks to help find the same position across stacks, they are linked across stacks.
    # if infLines:

    #     def updateLinePosX(event):
    #         for ixLine in ixLineItems:
    #             ixLine.setValue(event.x())

    #     def updateLinePosY(event):
    #         for iyLine in iyLineItems:
    #             iyLine.setValue(event.y())

    #     # start with the lines in the center (0,0) position
    #     xPosition = stacks[0].shape[2] / 2
    #     yPosition = stacks[0].shape[1] / 2
    #     # create the lines, and add callbacks
    #     ixLineItems = [pg.InfiniteLine(pos=xPosition, angle=90, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
    #     iyLineItems = [pg.InfiniteLine(pos=yPosition, angle=0, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
    #     for ixLine, iyLine in zip(ixLineItems, iyLineItems):
    #         ixLine.sigPositionChangeFinished.connect(updateLinePosX)
    #         iyLine.sigPositionChangeFinished.connect(updateLinePosY)

    # # This is the main GUI window, each component of the GUI will be added as a graphics layout in successive rows
    # window = pg.GraphicsLayoutWidget(size=(1200, 800))

    # # Create graphics layout with viewboxes for the image stacks
    # stackLayout = pg.GraphicsLayout()
    # window.addItem(stackLayout, row=1, col=0)
    # # create a viewbox for each stack, add the appropriate image to it, link images so they all move together
    # views = [
    #     stackLayout.addViewBox(
    #         row=0,
    #         col=stack,
    #         enableMouse=enableMouse,
    #         lockAspect=lockAspect,
    #         invertY=True,
    #         name=stackTitles[stack],
    #     )
    #     for stack in range(numStacks)
    # ]
    # for image, view in zip(imageItems, views):
    #     view.addItem(image)
    # for view in views[1:]:
    #     view.linkView(view.XAxis, views[0])
    #     view.linkView(view.YAxis, views[0])
    # # add infinite lines to mark positions if requested
    # if infLines:
    #     for ixLine, iyLine, view in zip(ixLineItems, iyLineItems, views):
    #         view.addItem(ixLine)
    #         view.addItem(iyLine)

    # # Create barplots for each feature
    # histCenters, histValues, histRange = [], [], []
    # for feature in features:
    #     # make histogram of each feature
    #     cHist, cEdges = np.histogram(feature, bins=50)
    #     histRange.append((cEdges[0], cEdges[-1]))  # min/max of histogram for each feature
    #     histCenters.append(helpers.edge2center(cEdges))  # center of histogram bin for each feature
    #     histValues.append(cHist)  # histogram value for each bin for each feature
    # featureHistograms = [
    #     pg.BarGraphItem(x=histCenter, height=histValue, width=histCenter[1] - histCenter[0]) for histCenter, histValue in zip(histCenters, histValues)
    # ]
    # featRedHistograms = [
    #     pg.BarGraphItem(x=histCenter, height=histValue / 2, width=histCenter[1] - histCenter[0], brush="r")
    #     for histCenter, histValue in zip(histCenters, histValues)
    # ]

    # # Create a graphics layout with bar graph plots for the features
    # featureLayout = pg.GraphicsLayout()
    # window.addItem(featureLayout, row=2, col=0)
    # featurePlots = [featureLayout.addPlot(row=0, col=feature, enableMouse=False, title=featureTitles[feature]) for feature in range(numFeatures)]
    # for featurePlot in featurePlots:
    #     featurePlot.setMouseEnabled(x=False, y=False)
    # # featurePlots = [featureLayout.addViewBox(row=0,col=feature) for feature in range(numFeatures)]
    # for featureHistogram, featurePlot in zip(featureHistograms, featurePlots):
    #     featurePlot.addItem(featureHistogram)
    # # for featureHistogram,featurePlot in zip(featRedHistograms, featurePlots): featurePlot.addItem(featureHistogram)

    # # Create vertical lines indicating the value of the currently presented cell
    # currentValueROI = [pg.InfiniteLine(pos=features[feature][0], angle=90, movable=False, pen=pg.mkPen(width=0.5)) for feature in range(numFeatures)]
    # for fplot, cv in zip(featurePlots, currentValueROI):
    #     fplot.addItem(cv)

    # # Create a slider label for indicating which ROI is being presented
    # sliderNameProxy = QGraphicsProxyWidget()
    # label = QLabel(f"ROI {1}/{numImages}")
    # label.setAlignment(QtCore.Qt.AlignCenter)
    # sliderNameProxy.setWidget(label)
    # window.addItem(sliderNameProxy, row=3, col=0)

    # # Create a slider with prev/next buttons and an edit field to change which ROI is being presented
    # def updateSlider(value):
    #     roi.update(value)  # first try updating roi value
    #     slider.setValue(roi.value)  # if it clipped, reset slider appropriately
    #     editField.setText(str(roi.value))  # update textfield
    #     updateStackIndex()  # update which ROI is presented

    # def prevROI():
    #     roi.update(roi.value - 1)  # try updating roi value
    #     slider.setValue(roi.value)  # update slider
    #     editField.setText(str(roi.value))  # update textfield
    #     updateStackIndex()  # update which ROI is presented

    # def nextROI():
    #     roi.update(roi.value + 1)  # try updating roi value
    #     slider.setValue(roi.value)  # update slider
    #     editField.setText(str(roi.value))  # update textfield
    #     updateStackIndex()  # update which ROI is presented

    # def gotoROI():
    #     if not editField.text().isdigit():
    #         editField.setText("invalid ROI")
    #         return
    #     textValue = int(editField.text())
    #     if (textValue < roi.minroi) or (textValue > roi.maxroi):
    #         editField.setText("invalid ROI")
    #         return
    #     # otherwise text is valid ROI
    #     roi.update(textValue)
    #     editField.setText(str(roi.value))
    #     slider.setValue(roi.value)  # update slider
    #     updateStackIndex()

    # slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    # slider.setMinimum(0)
    # slider.setMaximum(numImages - 1)
    # slider.setSingleStep(1)
    # slider.setPageStep(int(numImages / 10))
    # slider.setValue(roi.value)
    # slider.valueChanged.connect(updateSlider)
    # sliderProxy = QGraphicsProxyWidget()
    # sliderProxy.setWidget(slider)

    # prevButtonProxy = QGraphicsProxyWidget()
    # prevButton = QPushButton("button", text="Prev ROI")
    # prevButton.clicked.connect(prevROI)
    # prevButtonProxy.setWidget(prevButton)

    # nextButtonProxy = QGraphicsProxyWidget()
    # nextButton = QPushButton("button", text="Next ROI")
    # nextButton.clicked.connect(nextROI)
    # nextButtonProxy.setWidget(nextButton)

    # editFieldProxy = QGraphicsProxyWidget()
    # editField = QLineEdit()
    # editField.setText("0")
    # editFieldProxy.setWidget(editField)

    # gotoEditProxy = QGraphicsProxyWidget()
    # gotoButton = QPushButton("button", text="go to ROI")
    # gotoButton.clicked.connect(gotoROI)
    # gotoEditProxy.setWidget(gotoButton)

    # # add shortcut for going to ROI without pressing the button...
    # shortcut = QShortcut(QKeySequence("G"), window)
    # shortcut.activated.connect(gotoROI)

    # roiSelectionLayout = pg.GraphicsLayout()
    # roiSelectionLayout.addItem(prevButtonProxy, row=0, col=0)
    # roiSelectionLayout.addItem(sliderProxy, row=0, col=1)
    # roiSelectionLayout.addItem(nextButtonProxy, row=0, col=2)
    # roiSelectionLayout.addItem(editFieldProxy, row=0, col=3)
    # roiSelectionLayout.addItem(gotoEditProxy, row=0, col=4)
    # window.addItem(roiSelectionLayout, row=4, col=0)

    # # show GUI and return window for programmatic interaction
    # window.show()
    # return window


if __name__ == "__main__":
    args = handle_inputs()
    example_data = prepare_pair_example_data(args.mouse_name)
    window = plot_pair_example_figure(*example_data)


# converting uiPlottingFunctions.scrollMatchedImages into a redSelection GUI made to be similar to the same named function in Matlab
def redCellViewer(stacks, features, enableMouse=False, lockAspect=1, infLines=True, preserveScale=True):
    # supporting class for storing and updating the ROI displayed in redCellViewer()
    class currentROI:
        def __init__(self, minroi=0, maxroi=None):
            self.minroi = minroi
            self.maxroi = maxroi if maxroi is not None else np.inf
            assert self.minroi < self.maxroi, "minimum roi value must be less than maximum roi value"
            self.value = 0

        def update(self, value):
            self.value = np.minimum(np.maximum(value, self.minroi), self.maxroi)

    # handle inputs
    msg = "stacks should be a length 3 iterable containing 3-d centered stacks of the reference image, the mask, and the phase correlation plots"
    assert len(stacks) == 3, msg
    msg = "features should be a length 4 iterable containing the suite2p red probability, dot product, correlation coefficient, and central pxc point for each ROI in stacks"
    assert len(features) == 4, msg
    assert type(enableMouse) == bool, "enableMouse must be a boolean"
    numStacks = len(stacks)
    numImages = stacks[0].shape[0]
    numFeatures = len(features)
    for stack in range(numStacks):
        assert stacks[stack].ndim == 3, "stacks are not all 3-dimensional"
    for stack in range(numStacks):
        assert stacks[stack].shape[0] == numImages, "number of ROIs to look through are not the same in each stack"
    for feature in features:
        assert isinstance(feature, np.ndarray), "features need to be numpy arrays"
    for feature in features:
        assert feature.size == numImages, "number of ROIs in each features array must be same as the number of ROIs in the stacks"
    stackTitles = ["reference", "mask", "phase-correlation"]
    featureTitles = ["S2P", "dot(ref,mask)", "corr(ref,mask)", "pxc"]

    # keep track of current ROI (I have  no idea why I can't do this with a python int...
    roi = currentROI(minroi=0, maxroi=numImages - 1)

    # measure minimum and maximum of each stack
    if preserveScale:
        imLevels = [(np.min(stack), np.max(stack)) for stack in stacks]
    else:
        imLevels = [None] * numStacks  # allocate list for simple code later on

    def updateStackIndex():
        # whenever the ROI is changed, update the images and the labels (and keep scale the same if necessary)
        for stack, image, imLevel, view in zip(stacks, imageItems, imLevels, views):
            image.setImage(stack[roi.value])
            label.setText(f"ROI {roi.value+1}/{numImages}")
            if preserveScale:
                image.setLevels(imLevel)
        # whenever the ROI is changed, update the infiniteLine position indicating the value of that particular ROI
        for feature, cvalROI in zip(features, currentValueROI):
            cvalROI.setValue(feature[roi.value])

    # create image items for each stack
    imageItems = [pg.ImageItem(image=stacks[stack][0], axisOrder="row-major") for stack in range(numStacks)]
    if preserveScale:
        for imLevel, image in zip(imLevels, imageItems):
            image.setLevels(imLevel)

    # infLines are drawn over the stacks to help find the same position across stacks, they are linked across stacks.
    if infLines:

        def updateLinePosX(event):
            for ixLine in ixLineItems:
                ixLine.setValue(event.x())

        def updateLinePosY(event):
            for iyLine in iyLineItems:
                iyLine.setValue(event.y())

        # start with the lines in the center (0,0) position
        xPosition = stacks[0].shape[2] / 2
        yPosition = stacks[0].shape[1] / 2
        # create the lines, and add callbacks
        ixLineItems = [pg.InfiniteLine(pos=xPosition, angle=90, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        iyLineItems = [pg.InfiniteLine(pos=yPosition, angle=0, movable=True, pen=pg.mkPen(width=0.5)) for stack in range(numStacks)]
        for ixLine, iyLine in zip(ixLineItems, iyLineItems):
            ixLine.sigPositionChangeFinished.connect(updateLinePosX)
            iyLine.sigPositionChangeFinished.connect(updateLinePosY)

    # This is the main GUI window, each component of the GUI will be added as a graphics layout in successive rows
    window = pg.GraphicsLayoutWidget(size=(1200, 800))

    # Create graphics layout with viewboxes for the image stacks
    stackLayout = pg.GraphicsLayout()
    window.addItem(stackLayout, row=1, col=0)
    # create a viewbox for each stack, add the appropriate image to it, link images so they all move together
    views = [
        stackLayout.addViewBox(
            row=0,
            col=stack,
            enableMouse=enableMouse,
            lockAspect=lockAspect,
            invertY=True,
            name=stackTitles[stack],
        )
        for stack in range(numStacks)
    ]
    for image, view in zip(imageItems, views):
        view.addItem(image)
    for view in views[1:]:
        view.linkView(view.XAxis, views[0])
        view.linkView(view.YAxis, views[0])
    # add infinite lines to mark positions if requested
    if infLines:
        for ixLine, iyLine, view in zip(ixLineItems, iyLineItems, views):
            view.addItem(ixLine)
            view.addItem(iyLine)

    # Create barplots for each feature
    histCenters, histValues, histRange = [], [], []
    for feature in features:
        # make histogram of each feature
        cHist, cEdges = np.histogram(feature, bins=50)
        histRange.append((cEdges[0], cEdges[-1]))  # min/max of histogram for each feature
        histCenters.append(edge2center(cEdges))  # center of histogram bin for each feature
        histValues.append(cHist)  # histogram value for each bin for each feature
    featureHistograms = [
        pg.BarGraphItem(x=histCenter, height=histValue, width=histCenter[1] - histCenter[0]) for histCenter, histValue in zip(histCenters, histValues)
    ]
    featRedHistograms = [
        pg.BarGraphItem(x=histCenter, height=histValue / 2, width=histCenter[1] - histCenter[0], brush="r")
        for histCenter, histValue in zip(histCenters, histValues)
    ]

    # Create a graphics layout with bar graph plots for the features
    featureLayout = pg.GraphicsLayout()
    window.addItem(featureLayout, row=2, col=0)
    featurePlots = [featureLayout.addPlot(row=0, col=feature, enableMouse=False, title=featureTitles[feature]) for feature in range(numFeatures)]
    for featurePlot in featurePlots:
        featurePlot.setMouseEnabled(x=False, y=False)
    # featurePlots = [featureLayout.addViewBox(row=0,col=feature) for feature in range(numFeatures)]
    for featureHistogram, featurePlot in zip(featureHistograms, featurePlots):
        featurePlot.addItem(featureHistogram)
    # for featureHistogram,featurePlot in zip(featRedHistograms, featurePlots): featurePlot.addItem(featureHistogram)

    # Create vertical lines indicating the value of the currently presented cell
    currentValueROI = [pg.InfiniteLine(pos=features[feature][0], angle=90, movable=False, pen=pg.mkPen(width=0.5)) for feature in range(numFeatures)]
    for fplot, cv in zip(featurePlots, currentValueROI):
        fplot.addItem(cv)

    # Create a slider label for indicating which ROI is being presented
    sliderNameProxy = QGraphicsProxyWidget()
    label = QLabel(f"ROI {1}/{numImages}")
    label.setAlignment(QtCore.Qt.AlignCenter)
    sliderNameProxy.setWidget(label)
    window.addItem(sliderNameProxy, row=3, col=0)

    # Create a slider with prev/next buttons and an edit field to change which ROI is being presented
    def updateSlider(value):
        roi.update(value)  # first try updating roi value
        slider.setValue(roi.value)  # if it clipped, reset slider appropriately
        editField.setText(str(roi.value))  # update textfield
        updateStackIndex()  # update which ROI is presented

    def prevROI():
        roi.update(roi.value - 1)  # try updating roi value
        slider.setValue(roi.value)  # update slider
        editField.setText(str(roi.value))  # update textfield
        updateStackIndex()  # update which ROI is presented

    def nextROI():
        roi.update(roi.value + 1)  # try updating roi value
        slider.setValue(roi.value)  # update slider
        editField.setText(str(roi.value))  # update textfield
        updateStackIndex()  # update which ROI is presented

    def gotoROI():
        if not editField.text().isdigit():
            editField.setText("invalid ROI")
            return
        textValue = int(editField.text())
        if (textValue < roi.minroi) or (textValue > roi.maxroi):
            editField.setText("invalid ROI")
            return
        # otherwise text is valid ROI
        roi.update(textValue)
        editField.setText(str(roi.value))
        slider.setValue(roi.value)  # update slider
        updateStackIndex()

    slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(numImages - 1)
    slider.setSingleStep(1)
    slider.setPageStep(int(numImages / 10))
    slider.setValue(roi.value)
    slider.valueChanged.connect(updateSlider)
    sliderProxy = QGraphicsProxyWidget()
    sliderProxy.setWidget(slider)

    prevButtonProxy = QGraphicsProxyWidget()
    prevButton = QPushButton("button", text="Prev ROI")
    prevButton.clicked.connect(prevROI)
    prevButtonProxy.setWidget(prevButton)

    nextButtonProxy = QGraphicsProxyWidget()
    nextButton = QPushButton("button", text="Next ROI")
    nextButton.clicked.connect(nextROI)
    nextButtonProxy.setWidget(nextButton)

    editFieldProxy = QGraphicsProxyWidget()
    editField = QLineEdit()
    editField.setText("0")
    editFieldProxy.setWidget(editField)

    gotoEditProxy = QGraphicsProxyWidget()
    gotoButton = QPushButton("button", text="go to ROI")
    gotoButton.clicked.connect(gotoROI)
    gotoEditProxy.setWidget(gotoButton)

    # add shortcut for going to ROI without pressing the button...
    shortcut = QShortcut(QKeySequence("G"), window)
    shortcut.activated.connect(gotoROI)

    roiSelectionLayout = pg.GraphicsLayout()
    roiSelectionLayout.addItem(prevButtonProxy, row=0, col=0)
    roiSelectionLayout.addItem(sliderProxy, row=0, col=1)
    roiSelectionLayout.addItem(nextButtonProxy, row=0, col=2)
    roiSelectionLayout.addItem(editFieldProxy, row=0, col=3)
    roiSelectionLayout.addItem(gotoEditProxy, row=0, col=4)
    window.addItem(roiSelectionLayout, row=4, col=0)

    # show GUI and return window for programmatic interaction
    window.show()
    return window
