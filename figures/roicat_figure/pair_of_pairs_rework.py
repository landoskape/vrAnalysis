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
    QApplication,
    QDialog,
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


# Saving some good possibilities for the figure (no plane 0):
# ATL027, sessions 10/11, Match Idx: [1878 1603] NoMatch Idx: [2433 2652]
# ATL027, sessions 8/9, Match Idx: [11079  8659] NoMatch Idx: [13208 11206]
# ATL027, sessions 8/9, Match Idx: [7640 5003] NoMatch Idx: [10729  5743]

# lost possibility:
# ATL027, sessions 8/9, they were in rough centroid [310, 60-80], the match had a place field > 180cm.

# Possibilities (no plane 0):
# ATL027, sessions 9/10, Match Idx: [11828 13054] NoMatch Idx: [9393 9689]
# ATL027, sessions 9/10, Match Idx: [4984 5500] NoMatch Idx: [5098 7138]
# ATL027, sessions 9/11, Match Idx: [265 524] NoMatch Idx: [132 891]


# GLOBAL PARAMETER DICTIONARIES
ROICAT_COMPARISON_PARAMETERS = dict(
    sim_name="sConj",
    cutoffs=(0.6, 0.85),
    both_reliable=False,
)

# pair parameters
PAIR_PARAMETERS = dict(
    mincorr=0.6,  # minimum correlation for matched pair
    maxcorr=0.2,  # maximum correlation for non-matched pair
    maxdist=6,  # maximum distance for any example pair
    # set parameters for pair of pairs (distance apart etc.)
    maxdistpp=16,  # maximum distance for pair of pairs
    mindistpp=6,  # minimum distance for pair of pairs
)


class PairPairDatahandler:
    def __init__(self, mouse_name, fake_data=False, prm_updates={}):
        """constructor for the PairPairDatahandler class"""
        self.mouse_name = mouse_name
        self.fake_data = fake_data
        self.roistat = analysis.RoicatStats(tracking.tracker(self.mouse_name))
        self._prepare_data(prm_updates=prm_updates)

    def _make_fake_data(self):
        """make fake data for testing"""
        fov_plot = [np.random.rand(100, 100, 3) for _ in range(2)]
        average_centroid = np.random.rand(2) * 100
        spkmaps_match = [np.random.rand(100, 100) for _ in range(2)]
        spkmaps_nomatch = [np.random.rand(100, 100) for _ in range(2)]
        extents = [[0, 100, 0, 100] for _ in range(2)]
        return fov_plot, average_centroid, spkmaps_match, spkmaps_nomatch, extents

    def make_plot_data(self, isespair, idxroipair, roi_scale, zscore_lim):
        if self.fake_data:
            return self._make_fake_data()

        average_centroid = self._get_centroid(isespair, idxroipair)

        spkmaps_match, spkmaps_nomatch, distedges = self._get_spkmaps(isespair, idxroipair)

        extents = [[0, sm.shape[0], distedges[0], distedges[-1]] for sm in spkmaps_match]

        # make heatmaps of FOV
        FOVs = [
            self.roistat.track.rundata[self.plane_pair[isespair][idxroipair]]["aligner"]["ims_registered_nonrigid"][i]
            for i in self.prms["idx_ses_pairs"][isespair]
        ]
        roi_match, roi_nomatch = self._get_masks(isespair, idxroipair)
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

    def _get_spkmaps(self, isespair, idxroipair):
        """get requested spike maps for a given pair of ROIs in a pair of sessions"""
        idx_ses_pair = self.prms["idx_ses_pairs"][isespair]
        ifirst, isecond = self._pair_to_master_idx(idx_ses_pair)
        spkmap_match = [
            self.spkmaps[ifirst][self.pp_roi_match[isespair][0, idxroipair]],
            self.spkmaps[isecond][self.pp_roi_match[isespair][1, idxroipair]],
        ]
        spkmap_nomatch = [
            self.spkmaps[ifirst][self.pp_roi_nomatch[isespair][0, idxroipair]],
            self.spkmaps[isecond][self.pp_roi_nomatch[isespair][1, idxroipair]],
        ]
        distedges = self.roistat.pcss[idx_ses_pair[0]].distedges
        return spkmap_match, spkmap_nomatch, distedges

    def _get_centroid(self, isespair, idxroipair):
        return np.mean(np.stack((self.pp_centroid_match[isespair][idxroipair], self.pp_centroid_nomatch[isespair][idxroipair])), axis=0)

    def _get_masks(self, isespair, idxroipair):
        ROIs = self.roistat.track.get_ROIs(as_coo=False, idx_ses=self.prms["idx_ses_pairs"][isespair], keep_planes=self.roistat.keep_planes)
        ROIs = [sp.sparse.vstack(rois, format="csr") for rois in ROIs]  # concatenate across planes

        roi_match = [
            ROIs[0][[self.pp_roi_match[isespair][0, idxroipair]]],
            ROIs[1][[self.pp_roi_match[isespair][1, idxroipair]]],
        ]
        roi_nomatch = [
            ROIs[0][[self.pp_roi_nomatch[isespair][0, idxroipair]]],
            ROIs[1][[self.pp_roi_nomatch[isespair][1, idxroipair]]],
        ]

        num_pixels = roi_match[0].shape[1]
        hw = int(np.sqrt(num_pixels))
        roi_match = [r.toarray().reshape(hw, hw) for r in roi_match]
        roi_nomatch = [r.toarray().reshape(hw, hw) for r in roi_nomatch]

        return roi_match, roi_nomatch

    def _prepare_data(self, prm_updates={}):
        """
        Example of matched / not-matched ROIs in aligned FOVs with color-coded place fields
        """
        if self.fake_data:
            self.prms = {}
            self.prms["idx_ses"] = [0, 1, 2, 3]
            self.prms["idx_ses_pairs"] = [(0, 1), (2, 3)]
            self.plane_pair = [np.ones(15), np.ones(25)]
            return

        # get parameters
        self.roicat_comparison_parameters, self.pair_parameters = self._parameters(**prm_updates)

        # processing methods for preloaded roistat object
        self.envnum = self.roistat.env_selector(envmethod="most")

        # get all sessions with chosen environment
        self.idx_ses = self.roistat.idx_ses_selector(self.envnum, sesmethod="all")

        # pick 4 environments, the later the better, but if there's lots of sessions don't do the last one (imaging quality dropped off in last couple sessions usually)
        if len(self.idx_ses) > 7:
            self.idx_ses = self.idx_ses[-6:-2]
        elif len(self.idx_ses) > 4:
            self.idx_ses = self.idx_ses[-4:]

        # report which environment and sessions are being used
        print("env:", self.envnum, "idx_ses:", self.idx_ses)

        _, corr, tracked, pwdist, _, pwind, self.prms = self.roistat.make_roicat_comparison(
            self.envnum, idx_ses=self.idx_ses, **self.roicat_comparison_parameters
        )
        centroids = self.roistat.track.get_centroids(
            idx_ses=self.prms["idx_ses"], cat_planes=True, combine=True, keep_planes=self.roistat.keep_planes
        )

        # get possible matches and not-matches
        idx_possible_match = [
            np.where(t & (c > self.pair_parameters["mincorr"]) & (pwd < self.pair_parameters["maxdist"]))[0]
            for t, c, pwd in zip(tracked, corr, pwdist)
        ]
        idx_possible_nomatch = [
            np.where(~t & (c < self.pair_parameters["maxcorr"]) & (pwd < self.pair_parameters["maxdist"]))[0]
            for t, c, pwd in zip(tracked, corr, pwdist)
        ]

        # get possible match/nomatch pair indices
        roi_idx_match = [pwi[:, i] for pwi, i in zip(pwind, idx_possible_match)]
        roi_idx_nomatch = [pwi[:, i] for pwi, i in zip(pwind, idx_possible_nomatch)]

        # get average centroid of pairs
        centroids_match = self._get_centroids_match(centroids, roi_idx_match)
        centroids_nomatch = self._get_centroids_match(centroids, roi_idx_nomatch)
        acentroid_match = [np.mean(c, axis=0) for c in centroids_match]
        acentroid_nomatch = [np.mean(c, axis=0) for c in centroids_nomatch]

        # get distance between pairs of possible match/nomatch for each pair of sessions
        dist_match_nomatch = [sp.spatial.distance.cdist(am, anm) for am, anm in zip(acentroid_match, acentroid_nomatch)]
        good_distance = [(dmn < self.pair_parameters["maxdistpp"]) & (dmn > self.pair_parameters["mindistpp"]) for dmn in dist_match_nomatch]

        # check if pairs include any of the same ROIs
        duplicate_idx = self._identify_duplicates(roi_idx_match, roi_idx_nomatch)

        # get plane indices for each pair
        plane_pair_match = self._get_plane_of_pair(self.roistat, roi_idx_match)
        plane_pair_nomatch = self._get_plane_of_pair(self.roistat, roi_idx_nomatch)
        same_plane = [ppm.reshape(-1, 1) == ppn.reshape(1, -1) for ppm, ppn in zip(plane_pair_match, plane_pair_nomatch)]

        # possible pair of pairs boolean array
        self.ipair_match, self.ipair_nomatch = named_transpose(
            [np.where(gd & ~di & smp) for gd, di, smp in zip(good_distance, duplicate_idx, same_plane)]
        )

        # indices to ROI pairs for good pairs of pairs
        self.pp_roi_match = [rim[:, ipm] for rim, ipm in zip(roi_idx_match, self.ipair_match)]
        self.pp_roi_nomatch = [rinm[:, ipnm] for rinm, ipnm in zip(roi_idx_nomatch, self.ipair_nomatch)]
        self.pp_centroid_match = [acm[ipm] for acm, ipm in zip(acentroid_match, self.ipair_match)]
        self.pp_centroid_nomatch = [acnm[ipnm] for acnm, ipnm in zip(acentroid_nomatch, self.ipair_nomatch)]

        # get plane indices for each pair
        self.plane_pair = self._get_plane_of_pair(self.roistat, self.pp_roi_match)

        # check that match and no-match pairs are in same plane
        assert all([np.all(pp == ppnm) for pp, ppnm in zip(self.plane_pair, self._get_plane_of_pair(self.roistat, self.pp_roi_nomatch))])

        # get spkmaps
        self.spkmaps = self.roistat.get_spkmaps(
            self.envnum, trials="full", average=False, tracked=False, idx_ses=self.prms["idx_ses"], by_plane=False
        )[0]

        # zscore spkmaps for each ROI
        self.spkmaps = [
            sp.stats.zscore(sm.reshape(sm.shape[0], -1), axis=1, nan_policy="omit").reshape(sm.shape[0], sm.shape[1], -1) for sm in self.spkmaps
        ]

    def _identify_duplicates(self, roi_idx_match, roi_idx_nomatch):
        duplicate_idx = []
        for im, inm in zip(roi_idx_match, roi_idx_nomatch):
            c_duplicates = im.reshape(2, -1, 1) == inm.reshape(2, 1, -1)
            duplicate_idx.append(np.any(c_duplicates, axis=0))
        return duplicate_idx

    def _get_centroids_match(self, centroids, idx):
        assert len(self.prms["idx_ses_pairs"]) == len(idx), "idx doesn't match number of session pairs"
        assert len(centroids) == len(self.prms["idx_ses"]), "centroids doesn't match number of sessions"

        # prepare lookup table for going from the absolute session index to the relative index in centroids
        session_lookup = lambda ises: {val: idx for idx, val in enumerate(self.prms["idx_ses"])}[ises]

        # prepare list for centroids of matched pairs
        centroids_matched = []
        for ipair, imatch in zip(self.prms["idx_ses_pairs"], idx):
            i1 = session_lookup(ipair[0])  # idx to first session in pair
            i2 = session_lookup(ipair[1])  # idx to second session in pair
            c1 = centroids[i1][imatch[0]]  # centroids in first session
            c2 = centroids[i2][imatch[1]]  # centroids in second session
            centroids_matched.append(np.stack((c1, c2)))

        return centroids_matched

    def _get_plane_of_pair(self, roistat, idx):
        assert len(idx) == len(self.prms["idx_ses_pairs"]), "idx and prms['idx_ses_pairs'] don't have same number of elements"

        # prepare lookup table for going from the absolute session index to the relative index in centroids
        session_lookup = lambda ises: {val: idx for idx, val in enumerate(self.prms["idx_ses"])}[ises]

        # get plane index for each roi from sessions
        roiPlaneIdx = roistat.get_from_pcss("roiPlaneIdx", self.prms["idx_ses"])

        # prepare list of roi plane index for each pair
        pair_plane = []
        for ipair, ii in zip(self.prms["idx_ses_pairs"], idx):
            i1 = session_lookup(ipair[0])
            i2 = session_lookup(ipair[1])
            pp1 = roiPlaneIdx[i1][ii[0]]
            pp2 = roiPlaneIdx[i2][ii[1]]
            if not np.all(pp1 == pp2):
                raise ValueError("Planes of pair don't all match")
            pair_plane.append(pp1)

        return pair_plane

    def _parameters(self, verbose=True, **kwargs):
        """central method for setting parameters for the analysis"""
        # get data for all roicat plots
        roicat_comparison_parameters = ROICAT_COMPARISON_PARAMETERS.copy()
        pair_parameters = PAIR_PARAMETERS.copy()

        # possible updates to roicat_comparison_parameters
        rcp_update_keys = ["sim_name", "cutoffs", "both_reliable"]
        pp_update_keys = ["mincorr", "maxcorr", "maxdist", "maxdistpp", "mindistpp"]

        # update key if it is provided by user in kwargs
        for key in rcp_update_keys:
            if key in kwargs:
                if verbose:
                    print(f"Updating roicat_comparison_parameter:{key} to {kwargs[key]} from {roicat_comparison_parameters[key]}")
                roicat_comparison_parameters[key] = kwargs[key]

        for key in pp_update_keys:
            if key in kwargs:
                if verbose:
                    print(f"Updating pair_parameter:{key} to {kwargs[key]} from {pair_parameters[key]}")
                pair_parameters[key] = kwargs[key]

        return roicat_comparison_parameters, pair_parameters

    def _pair_to_master_idx(self, idx_ses_pair):
        """convert pair index to master index"""
        get_master_idx = {val: idx for idx, val in enumerate(self.prms["idx_ses"])}
        idx_to_master = [get_master_idx[isp] for isp in idx_ses_pair]
        return idx_to_master


class PairPairInteractivePlot(QDialog):
    def __init__(self, handler, parent=None):
        super(PairPairDatahandler, self).__init__(parent)
        self.originalPalette = QApplication.palette()

        self.handler = handler
        self.window = None
        self._construct_selectors()
        self._build_gui()

    def _construct_selectors(self):
        self.isespair = CurrentSelection(minval=0, maxval=len(self.handler.plane_pair))
        self.idxroipair = CurrentSelection(minval=0, maxval=len(self.handler.plane_pair[self.isespair()]))
        self.roi_scale = CurrentSelection(value=2.5, minval=0.0, maxval=100.0)
        self.zscore_lim = CurrentSelection(value=3, minval=0.1, maxval=100.0)

    def _rebuild_roipair_selector(self):
        self.idxroipair = CurrentSelection(minval=0, maxval=len(self.handler.plane_pair[self.isespair()]))
        self.roipair_slider = SliderSelector(self.window, self.idxroipair, "ROI Pair", callback=self._update_image_data, row=5, col=0)

    def print_selection(pp_roi_match, pp_roi_nomatch, prms, isespair, idxroipair):
        """simple method for printing output to terminal"""
        print("Session Pair:", prms["idx_ses_pairs"][isespair], "ROI Pair:", idxroipair)
        print(
            "Match Idx:",
            pp_roi_match[isespair][:, idxroipair],
            "NoMatch Idx:",
            pp_roi_nomatch[isespair][:, idxroipair],
        )

    def _build_gui(self):

        # This is the main GUI window, each component of the GUI will be added as a graphics layout in successive rows
        fov_plot, _, spkmaps_match, spkmaps_nomatch, _ = self.handler.make_plot_data(
            self.isespair(),
            self.idxroipair(),
            self.roi_scale(),
            self.zscore_lim(),
        )

        # create image items for each FOV
        self.fov_images = [pg.ImageItem(image=fplot, axisOrder="row-major") for fplot in fov_plot]
        self.spkmap_match_images = [pg.ImageItem(image=sm_match, axisOrder="row-major") for sm_match in spkmaps_match]
        self.spkmap_nomatch_images = [pg.ImageItem(image=sm_match, axisOrder="row-major") for sm_match in spkmaps_nomatch]

        self.window = pg.GraphicsLayoutWidget(size=(600, 800))

        # Create graphics layout with viewboxes for the FOV images
        self.fov_layout = pg.GraphicsLayout()
        self.window.addItem(self.fov_layout, row=1, col=0, rowspan=1, colspan=1)
        self.fov_views = [
            self.fov_layout.addViewBox(row=0, col=ii, enableMouse=True, lockAspect=True, invertY=True, name=f"Session {ises}")
            for ii, ises in enumerate(self.handler.prms["idx_ses_pairs"][self.isespair()])
        ]
        for image, view in zip(self.fov_images, self.fov_views):
            view.addItem(image)
        self.fov_views[1].linkView(self.fov_views[1].XAxis, self.fov_views[0])
        self.fov_views[1].linkView(self.fov_views[1].YAxis, self.fov_views[0])

        # Create graphics layout with viewboxes for the activity data (match)
        self.spkmap_match_layout = pg.GraphicsLayout()
        self.window.addItem(self.spkmap_match_layout, row=2, col=0)
        self.spkmap_match_views = [
            self.spkmap_match_layout.addViewBox(row=0, col=ii, enableMouse=False, lockAspect=False, invertY=True, name=f"Session {ises}")
            for ii, ises in enumerate(self.handler.prms["idx_ses_pairs"][self.isespair()])
        ]
        for image, view in zip(self.spkmap_match_images, self.spkmap_match_views):
            view.addItem(image)
        self.spkmap_match_views[1].linkView(self.spkmap_match_views[1].YAxis, self.spkmap_match_views[0])

        # Create graphics layout with viewboxes for the activity data (no-match)
        self.spkmap_nomatch_layout = pg.GraphicsLayout()
        self.window.addItem(self.spkmap_nomatch_layout, row=3, col=0)
        self.spkmap_nomatch_views = [
            self.spkmap_nomatch_layout.addViewBox(row=0, col=ii, enableMouse=False, lockAspect=False, invertY=True, name=f"Session {ises}")
            for ii, ises in enumerate(self.handler.prms["idx_ses_pairs"][self.isespair()])
        ]
        for image, view in zip(self.spkmap_nomatch_images, self.spkmap_nomatch_views):
            view.addItem(image)
        self.spkmap_nomatch_views[1].linkView(self.spkmap_nomatch_views[1].YAxis, self.spkmap_nomatch_views[0])

        self.sespair_slider = SliderSelector(
            self.window,
            self.isespair,
            "Session Pair",
            callback=self._update_session_pair,
            callback_requires_input=False,
            row=4,
            col=0,
        )
        self.roipair_slider = SliderSelector(
            self.window,
            self.idxroipair,
            "ROI Pair",
            callback=self._update_image_data,
            callback_requires_input=False,
            row=5,
            col=0,
        )

        # show GUI and return window for programmatic interaction
        self.window.show()

    def _update_session_pair(self):
        self._rebuild_roipair_selector()
        self._update_image_data()

    def _update_image_data(self):
        fov_plot, _, spkmaps_match, spkmaps_nomatch, _ = self.handler.make_plot_data(
            self.isespair(),
            self.idxroipair(),
            self.roi_scale(),
            self.zscore_lim(),
        )

        # create image items for each FOV
        for fi, fplot in zip(self.fov_images, fov_plot):
            fi.setImage(fplot)
        for si, sm_match in zip(self.spkmap_match_images, spkmaps_match):
            si.setImage(sm_match)
        for si, sm_nomatch in zip(self.spkmap_nomatch_images, spkmaps_nomatch):
            si.setImage(sm_nomatch)


# def plot_pair_example_figure(
#     roistat,
#     envnum,
#     plane_pair,
#     pp_roi_match,
#     pp_roi_nomatch,
#     pp_centroid_match,
#     pp_centroid_nomatch,
#     prms,
# ):
#     """this is the script to use to make a nice figure once you've chosen a pair of cells (it's not nice right now, but will be)"""
#     # pick random session pair and random index
#     isespair = 5
#     idxroipair = np.random.choice(pp_roi_match[isespair].shape[1])
#     roi_scale = 2.5
#     lim = 3  # zscore limit of colormap
#     print("Session Pair:", prms["idx_ses_pairs"][isespair], "ROI Pair:", idxroipair)
#     print(
#         "Match Idx:",
#         pp_roi_match[isespair][:, idxroipair],
#         "NoMatch Idx:",
#         pp_roi_nomatch[isespair][:, idxroipair],
#     )

#     ccpair = np.mean(
#         np.stack((pp_centroid_match[isespair][idxroipair], pp_centroid_nomatch[isespair][idxroipair])),
#         axis=0,
#     )

#     spkmaps_match, spkmaps_nomatch, distedges = _get_spkmaps(roistat, envnum, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch)
#     extents = [[0, sm.shape[0], distedges[0], distedges[-1]] for sm in spkmaps_match]

#     # make heatmaps of FOV
#     FOVs = [roistat.track.rundata[plane_pair[isespair][idxroipair]]["aligner"]["ims_registered_nonrigid"][i] for i in prms["idx_ses_pairs"][isespair]]
#     roi_match, roi_nomatch = _get_masks(roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch)
#     normalize = lambda data: [((d - d.min()) / (d.max() - d.min())).astype(np.float32) for d in data]
#     FOVs = normalize(FOVs)
#     roi_match = normalize(roi_match)
#     roi_nomatch = normalize(roi_nomatch)

#     fov_plot = [np.tile(fov.reshape(fov.shape[0], fov.shape[1], 1), (1, 1, 3)) for fov in FOVs]
#     fov_plot[0][:, :, 0] += roi_match[0] * roi_scale
#     fov_plot[1][:, :, 0] += roi_match[1] * roi_scale
#     fov_plot[0][:, :, 1] += roi_nomatch[0] * roi_scale
#     fov_plot[1][:, :, 2] += roi_nomatch[1] * roi_scale
#     fov_plot = normalize(fov_plot)

#     fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained", sharex=True, sharey=True)
#     ax[0].imshow(fov_plot[0])
#     ax[1].imshow(fov_plot[1])
#     ax[0].set_xlim(ccpair[0] + [-20, 20])
#     ax[0].set_ylim(ccpair[1] + [-20, 20])
#     ax[0].set_title("Session 1")
#     ax[1].set_title("Session 2")
#     plt.show()

#     fig, ax = plt.subplots(2, 2, figsize=(6, 6), layout="constrained", sharey=True)
#     ax[0, 0].imshow(spkmaps_match[0].T, cmap="Reds", aspect="auto", extent=extents[0], vmin=0, vmax=lim)
#     ax[0, 1].imshow(spkmaps_match[1].T, cmap="Reds", aspect="auto", extent=extents[1], vmin=0, vmax=lim)
#     ax[1, 0].imshow(spkmaps_nomatch[0].T, cmap="Greens", aspect="auto", extent=extents[0], vmin=0, vmax=lim)
#     ax[1, 1].imshow(spkmaps_nomatch[1].T, cmap="Blues", aspect="auto", extent=extents[1], vmin=0, vmax=lim)
#     ax[0, 0].set_ylabel("Virtual Position (cm)")
#     ax[1, 0].set_ylabel("Virtual Position (cm)")
#     ax[1, 0].set_xlabel("Trials")
#     ax[1, 1].set_xlabel("Trials")
#     ax[0, 0].set_title("Session 1")
#     ax[0, 1].set_title("Session 2")
#     plt.show()


if __name__ == "__main__":
    args = handle_inputs()
    # example_data = prepare_pair_example_data(args.mouse_name)
    pp_handler = PairPairDatahandler(args.mouse_name, fake_data=True)

    app = QApplication([])
    pp_window = PairPairInteractivePlot(pp_handler)
    pp_window.show()
    sys.exit(app.exec())
