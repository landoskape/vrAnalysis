# add path that contains the vrAnalysis package
import sys
import os

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

import numpy as np
import scipy as sp
from scipy.stats import ttest_rel
import matplotlib as mpl
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import helpers
from vrAnalysis import analysis
from vrAnalysis import fileManagement as fm

mousedb = database.vrDatabase("vrMice")

# list of mice with tracked sessions
TRACKED_MICE = mousedb.getTable(tracked=True)["mouseName"].tolist()
CUTOFFS = (0.4, 0.7)
MAXCUTOFFS = None
KEEP_PLANES = [1, 2, 3, 4]

# this value is validated by a hacky way of looking at neurons diameter
# neurons average diameter (i.e. average width and height of ROI mask) is uniform between ~3-10 pixels, then drops off
# so it's fair to assume that a nearest neighbors approach would try to identify ROIs within this range from each other
DIST_LIMIT = 10


def save_path(name):
    """helper for getting a save path for data produced by this object"""
    dir_name = "ROICaT_Stats"  # inherited from the RoicatStats analysis object...
    path = fm.analysisPath() / dir_name / name
    if not path.parent.exists():
        path.mkdir(parents=True)
    return path


def handle_inputs():
    parser = ArgumentParser(description="do pcm analyses")
    parser.add_argument(
        "--mice",
        nargs="*",
        type=str,
        default=TRACKED_MICE,
        help="which mice to run script on (default: TRACKED_MICE)",
    )
    parser.add_argument(
        "--cutoffs",
        nargs=2,
        type=float,
        default=CUTOFFS,
        help=f"min cutoffs for restricting cells (where relevant) to those with high reliability (default: {CUTOFFS})",
    )
    parser.add_argument(
        "--maxcutoffs",
        nargs=2,
        type=float,
        default=MAXCUTOFFS,
        help=f"max cutoffs for restricting cells to those with weak reliability (default: {MAXCUTOFFS})",
    )
    parser.add_argument(
        "--keep-planes",
        nargs="*",
        type=int,
        default=KEEP_PLANES,
        help=f"which planes from the multiplane imaging to use (default: {KEEP_PLANES})",
    )
    parser.add_argument(
        "--dist-limit",
        nargs=1,
        default=DIST_LIMIT,
        type=float,
        help=f"determines distance limit (in pixels) for 'nearest-neighbor' approach (default: {DIST_LIMIT})",
    )
    parser.add_argument(
        "--full-plots",
        default=False,
        action="store_true",
        help="will make all plots if used, including ones that take a long time and aren't in the main figure (default=False)",
    )
    parser.add_argument(
        "--show-plots",
        default=False,
        action="store_true",
        help="show plots if used, note that there may be a lot depending on the other settings (default: False)",
    )
    parser.add_argument(
        "--save-plots",
        default=False,
        action="store_true",
        help="save plots if used, will overwrite automatically! (default: False)",
    )
    parser.add_argument(
        "--save-data",
        default=False,
        action="store_true",
        help="save data produced by script if used (default: False)",
    )
    parser.add_argument(
        "--analyze-mouse-data",
        default=False,
        action="store_true",
        help="will process and analyze all mouse data instead of reusing saved data (default=False)",
    )
    return parser.parse_args()


def plot_loop_each_mouse(args):
    """
    plot loop for each mouse
    """
    kwargs = dict(
        sim_name="sConj",
        cutoffs=(0.4, 0.7),
        both_reliable=False,
    )

    plot_kwargs = dict(
        with_show=args.show_plots,
        with_save=args.save_plots,
    )

    # save means/serrors from pfcorr by classification test for across mouse plot of results
    all_means = []
    all_serrors = []
    all_mice = []
    all_prms = []

    for mouse_name in args.mice:
        print(f"Working on {mouse_name}...")
        track = tracking.tracker(mouse_name)
        roistat = analysis.RoicatStats(track, keep_planes=args.keep_planes)

        # select environment (CR mice had different structure and should use environment 2)
        if "CR" in mouse_name:
            envnum = 2
        else:
            # otherwise, choose the virtual enviroment with the most number of sessions
            # (we could do all of the below analyses for multiple environments, but I'd rather focus the analysis on environments
            # with the mostly stable functional tuning because the errors will be dominated by ROICaT rather than plasticity)
            envnum = roistat.env_selector(envmethod="most")

        # get all sessions with chosen environment
        idx_ses = roistat.idx_ses_selector(envnum, sesmethod="all")

        # pick 4 environments, the later the better, but if there's lots of sessions don't do the last one (imaging quality dropped off in last couple sessions usually)
        if len(idx_ses) > 7:
            idx_ses = idx_ses[-6:-2]
        elif len(idx_ses) > 4:
            idx_ses = idx_ses[-4:]
        else:
            print(f"skipping {mouse_name}, because of the small number of sessions: {idx_ses}")
            continue

        # report which environment and sessions are being used
        print("env:", envnum, "idx_ses:", idx_ses)

        # get data for all roicat plots
        sim, corr, tracked, pwdist, nnpair, pwind, prms = roistat.make_roicat_comparison(
            envnum, idx_ses=idx_ses, **kwargs
        )

        # simple plot of place field correlation mean for each session pair grouped by tracked, not-tracked, and nearest-neighbor comparison
        # required to return means across sessions for an across mouse plot
        c_means, c_serrors = roistat.plot_pfcorrmean_by_samediff(
            corr,
            tracked,
            nnpair,
            pwdist,
            prms,
            dist_limit=args.dist_limit,
            return_data=True,
            **plot_kwargs,
        )

        # scatter plot of roicat similarity vs. place field correlation, color-coded by whether the ROIs are tracked
        roistat.plot_sim_vs_pfcorr(sim, corr, tracked, prms, color_mode="tracked", **plot_kwargs)

        # scatter plot of ROI centroid pair-wise distance (post ROICaT alignment) vs. place field correlation, color-coded by tracked
        roistat.plot_pwdist_vs_pfcorr(
            pwdist, corr, tracked, prms, max_dist=50, color_mode="tracked", **plot_kwargs
        )

        if args.full_plots:
            # distribution plot (seaborn boxen) of place field correlation for each pair of session organized by tracked vs. not-tracked
            # this takes a while to compute the distributions in seaborn...
            roistat.plot_pfcorr_by_samediff(corr, tracked, nnpair, prms, **plot_kwargs)

        if args.full_plots:
            # histogram of similarity metric
            # it's just not a very useful plot in comparison to others
            roistat.plot_similarity_histograms(sim, prms, **plot_kwargs)

        if args.full_plots:
            # plot of place field correlation given pair-wise distance, grouped by tracked, not-tracked, and nearest-neighbors
            roistat.plot_pfcorr_vs_pwdist_by_group(
                corr, tracked, pwdist, nnpair, prms, **plot_kwargs
            )

        # collect data from each mouse
        all_means.append(c_means)
        all_serrors.append(c_serrors)
        all_mice.append(mouse_name)
        all_prms.append(prms)

        print("")

        # clean anything up that was plotted if not showing plots
        if not args.show_plots:
            plt.close("all")

    return all_mice, all_prms, all_means, all_serrors


def example_figure(mouse_name):
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
    _, corr, tracked, pwdist, _, pwind, prms = roistat.make_roicat_comparison(
        envnum, idx_ses=idx_ses, **kwargs
    )
    centroids = roistat.track.get_centroids(
        idx_ses=prms["idx_ses"], cat_planes=True, combine=True, keep_planes=roistat.keep_planes
    )

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
        assert len(idx) == len(
            prms["idx_ses_pairs"]
        ), "idx and prms['idx_ses_pairs'] don't have same number of elements"

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

    # set parameters for possible pairs
    mincorr = 0.6  # minimum correlation for matched pair
    maxcorr = 0.2  # maximum correlation for non-matched pair
    maxdist = 6  # maximum distance for any example pair

    # set parameters for pair of pairs (distance apart etc.)
    maxdistpp = 16  # maximum distance for pair of pairs
    mindistpp = 6  # minimum distance for pair of pairs

    # get possible matches and not-matches
    idx_possible_match = [
        np.where(t & (c > mincorr) & (pwd < maxdist))[0]
        for t, c, pwd in zip(tracked, corr, pwdist)
    ]
    idx_possible_nomatch = [
        np.where(~t & (c < maxcorr) & (pwd < maxdist))[0]
        for t, c, pwd in zip(tracked, corr, pwdist)
    ]

    # get possible match/nomatch pair indices
    roi_idx_match = [pwi[:, i] for pwi, i in zip(pwind, idx_possible_match)]
    roi_idx_nomatch = [pwi[:, i] for pwi, i in zip(pwind, idx_possible_nomatch)]

    # get average centroid of pairs
    centroids_match = _get_centroids_match(centroids, roi_idx_match, prms)
    centroids_nomatch = _get_centroids_match(centroids, roi_idx_nomatch, prms)
    acentroid_match = [np.mean(c, axis=0) for c in centroids_match]
    acentroid_nomatch = [np.mean(c, axis=0) for c in centroids_nomatch]

    # get distance between pairs of possible match/nomatch for each pair of sessions
    dist_match_nomatch = [
        sp.spatial.distance.cdist(am, anm) for am, anm in zip(acentroid_match, acentroid_nomatch)
    ]
    good_distance = [(dmn < maxdistpp) & (dmn > mindistpp) for dmn in dist_match_nomatch]

    # check if pairs include any of the same ROIs
    no_duplicates = _check_no_duplicates(roi_idx_match, roi_idx_nomatch)

    # get plane indices for each pair
    plane_pair_match = _get_plane_of_pair(roistat, roi_idx_match, prms)
    plane_pair_nomatch = _get_plane_of_pair(roistat, roi_idx_nomatch, prms)
    same_plane = [
        ppm.reshape(-1, 1) == ppn.reshape(1, -1)
        for ppm, ppn in zip(plane_pair_match, plane_pair_nomatch)
    ]

    # possible pair of pairs boolean array
    ipair_match, ipair_nomatch = helpers.named_transpose(
        [np.where(gd & nd & smp) for gd, nd, smp in zip(good_distance, no_duplicates, same_plane)]
    )

    # indices to ROI pairs for good pairs of pairs
    pp_roi_match = [rim[:, ipm] for rim, ipm in zip(roi_idx_match, ipair_match)]
    pp_roi_nomatch = [rinm[:, ipnm] for rinm, ipnm in zip(roi_idx_nomatch, ipair_nomatch)]
    pp_centroid_match = [acm[ipm] for acm, ipm in zip(acentroid_match, ipair_match)]
    pp_centroid_nomatch = [acnm[ipnm] for acnm, ipnm in zip(acentroid_nomatch, ipair_nomatch)]

    # get plane indices for each pair
    plane_pair = _get_plane_of_pair(roistat, pp_roi_match, prms)

    # check that match and no-match pairs are in same plane
    assert all(
        [
            np.all(pp == ppnm)
            for pp, ppnm in zip(plane_pair, _get_plane_of_pair(roistat, pp_roi_nomatch, prms))
        ]
    )

    # Now make figures
    def _get_masks(roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch):
        ROIs = roistat.track.get_ROIs(
            as_coo=False, idx_ses=prms["idx_ses_pairs"][isespair], keep_planes=roistat.keep_planes
        )
        ROIs = [sp.sparse.vstack(rois, format="csr") for rois in ROIs]  # concatenate across planes

        roi_match = [
            ROIs[0][pp_roi_match[isespair][0, idxroipair]],
            ROIs[1][pp_roi_match[isespair][1, idxroipair]],
        ]
        roi_nomatch = [
            ROIs[0][pp_roi_nomatch[isespair][0, idxroipair]],
            ROIs[1][pp_roi_nomatch[isespair][1, idxroipair]],
        ]

        num_pixels = roi_match[0].shape[1]
        hw = int(np.sqrt(num_pixels))
        roi_match = [r.toarray().reshape(hw, hw) for r in roi_match]
        roi_nomatch = [r.toarray().reshape(hw, hw) for r in roi_nomatch]

        return roi_match, roi_nomatch

    def _get_spkmaps(roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch):
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
        spkmap_match = [sp.stats.zscore(sm, axis=None) for sm in spkmap_match]
        spkmap_nomatch = [sp.stats.zscore(sm, axis=None) for sm in spkmap_nomatch]
        distedges = roistat.pcss[prms["idx_ses_pairs"][isespair][0]].distedges
        return spkmap_match, spkmap_nomatch, distedges

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
        np.stack(
            (pp_centroid_match[isespair][idxroipair], pp_centroid_nomatch[isespair][idxroipair])
        ),
        axis=0,
    )

    spkmaps_match, spkmaps_nomatch, distedges = _get_spkmaps(
        roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch
    )
    extents = [[0, sm.shape[0], distedges[0], distedges[-1]] for sm in spkmaps_match]

    # make heatmaps of FOV
    FOVs = [
        roistat.track.rundata[plane_pair[isespair][idxroipair]]["aligner"][
            "ims_registered_nonrigid"
        ][i]
        for i in prms["idx_ses_pairs"][isespair]
    ]
    roi_match, roi_nomatch = _get_masks(
        roistat, isespair, idxroipair, prms, pp_roi_match, pp_roi_nomatch
    )
    normalize = lambda data: [
        ((d - d.min()) / (d.max() - d.min())).astype(np.float32) for d in data
    ]
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
    ax[0, 0].imshow(
        spkmaps_match[0].T, cmap="Reds", aspect="auto", extent=extents[0], vmin=0, vmax=lim
    )
    ax[0, 1].imshow(
        spkmaps_match[1].T, cmap="Reds", aspect="auto", extent=extents[1], vmin=0, vmax=lim
    )
    ax[1, 0].imshow(
        spkmaps_nomatch[0].T, cmap="Greens", aspect="auto", extent=extents[0], vmin=0, vmax=lim
    )
    ax[1, 1].imshow(
        spkmaps_nomatch[1].T, cmap="Blues", aspect="auto", extent=extents[1], vmin=0, vmax=lim
    )
    ax[0, 0].set_ylabel("Virtual Position (cm)")
    ax[1, 0].set_ylabel("Virtual Position (cm)")
    ax[1, 0].set_xlabel("Trials")
    ax[1, 1].set_xlabel("Trials")
    ax[0, 0].set_title("Session 1")
    ax[0, 1].set_title("Session 2")
    plt.show()


def mouse_summary_plot(mouse_data):
    """
    make plots and do statistics on each mouse in the dataset
    """
    used_args = mouse_data["args"]
    group_names = ["tracked", f"nearest neighbors", "random pairs"]
    group_colors = ["b", "r", "k"]

    num_groups = len(group_colors)
    num_mice = len(mouse_data["means"])

    mouse_colors = mpl.colormaps["Dark2"]

    # summary data
    mn_per_mouse = np.stack([np.mean(mdata, axis=1) for mdata in mouse_data["means"]])
    cmp2random = ttest_rel(mn_per_mouse[:, 0], mn_per_mouse[:, 2])
    cmp2nearest = ttest_rel(mn_per_mouse[:, 0], mn_per_mouse[:, 1])
    cmp_nearest2random = ttest_rel(mn_per_mouse[:, 1], mn_per_mouse[:, 2])

    print("paired t-test ROICaT pairs compared to random pairs:")
    print(cmp2random)

    print("paired t-test ROICaT pairs compared to nearest neighbor post alignment:")
    print(cmp2nearest)

    print("paired t-test nearest pairs compared to random pairs:")
    print(cmp_nearest2random)

    fig = plt.figure(layout="constrained")
    for imouse, (mdata, mname) in enumerate(zip(mouse_data["means"], mouse_data["mice"])):
        plt.plot(
            range(num_groups),
            mdata,
            color=(mouse_colors(imouse), 0.3),
            linewidth=1.5,
            linestyle="-.",
            zorder=0,
        )
        plt.plot(
            range(num_groups),
            np.mean(mdata, axis=1),
            color=mouse_colors(imouse),
            marker="o",
            linewidth=1.5,
            label=f"Mouse {imouse}",
            zorder=1,
        )
    plt.xlim(-0.25, num_groups - 0.75)
    # plt.xlabel('Group')
    plt.ylabel("Place Field Correlation")
    plt.xticks(range(num_groups), group_names)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    args = handle_inputs()

    if args.analyze_mouse_data:
        # if collecting mouse data:
        mice, prms, means, errors = plot_loop_each_mouse(args)

        # create mouse_data dictionary for saving and/or plotting
        mouse_data = dict(
            mice=mice,
            prms=prms,
            means=means,
            errors=errors,
            args=args,
        )

        # save if requested
        if args.save_data:
            np.save(save_path("mouse_data"), mouse_data, allow_pickle=True)
            print("saved mouse data!")

    else:
        # if using pre-saved mouse data:
        mouse_data = np.load(save_path("mouse_data.npy"), allow_pickle=True).item()
        print("loaded mouse data!")

    # make summary plots
    mouse_summary_plot(mouse_data)
