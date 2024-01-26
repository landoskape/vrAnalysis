# add path that contains the vrAnalysis package
import sys
import os
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

import numpy as np
from scipy.stats import ttest_rel
import matplotlib as mpl
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import analysis
from vrAnalysis import fileManagement as fm

mousedb = database.vrDatabase('vrMice')

TRACKED_MICE = mousedb.getTable(tracked=True)['mouseName'].tolist() # list of mice with tracked sessions
CUTOFFS = (0.4, 0.7)
MAXCUTOFFS = None
KEEP_PLANES = [1, 2, 3, 4]

# this is validated by a hacky way of looking at neurons diameter
# neurons average diameter (i.e. average width and height of ROI mask) is uniform between ~3-10 pixels, then drops off
# so it's fair to assume that a nearest neighbors approach would try to identify ROIs within this range from each other
DIST_LIMIT = 10 

def save_path(name):
    """helper for getting a save path for data produced by this object"""
    dir_name = 'ROICaT_Stats' # inherited from the RoicatStats analysis object...
    path = fm.analysisPath() / dir_name / name
    if not path.parent.exists():
        path.mkdir(parents=True)
    return path


def handle_inputs():
    parser = ArgumentParser(description='do pcm analyses')
    parser.add_argument('--mice', nargs="*", type=str, default=TRACKED_MICE, help='which mice to run script on')
    parser.add_argument('--cutoffs', nargs=2, type=float, default=CUTOFFS, help='min cutoffs for restricting cells (where relevant) to those with high reliability')
    parser.add_argument('--maxcutoffs', nargs=2, type=float, default=MAXCUTOFFS, help='max cutoffs for restricting cells to those with weak reliability')
    parser.add_argument('--keep-planes', nargs="*", type=int, default=KEEP_PLANES, help='which planes from the multiplane imaging to use')
    parser.add_argument('--dist-limit', nargs=1, default=DIST_LIMIT, type=float, help='determines distance limit (in pixels) for "nearest-neighbor" approach')
    parser.add_argument('--full-plots', default=False, action='store_true', help='will make all plots if used, including ones that take a long time and aren''t in the main figure')
    parser.add_argument('--show-plots', default=False, action='store_true', help='show plots if used, note that there may be a lot depending on the other settings....')
    parser.add_argument('--save-plots', default=False, action='store_true', help='save plots if used, will overwrite automatically!')
    parser.add_argument('--save-data', default=False, action='store_true', help='save data produced by script if used')

    parser.add_argument('--use-saved-mouse-data', default=False, action='store_true', help='will use saved mouse data instead of remaking it if used')

    return parser.parse_args()

def plot_loop_each_mouse(args):
    """
    plot loop for each mouse
    """
    kwargs = dict(
        sim_name='sConj',
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
        if 'CR' in mouse_name:
            envnum = 2
        else:
            # otherwise, choose the virtual enviroment with the most number of sessions
            # (we could do all of the below analyses for multiple environments, but I'd rather focus the analysis on environments
            # with the mostly stable functional tuning because the errors will be dominated by ROICaT rather than plasticity) 
            envnum = roistat.env_selector(envmethod='most')

        # get all sessions with chosen environment
        idx_ses = roistat.idx_ses_selector(envnum, sesmethod='all')

        # pick 4 environments, the later the better, but if there's lots of sessions don't do the last one (imaging quality dropped off in last couple sessions usually)
        if len(idx_ses)>7:
            idx_ses = idx_ses[-6:-2]
        elif len(idx_ses)>4:
            idx_ses = idx_ses[-4:]
        else:
            print(f'skipping {mouse_name}, because of the small number of sessions: {idx_ses}')
            continue
        
        # report which environment and sessions are being used
        print('env:', envnum, 'idx_ses:', idx_ses)
        
        # get data for all roicat plots
        sim, corr, tracked, pwdist, nnpair, prms = roistat.make_roicat_comparison(envnum, idx_ses=idx_ses, **kwargs)
        
        # simple plot of place field correlation mean for each session pair grouped by tracked, not-tracked, and nearest-neighbor comparison
        # required to return means across sessions for an across mouse plot
        c_means, c_serrors = roistat.plot_pfcorrmean_by_samediff(corr, tracked, nnpair, pwdist, prms, dist_limit=args.dist_limit, return_data=True, **plot_kwargs)

        # scatter plot of roicat similarity vs. place field correlation, color-coded by whether the ROIs are tracked
        roistat.plot_sim_vs_pfcorr(sim, corr, tracked, prms, color_mode='tracked', **plot_kwargs)

        # scatter plot of ROI centroid pair-wise distance (post ROICaT alignment) vs. place field correlation, color-coded by tracked
        roistat.plot_pwdist_vs_pfcorr(pwdist, corr, tracked, prms, max_dist=50, color_mode='tracked', **plot_kwargs)

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
            roistat.plot_pfcorr_vs_pwdist_by_group(corr, tracked, pwdist, nnpair, prms, **plot_kwargs)

        # collect data from each mouse
        all_means.append(c_means)
        all_serrors.append(c_serrors)
        all_mice.append(mouse_name)
        all_prms.append(prms)

        print('')

        # clean anything up that was plotted if not showing plots
        if not args.show_plots:
            plt.close('all')

    return all_mice, all_prms, all_means, all_serrors

def mouse_summary_plot(mouse_data):
    """
    make plots and do statistics on each mouse in the dataset
    """
    used_args = mouse_data['args']
    group_names = ['tracked', f'nearest neighbors', 'random pairs']
    group_colors = ['b', 'r', 'k']
    
    num_groups = len(group_colors)
    num_mice = len(mouse_data['means'])

    mouse_colors = mpl.colormaps['Dark2']

    # summary data
    mn_per_mouse = np.stack([np.mean(mdata, axis=1) for mdata in mouse_data['means']])
    cmp2random = ttest_rel(mn_per_mouse[:, 0], mn_per_mouse[:, 2])
    cmp2nearest = ttest_rel(mn_per_mouse[:, 0], mn_per_mouse[:, 1])
    cmp_nearest2random = ttest_rel(mn_per_mouse[:, 1], mn_per_mouse[:, 2])

    print('paired t-test ROICaT pairs compared to random pairs:')
    print(cmp2random)

    print('paired t-test ROICaT pairs compared to nearest neighbor post alignment:')
    print(cmp2nearest)

    print('paired t-test nearest pairs compared to random pairs:')
    print(cmp_nearest2random)

    fig = plt.figure(layout='constrained')
    for imouse, (mdata, mname) in enumerate(zip(mouse_data['means'], mouse_data['mice'])):
        plt.plot(range(num_groups), mdata, color=(mouse_colors(imouse), 0.3), linewidth=1.5, linestyle='-.', zorder=0)
        plt.plot(range(num_groups), np.mean(mdata, axis=1), color=mouse_colors(imouse), marker='o', linewidth=1.5, label=f"Mouse {imouse}", zorder=1)    
    plt.xlim(-0.25, num_groups-0.75)
    # plt.xlabel('Group')
    plt.ylabel('Place Field Correlation')
    plt.xticks(range(num_groups), group_names)
    plt.legend(loc='upper right')
    plt.show()




if __name__ == "__main__":
    args = handle_inputs()

    if not args.use_saved_mouse_data:
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
            np.save(save_path('mouse_data'), mouse_data, allow_pickle=True)
            print('saved mouse data!')

    else:
        # if using pre-saved mouse data:
        mouse_data = np.load(save_path('mouse_data.npy'), allow_pickle=True).item()
        print('loaded mouse data!')

    # make summary plots 
    mouse_summary_plot(mouse_data)

