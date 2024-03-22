# add path that contains the vrAnalysis package
import sys
import os

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from argparse import ArgumentParser
import numpy as np
from vrAnalysis import database
from vrAnalysis.analysis import SpikemapMethods

mousedb = database.vrDatabase("vrMice")
sessiondb = database.vrDatabase("vrSessions")

MICE = mousedb.getTable(experimentID=1)["mouseName"].tolist()  # list of mice with Imaging data
CUTOFFS = (0.4, 0.7)
MAXCUTOFFS = None


SETTINGS = [
    dict(
        distStep=1,
        smooth=None,
    ),
    dict(
        distStep=1,
        smooth=0.1,
    ),
    dict(
        distStep=1,
        smooth=1,
    ),
    dict(
        distStep=1,
        smooth=5,
    ),
    dict(
        distStep=2,
        smooth=1,
    ),
    dict(
        distStep=2,
        smooth=5,
    ),
    dict(
        distStep=5,
        smooth=1,
    ),
    dict(
        distStep=5,
        smooth=5,
    ),
]


def handle_inputs():
    parser = ArgumentParser(description="do analysis of spikemap comparison")
    parser.add_argument("--mice", nargs="*", type=str, default=MICE, help="which mice to do analyses for (default=All with imaging)")
    parser.add_argument("--sessions-per-mouse", type=int, default=1, help="number of (randomly selected) sessions for each mouse (default=1)")
    parser.add_argument("--rois-per-session", type=int, default=10, help="number of (randomly selected) ROIs for each session (default=10)")
    parser.add_argument("--cutoffs", nargs=2, type=float, default=CUTOFFS, help="cutoffs for reliability (default=(0.4, 0.7))")
    parser.add_argument("--maxcutoffs", nargs=2, type=float, default=MAXCUTOFFS, help="maxcutoffs for reliability (default=None)")
    parser.add_argument("--save", default=False, action="store_true", help="whether to save figures (default=False)")
    parser.add_argument("--no-show", default=False, action="store_true", help="if used, won't show figures (default=False)")
    return parser.parse_args()


if __name__ == "__main__":
    # Handle input arguments
    args = handle_inputs()
    mice = args.mice
    sessions_per_mouse = args.sessions_per_mouse
    rois_per_session = args.rois_per_session
    cutoffs = args.cutoffs
    maxcutoffs = args.maxcutoffs
    withSave = args.save
    withShow = not args.no_show

    # For each mouse in list,
    for mouse_name in mice:
        # Get list of possible sessions and choose a few randomly
        ises = sessiondb.iterSessions(mouseName=mouse_name)
        ses_list = np.random.choice(ises, sessions_per_mouse)

        # then go through sessions,
        for ses in ses_list:
            # Make a spikemap method analysis object
            pcss = SpikemapMethods(ses, autoload=False)

            # load spkmaps and reliability with different settings
            spkmaps, rawspkmaps, relmse, relcor, distcenters, settings = pcss.compare_methods(SETTINGS)

            # plot example ROIs place fields from the different methods
            pcss.plot_examples(
                spkmaps,
                rawspkmaps,
                relmse,
                relcor,
                distcenters,
                settings,
                num_to_plot=rois_per_session,
                cutoffs=cutoffs,
                maxcutoffs=maxcutoffs,
                withSave=True,
                withShow=False,
            )

            # plot reliability distribution using the different methods
            pcss.plot_reliability_distribution(relmse, relcor, settings, withSave=True, withShow=False)
