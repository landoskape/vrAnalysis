# add path that contains the vrAnalysis package
import sys
import os
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from argparse import ArgumentParser
from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import analysis

mousedb = database.vrDatabase('vrMice')

TRACKED_MICE = mousedb.getTable(tracked=True)['mouseName'].tolist() # list of mice with tracked sessions
CUTOFFS = (0.2, 0.5)
MAXCUTOFFS = (None, None)

def handle_inputs():
    parser = ArgumentParser(description='do pcm analyses')
    parser.add_argument('--mice', nargs="*", type=str, default=TRACKED_MICE)
    parser.add_argument('--cutoffs', nargs=2, type=float, default=CUTOFFS)
    parser.add_argument('--maxcutoffs', nargs=2, type=float, default=MAXCUTOFFS)
    return parser.parse_args()

if __name__ == "__main__":
    args = handle_inputs()
    tracked_mice = args.mice
    cutoffs = args.cutoffs
    maxcutoffs = args.maxcutoffs
    for mouseName in tracked_mice:
        track = tracking.tracker(mouseName) # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False) # open up place cell multi session analysis object (don't autoload!!!)
        for envnum in pcm.environments[pcm.environments > 0]:
            # pcm.plot_pfplasticity(envnum, idx_ses=None, cutoffs=cutoffs, both_reliable=False, withShow=False, withSave=True)
            # pcm.plot_pfplasticity(envnum, idx_ses=None, cutoffs=cutoffs, both_reliable=True, withShow=False, withSave=True)
            # pcm.plot_pfreliability(envnum, idx_ses=None, cutoffs=None, reduction='median', withShow=False, withSave=True)
            # pcm.compare_pfplasticity(envnum, idx_ses=None, cutoffs=cutoffs, both_reliable=False, withShow=False, withSave=True) 
            # pcm.compare_pfplasticity(envnum, idx_ses=None, cutoffs=cutoffs, both_reliable=True, withShow=False, withSave=True) 
            # pcm.plot_rel_plasticity(envnum, idx_ses=None, cutoffs=cutoffs, maxcutoffs=maxcutoffs, withShow=False, withSave=True)
            # for present in ['r2', 'pc']:
            #    pcm.hist_pfplasticity(envnum, idx_ses=None, cutoffs=cutoffs, present=present, split_red=True, withShow=False, withSave=True)
            pass