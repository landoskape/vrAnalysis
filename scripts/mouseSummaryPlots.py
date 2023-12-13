# add path that contains the vrAnalysis package
import sys
import os
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from argparse import ArgumentParser
from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import analysis

CUTOFFS = (0.2, 0.5)
MAXCUTOFFS = None

def handle_inputs():
    parser = ArgumentParser(description='do summary plots for a mouse')
    parser.add_argument('--mouse', type=str, nargs=1, required=True, type=str, default=None)
    parser.add_argument('--cutoffs', nargs=2, type=float, default=CUTOFFS)
    parser.add_argument('--maxcutoffs', nargs=2, type=float, default=MAXCUTOFFS)
    return parser.parse_args()

if __name__ == "__main__":
    args = handle_inputs()
    mouse = args.mouse
    cutoffs = args.cutoffs
    maxcutoffs = args.maxcutoffs

    track = tracking.tracker(mouse) # get tracker object for mouse
    pcm = analysis.placeCellMultiSession(track, autoload=False) # open up place cell multi session analysis object (don't autoload!!!)
    