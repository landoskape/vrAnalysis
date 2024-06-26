import os, sys
from argparse import ArgumentParser

from helpers import get_sessions, make_and_save_populations
from rrr_optimization import do_rrr_optimization
from network_optimization import do_network_optimization

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from vrAnalysis.helpers import argbool


def parse_args():
    parser = ArgumentParser(description="Run analysis on all sessions.")
    parser.add_argument("--redo_pop_splits", default=False, action="store_true", help="Remake population objects and train/val/test splits.")
    parser.add_argument("--rrr", default=False, action="store_true", help="Run reduced rank regression optimization.")
    parser.add_argument("--networks", default=False, action="store_true", help="Run network optimization.")
    parser.add_argument("--skip_completed", type=argbool, default=True, help="Skip completed sessions (default=True)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_sessions = get_sessions()

    # this set of analyses requires consistent train/val/test splits.
    # make_and_save_populations will generate these splits and save them to a temp file in placeCellSingleSession
    if args.redo_pop_splits:
        make_and_save_populations(all_sessions)

    # this set performs optimization and testing of reduced rank regression. It will cache results and save a
    # temporary file in placeCellSingleSession containing the scores and best alpha for each session.
    if args.rrr:
        do_rrr_optimization(all_sessions)

    if args.networks:
        do_network_optimization(all_sessions, skip_completed=args.skip_completed)
