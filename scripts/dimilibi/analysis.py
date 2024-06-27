from argparse import ArgumentParser

from helpers import get_sessions, make_and_save_populations
from rrr_optimization import do_rrr_optimization

# from .network_optimization import do_network_optimization


def parse_args():
    parser = ArgumentParser(description="Run analysis on all sessions.")
    parser.add_argument("--redo_pop_splits", default=False, action="store_true", help="Remake population objects and train/val/test splits.")
    parser.add_argument("--rrr", default=False, action="store_true", help="Run reduced rank regression optimization.")
    parser.add_argument("--network", default=False, action="store_true", help="Run network optimization.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_sessions = get_sessions()
    if args.redo_pop_splits:
        make_and_save_populations(all_sessions)
    if args.rrr:
        do_rrr_optimization(all_sessions)
    if args.network:
        pass
        # do_network_optimization(all_sessions)
