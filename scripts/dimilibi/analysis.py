import os, sys
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from helpers import get_sessions, make_and_save_populations
from rrr_optimization import do_rrr_optimization, load_rrr_results
from network_optimization import do_network_optimization, load_network_results

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from vrAnalysis.helpers import argbool


import torch

def parse_args():
    parser = ArgumentParser(description="Run analysis on all sessions.")
    parser.add_argument("--redo_pop_splits", default=False, action="store_true", help="Remake population objects and train/val/test splits.")
    parser.add_argument("--rrr", default=False, action="store_true", help="Run reduced rank regression optimization.")
    parser.add_argument("--networks", default=False, action="store_true", help="Run network optimization.")
    parser.add_argument("--skip_completed", type=argbool, default=True, help="Skip completed sessions (default=True)")
    parser.add_argument("--compare_rrr_to_networks", type=argbool, default=False, help="Do rrr to network comparison.")
    return parser.parse_args()

def plot_comparison_test(rrr_results, network_results, network_parameters):
    ranks = rrr_results["ranks"]
    assert all([ranks == nres["ranks"] for nres in network_results]), "ranks don't match between rrr and networks"
    
    network_names = list(network_parameters.keys())

    mn_rrr_score = rrr_results["test_scores"].nanmean(dim=0)
    mn_network_score = [nres["test_scores"].nanmean(dim=0) for nres in network_results]
    
    mn_rrr_scaled_mse = rrr_results["test_scaled_mses"].nanmean(dim=0)
    mn_network_scaled_mse = [nres["test_scaled_mses"].nanmean(dim=0) for nres in network_results]
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained", sharex=True)

    ax[0].plot(ranks, mn_rrr_score, color='k', label='RRR')
    for i, mn_score in enumerate(mn_network_score):
        ax[0].plot(ranks, mn_score, label=f"Network {network_names[i]}")
    ax[0].set_ylabel("Test Score")
    ax[0].set_xlabel("Rank")
    ax[0].set_ylim(-1.1, 1.0)
    ax[0].legend()
    
    ax[1].plot(ranks, mn_rrr_scaled_mse, color='k', label='RRR')
    for i, mn_scaled_mse in enumerate(mn_network_scaled_mse):
        ax[1].plot(ranks, mn_scaled_mse, label=f"Network {network_names[i]}")
    ax[1].set_ylabel("Test Scaled MSE")
    ax[1].set_xlabel("Rank")
    ax[1].set_ylim(0.0, 1e5)
    ax[1].legend()

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

    if args.compare_rrr_to_networks:
        rrr_results = load_rrr_results(all_sessions, results="test_by_mouse")
        network_results, network_parameters = load_network_results(all_sessions, results="test_by_mouse")
        plot_comparison_test(rrr_results, network_results, network_parameters)

        