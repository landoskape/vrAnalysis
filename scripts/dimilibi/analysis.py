import os, sys
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import matplotlib as mpl

from helpers import get_sessions, make_and_save_populations, load_population, get_ranks
from rrr_optimization import do_rrr_optimization, load_rrr_results
from network_optimization import do_network_optimization, load_network_results
from rrr_state_optimization import do_rrr_state_optimization


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from vrAnalysis.helpers import argbool, refline


import torch

# NOTE:
# I ran a second round of optimizations with a different population split -- using the population_name="redo1" to distinguish them.


def parse_args():
    parser = ArgumentParser(description="Run analysis on all sessions.")
    parser.add_argument("--redo_pop_splits", default=False, action="store_true", help="Remake population objects and train/val/test splits.")
    parser.add_argument(
        "--redo_pop_splits_behavior",
        default=False,
        action="store_true",
        help="Remake population objects and train/val/test splits with behavior data.",
    )
    parser.add_argument(
        "--population_name", default=None, type=str, help="Name of population object to save (default=None, just uses name of session)"
    )
    parser.add_argument("--rrr", default=False, action="store_true", help="Run reduced rank regression optimization.")
    parser.add_argument("--networks", default=False, action="store_true", help="Run network optimization.")
    parser.add_argument("--rrr_state", default=False, action="store_true", help="Run rrr (state) optimization.")
    parser.add_argument("--skip_completed", type=argbool, default=True, help="Skip completed sessions (default=True)")
    parser.add_argument("--retest_only", type=argbool, default=False, help="Only retest sessions that have already been optimized.")
    parser.add_argument("--compare_rrr_to_networks", type=argbool, default=False, help="Do rrr to network comparison.")
    parser.add_argument("--analyze_rrr_fits", type=argbool, default=False, help="Do analysis of rrr fits.")
    parser.add_argument("--analyze_networks", type=argbool, default=False, help="Do analysis of networks.")
    parser.add_argument("--save", type=argbool, default=True, help="Whether to save results (default=True).")
    return parser.parse_args()


def plot_comparison_test(rrr_results, network_results, rrr_all, network_all, session_ids):
    ranks = rrr_results["ranks"]
    assert ranks == network_results["ranks"], "ranks don't match between rrr and hyperparameters"

    mn_rrr_score = rrr_results["test_scores"].nanmean(dim=0)
    mn_network_score = network_results["test_scores"].nanmean(dim=0)
    std_rrr_score = rrr_results["test_scores"].std(dim=0) / rrr_results["test_scores"].size(0) ** 0.5
    std_network_score = network_results["test_scores"].std(dim=0) / network_results["test_scores"].size(0) ** 0.5

    mn_rrr_scaled_mse = rrr_results["test_scaled_mses"].nanmean(dim=0)
    mn_network_scaled_mse = network_results["test_scaled_mses"].nanmean(dim=0)
    std_rrr_scaled_mse = rrr_results["test_scaled_mses"].std(dim=0) / rrr_results["test_scores"].size(0) ** 0.5
    std_network_scaled_mse = network_results["test_scaled_mses"].std(dim=0) / network_results["test_scores"].size(0) ** 0.5

    # Also compare each session individually (for each rank)
    mouse_names = sorted(list(set(map(lambda x: x[0], session_ids))))
    num_mice = len(mouse_names)
    num_ranks = len(ranks)
    num_sessions = len(rrr_all)
    assert num_sessions == len(network_all), "number of sessions don't match between rrr and networks"
    rrr_score_all = torch.zeros(num_sessions, num_ranks)
    network_score_all = torch.zeros(num_sessions, num_ranks)
    mouse_index = torch.zeros(num_sessions)
    for ises in range(num_sessions):
        rrr_score_all[ises] = torch.tensor(rrr_all[ises]["test_scores"])
        network_score_all[ises] = torch.tensor([torch.mean(nar["test_score"]) for nar in network_all[ises]])
        mouse_index[ises] = mouse_names.index(session_ids[ises][0])

    # Compare the two models across ranks
    model_improvement = network_score_all - rrr_score_all
    session_number = torch.arange(num_sessions).view(-1, 1).expand(-1, num_ranks)
    sort_by_improvement = torch.argsort(torch.mean(model_improvement, dim=1))
    model_mouse_id = mouse_index.clone().view(-1, 1).expand(-1, num_ranks)

    cols = "kb"
    colmouse = mpl.colormaps["tab20"].resampled(num_mice)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained", sharex=False)

    ax[0].plot(ranks, mn_rrr_score, color=cols[0], label="RRR")
    ax[0].fill_between(ranks, mn_rrr_score - std_rrr_score, mn_rrr_score + std_rrr_score, color=cols[0], alpha=0.2)
    ax[0].plot(ranks, mn_network_score, color=cols[1], label="Optimized Networks")
    ax[0].fill_between(ranks, mn_network_score - std_network_score, mn_network_score + std_network_score, color=cols[1], alpha=0.2)
    ax[0].set_ylabel("Test Score")
    ax[0].set_xlabel("Rank")
    # ax[0].set_ylim(-1.1, 1.0)
    ax[0].set_xscale("log")
    ax[0].legend()

    irank = 5
    min_val = min(rrr_score_all[:, irank].min(), network_score_all[:, irank].min())
    max_val = max(rrr_score_all[:, irank].max(), network_score_all[:, irank].max())
    range_val = max_val - min_val
    min_val = min_val - 0.1 * range_val
    max_val = max_val + 0.1 * range_val
    ax[1].scatter(rrr_score_all[:, irank], network_score_all[:, irank], c=mouse_index, cmap=colmouse)
    refline(1, 0, ax=ax[1], color="k", linestyle="--")
    ax[1].set_xlabel("RRR Test Score")
    ax[1].set_ylabel("Network Test Score")
    ax[1].set_title(f"Rank {ranks[irank]}")
    ax[1].set_xlim(min_val, max_val)
    ax[1].set_ylim(min_val, max_val)

    plt.show()

    fig = plt.figure(figsize=(4, 4), layout="constrained")
    plt.scatter(session_number, model_improvement[sort_by_improvement], c=model_mouse_id[sort_by_improvement], s=5, cmap=colmouse)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Session (Sorted by NL Improvement)")
    plt.ylabel("Network Test Score - RRR Test Score")
    plt.title("Model Improvement by Session")
    plt.show()


def plot_rrr_analysis(rrr_results):
    best_alpha = [res["best_alpha"] for res in rrr_results]
    population = [load_population(res["mouse_name"], res["datestr"], res["sessionid"]) for res in rrr_results]
    num_neurons = [pop.size(0) for pop in population]

    plt.scatter(num_neurons, best_alpha)
    plt.yscale("log")
    plt.show()

    from dimilibi import PCA

    random_selection = torch.randperm(len(population))[:10]
    npop_evals = [PCA().fit(population[rsidx].data).get_eigenvalues() for rsidx in random_selection]

    plt.scatter([num_neurons[rsidx] for rsidx in random_selection], [torch.sum(nevals) for nevals in npop_evals])
    plt.show()


def analyze_hyperparameters(results, sessionids):
    ranks = get_ranks()
    mouse_names = sorted(list(set(map(lambda x: x[0], sessionids))))
    num_ranks = len(ranks)
    num_mice = len(mouse_names)

    hprm_names = results["params"].keys()
    num_hprms = len(hprm_names)

    params = {}
    for hprm_name in hprm_names:
        params[hprm_name] = torch.stack([torch.tensor(r) for r in results["params"][hprm_name]]).T

    # organize by mouse
    params_by_mouse = dict(zip(hprm_names, [torch.zeros(num_mice, num_ranks) for _ in hprm_names]))
    for hprm_name in hprm_names:
        mouse_counts = torch.zeros(num_mice)
        for sid, prm_values in zip(sessionids, params[hprm_name]):
            mouse_idx = mouse_names.index(sid[0])
            params_by_mouse[hprm_name][mouse_idx] += prm_values
            mouse_counts[mouse_idx] += 1
        params_by_mouse[hprm_name] /= mouse_counts.unsqueeze(1)

    ignore_yscale = ["transparent_relu"]
    fig, ax = plt.subplots(1, num_hprms, figsize=(num_hprms * 3, 3), layout="constrained")
    for iprm, hprm_name in enumerate(hprm_names):
        ax[iprm].plot(ranks, params_by_mouse[hprm_name].T)
        ax[iprm].set_title(hprm_name)
        ax[iprm].set_xlabel("Rank")
        ax[iprm].set_ylabel(hprm_name)
        ax[iprm].set_xscale("log")
        if hprm_name not in ignore_yscale:
            ax[iprm].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    all_sessions = get_sessions()

    # this set of analyses requires consistent train/val/test splits.
    # make_and_save_populations will generate these splits and save them to a temp file in placeCellSingleSession
    if args.redo_pop_splits:
        make_and_save_populations(all_sessions, args.population_name)

    if args.redo_pop_splits_behavior:
        make_and_save_populations(all_sessions, get_behavior=True, population_name=args.population_name)

    # this set performs optimization and testing of reduced rank regression. It will cache results and save a
    # temporary file in placeCellSingleSession containing the scores and best alpha for each session.
    if args.rrr:
        do_rrr_optimization(all_sessions, skip_completed=args.skip_completed, save=args.save, population_name=args.population_name)

    if args.networks:
        do_network_optimization(
            all_sessions, retest_only=args.retest_only, population_name=args.population_name, skip_completed=args.skip_completed, save=args.save
        )

    if args.rrr_state:
        do_rrr_state_optimization(all_sessions, skip_completed=args.skip_completed, save=args.save, population_name=args.population_name)

    if args.compare_rrr_to_networks:
        rrr_results = load_rrr_results(all_sessions, results="test_by_mouse", population_name=args.population_name)
        network_results = load_network_results(all_sessions, results="test_by_mouse", population_name=args.population_name)[0]
        rrr_all = load_rrr_results(all_sessions, results="all", population_name=args.population_name)
        network_all, session_ids, tested_ranks = load_network_results(all_sessions, results="all", population_name=args.population_name)
        plot_comparison_test(rrr_results, network_results, rrr_all, network_all, session_ids)

    if args.analyze_rrr_fits:
        rrr_results = load_rrr_results(all_sessions, results="all", population_name=args.population_name)

    if args.analyze_networks:
        network_results = load_network_results(all_sessions, results="test_by_mouse", population_name=args.population_name)[0]
        analyze_hyperparameters(network_results)
