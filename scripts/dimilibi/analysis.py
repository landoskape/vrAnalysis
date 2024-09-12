import os, sys
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from helpers import get_sessions, make_and_save_populations, load_population, get_ranks
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
    parser.add_argument("--hyperparameters", default=False, action="store_true", help="Run hyperparameter optimization.")
    parser.add_argument("--skip_completed", type=argbool, default=True, help="Skip completed sessions (default=True)")
    parser.add_argument("--retest_only", type=argbool, default=False, help="Only retest sessions that have already been optimized.")
    parser.add_argument("--compare_rrr_to_networks", type=argbool, default=False, help="Do rrr to network comparison.")
    parser.add_argument("--analyze_rrr_fits", type=argbool, default=False, help="Do analysis of rrr fits.")
    parser.add_argument("--analyze_hyperparameters", type=argbool, default=False, help="Do analysis of hyperparameters.")
    parser.add_argument("--save", type=argbool, default=True, help="Whether to save results (default=True).")
    return parser.parse_args()

def plot_comparison_test(rrr_results, network_results, network_parameters, hyperparameter_results, ignore_bvae=True):
    ranks = rrr_results["ranks"]
    assert all([ranks == nres["ranks"] for nres in network_results]), "ranks don't match between rrr and networks"
    assert ranks == hyperparameter_results["ranks"], "ranks don't match between rrr and hyperparameters"
    
    network_names = list(network_parameters.keys())
    if ignore_bvae:
        if "betavae" in network_names:
            idx_bvae = network_names.index("betavae")
            network_names.pop(idx_bvae)
            network_results.pop(idx_bvae)

    mn_rrr_score = rrr_results["test_scores"].nanmean(dim=0)
    mn_network_score = [nres["test_scores"].nanmean(dim=0) for nres in network_results]
    mn_hyperparameter_score = hyperparameter_results["test_scores"].nanmean(dim=0)
    std_rrr_score = rrr_results["test_scores"].std(dim=0) / rrr_results["test_scores"].size(0)**0.5
    std_network_score = [nres["test_scores"].std(dim=0) / nres["test_scores"].size(0)**0.5 for nres in network_results]
    std_hyperparameter_score = hyperparameter_results["test_scores"].std(dim=0) / hyperparameter_results["test_scores"].size(0)**0.5
    
    mn_rrr_scaled_mse = rrr_results["test_scaled_mses"].nanmean(dim=0)
    mn_network_scaled_mse = [nres["test_scaled_mses"].nanmean(dim=0) for nres in network_results]
    mn_hyperparameter_scaled_mse = hyperparameter_results["test_scaled_mses"].nanmean(dim=0)
    std_rrr_scaled_mse = rrr_results["test_scaled_mses"].std(dim=0)
    std_network_scaled_mse = [nres["test_scaled_mses"].std(dim=0) for nres in network_results]
    std_hyperparameter_scaled_mse = hyperparameter_results["test_scaled_mses"].std(dim=0)
    
    cols = "krb"

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained", sharex=True)

    # ax[0].plot(ranks, mn_rrr_score, color=cols[0], label='RRR')
    # ax[0].fill_between(ranks, mn_rrr_score - std_rrr_score, mn_rrr_score + std_rrr_score, color=cols[0], alpha=0.2)
    # for i, mn_score in enumerate(mn_network_score):
    #     ax[0].plot(ranks, mn_score, color=cols[1], label=f"Network ({network_names[i]})")
    #     ax[0].fill_between(ranks, mn_score - std_network_score[i], mn_score + std_network_score[i], color=cols[1], alpha=0.2)
    ax[0].plot(ranks, mn_hyperparameter_score, color=cols[2], label="Optimized Networks")
    ax[0].fill_between(ranks, mn_hyperparameter_score - std_hyperparameter_score, mn_hyperparameter_score + std_hyperparameter_score, color=cols[2], alpha=0.2)
    ax[0].set_ylabel("Test Score")
    ax[0].set_xlabel("Rank")
    # ax[0].set_ylim(-1.1, 1.0)
    ax[0].set_xscale('log')
    ax[0].legend()
    
    # ax[1].plot(ranks, mn_rrr_scaled_mse, color=cols[0], label='RRR')
    # for i, mn_scaled_mse in enumerate(mn_network_scaled_mse):
    #     ax[1].plot(ranks, mn_scaled_mse, label=f"Network ({network_names[i]})")
    # ax[1].plot(ranks, mn_hyperparameter_scaled_mse, label="Optimized Networks")
    ax[1].set_ylabel("Test Scaled MSE")
    ax[1].set_xlabel("Rank")
    ax[1].set_ylim(0.0, 1.1)
    ax[1].legend()

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
    fig, ax = plt.subplots(1, num_hprms, figsize=(num_hprms*3, 3), layout="constrained")
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
        make_and_save_populations(all_sessions)

    # this set performs optimization and testing of reduced rank regression. It will cache results and save a
    # temporary file in placeCellSingleSession containing the scores and best alpha for each session.
    if args.rrr:
        do_rrr_optimization(all_sessions, skip_completed=args.skip_completed, save=args.save)

    if args.networks:
        do_network_optimization(all_sessions, retest_only=args.retest_only, skip_completed=args.skip_completed, save=args.save)

    if args.compare_rrr_to_networks:
        rrr_results = load_rrr_results(all_sessions, results="test_by_mouse")
        network_results = load_network_results(all_sessions, results="test_by_mouse")[0]
        plot_comparison_test(rrr_results, network_results)

    if args.analyze_rrr_fits:
        rrr_results = load_rrr_results(all_sessions, results="all")

    if args.analyze_networks:
        network_results = load_network_results(all_sessions, results="test_by_mouse")
        analyze_hyperparameters(network_results)
        
        