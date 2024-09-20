import os, sys
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import matplotlib as mpl

from helpers import get_sessions, make_and_save_populations, load_population, get_ranks
from rrr_optimization import do_rrr_optimization, load_rrr_results
from network_optimization import do_network_optimization, load_network_results
from rrr_state_optimization import do_rrr_state_optimization, load_rrr_state_results, make_rrr_state_example, add_rrr_state_results


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from vrAnalysis.helpers import argbool, refline

import numpy as np
import torch

# NOTE:
# I ran a second round of optimizations with a different population split -- using the population_name="redo1" to distinguish them.
# For "state" population splits, I added one with only planes 1/2 called "fast"...


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


def analyze_rrr_state(results):
    mouse_name = [res["mouse_name"] for res in results]
    test_score = torch.tensor([res["test_score"] for res in results])
    test_scaled_mse = torch.tensor([res["test_scaled_mse"] for res in results])
    test_score_direct = torch.tensor([res["test_score_direct"] for res in results])
    test_scaled_mse_direct = torch.tensor([res["test_scaled_mse_direct"] for res in results])
    test_score_pfpred = torch.tensor([res["pf_pred_score"] for res in results])
    pos_score_encoder = torch.tensor([res["encoder_position_score"] for res in results])
    pos_score_latent = torch.tensor([res["latent_position_score"] for res in results])

    # Use scipy to run ttest on the scores
    from scipy.stats import ttest_rel

    tres = ttest_rel(test_score, test_score_direct)
    print("Test score compared to direct model pvalue: ", tres.pvalue, test_score.mean(), test_score_direct.mean())

    idx_valid = (torch.abs(pos_score_encoder) < 10) & (torch.abs(pos_score_latent) < 10)
    tres = ttest_rel(pos_score_encoder[idx_valid], pos_score_latent[idx_valid])
    print("Position score encoder compared to latent pvalue: ", tres.pvalue, pos_score_encoder[idx_valid].mean(), pos_score_latent[idx_valid].mean())

    mice = sorted(list(set(mouse_name)))
    num_mice = len(mice)
    cmap = mpl.colormaps["turbo"].resampled(num_mice)
    cols = [cmap(mice.index(mn)) for mn in mouse_name]

    # Average across mice
    mouse_score = torch.zeros(num_mice)
    mouse_scaled_mse = torch.zeros(num_mice)
    mouse_score_direct = torch.zeros(num_mice)
    mouse_scaled_mse_direct = torch.zeros(num_mice)
    mouse_score_pfpred = torch.zeros(num_mice)
    mouse_pos_score_encoder = torch.zeros(num_mice)
    mouse_pos_score_latent = torch.zeros(num_mice)
    for imouse, mouse in enumerate(mice):
        idx = torch.tensor([mn == mouse for mn in mouse_name])
        mouse_score[imouse] = test_score[idx].mean()
        mouse_scaled_mse[imouse] = test_scaled_mse[idx].mean()
        mouse_score_direct[imouse] = test_score_direct[idx].mean()
        mouse_scaled_mse_direct[imouse] = test_scaled_mse_direct[idx].mean()
        mouse_score_pfpred[imouse] = test_score_pfpred[idx].mean()
        mouse_pos_score_encoder[imouse] = pos_score_encoder[idx].mean()
        mouse_pos_score_latent[imouse] = pos_score_latent[idx].mean()

    plt.rcParams.update({"font.size": 12})

    xd = [0, 1, 2]
    fig, ax = plt.subplots(1, figsize=(4, 4), layout="constrained")
    # for ts, tsd, c in zip(test_score, test_score_direct, cols):
    #     ax.plot(xd, [ts, tsd], color=c, linestyle="-", alpha=0.1, linewidth=0.1)
    for imouse, mouse in enumerate(mice):
        yd = [mouse_score_pfpred[imouse], mouse_score[imouse], mouse_score_direct[imouse]]
        ax.plot(xd, yd, color=cmap(imouse), label=mouse, linewidth=1.5, linestyle="-", marker="o")
    # ax.plot(xd, [mouse_score.mean(), mouse_score_direct.mean()], color="k", linestyle="-", marker="o", label="Mean")
    ax.set_xticks(xd)
    ax.set_xticklabels(["PF-Pred", "Pos-Model", "RRR"])
    ax.set_ylabel("Test Score")
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.01, max(mouse_score.max(), mouse_score_direct.max()) + 0.01)
    # ax.legend()
    plt.show()


def create_rrr_diagram(sizes=[5, 2, 5], ball_size=0.2, line_width=2, colors=["k", "k", "k"], figsize=(8, 8), dpi=100, xscale=2, fontsize=18):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(-0.5, 2 * xscale + 0.5)
    ax.set_ylim(0, max(sizes) + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Calculate vertical offsets to center the balls
    offsets = [(max(sizes) - size) / 2 for size in sizes]

    # Create balls
    for x, size in enumerate(sizes):
        for j in range(size):
            y = j + 1 + offsets[x]
            circle = plt.Circle((x * xscale, y), ball_size, facecolor=colors[x], edgecolor="black", linewidth=line_width)
            ax.add_patch(circle)

    # Create connections
    for i in range(len(sizes) - 1):
        for j in range(sizes[i]):
            for k in range(sizes[i + 1]):
                xpos = np.array([i, i + 1]) * xscale
                ax.plot(xpos, [j + 1 + offsets[i], k + 1 + offsets[i + 1]], color="k", linewidth=line_width, alpha=1)

    # Add labels
    ax.text(0 * xscale, 0.3, "Input", ha="center", va="center", fontsize=fontsize)
    ax.text(1 * xscale, 0.6, "Latent:", ha="center", va="center", fontsize=fontsize)
    ax.text(1 * xscale, 0.3, "Encode Position", ha="center", va="center", fontsize=fontsize)
    ax.text(1 * xscale, 0.0, "Unconstrained", ha="center", va="center", fontsize=fontsize)

    ax.text(2 * xscale, 0.3, "Output", ha="center", va="center", fontsize=fontsize)

    plt.tight_layout()
    return fig, ax


def parse_args():
    parser = ArgumentParser(description="Run analysis on all sessions.")
    parser.add_argument("--redo_pop_splits", default=False, action="store_true", help="Remake population objects and train/val/test splits.")
    parser.add_argument(
        "--redo_pop_splits_behavior",
        default=False,
        action="store_true",
        help="Remake population objects and train/val/test splits with behavior data.",
    )
    parser.add_argument("--keep_planes", nargs="+", default=[1, 2, 3, 4], type=int, help="Which planes to keep (default=[1, 2, 3, 4])")
    parser.add_argument(
        "--population_name", default=None, type=str, help="Name of population object to save (default=None, just uses name of session)"
    )
    parser.add_argument("--rrr", default=False, action="store_true", help="Run reduced rank regression optimization.")
    parser.add_argument("--networks", default=False, action="store_true", help="Run network optimization.")
    parser.add_argument("--rrr_state", default=False, action="store_true", help="Run rrr (state) optimization.")
    parser.add_argument("--rrr_state_add_results", default=False, action="store_true", help="Add rrr state results to the session.")
    parser.add_argument("--skip_completed", type=argbool, default=True, help="Skip completed sessions (default=True)")
    parser.add_argument("--retest_only", type=argbool, default=False, help="Only retest sessions that have already been optimized.")
    parser.add_argument("--compare_rrr_to_networks", type=argbool, default=False, help="Do rrr to network comparison.")
    parser.add_argument("--analyze_rrr_fits", type=argbool, default=False, help="Do analysis of rrr fits.")
    parser.add_argument("--analyze_networks", type=argbool, default=False, help="Do analysis of networks.")
    parser.add_argument("--analyze_rrr_state", type=argbool, default=False, help="Do analysis of rrr state.")
    parser.add_argument("--make_rrr_state_example", type=argbool, default=False, help="Make an example of the RRR state model.")
    parser.add_argument("--make_rrr_diagram", type=argbool, default=False, help="Make a diagram of the RRR model.")
    parser.add_argument("--save", type=argbool, default=True, help="Whether to save results (default=True).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_sessions = get_sessions()

    # this set of analyses requires consistent train/val/test splits.
    # make_and_save_populations will generate these splits and save them to a temp file in placeCellSingleSession
    if args.redo_pop_splits:
        make_and_save_populations(all_sessions, args.keep_planes, args.population_name)

    if args.redo_pop_splits_behavior:
        make_and_save_populations(all_sessions, args.keep_planes, get_behavior=True, population_name=args.population_name)

    # this set performs optimization and testing of reduced rank regression. It will cache results and save a
    # temporary file in placeCellSingleSession containing the scores and best alpha for each session.
    if args.rrr:
        do_rrr_optimization(all_sessions, skip_completed=args.skip_completed, save=args.save, population_name=args.population_name)

    if args.networks:
        do_network_optimization(
            all_sessions, retest_only=args.retest_only, population_name=args.population_name, skip_completed=args.skip_completed, save=args.save
        )

    if args.rrr_state:
        do_rrr_state_optimization(
            all_sessions,
            skip_completed=args.skip_completed,
            save=args.save,
            keep_planes=args.keep_planes,
            population_name=args.population_name,
        )

    if args.rrr_state_add_results:
        add_rrr_state_results(all_sessions, population_name=args.population_name, keep_planes=args.keep_planes)

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

    if args.analyze_rrr_state:
        rrr_state_results = load_rrr_state_results(all_sessions, results="all", population_name=args.population_name)
        analyze_rrr_state(rrr_state_results)

    if args.make_rrr_state_example:
        make_rrr_state_example(all_sessions, population_name=args.population_name, keep_planes=args.keep_planes)

    if args.make_rrr_diagram:
        fig, ax = create_rrr_diagram(ball_size=0.3, line_width=5, xscale=2, fontsize=24)
