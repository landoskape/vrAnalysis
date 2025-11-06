import os, sys
from copy import copy
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import matplotlib as mpl

from helpers import get_sessions, make_and_save_populations, load_population, get_ranks
from helpers import figure_folder
from rrr_optimization import do_rrr_optimization, load_rrr_results
from network_optimization import do_network_optimization, load_network_results
from rrr_state_optimization import do_rrr_state_optimization, load_rrr_state_results, make_rrr_state_example, add_rrr_state_results


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from _old_vrAnalysis.helpers import argbool, refline, save_figure

import numpy as np
import torch


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

    plt.rcParams.update({"font.size": 18})
    fig = plt.figure(figsize=(6, 6), layout="constrained")
    plt.scatter(session_number, model_improvement[sort_by_improvement], c=model_mouse_id[sort_by_improvement], s=5, cmap=colmouse)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Session (Sorted by NL Improvement)")
    plt.ylabel("Network Test Score - RRR Test Score")
    plt.title("Model Improvement by Session")
    plt.show()


def compare_reshuffled_populations(rrr_all, network_all, rrr_all_cmp, network_all_cmp, session_ids, session_ids_cmp):
    ranks = rrr_all[0]["ranks"]
    for rrr, ntw, rrrcmp, ntwcmp in zip(rrr_all, network_all, rrr_all_cmp, network_all_cmp):
        assert rrr["ranks"] == ranks, "ranks don't match between rrr and hyperparameters"
        assert rrrcmp["ranks"] == ranks, "ranks don't match between rrr and hyperparameters (comparison population)"
        c_ntw_ranks = tuple([nar["rank"] for nar in ntw])
        c_ntw_ranks_cmp = tuple([nar["rank"] for nar in ntwcmp])
        assert c_ntw_ranks == ranks, "ranks don't match between networks"
        assert c_ntw_ranks_cmp == ranks, "ranks don't match between networks (comparison population)"

    assert session_ids == session_ids_cmp, "session ids don't match between populations"

    # Also compare each session individually (for each rank)
    mouse_names = sorted(list(set(map(lambda x: x[0], session_ids))))
    num_mice = len(mouse_names)
    num_ranks = len(ranks)
    num_sessions = len(rrr_all)
    assert num_sessions == len(network_all), "number of sessions don't match between rrr and networks"
    rrr_score_all = torch.zeros(num_sessions, num_ranks)
    network_score_all = torch.zeros(num_sessions, num_ranks)
    rrr_score_all_cmp = torch.zeros(num_sessions, num_ranks)
    network_score_all_cmp = torch.zeros(num_sessions, num_ranks)
    mouse_index = torch.zeros(num_sessions)
    for ises in range(num_sessions):
        rrr_score_all[ises] = torch.tensor(rrr_all[ises]["test_scores"])
        network_score_all[ises] = torch.tensor([torch.mean(nar["test_score"]) for nar in network_all[ises]])
        mouse_index[ises] = mouse_names.index(session_ids[ises][0])

        rrr_score_all_cmp[ises] = torch.tensor(rrr_all_cmp[ises]["test_scores"])
        network_score_all_cmp[ises] = torch.tensor([torch.mean(nar["test_score"]) for nar in network_all_cmp[ises]])

    # Compare the two models across ranks
    model_improvement = network_score_all - rrr_score_all
    model_improvement_cmp = network_score_all_cmp - rrr_score_all_cmp
    session_number = torch.arange(num_sessions).view(-1, 1).expand(-1, num_ranks)
    sort_by_improvement = torch.argsort(torch.mean(model_improvement, dim=1))
    model_mouse_id = mouse_index.clone().view(-1, 1).expand(-1, num_ranks)

    colmouse = mpl.colormaps["tab20"].resampled(num_mice)

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    ax[0].scatter(session_number, model_improvement[sort_by_improvement], c=model_mouse_id[sort_by_improvement], s=5, cmap=colmouse)
    ax[0].axhline(0, color="k", linestyle="--")
    ax[0].set_xlabel("Session (Sort by $\Delta$ Shuf0)")
    ax[0].set_ylabel("DNN - RRR Test Score")
    ax[0].set_title("Shuffle 0")

    ax[1].scatter(session_number, model_improvement_cmp[sort_by_improvement], c=model_mouse_id[sort_by_improvement], s=5, cmap=colmouse)
    ax[1].axhline(0, color="k", linestyle="--")
    ax[1].set_xlabel("Session (Sort by $\Delta$ Shuf0)")
    ax[1].set_title("Shuffle 1")

    ax[2].scatter(model_improvement, model_improvement_cmp, c=model_mouse_id, s=5, cmap=colmouse)
    refline(1, 0, color="k", linestyle="--", ax=ax[2])
    ax[2].set_xlabel("Shuffle 0 Improvement")
    ax[2].set_ylabel("Shuffle 1 Improvement")
    ax[2].set_title("Shuffle Comp.")

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
    test_score_pfpred_withcvgain = torch.tensor([res["pf_pred_score_withcvgain"] for res in results])
    test_score_pfpred_withgain = torch.tensor([res["pf_pred_score_withgain"] for res in results])
    pos_score_encoder = torch.tensor([res["encoder_position_score"] for res in results])
    pos_score_latent = torch.tensor([res["latent_position_score"] for res in results])
    opt_pos_est_source = torch.tensor([res["opt_pos_estimate_source"] for res in results])
    opt_pos_est_target = torch.tensor([res["opt_pos_estimate_target"] for res in results])
    opt_pos_est_target_withcvgain = torch.tensor([res["opt_pos_estimate_target_withcvgain"] for res in results])
    opt_pos_est_target_withgain = torch.tensor([res["opt_pos_estimate_target_withgain"] for res in results])
    opt_pos_est_position = torch.tensor([res["opt_pos_estimate_position"] for res in results])
    test_score_doublecv = torch.tensor([res["test_score_doublecv"] for res in results])
    rbfpos_to_target_score = torch.tensor([res["rbfpos_to_target_score"] for res in results])

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

    # Average across mice
    mouse_score = torch.zeros(num_mice)
    mouse_scaled_mse = torch.zeros(num_mice)
    mouse_score_direct = torch.zeros(num_mice)
    mouse_scaled_mse_direct = torch.zeros(num_mice)
    mouse_score_pfpred = torch.zeros(num_mice)
    mouse_score_pfpred_withcvgain = torch.zeros(num_mice)
    mouse_score_pfpred_withgain = torch.zeros(num_mice)
    mouse_pos_score_encoder = torch.zeros(num_mice)
    mouse_pos_score_latent = torch.zeros(num_mice)
    mouse_score_opt_pos_est_source = torch.zeros(num_mice)
    mouse_score_opt_pos_est_target = torch.zeros(num_mice)
    mouse_score_opt_pos_est_target_withcvgain = torch.zeros(num_mice)
    mouse_score_opt_pos_est_target_withgain = torch.zeros(num_mice)
    mouse_score_opt_pos_est_position = torch.zeros(num_mice)
    mouse_score_doublecv = torch.zeros(num_mice)
    mouse_rbfpos_to_target_score = torch.zeros(num_mice)
    for imouse, mouse in enumerate(mice):
        idx = torch.tensor([mn == mouse for mn in mouse_name])
        mouse_score[imouse] = test_score[idx].mean()
        mouse_scaled_mse[imouse] = test_scaled_mse[idx].mean()
        mouse_score_direct[imouse] = test_score_direct[idx].mean()
        mouse_scaled_mse_direct[imouse] = test_scaled_mse_direct[idx].mean()
        mouse_score_pfpred[imouse] = test_score_pfpred[idx].mean()
        mouse_score_pfpred_withcvgain[imouse] = test_score_pfpred_withcvgain[idx].mean()
        mouse_score_pfpred_withgain[imouse] = test_score_pfpred_withgain[idx].mean()
        mouse_pos_score_encoder[imouse] = pos_score_encoder[idx].mean()
        mouse_pos_score_latent[imouse] = pos_score_latent[idx].mean()
        mouse_score_opt_pos_est_source[imouse] = opt_pos_est_source[idx].mean()
        mouse_score_opt_pos_est_target[imouse] = opt_pos_est_target[idx].mean()
        mouse_score_opt_pos_est_target_withcvgain[imouse] = opt_pos_est_target_withcvgain[idx].mean()
        mouse_score_opt_pos_est_target_withgain[imouse] = opt_pos_est_target_withgain[idx].mean()
        mouse_score_opt_pos_est_position[imouse] = opt_pos_est_position[idx].mean()
        mouse_score_doublecv[imouse] = test_score_doublecv[idx].mean()
        mouse_rbfpos_to_target_score[imouse] = rbfpos_to_target_score[idx].mean()

    include_extras = False
    if include_extras:
        labels = ["PF", "Opt-PF", "PF", "Opt-PF", "DCV-RBF", "RBF", "Opt-RBF", "RRR"]
        colors = ["black", "sienna", "black", "sienna", "crimson", "mediumvioletred", "darkmagenta", "orangered"]
        with_gain = [False, False, True, True, False, False, False, False]
        xd = [0, 1, 2, 3, 4, 5, 6, 7]
        allyd = np.stack(
            list(
                map(
                    lambda x: np.array(x),
                    zip(
                        mouse_score_pfpred,
                        mouse_score_opt_pos_est_target,
                        mouse_score_pfpred_withcvgain,
                        mouse_score_opt_pos_est_target_withcvgain,
                        mouse_score_doublecv,
                        mouse_rbfpos_to_target_score,
                        mouse_score,
                        mouse_score_direct,
                    ),
                )
            )
        )
    else:
        labels = ["PF", "Opt-PF", "PF", "Opt-PF", "RBF(Pos)", "RRR"]
        colors = ["black", "sienna", "black", "sienna", "mediumvioletred", "orangered"]
        with_gain = [False, False, True, True, False, False]
        xd = [0, 1, 2, 3, 4, 5]
        allyd = np.stack(
            list(
                map(
                    lambda x: np.array(x),
                    zip(
                        mouse_score_pfpred,
                        mouse_score_opt_pos_est_target,
                        mouse_score_pfpred_withcvgain,
                        mouse_score_opt_pos_est_target_withcvgain,
                        mouse_score,
                        mouse_score_direct,
                    ),
                )
            )
        )

    # Model Level controls which part of plot to include
    # Model level 0 is just PF model
    # Model level 1 adds the optimized PF model
    # Model level 3 adds the gain terms
    # Model level 4 adds the RBF term
    # Model level 5 adds the RRR term
    # NOTE!!!: This only works when include_extras is False!!!
    # because model_level is used to index into xd and allyd and the other related components
    model_level = np.inf


for model_level in [0, 1, 3, 4, 5]:

    plt.rcParams.update({"font.size": 24})

    cosyne = True
    if cosyne:
        fig_height = 6.5
        fig_width = 8 / 8.5 * 6.5
    else:
        fig_height = 8.5
        fig_width = 8

    if not include_extras and model_level < np.inf:
        xd_plot = xd[: model_level + 1]
        allyd_plot = allyd[:, : model_level + 1]
        labels_plot = labels[: model_level + 1]
        colors_plot = colors[: model_level + 1]
        show_gain = model_level >= 3
    else:
        xd_plot = copy(xd)
        allyd_plot = copy(allyd)
        labels_plot = copy(labels)
        colors_plot = copy(colors)
        show_gain = True

    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), layout="constrained")
    ax.plot(xd_plot, allyd_plot.T, color="k", linestyle="-", marker="o", markersize=12)
    for x, eachdata, color in zip(xd_plot, allyd_plot.T, colors_plot):
        ax.plot(x * np.ones_like(eachdata), eachdata, color=color, linestyle="none", marker="o", markersize=13)
    ylim = (-0.1, 0.2)
    # Make a gray patch around the x value spanning the full ylim
    if show_gain:
        for x, wg in zip(xd, with_gain):
            if wg:
                ax.fill_between([x - 0.5, x + 0.5], ylim[0], ylim[1], color="gray", edgecolor="none", alpha=0.2)
        ax.text(2.5, ylim[1] * 0.95, "+Gain", ha="center", va="top")
    ax.set_xticks(ticks=xd_plot, labels=labels_plot, rotation=45, ha="center")
    ax.set_ylabel("Test Score")
    ax.set_xlim(-0.5, max(xd) + 0.5)
    for ixtick, color in enumerate(colors_plot):
        plt.setp(ax.get_xticklabels()[ixtick], color=color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Make a rightward, black, thick arrow pointing right on the bottom of the plot
    ax.annotate(
        "",
        xy=(max(xd), ymin - 0.05 * yrange),
        xytext=(0, ymin - 0.05 * yrange),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
    )
    ax.text(np.mean(xd), ymin - 0.02 * yrange, "less constrained\nby position", ha="center", va="bottom")
    ax.set_ylim(ylim)

    plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = figure_folder()
        save_name = f"decoder_test_score_comparison"
        save_name += "_with_extras" if include_extras else ""
        save_name += "_cosyne" if cosyne else ""
        if (model_level < np.inf) and (not include_extras):
            save_name += f"_model_level_{model_level}"
        save_path = save_directory / save_name
        save_figure(fig, save_path)


def parse_args():
    parser = ArgumentParser(description="Run analysis on all sessions.")
    parser.add_argument(
        "--redo_pop_splits",
        default=False,
        action="store_true",
        help="Remake population objects and train/val/test splits.",
    )
    parser.add_argument(
        "--redo_pop_splits_behavior",
        default=False,
        action="store_true",
        help="Remake population objects and train/val/test splits with behavior data.",
    )
    parser.add_argument(
        "--keep_planes",
        nargs="+",
        default=[1, 2, 3, 4],
        type=int,
        help="Which planes to keep (default=[1, 2, 3, 4])",
    )
    parser.add_argument(
        "--population_name",
        default=None,
        type=str,
        help="Name of population object to save (default=None, just uses name of session)",
    )
    parser.add_argument(
        "--population_name_compare",
        default=None,
        type=str,
        help="Name of population object to compare with, require if running compare_reshuffled_populations",
    )
    parser.add_argument("--rrr", default=False, action="store_true", help="Run reduced rank regression optimization.")
    parser.add_argument("--networks", default=False, action="store_true", help="Run network optimization.")
    parser.add_argument("--rrr_state", default=False, action="store_true", help="Run rrr (state) optimization.")
    parser.add_argument("--rrr_state_add_results", default=False, action="store_true", help="Add rrr state results to the session.")
    parser.add_argument("--skip_completed", type=argbool, default=True, help="Skip completed sessions (default=True)")
    parser.add_argument("--retest_only", type=argbool, default=False, help="Only retest sessions that have already been optimized.")
    parser.add_argument("--compare_rrr_to_networks", type=argbool, default=False, help="Do rrr to network comparison.")
    parser.add_argument("--compare_reshuffled_populations", type=argbool, default=False, help="Compare reshuffled populations.")
    parser.add_argument("--analyze_rrr_fits", type=argbool, default=False, help="Do analysis of rrr fits.")
    parser.add_argument("--analyze_networks", type=argbool, default=False, help="Do analysis of networks.")
    parser.add_argument("--analyze_rrr_state", type=argbool, default=False, help="Do analysis of rrr state.")
    parser.add_argument("--make_rrr_state_example", type=argbool, default=False, help="Make an example of the RRR state model.")
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

    if args.compare_reshuffled_populations:
        assert args.population_name_compare is not None, "Must provide a population name to compare to."
        rrr_all = load_rrr_results(all_sessions, results="all", population_name=args.population_name)
        network_all, session_ids, tested_ranks = load_network_results(all_sessions, results="all", population_name=args.population_name)
        rrr_all_cmp = load_rrr_results(all_sessions, results="all", population_name=args.population_name_compare)
        network_all_cmp, session_ids_cmp, tested_ranks_cmp = load_network_results(
            all_sessions, results="all", population_name=args.population_name_compare
        )
        compare_reshuffled_populations(rrr_all, network_all, rrr_all_cmp, network_all_cmp, session_ids, session_ids_cmp)

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
