import os
import sys
import time
from tqdm import tqdm
import torch

from helpers import load_population, get_ranks, memory


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from vrAnalysis import analysis
from vrAnalysis import session

from dimilibi import ReducedRankRegression, scaled_mse


def get_init_alphas():
    """initial alphas to test for ridge regression"""
    return torch.logspace(6, 10, 5)


def get_next_alphas(init_alpha):
    """next alphas to test for ridge regression based on best initial alpha"""
    return torch.logspace(torch.log10(init_alpha) - 0.2, torch.log10(init_alpha) + 0.2, 5)


@memory.cache
def optimize_rrr(mouse_name, datestr, sessionid):
    """
    Optimize ridge regression for a given session.

    Performs a two-stage targeted grid search over the ridge regression alpha parameter.
    First uses a wide logarithmic range over 5 orders of magnitude to find the initial alpha,
    then uses a focused search around the initial alpha to find the best alpha.

    Parameters
    ----------
    mouse_name : str
        Name of the mouse.
    datestr : str
        Date string of the session.
    sessionid : int
        Session identifier.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:
        - mouse_name : str
            Name of the mouse.
        - datestr : str
            Date string of the session.
        - sessionid : int
            Session identifier.
        - best_alpha : float
            Best alpha for ridge regression.
        - val_score : float
            Validation score for the best alpha.
        - indices_dict : dict
            Dictionary containing indices for splitting the data.
    """
    npop = load_population(mouse_name, datestr, sessionid)

    # split the data into training and validation sets
    train_source, train_target = npop.get_split_data(0, center=True, scale=False)
    val_source, val_target = npop.get_split_data(1, center=True, scale=False)

    # get initial estimate for the best alpha
    init_alphas = get_init_alphas()
    init_rrr = []
    for a in tqdm(init_alphas):
        init_rrr.append(ReducedRankRegression(alpha=a, fit_intercept=False).fit(train_source.T, train_target.T))
    init_scores = [rrr.score(val_source.T, val_target.T) for rrr in init_rrr]
    best_init_alpha = init_alphas[init_scores.index(max(init_scores))]

    # improve best alpha in focused search
    next_alpha = get_next_alphas(best_init_alpha)
    next_rrr = []
    for a in tqdm(next_alpha):
        next_rrr.append(ReducedRankRegression(alpha=a, fit_intercept=False).fit(train_source.T, train_target.T))
    next_scores = [rrr.score(val_source.T, val_target.T) for rrr in next_rrr]
    best_index = next_scores.index(max(next_scores))
    best_alpha = next_alpha[best_index]

    results = dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        best_alpha=best_alpha,
        val_score=max(next_scores),
    )
    return results


@memory.cache
def test_rrr(mouse_name, datestr, sessionid, alpha, ranks):
    """
    Test ridge regression for a given session.

    Parameters
    ----------
    mouse_name : str
        Name of the mouse.
    datestr : str
        Date string of the session.
    sessionid : int
        Session identifier.
    alpha : float
        Alpha parameter for ridge regression.
    ranks : list
        List of ranks to test ridge regression.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:
        - test_score : float
            Test score for the full model.
        - test_scaled_mse : float
            Test scaled mean squared error for the full model.
        - test_score_by_rank : list
            List of test scores for reduced rank models.
        - test_scaled_mse_by_rank : list
            List of test scaled mean squared errors for reduced rank models.
    """

    npop = load_population(mouse_name, datestr, sessionid)

    train_source, train_target = npop.get_split_data(0, center=True, scale=False)
    test_source, test_target = npop.get_split_data(2, center=True, scale=False)

    rrr = ReducedRankRegression(alpha=alpha, fit_intercept=False).fit(train_source.T, train_target.T)

    # test full model
    test_score = rrr.score(test_source.T, test_target.T)
    test_scaled_mse = scaled_mse(rrr.predict(test_source.T).T, test_target, reduce="mean")

    # test reduced rank models
    test_score_by_rank = [rrr.score(test_source.T, test_target.T, rank=r) for r in tqdm(ranks)]
    test_scaled_mse_by_rank = [scaled_mse(rrr.predict(test_source.T, rank=r).T, test_target, reduce="mean") for r in tqdm(ranks)]

    results = dict(
        ranks=ranks,
        test_score=test_score,
        test_scaled_mse=test_scaled_mse,
        test_score_by_rank=test_score_by_rank,
        test_scaled_mse_by_rank=test_scaled_mse_by_rank,
    )

    return results


def rrr_tempfile_name(vrexp):
    """generate temporary file name for rrr results"""
    return f"rrr_optimization_results_{str(vrexp)}"


def do_rrr_optimization(all_sessions):
    """
    Perform ridge regression optimization and testing.

    Will optimize ridge regression for each session in all_sessions and then test the best model
    on the test set with a range of ranks and the full model. The results are saved as a dictionary
    and stored in a temporary file using the standard analysis temporary file storage system.

    Parameters
    ----------
    all_sessions : dict
        Dictionary containing session identifiers for each mouse.
    """
    ranks = get_ranks()
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            print(f"Optimizing ridge regression for: {mouse_name}, {datestr}, {sessionid}:")
            t = time.time()
            optimize_results = optimize_rrr(mouse_name, datestr, sessionid)
            print(f"Time: {time.time() - t : .2f}, Best alpha: {optimize_results['best_alpha']}")
            print(f"Testing ridge regression for: {mouse_name}, {datestr}, {sessionid}:")
            alpha = optimize_results["best_alpha"].item()
            t = time.time()
            test_results = test_rrr(mouse_name, datestr, sessionid, alpha, ranks)
            print(f"Time: {time.time() - t : .2f}")
            rrr_results = {**optimize_results, **test_results}
            pcss.save_temp_file(rrr_results, rrr_tempfile_name(pcss.vrexp))

def load_rrr_results(all_sessions, results='all'):
    rrr_results = []
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            rrr_filename = rrr_tempfile_name(pcss.vrexp)
            if not pcss.check_temp_file(rrr_filename):
                print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} (not found)")
                continue
            print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
            rrr_results.append(pcss.load_temp_file(rrr_filename))
    if results=='all':
        return rrr_results
    if results=='test_by_mouse':
        ranks = rrr_results[0]["ranks"]
        for rrr_res in rrr_results:
            assert rrr_res["ranks"] == ranks, "ranks are not all equal"
        num_ranks = len(ranks)
        mouse_names = list(set([res["mouse_name"] for res in rrr_results]))
        num_mice = len(mouse_names)
        test_scores = torch.zeros((num_mice, num_ranks))
        test_scaled_mses = torch.zeros((num_mice, num_ranks))
        num_samples = torch.zeros(num_mice)
        for rrr_res in rrr_results:
            mouse_idx = mouse_names.index(rrr_res["mouse_name"])
            test_scores[mouse_idx] += torch.tensor(rrr_res["test_score_by_rank"])
            test_scaled_mses[mouse_idx] += torch.tensor(rrr_res["test_scaled_mse_by_rank"])
            num_samples[mouse_idx] += 1
        # get average for each mouse
        test_scores /= num_samples.unsqueeze(1)
        test_scaled_mses /= num_samples.unsqueeze(1)
        return dict(
            mouse_names=mouse_names,
            ranks=ranks,
            test_scores=test_scores,
            test_scaled_mses=test_scaled_mses,
        )
    raise ValueError(f"results must be 'all' or 'test_by_mouse', got {results}")