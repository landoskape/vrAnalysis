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


def get_alphas():
    """initial alphas to test for ridge regression"""
    return torch.logspace(1, 9, 9)


@memory.cache
def optimize_rrr(mouse_name, datestr, sessionid, rank, population_name=None):
    """
    Optimize ridge regression for a given session.

    Performs a targeted grid search over the ridge regression alpha parameter.

    Parameters
    ----------
    mouse_name : str
        Name of the mouse.
    datestr : str
        Date string of the session.
    sessionid : int
        Session identifier.
    rank : int
        Rank of the reduced rank regression for validation.
    population_name : Optional[str]
        Name of the population object to load. If None, the default population object will be loaded.

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
    npop = load_population(mouse_name, datestr, sessionid, population_name=population_name)

    # split the data into training and validation sets
    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    val_source, val_target = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")

    # do an intelligent grid search over alphas
    alphas = torch.sort(get_alphas()).values

    # inline for evaluating alpha
    def evaluate_alpha(alpha_index):
        c_alpha = alphas[alpha_index]
        model = ReducedRankRegression(alpha=c_alpha, fit_intercept=True).fit(train_source.T, train_target.T)
        return model.score(val_source.T, val_target.T, rank=rank, nonnegative=True), model

    # set initial best alpha to the middle of the range
    best_alpha_index = len(alphas) // 2
    best_score, best_model = evaluate_alpha(best_alpha_index)

    # try going lower or higher with alpha as long as it's possible
    go_lower = best_alpha_index > 0
    go_higher = best_alpha_index < len(alphas) - 1
    while go_lower or go_higher:
        # test one left of mid and one right of mid
        if go_lower:
            left_score, left_model = evaluate_alpha(best_alpha_index - 1)
        else:
            left_score = -float("inf")
        if go_higher:
            right_score, right_model = evaluate_alpha(best_alpha_index + 1)
        else:
            right_score = -float("inf")

        if (left_score > best_score) and (right_score > best_score):
            print("Both left and right scores are better than the best score, I didn't think this would ever happen...")
            print(f"Left score: {left_score}, Right score: {right_score}, Best score: {best_score}")
            if left_score > right_score:
                # if left is better, go left
                best_score = left_score
                best_model = left_model
                best_alpha_index -= 1
                go_lower = best_alpha_index > 0
                go_higher = False
            else:
                # if right is better, go right
                best_score = right_score
                best_model = right_model
                best_alpha_index += 1
                go_lower = False
                go_higher = best_alpha_index < len(alphas) - 1

        elif (left_score > best_score) or (right_score > best_score):
            # If one of the directions gives an improvement, go that way
            if left_score > best_score:
                # reset the best score, model, and alpha index, stop going higher
                best_score = left_score
                best_model = left_model
                best_alpha_index -= 1
                go_lower = best_alpha_index > 0
                go_higher = False

            else:
                # reset the best score, model, and alpha index, stop going lower
                best_score = right_score
                best_model = right_model
                best_alpha_index += 1
                go_lower = False
                go_higher = best_alpha_index < len(alphas) - 1
        else:
            # If neither direction gives an improvement, stop
            go_lower = False
            go_higher = False

    return alphas[best_alpha_index], best_score, best_model


@memory.cache
def test_rrr(mouse_name, datestr, sessionid, alphas, ranks, models=None, population_name=None):
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
    alphas : float
        Alpha parameters for ridge regression for each rank.
    ranks : list
        List of ranks to test ridge regression.
    models : list[ReducedRankRegression]
        List of ReducedRankRegression models for each rank. Will be used if provided,
        otherwise the models will be refit to the training data.
    population_name : Optional[str]
        Name of the population object to load. If None, the default population object will be loaded.

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
    assert len(alphas) == len(ranks), "alphas and ranks must have the same length"

    npop = load_population(mouse_name, datestr, sessionid, population_name=population_name)

    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    test_source, test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

    if models is None:
        models = [ReducedRankRegression(alpha=alpha, fit_intercept=True).fit(train_source.T, train_target.T) for alpha in alphas]

    # test reduced rank models
    test_scores = [rmodel.score(test_source.T, test_target.T, rank=rank, nonnegative=True) for rmodel, rank in tqdm(zip(models, ranks))]
    test_scaled_mses = [
        scaled_mse(rmodel.predict(test_source.T, rank=rank, nonnegative=True).T, test_target, reduce="mean")
        for rmodel, rank in tqdm(zip(models, ranks))
    ]

    results = dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        ranks=ranks,
        alphas=alphas,
        test_scores=test_scores,
        test_scaled_mses=test_scaled_mses,
    )

    return results


def rrr_tempfile_name(vrexp, population_name=None):
    """generate temporary file name for rrr results"""
    name = f"rrr_optimization_results_{str(vrexp)}"
    if population_name is not None:
        name += f"_{population_name}"
    return name


def do_rrr_optimization(all_sessions, skip_completed=True, save=True, population_name=None):
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
            alphas = []
            scores = []
            models = []
            if skip_completed and pcss.check_temp_file(rrr_tempfile_name(pcss.vrexp, population_name=population_name)):
                print(f"Found completed RRR optimization for: {mouse_name}, {datestr}, {sessionid}")
                continue
            for rank in ranks:
                print(f"Optimizing ridge regression for: {mouse_name}, {datestr}, {sessionid}, rank:{rank} :")
                t = time.time()
                best_alpha, best_score, best_model = optimize_rrr(mouse_name, datestr, sessionid, rank, population_name=population_name)
                print(f"Time: {time.time() - t : .2f}, Best alpha: {best_alpha}, Best score: {best_score}")
                alphas.append(best_alpha)
                scores.append(best_score)
                models.append(best_model)
            print(f"Testing ridge regression for: {mouse_name}, {datestr}, {sessionid} across all ranks:")
            t = time.time()
            test_results = test_rrr(mouse_name, datestr, sessionid, alphas, ranks, models=models, population_name=population_name)
            print(f"Time: {time.time() - t : .2f}")
            if save:
                pcss.save_temp_file(test_results, rrr_tempfile_name(pcss.vrexp, population_name=population_name))


def load_rrr_results(all_sessions, results="all", population_name=None):
    rrr_results = []
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            rrr_filename = rrr_tempfile_name(pcss.vrexp, population_name=population_name)
            if not pcss.check_temp_file(rrr_filename):
                print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} (not found)")
                continue
            print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
            rrr_results.append(pcss.load_temp_file(rrr_filename))
    if results == "all":
        return rrr_results
    if results == "test_by_mouse":
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
            test_scores[mouse_idx] += torch.tensor(rrr_res["test_scores"])
            test_scaled_mses[mouse_idx] += torch.tensor(rrr_res["test_scaled_mses"])
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
