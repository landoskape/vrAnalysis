import os
import sys
import time
from tqdm import tqdm
import numpy as np
import torch
import optuna

from helpers import load_population, get_ranks, memory, make_position_basis


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from vrAnalysis import analysis
from vrAnalysis import session

from dimilibi import ReducedRankRegression, scaled_mse


def get_alphas():
    """initial alphas to test for ridge regression"""
    return torch.logspace(1, 9, 9)


def optimize_rrr_state(trial, mouse_name, datestr, sessionid, population_name=None):
    """
    Optimize ridge regression for a given session, where a latent variable is fit to the mouse's virtual position.

    The optimization works as follows:
    0. The position is converted into a basis representation (for each environment) that can be used as the target for the decoder.
    1. An encoder is trained to predict the position basis from the source neural data.
    2. A decoder is trained to predict the target neural data from the position basis.
    3. The encoder is used to predict the position on the validation set.
    4. The decoder is used to predict the target neural data on the validation set.
    5. The validation score is calculated as the score of the decoder on the predicted target data.

    Note: The idea is to train the encoder without considering it's effect on decoding the target data. The only way the encoder model
    depends on the fit to target neurons is due to the fact that hyperparameters are only chosen based on the validation score of the decoder.
    Therefore, if the position score happens to be bad in a way that allows the target score to be good, this will be selected as the best model.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object...
    mouse_name : str
        Name of the mouse.
    datestr : str
        Date string of the session.
    sessionid : int
        Session identifier.
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
    # Set up optuna study variables
    num_basis = trial.suggest_int("num_basis", 5, 50, log=True)
    basis_width = trial.suggest_float("basis_width", 1.0, 100.0, log=True)
    alpha_encoder = trial.suggest_float("alpha_encoder", 1e-1, 1e5, log=True)
    alpha_decoder = trial.suggest_float("alpha_decoder", 1e-1, 1e5, log=True)
    fit_intercept_encoder = trial.suggest_categorical("fit_intercept_encoder", [True, False])
    fit_intercept_decoder = trial.suggest_categorical("fit_intercept_decoder", [True, False])
    nonnegative_encoder = trial.suggest_categorical("nonnegative_encoder", [True, False])
    nonnegative_decoder = trial.suggest_categorical("nonnegative_decoder", [True, False])

    # Load population with behavior data
    npop, behavior_data = load_population(mouse_name, datestr, sessionid, population_name=population_name, get_behavior=True)

    # Make the position basis with current hyperparameters
    position_basis = make_position_basis(behavior_data["position"], behavior_data["environment"], num_basis=num_basis, basis_width=basis_width)

    # split the data into training and validation sets
    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    val_source, val_target = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")

    # Get the train / validation data for the behavior position
    train_position = npop.apply_split(position_basis.T, 0)

    # build an encoder on the training data
    encoder = ReducedRankRegression(alpha=alpha_encoder, fit_intercept=fit_intercept_encoder).fit(train_source.T, train_position.T)
    decoder = ReducedRankRegression(alpha=alpha_decoder, fit_intercept=fit_intercept_decoder).fit(train_position.T, train_target.T)

    # get the encoder and decoder predictions
    val_position_hat = encoder.predict(val_source.T, nonnegative=nonnegative_encoder)
    val_target_score = decoder.score(val_position_hat, val_target.T, nonnegative=nonnegative_decoder)

    # return the validation score
    return val_target_score


def test_rrr_state(mouse_name, datestr, sessionid, params, population_name=None):
    """
    Test the ridge regression model on the test set.

    Parameters
    ----------
    mouse_name : str
        Name of the mouse.
    datestr : str
        Date string of the session.
    sessionid : int
        Session identifier.
    params : dict
        Dictionary containing the following keys:
        - num_basis : int
            Number of basis functions to use for the position.
        - basis_width : float
            Width of the basis functions.
        - alpha_encoder : float
            Alpha value for the encoder.
        - alpha_decoder : float
            Alpha value for the decoder.
        - fit_intercept_encoder : bool
            Whether to fit an intercept for the encoder.
        - fit_intercept_decoder : bool
            Whether to fit an intercept for the decoder.
        - nonnegative_encoder : bool
            Whether to enforce nonnegativity for the encoder.
        - nonnegative_decoder : bool
            Whether to enforce nonnegativity for the decoder.
    population_name : Optional[str]
        Name of the population object to load. If None, the default population object will be loaded.

    Returns
    -------
    """
    # Load population with behavior data
    npop, behavior_data = load_population(mouse_name, datestr, sessionid, population_name=population_name, get_behavior=True)

    # Make the position basis with optimal hyperparameters
    position_basis = make_position_basis(
        behavior_data["position"], behavior_data["environment"], num_basis=params["num_basis"], basis_width=params["basis_width"]
    )

    # split the data into training and validation sets
    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    test_source, test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

    # Get the train / validation data for the behavior position
    train_position = npop.apply_split(position_basis.T, 0)

    # build an encoder on the training data
    encoder = ReducedRankRegression(alpha=params["alpha_encoder"], fit_intercept=params["fit_intercept_encoder"]).fit(
        train_source.T, train_position.T
    )
    decoder = ReducedRankRegression(alpha=params["alpha_decoder"], fit_intercept=params["fit_intercept_decoder"]).fit(
        train_position.T, train_target.T
    )

    # get the encoder and decoder predictions
    test_position_hat = encoder.predict(test_source.T, nonnegative=params["nonnegative_encoder"])
    test_target_hat = decoder.predict(test_position_hat, nonnegative=params["nonnegative_decoder"])

    # Measure score directly (redo prediction, but whatever it's fast)
    test_target_score = decoder.score(test_position_hat, test_target.T, nonnegative=params["nonnegative_decoder"])

    # Measure the scaled mse of the prediction
    test_target_scaled_mse = scaled_mse(test_target_hat, test_target.T, reduce="mean")

    # Return test score
    return test_target_score, test_target_scaled_mse


def optimize_direct_model(mouse_name, datestr, sessionid, params, population_name=None, n_trials=20, n_jobs=2, show_progress_bar=True):
    """
    Optimize a direct model from source to target using the same rank as the optimal position basis model.
    """
    # Load population with behavior data
    npop, behavior_data = load_population(mouse_name, datestr, sessionid, population_name=population_name, get_behavior=True)

    # this is a bit of a hack to get the rank of the position basis, instead we could multiply num_basis by num_environments but this is fast enough
    position_basis = make_position_basis(behavior_data["position"], behavior_data["environment"], num_basis=params["num_basis"])
    rank = position_basis.shape[1]

    # split the data into training and validation sets
    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    val_source, val_target = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")
    test_source, test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

    # method for evaluating hyperparameters
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e1, 1e9, log=True)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        nonnegative = trial.suggest_categorical("nonnegative", [True, False])
        model = ReducedRankRegression(alpha=alpha, fit_intercept=fit_intercept).fit(train_source.T, train_target.T)
        val_score = model.score(val_source.T, val_target.T, rank=rank, nonnegative=nonnegative)
        return val_score

    # Optimize the model
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=show_progress_bar)

    # Evaluate the best model on the test set
    best_params = study.best_params
    model = ReducedRankRegression(alpha=best_params["alpha"], fit_intercept=best_params["fit_intercept"]).fit(train_source.T, train_target.T)
    test_score = model.score(test_source.T, test_target.T, rank=rank, nonnegative=best_params["nonnegative"])
    test_prediction = model.predict(test_source.T, rank=rank, nonnegative=best_params["nonnegative"])
    test_scaled_mse = scaled_mse(test_prediction, test_target.T, reduce="mean")

    return test_score, test_scaled_mse, best_params


def optuna_study_rrr_state(mouse_name, datestr, sessionid, population_name=None, n_trials=40, n_jobs=2, show_progress_bar=True):
    """
    mouse_name : str
        Name of the mouse.
    datestr : str
        Date string of the session.
    sessionid : int
        Session identifier.
    population_name : Optional[str]
        Name of the population object to load. If None, the default population object will be loaded.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optimize_rrr_state(trial, mouse_name, datestr, sessionid, population_name=population_name),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # Evaluate the best model on the test set
    test_score, test_scaled_mse = test_rrr_state(mouse_name, datestr, sessionid, study.best_params, population_name=population_name)

    # Now compare to a direct model
    test_score_direct, test_scaled_mse_direct, best_params_direct = optimize_direct_model(
        mouse_name,
        datestr,
        sessionid,
        study.best_params,
        population_name=population_name,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # return the results of the full study
    results = dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        study=study,
        params=study["best_params"],
        test_score=test_score,
        test_scaled_mse=test_scaled_mse,
        test_score_direct=test_score_direct,
        test_scaled_mse_direct=test_scaled_mse_direct,
        params_direct=best_params_direct,
    )

    return results


def rrr_state_tempfile_name(vrexp, population_name=None):
    """generate temporary file name for rrr results"""
    name = f"rrr_optimization_results_state_{str(vrexp)}"
    if population_name is not None:
        name += f"_{population_name}"
    return name


def do_rrr_state_optimization(all_sessions, skip_completed=True, save=True, population_name=None):
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
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            if skip_completed and pcss.check_temp_file(rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)):
                print(f"Found completed RRR(state) optimization for: {mouse_name}, {datestr}, {sessionid}")
                continue

            print(f"Optimizing RRR(state) for: {mouse_name}, {datestr}, {sessionid}:")
            t = time.time()
            results = optuna_study_rrr_state(mouse_name, datestr, sessionid, population_name=population_name)
            if save:
                pcss.save_temp_file(results, rrr_state_tempfile_name(pcss.vrexp, population_name=population_name))


def load_rrr_results(all_sessions, results="all", population_name=None):
    rrr_results = []
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            rrr_filename = rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)
            if not pcss.check_temp_file(rrr_filename):
                print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} (not found)")
                continue
            print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
            rrr_results.append(pcss.load_temp_file(rrr_filename))
    if results == "all":
        return rrr_results
    if results == "test_by_mouse":
        mouse_names = list(set([res["mouse_name"] for res in rrr_results]))
        num_mice = len(mouse_names)
        test_scores = torch.zeros((num_mice,))
        test_scaled_mses = torch.zeros((num_mice,))
        test_scores_direct = torch.zeros((num_mice,))
        test_scaled_mses_direct = torch.zeros((num_mice,))
        num_samples = torch.zeros(num_mice)
        for rrr_res in rrr_results:
            mouse_idx = mouse_names.index(rrr_res["mouse_name"])
            test_scores[mouse_idx] += torch.tensor(rrr_res["test_score"])
            test_scaled_mses[mouse_idx] += torch.tensor(rrr_res["test_scaled_mse"])
            test_scores_direct[mouse_idx] += torch.tensor(rrr_res["test_score_direct"])
            test_scaled_mses_direct[mouse_idx] += torch.tensor(rrr_res["test_scaled_mse_direct"])
            num_samples[mouse_idx] += 1
        # get average for each mouse
        test_scores /= num_samples
        test_scaled_mses /= num_samples
        test_scores_direct /= num_samples
        test_scaled_mses_direct /= num_samples
        return dict(
            mouse_names=mouse_names,
            test_scores=test_scores,
            test_scaled_mses=test_scaled_mses,
            test_scores_direct=test_scores_direct,
            test_scaled_mses_direct=test_scaled_mses_direct,
        )
    raise ValueError(f"results must be 'all' or 'test_by_mouse', got {results}")
