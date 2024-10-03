import os
import sys
import time
import torch
import optuna
import numpy as np
from scipy.stats import zscore
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from rastermap import Rastermap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from scripts.dimilibi.helpers import load_population, make_position_basis
from scripts.dimilibi.helpers import SPEEDTHRESHOLD



from vrAnalysis import analysis
from vrAnalysis import session
from vrAnalysis.helpers import save_figure
from vrAnalysis.helpers import refline

from dimilibi import ReducedRankRegression, scaled_mse, measure_r2


def optimize_rrr_state(trial, mouse_name, datestr, sessionid, keep_planes=[1, 2, 3, 4], population_name=None):
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
    fit_intercept_encoder = True  # trial.suggest_categorical("fit_intercept_encoder", [True, False])
    fit_intercept_decoder = True  # trial.suggest_categorical("fit_intercept_decoder", [True, False])
    nonnegative_encoder = True  # trial.suggest_categorical("nonnegative_encoder", [True, False])
    nonnegative_decoder = True  # trial.suggest_categorical("nonnegative_decoder", [True, False])

    # Load population with behavior data
    npop, behavior_data = load_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True)

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


def test_rrr_state(mouse_name, datestr, sessionid, params, keep_planes=[1, 2, 3, 4], population_name=None):
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
    npop, behavior_data = load_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True)

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
    encoder = ReducedRankRegression(alpha=params["alpha_encoder"], fit_intercept=True).fit(train_source.T, train_position.T)
    decoder = ReducedRankRegression(alpha=params["alpha_decoder"], fit_intercept=True).fit(train_position.T, train_target.T)

    # get the encoder and decoder predictions
    test_position_hat = encoder.predict(test_source.T, nonnegative=True)
    test_target_hat = decoder.predict(test_position_hat, nonnegative=True)

    # Measure score directly (redo prediction, but whatever it's fast)
    test_target_score = decoder.score(test_position_hat, test_target.T, nonnegative=True)

    # Measure the scaled mse of the prediction
    test_target_scaled_mse = scaled_mse(test_target_hat, test_target.T, reduce="mean")

    # Return test score
    return test_target_score, test_target_scaled_mse


def optimize_direct_model(
    mouse_name, datestr, sessionid, params, keep_planes=[1, 2, 3, 4], population_name=None, n_trials=12, n_jobs=2, show_progress_bar=True
):
    """
    Optimize a direct model from source to target using the same rank as the optimal position basis model.
    """
    # Load population with behavior data
    npop, behavior_data = load_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True)

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
        fit_intercept = True  # trial.suggest_categorical("fit_intercept", [True, False])
        nonnegative = True  # trial.suggest_categorical("nonnegative", [True, False])
        model = ReducedRankRegression(alpha=alpha, fit_intercept=fit_intercept).fit(train_source.T, train_target.T)
        val_score = model.score(val_source.T, val_target.T, rank=rank, nonnegative=nonnegative)
        return val_score

    # Optimize the model
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=show_progress_bar)

    # Evaluate the best model on the test set
    best_params = study.best_params
    model = ReducedRankRegression(alpha=best_params["alpha"], fit_intercept=True).fit(train_source.T, train_target.T)
    test_score = model.score(test_source.T, test_target.T, rank=rank, nonnegative=True)
    test_prediction = model.predict(test_source.T, rank=rank, nonnegative=True)
    test_scaled_mse = scaled_mse(test_prediction, test_target.T, reduce="mean")

    return test_score, test_scaled_mse, best_params


def optuna_study_rrr_state(
    mouse_name,
    datestr,
    sessionid,
    keep_planes=[1, 2, 3, 4],
    population_name=None,
    n_trials=36,
    n_jobs=2,
    show_progress_bar=True,
):
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
        lambda trial: optimize_rrr_state(trial, mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # Evaluate the best model on the test set
    test_score, test_scaled_mse = test_rrr_state(
        mouse_name, datestr, sessionid, study.best_params, keep_planes=keep_planes, population_name=population_name
    )

    # Now compare to a direct model
    test_score_direct, test_scaled_mse_direct, best_params_direct = optimize_direct_model(
        mouse_name,
        datestr,
        sessionid,
        study.best_params,
        keep_planes=keep_planes,
        population_name=population_name,
        # use default value (it's less than the full model) n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # return the results of the full study
    results = dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        study=study,
        params=study.best_params,
        test_score=test_score,
        test_scaled_mse=test_scaled_mse,
        test_score_direct=test_score_direct,
        test_scaled_mse_direct=test_scaled_mse_direct,
        params_direct=best_params_direct,
    )

    return results


def add_rrr_state_results(all_sessions, keep_planes=[1, 2, 3, 4], population_name=None):
    """
    This is a helper function to add a result to the results list
    It's useful to avoid having to completely reload all the results when adding a single result
    Right now, I'm adding a measurement of the test_score for a place field lookup model
    When adding new things, add a boolean which can be set to True/False to determine which to do
    """
    add_place_field_score = False
    add_position_decoder_from_direct_model = False
    add_simple_position_estimator = False
    add_rbfpos_to_target = False
    add_doublecv_rbf_estimator = True
    if add_place_field_score:
        for mouse_name, sessions in all_sessions.items():
            for datestr, sessionid in sessions:
                pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), speedThreshold=SPEEDTHRESHOLD, keep_planes=keep_planes, autoload=False)
                rrr_filename = rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)
                if not pcss.check_temp_file(rrr_filename):
                    print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} -- (temp file not found)")
                    continue
                else:
                    print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
                    results = pcss.load_temp_file(rrr_filename)

                    # Get the population data etc
                    npop, behavior_data = load_population(
                        mouse_name,
                        datestr,
                        sessionid,
                        keep_planes=keep_planes,
                        population_name=population_name,
                        get_behavior=True,
                    )

                    environments = np.unique(behavior_data["environment"])
                    num_environments = len(environments)
                    num_timepoints = behavior_data["position"].shape[0]
                    pos_by_env = torch.full((num_timepoints, num_environments), np.nan, dtype=torch.float32)
                    for ienv, envnum in enumerate(environments):
                        idx_env = behavior_data["environment"] == envnum
                        pos_by_env[idx_env, ienv] = torch.tensor(behavior_data["position"][idx_env]).float()

                    # split the data into training and validation sets
                    test_pos_by_env = npop.apply_split(pos_by_env.T, 2)
                    test_source, test_target = npop.get_split_data(2, center=False, scale=True)

                    # Get place fields from ROIs (use all trials)
                    spkmaps = pcss.get_spkmap(average=True, trials="full")
                    spkmaps_source = [torch.tensor(spkmap[npop.cell_split_indices[0]], dtype=torch.float32) for spkmap in spkmaps]
                    spkmaps_target = [torch.tensor(spkmap[npop.cell_split_indices[1]], dtype=torch.float32) for spkmap in spkmaps]

                    # place field prediction
                    pf_source_pred = torch.full(test_source.shape, np.nan)
                    pf_target_pred = torch.full(test_target.shape, np.nan)
                    for ienv, envnum in enumerate(environments):
                        idx_env = ~torch.isnan(test_pos_by_env[ienv])
                        pos = test_pos_by_env[ienv][idx_env].floor().long()
                        pf_source_pred[:, idx_env] = torch.gather(spkmaps_source[ienv], 1, pos.view(1, -1).expand(spkmaps_source[ienv].shape[0], -1))
                        pf_target_pred[:, idx_env] = torch.gather(spkmaps_target[ienv], 1, pos.view(1, -1).expand(spkmaps_target[ienv].shape[0], -1))

                    not_nan_position = ~torch.any(torch.isnan(pf_target_pred), dim=0)
                    y_target = test_target[:, not_nan_position]
                    y_hat = pf_target_pred[:, not_nan_position]
                    y_source = test_source[:, not_nan_position]
                    y_hat_source = pf_source_pred[:, not_nan_position]

                    # Write an optimization using gradient descent for the gain variable
                    def objective_gain(gain, y_hat, y_target):
                        # gain is a vector that multiplies each sample of y_hat equally
                        # where y_hat is the neurons by samples prediction
                        assert gain.shape[0] == y_hat.shape[1], "Gain must have the same number of samples as y_hat"
                        assert gain.ndim == 1, "Gain must be a vector"
                        return measure_r2(y_hat * gain, y_target)
                    
                    # Fit gain on the source data
                    gain_source = torch.ones(y_hat_source.shape[1], dtype=torch.float32, requires_grad=True)
                    optimizer = torch.optim.Adam([gain_source], lr=1e-2, maximize=True)
                    for _ in range(1000):
                        optimizer.zero_grad()
                        loss = objective_gain(gain_source, y_hat_source, y_source)
                        loss.backward()
                        optimizer.step()

                    # Fit gain on the target data
                    gain_target = torch.ones(y_hat_source.shape[1], dtype=torch.float32, requires_grad=True)
                    optimizer = torch.optim.Adam([gain_target], lr=1e-2, maximize=True)
                    for _ in range(1000):
                        optimizer.zero_grad()
                        loss = objective_gain(gain_target, y_hat, y_target)
                        loss.backward()
                        optimizer.step()
                    
                    y_hat_source_gain = (y_hat * gain_source).detach()
                    y_hat_target_gain = (y_hat * gain_target).detach()
                    print("Place field prediction score:", measure_r2(y_hat, y_target))
                    print("Place field prediction score (cvgain):", measure_r2(y_hat_source_gain, y_target))
                    print("Place field prediction score (gain):", measure_r2(y_hat_target_gain, y_target))

                    # Add the place field prediction score to the results
                    results["pf_pred_score"] = measure_r2(y_hat, y_target)
                    results["pf_pred_score_withcvgain"] = measure_r2(y_hat_source_gain, y_target)
                    results["pf_pred_score_withgain"] = measure_r2(y_hat_target_gain, y_target)

                    # Save the results
                    pcss.save_temp_file(results, rrr_filename)


    if add_position_decoder_from_direct_model:
        for mouse_name, sessions in all_sessions.items():
            for datestr, sessionid in sessions:
                pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), speedThreshold=SPEEDTHRESHOLD, keep_planes=keep_planes, autoload=False)
                rrr_filename = rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)
                if not pcss.check_temp_file(rrr_filename):
                    print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} -- (temp file not found)")
                    continue
                else:
                    print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
                    results = pcss.load_temp_file(rrr_filename)
                    params = results["params"]
                    params_direct = results["params_direct"]

                    # Get the population data etc
                    npop, behavior_data = load_population(
                        mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True
                    )

                    # Make the position basis with optimal hyperparameters
                    position_basis = make_position_basis(
                        behavior_data["position"], behavior_data["environment"], num_basis=params["num_basis"], basis_width=params["basis_width"]
                    )

                    # split the data into training and validation sets
                    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
                    val_source, _ = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")
                    test_source, test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

                    # Get the train / validation data for the behavior position
                    train_position = npop.apply_split(position_basis.T, 0)
                    val_position = npop.apply_split(position_basis.T, 1)
                    test_position = npop.apply_split(position_basis.T, 2)

                    # method for evaluating hyperparameters
                    def objective_encoder(trial):
                        alpha = trial.suggest_float("alpha", 1e-1, 1e9, log=True)
                        fit_intercept = True  # trial.suggest_categorical("fit_intercept", [True, False])
                        nonnegative = True  # trial.suggest_categorical("nonnegative", [True, False])
                        model = ReducedRankRegression(alpha=alpha, fit_intercept=fit_intercept).fit(train_source.T, train_position.T)
                        val_score = model.score(val_source.T, val_position.T, nonnegative=nonnegative)
                        return val_score

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective_encoder, n_trials=24, n_jobs=2, show_progress_bar=True)

                    best_params = study.best_params
                    encoder = ReducedRankRegression(alpha=best_params["alpha"], fit_intercept=True).fit(train_source.T, train_position.T)
                    test_prediction = encoder.predict(test_source.T, nonnegative=True)
                    encoder_score = measure_r2(test_prediction.T, test_position)
                    print("Encoder score:", encoder_score, "Alpha: ", best_params["alpha"], "<-- make sure this isn't 1e-1 or 1e9!!")

                    rank = position_basis.shape[1]
                    direct_model = ReducedRankRegression(alpha=params_direct["alpha"], fit_intercept=True).fit(train_source.T, train_target.T)
                    train_latent = direct_model.predict_latent(train_source.T, rank=rank)
                    val_latent = direct_model.predict_latent(val_source.T, rank=rank)
                    test_latent = direct_model.predict_latent(test_source.T, rank=rank)

                    # method for evaluating hyperparameters
                    def objective_latent(trial):
                        alpha = trial.suggest_float("alpha", 1e-9, 1e9, log=True)
                        fit_intercept = True  # trial.suggest_categorical("fit_intercept", [True, False])
                        nonnegative = True  # trial.suggest_categorical("nonnegative", [True, False])
                        model = ReducedRankRegression(alpha=alpha, fit_intercept=fit_intercept).fit(train_latent, train_position.T)
                        val_score = model.score(val_latent, val_position.T, rank=rank, nonnegative=nonnegative)
                        return val_score

                    # Optimize the model
                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective_latent, n_trials=24, n_jobs=2, show_progress_bar=True)

                    # Evaluate the best model on the test set
                    best_params = study.best_params
                    model = ReducedRankRegression(alpha=best_params["alpha"], fit_intercept=True).fit(train_latent, train_position.T)
                    test_prediction = model.predict(test_latent, rank=rank, nonnegative=True)
                    test_score = measure_r2(test_prediction.T, test_position)
                    print("Latent variable score:", test_score)

                    # Add the place field prediction score to the results
                    results["encoder_position_score"] = encoder_score
                    results["latent_position_score"] = test_score

                    # Save the results
                    pcss.save_temp_file(results, rrr_filename)

    if add_simple_position_estimator:
        """
        This model is one where we predict a neurons activity from their place field, where we optimize
        the position estimate from half the cells and test on the other half of cells. 

        Let f_{nt} = p_n(x_t) + \epsilon_{nt} be the activity of neuron n at time t
        - where p_n(x_t) is the place field of neuron n at position x_t
        - and \epsilon_{nt} is the independent (usually gaussian) noise in the system

        p_n(x) is determined "deterministically" by a method established elsewhere
        however, x_t is determined by minimizing the following loss function:

        L = \sum_{n,t} (f_{nt} - p_n(x_t))^2

        Once x_t is determined, we can calculate the R^2 score of the prediction for an 
        independent set of cells. This is the score we will add to the results.
        """
        for mouse_name, sessions in all_sessions.items():
            for datestr, sessionid in sessions:
                pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), speedThreshold=SPEEDTHRESHOLD, keep_planes=keep_planes, autoload=False)
                rrr_filename = rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)
                if not pcss.check_temp_file(rrr_filename):
                    print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} -- (temp file not found)")
                    continue
                else:
                    print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
                    results = pcss.load_temp_file(rrr_filename)

                    # Get the population data etc
                    npop, behavior_data = load_population(
                        mouse_name,
                        datestr,
                        sessionid,
                        keep_planes=keep_planes,
                        population_name=population_name,
                        get_behavior=True,
                    )
                    num_environments = len(pcss.environments)
                    num_timepoints = behavior_data["position"].shape[0]
                    pos_by_env = torch.full((num_timepoints, num_environments), np.nan, dtype=torch.float32)
                    for ienv, envnum in enumerate(pcss.environments):
                        idx_env = behavior_data["environment"] == envnum
                        pos_by_env[idx_env, ienv] = torch.tensor(behavior_data["position"][idx_env]).float()

                    test_source, test_target = npop.get_split_data(2, center=False, scale=True)
                    test_environment = npop.apply_split(behavior_data["environment"].reshape(1, -1), 2).squeeze()

                    test_pos_by_env = npop.apply_split(pos_by_env.T, 2)

                    # Get place fields from ROIs (use all trials)
                    spkmaps = pcss.get_spkmap(average=True, trials="full")
                    spkmaps_source = [torch.tensor(spkmap[npop.cell_split_indices[0]], dtype=torch.float32) for spkmap in spkmaps]
                    spkmaps_target = [torch.tensor(spkmap[npop.cell_split_indices[1]], dtype=torch.float32) for spkmap in spkmaps]
                    
                    # Clear out nan positions
                    idx_nan = torch.any(
                        torch.stack([torch.any(torch.isnan(s), dim=0) for s in spkmaps_source], dim=0)
                        | torch.stack([torch.any(torch.isnan(s), dim=0) for s in spkmaps_target], dim=0), dim=0
                    )
                    idx_to_original_position = torch.where(~idx_nan)[0]
                    spkmaps_source = [s[:, ~idx_nan] for s in spkmaps_source]
                    spkmaps_target = [s[:, ~idx_nan] for s in spkmaps_target]

                    # Optimize position estimate from source neurons
                    # For each environment, pick the position index to the spkmap that minimizes loss
                    pos_estimate = torch.full(test_pos_by_env.shape, np.nan, dtype=torch.float32)
                    source_estimate = torch.zeros_like(test_source)
                    target_estimate = torch.zeros_like(test_target)
                    for ienv, envnum in enumerate(pcss.environments):
                        idx_env = test_environment == envnum
                        env_target = test_source[:, idx_env].unsqueeze(2).expand(-1, -1, spkmaps_source[ienv].shape[1])
                        spkmap_pred = spkmaps_source[ienv].unsqueeze(1).expand(-1, env_target.shape[1], -1)
                        mse = torch.mean((env_target - spkmap_pred) ** 2, dim=0)
                        best_pos = torch.argmin(mse, dim=1)
                        pos_estimate[ienv, idx_env] = idx_to_original_position[best_pos].float()
                        source_estimate[:, idx_env] = spkmaps_source[ienv][:, best_pos]
                        target_estimate[:, idx_env] = spkmaps_target[ienv][:, best_pos]

                    # Write an optimization using gradient descent for the gain variable
                    def objective_gain(gain, y_hat, y_target):
                        # gain is a vector that multiplies each sample of y_hat equally
                        # where y_hat is the neurons by samples prediction
                        assert gain.shape[0] == y_hat.shape[1], "Gain must have the same number of samples as y_hat"
                        assert gain.ndim == 1, "Gain must be a vector"
                        return measure_r2(y_hat * gain, y_target)
                    
                    # Fit gain on the source data
                    gain_source = torch.ones(source_estimate.shape[1], dtype=torch.float32, requires_grad=True)
                    optimizer = torch.optim.Adam([gain_source], lr=1e-2, maximize=True)
                    for _ in range(1000):
                        optimizer.zero_grad()
                        loss = objective_gain(gain_source, source_estimate, test_source)
                        loss.backward()
                        optimizer.step()

                    gain_target = torch.ones(target_estimate.shape[1], dtype=torch.float32, requires_grad=True)
                    optimizer = torch.optim.Adam([gain_target], lr=1e-2, maximize=True)
                    for _ in range(1000):
                        optimizer.zero_grad()
                        loss = objective_gain(gain_target, target_estimate, test_target)
                        loss.backward()
                        optimizer.step()

                    target_estimate_cvgain = (target_estimate * gain_source).detach()
                    target_estimate_gain = (target_estimate * gain_target).detach()

                    flattened_pos_estimate = pos_estimate.flatten()
                    flattened_true_position = test_pos_by_env.flatten()
                    idx_nan_flattened = torch.isnan(flattened_pos_estimate) | torch.isnan(flattened_true_position)
                    pos_estimate = flattened_pos_estimate[~idx_nan_flattened]
                    true_position = flattened_true_position[~idx_nan_flattened]
                    print("Position estimate score (source):", measure_r2(source_estimate, test_source))
                    print("Position estimate score (target):", measure_r2(target_estimate, test_target))
                    print("Position estimate score (target, cvgain):", measure_r2(target_estimate_cvgain, test_target))
                    print("Position estimate score (target, gain):", measure_r2(target_estimate_gain, test_target))
                    print("Position estimate score (position):", measure_r2(pos_estimate, true_position))

                    # Add the place field prediction score to the results
                    results["opt_pos_estimate_source"] = measure_r2(source_estimate, test_source)
                    results["opt_pos_estimate_target"] = measure_r2(target_estimate, test_target)
                    results["opt_pos_estimate_target_withcvgain"] = measure_r2(target_estimate_cvgain, test_target)
                    results["opt_pos_estimate_target_withgain"] = measure_r2(target_estimate_gain, test_target)
                    results["opt_pos_estimate_position"] = measure_r2(pos_estimate, true_position)
                    results.pop("opt_pos_estimate_source_withgain", None)

                    # Save the results
                    pcss.save_temp_file(results, rrr_filename)
    
    if add_rbfpos_to_target:
        """
        To fill out the continuum of "progressively less constrained by spatial information", here we add a model that is exclusively
        trained on predicting target ROIs from the true RBF(Pos) data. This is a model that is more flexible than the PF models, because
        it can use this distribution of position information, but doesn't use any information about the neural state (i.e. from source
        ROIs) to make the prediction.
        """
        for mouse_name, sessions in all_sessions.items():
            for datestr, sessionid in sessions:
                pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), speedThreshold=SPEEDTHRESHOLD, keep_planes=keep_planes, autoload=False)
                rrr_filename = rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)
                if not pcss.check_temp_file(rrr_filename):
                    print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} -- (temp file not found)")
                    continue
                else:
                    print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
                    results = pcss.load_temp_file(rrr_filename)

                    def fit_decoder_only(trial, mouse_name, datestr, sessionid, keep_planes, population_name):
                        # Set up optuna study variables
                        num_basis = trial.suggest_int("num_basis", 5, 50, log=True)
                        basis_width = trial.suggest_float("basis_width", 1.0, 100.0, log=True)
                        alpha = trial.suggest_float("alpha", 1e-1, 1e5, log=True)
                        fit_intercept = True  # trial.suggest_categorical("fit_intercept_decoder", [True, False])
                        nonnegative = True  # trial.suggest_categorical("nonnegative_decoder", [True, False])

                        # Load population with behavior data
                        npop, behavior_data = load_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True)

                        # Make the position basis with current hyperparameters
                        position_basis = make_position_basis(behavior_data["position"], behavior_data["environment"], num_basis=num_basis, basis_width=basis_width)

                        # split the data into training and validation sets for the target ROIs only
                        train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")[1]
                        val_target = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")[1]

                        # Get the train / validation data for the behavior position
                        train_position = npop.apply_split(position_basis.T, 0)
                        val_position = npop.apply_split(position_basis.T, 1)

                        # build an encoder on the training data
                        decoder = ReducedRankRegression(alpha=alpha, fit_intercept=fit_intercept).fit(train_position.T, train_target.T)

                        # get decoder predictions
                        val_target_score = decoder.score(val_position.T, val_target.T, nonnegative=nonnegative)

                        # return the validation score
                        return val_target_score

                    def test_decoder_only(mouse_name, datestr, sessionid, params, keep_planes, population_name):
                        # Load population with behavior data
                        npop, behavior_data = load_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True)

                        # Make the position basis with optimal hyperparameters
                        position_basis = make_position_basis(
                            behavior_data["position"], behavior_data["environment"], num_basis=params["num_basis"], basis_width=params["basis_width"]
                        )

                        # split the data into training and validation sets
                        train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")[1]
                        test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")[1]

                        # Get the train / validation data for the behavior position
                        train_position = npop.apply_split(position_basis.T, 0)
                        test_position = npop.apply_split(position_basis.T, 2)

                        # build an encoder on the training data
                        decoder = ReducedRankRegression(alpha=params["alpha"], fit_intercept=True).fit(train_position.T, train_target.T)

                        # Measure score directly (redo prediction, but whatever it's fast)
                        test_target_score = decoder.score(test_position.T, test_target.T, nonnegative=True)

                        # Return test score
                        return test_target_score
                
                    # Optimize the model
                    study = optuna.create_study(direction="maximize")
                    study.optimize(
                        lambda trial: fit_decoder_only(trial, mouse_name, datestr, sessionid, keep_planes, population_name),
                        n_trials=24,
                        n_jobs=2,
                        show_progress_bar=True,
                    )

                    # Evaluate the best model on the test set
                    test_score = test_decoder_only(mouse_name, datestr, sessionid, study.best_params, keep_planes, population_name)

                    # Add the place field prediction score to the results
                    results["rbfpos_to_target_score"] = test_score

                    # Save temp file
                    pcss.save_temp_file(results, rrr_filename)

    if add_doublecv_rbf_estimator:
        """
        Perform a double cross-validation on the RBF position estimator to get a more accurate estimate of the performance 
        when limiting it to spatial information. Here, the encoder and decoder are trained on different time points, such that
        any information about the neural state that is independent of position (but "yoked" to position since we don't have that 
        many samples) is hopefully broken and the performance will be position-specific. 
        """
        for mouse_name, sessions in all_sessions.items():
            for datestr, sessionid in sessions:
                pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), speedThreshold=SPEEDTHRESHOLD, keep_planes=keep_planes, autoload=False)
                rrr_filename = rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)
                if not pcss.check_temp_file(rrr_filename):
                    print(f"Skipping rrr_results from {mouse_name}, {datestr}, {sessionid} -- (temp file not found)")
                    continue
                else:
                    print(f"Loading rrr_results from {mouse_name}, {datestr}, {sessionid}")
                    results = pcss.load_temp_file(rrr_filename)

                # Load population with behavior data
                npop, behavior_data = load_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True)

                def _split_by_chunks(idx):
                    start_chunk = torch.where(torch.cat((torch.tensor(True).view(1), torch.diff(idx) != 1)))[0]
                    end_chunk = torch.where(torch.cat((torch.diff(idx) != 1, torch.tensor(True).view(1))))[0]
                    chunks = [idx[s:e] for s, e in zip(start_chunk, end_chunk)]
                    idx_even = torch.cat(chunks[::2])
                    idx_odd = torch.cat(chunks[1::2])
                    return idx_even, idx_odd
                
                train_idx_encoder, train_idx_decoder = _split_by_chunks(npop.time_split_indices[0])

                # Overwrite these indices
                time_split_indices = [train_idx_encoder, train_idx_decoder, npop.time_split_indices[1], npop.time_split_indices[2]]
                npop.time_split_indices = time_split_indices

                def optimize_rrr_state(trial):
                    """
                    Optimize the reduced rank regression model using double cross-validation, where the encoder and decoder
                    are trained & validated on different timepoints.
                    """
                    # Set up optuna study variables
                    num_basis = trial.suggest_int("num_basis", 5, 50, log=True)
                    basis_width = trial.suggest_float("basis_width", 1.0, 100.0, log=True)
                    alpha_encoder = trial.suggest_float("alpha_encoder", 1e-1, 1e5, log=True)
                    alpha_decoder = trial.suggest_float("alpha_decoder", 1e-1, 1e5, log=True)
                    fit_intercept_encoder = True  # trial.suggest_categorical("fit_intercept_encoder", [True, False])
                    fit_intercept_decoder = True  # trial.suggest_categorical("fit_intercept_decoder", [True, False])
                    nonnegative_encoder = True  # trial.suggest_categorical("nonnegative_encoder", [True, False])
                    nonnegative_decoder = True  # trial.suggest_categorical("nonnegative_decoder", [True, False])

                    # Make the position basis with current hyperparameters
                    position_basis = make_position_basis(behavior_data["position"], behavior_data["environment"], num_basis=num_basis, basis_width=basis_width)

                    # split the data into training and validation sets
                    train_source_enc = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")[0]
                    train_target_dec = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")[1]
                    val_source, val_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

                    # Get the train / validation data for the behavior position
                    train_position_enc = npop.apply_split(position_basis.T, 0)
                    train_position_dec = npop.apply_split(position_basis.T, 1)

                    # build an encoder on the training data
                    encoder = ReducedRankRegression(alpha=alpha_encoder, fit_intercept=fit_intercept_encoder).fit(train_source_enc.T, train_position_enc.T)
                    decoder = ReducedRankRegression(alpha=alpha_decoder, fit_intercept=fit_intercept_decoder).fit(train_position_dec.T, train_target_dec.T)

                    # get the encoder and decoder predictions
                    val_position_hat = encoder.predict(val_source.T, nonnegative=nonnegative_encoder)
                    val_target_score = decoder.score(val_position_hat, val_target.T, nonnegative=nonnegative_decoder)

                    # return the validation score
                    return val_target_score


                def test_rrr_state(params):
                    # Make the position basis with optimal hyperparameters
                    position_basis = make_position_basis(
                        behavior_data["position"], behavior_data["environment"], num_basis=params["num_basis"], basis_width=params["basis_width"]
                    )

                    # split the data into training and validation sets
                    train_source_enc = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")[0]
                    train_target_dec = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")[1]

                    test_source, test_target = npop.get_split_data(3, center=False, scale=True, scale_type="preserve")

                    # Get the train / validation data for the behavior position
                    train_position_enc = npop.apply_split(position_basis.T, 0)
                    train_position_dec = npop.apply_split(position_basis.T, 1)

                    # build an encoder on the training data
                    encoder = ReducedRankRegression(alpha=params["alpha_encoder"], fit_intercept=True).fit(train_source_enc.T, train_position_enc.T)
                    decoder = ReducedRankRegression(alpha=params["alpha_decoder"], fit_intercept=True).fit(train_position_dec.T, train_target_dec.T)

                    # get the encoder and decoder predictions
                    test_position_hat = encoder.predict(test_source.T, nonnegative=True)

                    # Measure score directly (redo prediction, but whatever it's fast)
                    test_target_score = decoder.score(test_position_hat, test_target.T, nonnegative=True)

                    # Return test score
                    return test_target_score
                
                # Optimize the model
                study = optuna.create_study(direction="maximize")
                study.optimize(optimize_rrr_state, n_trials=24, n_jobs=2, show_progress_bar=True)

                # Test the model
                test_score = test_rrr_state(study.best_params)

                # Add the place field prediction score to the results
                results["test_score_doublecv"] = test_score

                # Save temp file
                pcss.save_temp_file(results, rrr_filename)


def rrr_state_tempfile_name(vrexp, population_name=None):
    """generate temporary file name for rrr results"""
    name = f"rrr_optimization_results_state_{str(vrexp)}"
    if population_name is not None:
        name += f"_{population_name}"
    return name


def do_rrr_state_optimization(all_sessions, skip_completed=True, save=True, keep_planes=[1, 2, 3, 4], population_name=None):
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
            results = optuna_study_rrr_state(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name)
            if save:
                pcss.save_temp_file(results, rrr_state_tempfile_name(pcss.vrexp, population_name=population_name))


def make_rrr_state_example(all_sessions, population_name=None, keep_planes=[1, 2, 3, 4]):
    """
    The idea is to hard code an example, but make it easy for the user to change the session in debugger mode...
    """
    idx_mouse = 3  # Change this if you want a different example!!!
    idx_session = 10  # Change this if you want a different example!!!
    mouse_names = list(all_sessions.keys())
    assert -len(mouse_names) <= idx_mouse < len(mouse_names), f"idx_mouse must be between -{len(mouse_names)} and {len(mouse_names) - 1}"
    date_strings, session_ids = zip(*all_sessions[mouse_names[idx_mouse]])
    assert -len(date_strings) <= idx_session < len(date_strings), f"idx_session must be between -{len(date_strings)} and {len(date_strings) - 1}"

    # We have our example now:
    mouse_name = mouse_names[idx_mouse]
    datestr = date_strings[idx_session]
    sessionid = session_ids[idx_session]
    print(f"Making example for: {mouse_name}, {datestr}, {sessionid}")
    pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
    rrr_filename = rrr_state_tempfile_name(pcss.vrexp, population_name=population_name)
    if not pcss.check_temp_file(rrr_filename):
        raise ValueError(f"RRR(state) optimization for {mouse_name}, {datestr}, {sessionid} not found.")

    # Get the results for the best parameters
    example_results = pcss.load_temp_file(rrr_filename)
    params = example_results["params"]

    # Load population with behavior data
    npop, behavior_data = load_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, population_name=population_name, get_behavior=True)

    # Make the position basis with optimal hyperparameters
    position_basis = make_position_basis(
        behavior_data["position"], behavior_data["environment"], num_basis=params["num_basis"], basis_width=params["basis_width"]
    )

    # optimize position estimate with place fields
    num_environments = len(pcss.environments)
    num_timepoints = behavior_data["position"].shape[0]
    pos_by_env = torch.full((num_timepoints, num_environments), np.nan, dtype=torch.float32)
    for ienv, envnum in enumerate(pcss.environments):
        idx_env = behavior_data["environment"] == envnum
        pos_by_env[idx_env, ienv] = torch.tensor(behavior_data["position"][idx_env]).float()

    test_source, test_target = npop.get_split_data(2, center=False, scale=True)
    test_environment = npop.apply_split(behavior_data["environment"].reshape(1, -1), 2).squeeze()

    test_pos_by_env = npop.apply_split(pos_by_env.T, 2)

    # Get place fields from ROIs (use all trials)
    spkmaps = pcss.get_spkmap(average=True, trials="full")
    spkmaps_source = [torch.tensor(spkmap[npop.cell_split_indices[0]], dtype=torch.float32) for spkmap in spkmaps]
    spkmaps_target = [torch.tensor(spkmap[npop.cell_split_indices[1]], dtype=torch.float32) for spkmap in spkmaps]
    
    # Clear out nan positions
    idx_nan = torch.any(
        torch.stack([torch.any(torch.isnan(s), dim=0) for s in spkmaps_source], dim=0)
        | torch.stack([torch.any(torch.isnan(s), dim=0) for s in spkmaps_target], dim=0), dim=0
    )
    idx_to_original_position = torch.where(~idx_nan)[0]
    spkmaps_source = [s[:, ~idx_nan] for s in spkmaps_source]

    # Optimize position estimate from source neurons
    # For each environment, pick the position index to the spkmap that minimizes loss
    pos_estimate = torch.full(test_pos_by_env.shape, np.nan, dtype=torch.float32)
    for ienv, envnum in enumerate(pcss.environments):
        idx_env = test_environment == envnum
        env_target = test_source[:, idx_env].unsqueeze(2).expand(-1, -1, spkmaps_source[ienv].shape[1])
        spkmap_pred = spkmaps_source[ienv].unsqueeze(1).expand(-1, env_target.shape[1], -1)
        mse = torch.mean((env_target - spkmap_pred) ** 2, dim=0)
        best_pos = torch.argmin(mse, dim=1)
        pos_estimate[ienv, idx_env] = idx_to_original_position[best_pos].float()

    # split the data into training and validation sets
    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    test_source, test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

    # Get the train / validation data for the behavior position
    train_position = npop.apply_split(position_basis.T, 0)
    test_position = npop.apply_split(position_basis.T, 2)
    

    # build an encoder on the training data
    encoder = ReducedRankRegression(alpha=params["alpha_encoder"], fit_intercept=True).fit(train_source.T, train_position.T)
    decoder = ReducedRankRegression(alpha=params["alpha_decoder"], fit_intercept=True).fit(train_position.T, train_target.T)
    direct = ReducedRankRegression(alpha=example_results["params_direct"]["alpha"], fit_intercept=True).fit(train_source.T, train_target.T)
    direct_target_hat = direct.predict(test_source.T, rank=position_basis.shape[1], nonnegative=True)

    # get the encoder and decoder predictions
    test_position_hat = encoder.predict(test_source.T, nonnegative=True)
    test_target_hat = decoder.predict(test_position_hat, nonnegative=True)

    # Measure score directly (redo prediction, but whatever it's fast)
    test_target_score = decoder.score(test_position_hat, test_target.T, nonnegative=True)
    print(test_target_score)

    z_train_source = zscore(np.array(train_source), axis=1)
    source_model = Rastermap(n_PCs=200, n_clusters=40, locality=0.75, time_lag_window=5).fit(z_train_source)
    source_sort = source_model.isort

    z_train_target = zscore(np.array(train_target), axis=1)
    target_model = Rastermap(n_PCs=200, n_clusters=40, locality=0.75, time_lag_window=5).fit(z_train_target)
    target_sort = target_model.isort

    test_position_environment = npop.apply_split(np.stack((behavior_data["position"], behavior_data["environment"]), axis=0), 2)
    environments = np.unique(behavior_data["environment"])
    num_environments = len(environments)
    test_position_by_env = np.full((test_position_hat.shape[0], num_environments), np.nan)
    for idx, envnum in enumerate(environments):
        i_env = test_position_environment[1] == envnum
        test_position_by_env[i_env, idx] = test_position_environment[0][i_env]

    fontsize = 24
    plt.rcParams.update({"font.size": fontsize})

    # Show prediction in alpha on top of the position
    vmin = 0
    vmax = 4
    vmax_pos = test_position_hat.max() * 0.5
    roi_extent = [0, test_source.shape[1], 0, test_source.shape[0]]


    from matplotlib.patches import ConnectionPatch

    def add_custom_arrow(ax1, ax2, xyA, xyB, color, width, arrowstyle, connectionstyle, mutation_scale=20, zorder=1):
        return ConnectionPatch(
            xyA=xyA, xyB=xyB, coordsA="axes fraction", coordsB="axes fraction", axesA=ax1, axesB=ax2, arrowstyle=arrowstyle, color=color, linewidth=width, connectionstyle=connectionstyle, zorder=zorder, mutation_scale=mutation_scale,
        )

    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

    # Source ROI
    ax[0].imshow(test_source[source_sort], extent=roi_extent, aspect="auto", cmap="gray_r", vmin=vmin, vmax=vmax)
    ax[0].set_ylabel("Source")

    # True Position & Prediction
    max_position = 200
    envcol = 'kr'
    for idx, envnum in enumerate(environments):
        pred_label = "Pred" if idx == 0 else None
        ax[1].plot(pos_estimate[idx] + max_position*idx, color="sienna", linestyle="--", label=pred_label, linewidth=2)
        true_label = "True" if idx == 0 else None
        ax[1].plot(test_position_by_env[:, idx] + max_position*idx, color=envcol[idx], label=true_label, linewidth=2)

    ax[1].set_ylabel("Pos")
    ax[1].legend(fontsize=20, loc="lower right", fancybox=True, shadow=True)
    ax[1].set_ylim(0, max_position * num_environments)

    # RBF(Position) & Prediction
    extent = [0, test_position.shape[1], 0, test_position_hat.shape[1]]
    ax[2].imshow(test_position, aspect="auto", cmap="gray_r", extent=extent, interpolation="none", vmin=vmin)
    ax[2].invert_yaxis()
    ax[2].set_ylabel("RBF(Pos)")

    ax[3].imshow(test_position_hat.T, aspect="auto", extent=extent, cmap="Purples", interpolation="none", vmin=vmin, vmax=vmax_pos)
    ax[3].invert_yaxis()
    ax[3].set_ylabel("Pred")

    # Target ROI
    ax[4].imshow(test_target[target_sort], extent=roi_extent, aspect="auto", cmap="gray_r", vmin=vmin, vmax=vmax)
    xlim = ax[4].get_xlim()
    xticks = np.arange(0, xlim[1], 200)
    ax[4].set_xticks(xticks)
    ax[4].set_xlabel("Imaging Frames - Test Set")
    ax[4].set_ylabel("Target")

    # Remove yticks
    for a in ax:
        a.set_yticks([])
    
    left_x_start = -0.06
    right_x_start = 1.01
    mutation_scale = 40
    inner_rad = 0.25
    outer_rad = 0.35

    black_arrow = add_custom_arrow(ax[1], ax[4], (left_x_start, 0.5), (left_x_start, 0.35), 'black', 5, '-|>', f'arc3,rad={outer_rad}', mutation_scale=mutation_scale)
    fig.add_artist(black_arrow)

    # Orange arrows from ax[0] to ax[1] then ax[1] to ax[4] on the left
    orange_arrow1 = add_custom_arrow(ax[0], ax[1], (left_x_start, 0.5), (left_x_start, 0.65), 'sienna', 5, '-|>', f'arc3,rad={inner_rad}', mutation_scale=mutation_scale)
    orange_arrow2 = add_custom_arrow(ax[1], ax[4], (left_x_start, 0.5), (left_x_start, 0.65), 'sienna', 5, '-|>', f'arc3,rad={inner_rad}', mutation_scale=mutation_scale)
    fig.add_artist(orange_arrow1)
    fig.add_artist(orange_arrow2)

    # Green arrows from ax[0] to ax[3] then ax[3] to ax[4] on the right
    green_arrow1 = add_custom_arrow(ax[0], ax[3], (right_x_start, 0.5), (right_x_start, 0.5), 'darkmagenta', 5, '-|>', f'arc3,rad=-{inner_rad}', mutation_scale=mutation_scale)
    green_arrow2 = add_custom_arrow(ax[3], ax[4], (right_x_start, 0.5), (right_x_start, 0.65), 'darkmagenta', 5, '-|>', f'arc3,rad=-{inner_rad}', mutation_scale=mutation_scale)
    fig.add_artist(green_arrow1)
    fig.add_artist(green_arrow2)

    # Blue arrow from ax[0] to ax[4] on the right
    blue_arrow = add_custom_arrow(ax[0], ax[4], (right_x_start, 0.5), (right_x_start, 0.35), 'orangered', 5, '-|>', f'arc3,rad=-{outer_rad}', mutation_scale=mutation_scale)
    fig.add_artist(blue_arrow)

    plt.tight_layout(h_pad=0.001)
    fig.subplots_adjust(left=0.15, right=0.85)

    plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = pcss.saveDirectory("example_plots")
        save_name = f"{pcss.vrexp.sessionPrint('_')}_state_pred_example"
        save_path = save_directory / save_name
        save_figure(fig, save_path)




    total_in_target = np.prod(test_target.shape)
    idx_random = np.random.choice(total_in_target, 250000, replace=False)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    ax.scatter(test_target.flatten()[idx_random], test_target_hat.T.flatten()[idx_random], s=1, c="k", label="Position Constrained Model")
    ax.scatter(test_target.flatten()[idx_random], direct_target_hat.T.flatten()[idx_random], s=1, c="r", label="Direct Model")
    refline(1, 0, ax=ax, color="k", linestyle="--")
    ax.set_xlabel("True Target Activity")
    ax.set_ylabel("Predicted Target Activity")
    ax.set_title("Decoder Performance")
    ax.legend()
    plt.show()

    # Measure r2 for the two models
    from sklearn.metrics import r2_score

    r2_score(test_target.flatten(), test_target_hat.T.flatten()), r2_score(test_target.flatten(), direct_target_hat.T.flatten())
    print(r2_score(test_target, test_target_hat.T), r2_score(test_target, direct_target_hat.T))


def load_rrr_state_results(all_sessions, results="all", population_name=None):
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
        test_scores_pfpred = torch.zeros((num_mice,))
        test_scores_pfpred_withgain = torch.zeros((num_mice,))
        pos_score_encoder = torch.zeros((num_mice,))
        pos_score_latent = torch.zeros((num_mice,))
        opt_pos_est_source_score = torch.zeros((num_mice,))
        opt_pos_est_target_score = torch.zeros((num_mice,))
        opt_pos_est_source_withgain_score = torch.zeros((num_mice,))
        opt_pos_est_target_withgain_score = torch.zeros((num_mice,))
        opt_pos_est_position_score = torch.zeros((num_mice,))
        test_scores_doublecv = torch.zeros((num_mice,))
        rbfpos_to_target_score = torch.zeros((num_mice,))
        num_samples = torch.zeros(num_mice)
        for rrr_res in rrr_results:
            mouse_idx = mouse_names.index(rrr_res["mouse_name"])
            test_scores[mouse_idx] += torch.tensor(rrr_res["test_score"])
            test_scaled_mses[mouse_idx] += torch.tensor(rrr_res["test_scaled_mse"])
            test_scores_direct[mouse_idx] += torch.tensor(rrr_res["test_score_direct"])
            test_scaled_mses_direct[mouse_idx] += torch.tensor(rrr_res["test_scaled_mse_direct"])
            test_scores_pfpred[mouse_idx] += torch.tensor(rrr_res["pf_pred_score"])
            test_scores_pfpred_withgain[mouse_idx] += torch.tensor(rrr_res["pf_pred_score_withgain"])
            pos_score_encoder[mouse_idx] += torch.tensor(rrr_res["encoder_position_score"])
            pos_score_latent[mouse_idx] += torch.tensor(rrr_res["latent_position_score"])
            opt_pos_est_source_score[mouse_idx] += torch.tensor(rrr_res["opt_pos_estimate_source"])
            opt_pos_est_target_score[mouse_idx] += torch.tensor(rrr_res["opt_pos_estimate_target"])
            opt_pos_est_source_withgain_score[mouse_idx] += torch.tensor(rrr_res["opt_pos_estimate_source_withgain"])
            opt_pos_est_target_withgain_score[mouse_idx] += torch.tensor(rrr_res["opt_pos_estimate_target_withgain"])
            opt_pos_est_position_score[mouse_idx] += torch.tensor(rrr_res["opt_pos_estimate_position"])
            test_scores_doublecv[mouse_idx] += torch.tensor(rrr_res["test_score_doublecv"])
            rbfpos_to_target_score[mouse_idx] += torch.tensor(rrr_res["rbfpos_to_target_score"])
            num_samples[mouse_idx] += 1

        # get average for each mouse
        test_scores /= num_samples
        test_scaled_mses /= num_samples
        test_scores_direct /= num_samples
        test_scaled_mses_direct /= num_samples
        test_scores_pfpred /= num_samples
        test_scores_pfpred_withgain /= num_samples
        pos_score_encoder /= num_samples
        pos_score_latent /= num_samples
        opt_pos_est_source_score /= num_samples
        opt_pos_est_target_score /= num_samples
        opt_pos_est_source_withgain_score /= num_samples
        opt_pos_est_target_withgain_score /= num_samples
        opt_pos_est_position_score /= num_samples
        test_scores_doublecv /= num_samples
        rbfpos_to_target_score /= num_samples
        return dict(
            mouse_names=mouse_names,
            test_scores=test_scores,
            test_scaled_mses=test_scaled_mses,
            test_scores_direct=test_scores_direct,
            test_scaled_mses_direct=test_scaled_mses_direct,
            test_scores_pfpred=test_scores_pfpred,
            test_scores_pfpred_withgain=test_scores_pfpred_withgain,
            pos_score_encoder=pos_score_encoder,
            pos_score_latent=pos_score_latent,
            opt_pos_est_source_score=opt_pos_est_source_score,
            opt_pos_est_target_score=opt_pos_est_target_score,
            opt_pos_est_source_withgain_score=opt_pos_est_source_withgain_score,
            opt_pos_est_target_withgain_score=opt_pos_est_target_withgain_score,
            opt_pos_est_position_score=opt_pos_est_position_score,
            test_scores_doublecv=test_scores_doublecv,
            rbfpos_to_target_score=rbfpos_to_target_score,
        )
    raise ValueError(f"results must be 'all' or 'test_by_mouse', got {results}")

