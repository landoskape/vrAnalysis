import optuna
from .regression import ReducedRankRegression


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for hyperparameter optimization of Reduced Rank Ridge Regression.

    Parameters
    ----------
    trial : optuna.trial.Trial
        A trial object from Optuna.
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
        The target values (class labels in classification, real numbers in regression).
    X_val : array-like of shape (n_samples, n_features)
        The validation input samples.
    y_val : array-like of shape (n_samples,) or (n_samples, n_targets)
        The validation target values.

    Returns
    -------
    float
        The mean squared error on the validation set.
    """
    # Define the hyperparameter to optimize
    alpha = trial.suggest_float("alpha", 1e8, 1e10)

    # Initialize and fit the model
    model = ReducedRankRegression(alpha=alpha, fit_intercept=True)
    model.fit(X_train.T, y_train.T)

    # Make predictions on validation set
    return model.score(X_val.T, y_val.T)


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, show_progress_bar=False):
    """
    Optimize hyperparameters for Reduced Rank Ridge Regression using Optuna.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
        The target values (class labels in classification, real numbers in regression).
    X_val : array-like of shape (n_samples, n_features)
        The validation input samples.
    y_val : array-like of shape (n_samples,) or (n_samples, n_targets)
        The validation target values.
    n_trials : int, optional
        The number of trials to run for optimization. Default is 100.
    show_progress_bar : bool, optional
        If True, show a progress bar during optimization. Default is False.

    Returns
    -------
    float
        The best alpha value found.
    float
        The best mean squared error achieved on the validation set.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=show_progress_bar)

    return study, study.best_params["alpha"], study.best_value
