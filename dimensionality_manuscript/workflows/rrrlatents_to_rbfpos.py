from typing import Optional, Union
from pathlib import Path
import joblib
import torch
from tqdm import tqdm

from vrAnalysis.database import get_database
from vrAnalysis.sessions import B2Session
from dimilibi import RidgeRegression
from dimensionality_manuscript.registry import PopulationRegistry, get_model
import optuna
import gc

# get session database
sessiondb = get_database("vrSessions")

# get population registry and models
registry = PopulationRegistry()
rbfpos_decoder_only_model = get_model("rbfpos_decoder_only", registry)
rbfpos_model = get_model("rbfpos", registry)
rbfpos_leak_model = get_model("rbfpos_leak", registry)
rrr_model = get_model("rrr", registry)


def zscore(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Z-score the data along the specified dimension."""
    return (x - x.mean(dim=dim, keepdims=True)) / x.std(dim=dim, keepdims=True)


def gather_latents(
    session: B2Session,
    rrr_variance: Optional[Union[float, str]] = None,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """Gather the latents from the RBFPos leak model and the RRR model.

    If rrr_variance is provided, we will use it as a cutoff for removing RRR latents
    that are not needed to explain that fraction of total variance. If it is set to
    the string "match", then it will use the number of latents from the RBFPos leak model.

    All latents are z-scored unless normalize is False.
    """
    if not normalize:
        normalize_func = lambda x, dim=None: x
    else:
        normalize_func = zscore

    # Get the best hyperparameters
    hyperparameters_leak = rbfpos_leak_model.get_best_hyperparameters(session, spks_type="oasis", method="best")[0]
    hyperparameters_rrr = rrr_model.get_best_hyperparameters(session, spks_type="oasis", method="best")[0]

    # Train the models and get test latents
    report_leak = rbfpos_leak_model.process(session, spks_type="oasis", hyperparameters=hyperparameters_leak)
    report_rrr = rrr_model.process(session, spks_type="oasis", hyperparameters=hyperparameters_rrr)

    # Get the train & validation latents
    train_extras_leak = rbfpos_leak_model.predict(session, report_leak.trained_model, split="train", hyperparameters=hyperparameters_leak)[1]
    train_extras_rrr = rrr_model.predict(session, report_rrr.trained_model, split="train", hyperparameters=hyperparameters_rrr)[1]
    val_extras_leak = rbfpos_leak_model.predict(session, report_leak.trained_model, split="validation", hyperparameters=hyperparameters_leak)[1]
    val_extras_rrr = rrr_model.predict(session, report_rrr.trained_model, split="validation", hyperparameters=hyperparameters_rrr)[1]

    # Z-Score the latents by feature (rrr latents are transposed compared to rbfpos latents)
    train_rbfpos_true = normalize_func(torch.tensor(train_extras_leak["position_basis"]), dim=0)
    train_rbfpos_pred = normalize_func(torch.tensor(train_extras_leak["position_basis_predicted"]), dim=0)
    val_rbfpos_true = normalize_func(torch.tensor(val_extras_leak["position_basis"]), dim=0)
    val_rbfpos_pred = normalize_func(torch.tensor(val_extras_leak["position_basis_predicted"]), dim=0)
    test_rbfpos_true = normalize_func(torch.tensor(report_leak.extras["position_basis"]), dim=0)
    test_rbfpos_pred = normalize_func(torch.tensor(report_leak.extras["position_basis_predicted"]), dim=0)

    # Handle RRR latents uniquely in case we need to remove some components
    if rrr_variance is not None:
        if rrr_variance == "match":
            # Use the number of latents from the RBFPos leak model
            # unless that is bigger than the latents in the RRR model
            idx_last_rrr_latent = min(
                train_rbfpos_true.shape[1],
                train_extras_rrr["latents"].shape[0],
            )
        elif isinstance(rrr_variance, float):
            rrr_latents = torch.tensor(train_extras_rrr["latents"])
            rrr_latents_var = rrr_latents.var(dim=1)
            rrr_latents_var_cumsum = rrr_latents_var.cumsum(dim=0)
            rrr_latents_var_cumsum = rrr_latents_var_cumsum / rrr_latents_var_cumsum[-1]
            idx_last_rrr_latent = torch.where(rrr_latents_var_cumsum <= rrr_variance)[0][-1]
        else:
            raise ValueError(f"Invalid value for rrr_variance: {rrr_variance}")
    else:
        idx_last_rrr_latent = train_extras_rrr["latents"].shape[0]

    # Z-Score the RRR latents (keeping only required amount)
    train_rrr_latents = normalize_func(torch.tensor(train_extras_rrr["latents"][:idx_last_rrr_latent]), dim=1)
    val_rrr_latents = normalize_func(torch.tensor(val_extras_rrr["latents"][:idx_last_rrr_latent]), dim=1)
    test_rrr_latents = normalize_func(torch.tensor(report_rrr.extras["latents"][:idx_last_rrr_latent]), dim=1)

    data = {
        "train_rbfpos_true": train_rbfpos_true,
        "train_rbfpos_pred": train_rbfpos_pred,
        "train_rrr": train_rrr_latents.T,
        "val_rbfpos_true": val_rbfpos_true,
        "val_rbfpos_pred": val_rbfpos_pred,
        "val_rrr": val_rrr_latents.T,
        "test_rbfpos_true": test_rbfpos_true,
        "test_rbfpos_pred": test_rbfpos_pred,
        "test_rrr": test_rrr_latents.T,
    }
    return data


def optimize_alpha(Xtrain: torch.Tensor, Xval: torch.Tensor, Ytrain: torch.Tensor, Yval: torch.Tensor) -> tuple[float, float]:
    """Optimize the alpha for the Ridge Regression model."""

    def _objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-3, 1e4, log=True)
        model = RidgeRegression(alpha=alpha, fit_intercept=True).fit(Xtrain, Ytrain)
        score = model.score(Xval, Yval, dim=None)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=100)
    best_alpha = study.best_params["alpha"]
    best_score = study.best_value
    return best_alpha, best_score


def process_session(
    session: B2Session,
    rrr_variance: Optional[float] = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Process a session and return the results.

    We determine what fraction of variance in the latents can be explained by latents of a different model.
    Specifically, we use RidgeRegression to fit the latents of the reduced rank regression model to the
    latents of the RBFPos leak model, and vice versa. We do this for RBFPos true basis and predicted basis.

    RidgeRegression models are optimized with Optuna to find the best alpha using train/val/test splits.
    """
    data = gather_latents(session, rrr_variance, normalize)
    alpha_rrr_to_pred, score_rrr_to_pred = optimize_alpha(
        data["train_rrr"],
        data["val_rrr"],
        data["train_rbfpos_pred"],
        data["val_rbfpos_pred"],
    )
    alpha_pred_to_rrr, score_pred_to_rrr = optimize_alpha(
        data["train_rbfpos_pred"],
        data["val_rbfpos_pred"],
        data["train_rrr"],
        data["val_rrr"],
    )
    alpha_rrr_to_true, score_rrr_to_true = optimize_alpha(
        data["train_rrr"],
        data["val_rrr"],
        data["train_rbfpos_true"],
        data["val_rbfpos_true"],
    )
    alpha_true_to_rrr, score_true_to_rrr = optimize_alpha(
        data["train_rbfpos_true"],
        data["val_rbfpos_true"],
        data["train_rrr"],
        data["val_rrr"],
    )

    # Fit final models with optimized alphas
    rrr_to_pred = RidgeRegression(alpha=alpha_rrr_to_pred, fit_intercept=True).fit(data["train_rrr"], data["train_rbfpos_pred"])
    pred_to_rrr = RidgeRegression(alpha=alpha_pred_to_rrr, fit_intercept=True).fit(data["train_rbfpos_pred"], data["train_rrr"])
    rrr_to_true = RidgeRegression(alpha=alpha_rrr_to_true, fit_intercept=True).fit(data["train_rrr"], data["train_rbfpos_true"])
    true_to_rrr = RidgeRegression(alpha=alpha_true_to_rrr, fit_intercept=True).fit(data["train_rbfpos_true"], data["train_rrr"])

    # Evaluate on test set
    test_score_rrr_to_pred = rrr_to_pred.score(data["test_rrr"], data["test_rbfpos_pred"], dim=None)
    test_score_pred_to_rrr = pred_to_rrr.score(data["test_rbfpos_pred"], data["test_rrr"], dim=None)
    test_score_rrr_to_true = rrr_to_true.score(data["test_rrr"], data["test_rbfpos_true"], dim=None)
    test_score_true_to_rrr = true_to_rrr.score(data["test_rbfpos_true"], data["test_rrr"], dim=None)

    results = {
        "alpha_rrr_to_pred": alpha_rrr_to_pred,
        "score_rrr_to_pred": score_rrr_to_pred,
        "alpha_rrr_to_true": alpha_rrr_to_true,
        "score_rrr_to_true": score_rrr_to_true,
        "alpha_pred_to_rrr": alpha_pred_to_rrr,
        "score_pred_to_rrr": score_pred_to_rrr,
        "alpha_true_to_rrr": alpha_true_to_rrr,
        "score_true_to_rrr": score_true_to_rrr,
        "test_score_rrr_to_pred": test_score_rrr_to_pred,
        "test_score_pred_to_rrr": test_score_pred_to_rrr,
        "test_score_rrr_to_true": test_score_rrr_to_true,
        "test_score_true_to_rrr": test_score_true_to_rrr,
    }
    return results


def get_filepath(session: B2Session, rrr_variance: Optional[float] = None, normalize: bool = True) -> Path:
    """Get the filepath for the results of a session."""
    name = session.session_print(joinby=".")
    if rrr_variance is not None:
        name += f"_rrr_variance_{rrr_variance}"
    if not normalize:
        name += "_no_normalize"
    return registry.registry_paths.rrrlatents_to_rbfpos_path / f"{name}.pkl"


process_sessions = True
force_remake = False
clear_cache = False
validate_results = False

rrr_variance_required = [
    0.95,
    "match",
]

normalize_required = [
    False,
    # True,
]

if __name__ == "__main__":
    for rrr_variance in rrr_variance_required:
        for normalize in normalize_required:
            for session in tqdm(sessiondb.iter_sessions(imaging=True), desc=f"Fitting Latent to Latent Models for RRR Variance {rrr_variance}"):
                _clear_session_cache = False
                results_path = get_filepath(session, rrr_variance=rrr_variance, normalize=normalize)

                # Clear cache if requested
                if clear_cache:
                    if results_path.exists():
                        results_path.unlink()

                # Validate results if requested
                if validate_results:
                    if results_path.exists():
                        results = joblib.load(results_path)
                        if results is not None:
                            required_keys = [
                                "alpha_rrr_to_pred",
                                "score_rrr_to_pred",
                                "alpha_rrr_to_true",
                                "score_rrr_to_true",
                                "alpha_pred_to_rrr",
                                "score_pred_to_rrr",
                                "alpha_true_to_rrr",
                                "score_true_to_rrr",
                                "test_score_rrr_to_pred",
                                "test_score_rrr_to_true",
                                "test_score_pred_to_rrr",
                                "test_score_true_to_rrr",
                            ]
                            results_valid = all(key in results for key in required_keys)
                            if not results_valid:
                                results_path.unlink()

                # Process session if requested (or not exists)
                if process_sessions:
                    if force_remake or not results_path.exists():
                        try:
                            _clear_session_cache = True
                            results = process_session(session, rrr_variance=rrr_variance, normalize=normalize)
                            joblib.dump(results, results_path)
                        except Exception as e:
                            print(f"Error processing session {session.session_print()}: {e}")
                            continue
                        finally:
                            if _clear_session_cache:
                                session.clear_cache()
                                torch.cuda.empty_cache()
                                gc.collect()
