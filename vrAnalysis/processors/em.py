from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm

from vrAnalysis.helpers import edge2center, cross_validate_trials
from vrAnalysis.processors.placefields import (
    get_frame_behavior,
    get_placefield,
    FrameBehavior,
    Placefield,
    get_placefield,
    get_placefield_prediction,
)
from vrAnalysis.processors import SpkmapProcessor
from vrAnalysis.sessions import B2Session
from dimilibi import measure_r2, measure_rms, mse


def _huber_irls_weights_from_pred(
    spks: np.ndarray,
    pred: np.ndarray,
    k: float = 2.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-frame Huber IRLS weights from prediction residuals."""
    r = np.sqrt(np.sum((spks - pred) ** 2, axis=1))
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma = 1.4826 * mad + eps

    u = r / (k * sigma)
    weights = np.ones_like(u)
    mask = u > 1.0
    weights[mask] = 1.0 / (u[mask] + eps)
    return np.clip(weights, 0.0, 1.0)


def _mse_scores(spks_t: torch.Tensor, pf_t: torch.Tensor) -> torch.Tensor:
    """MSE scores via expanded ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>. Lower = better."""
    spks_sq = (spks_t**2).sum(dim=1)[:, None, None]
    pf_sq = (pf_t**2).sum(dim=0)[None, ...]
    cross_term = torch.einsum("ij,jkl->ikl", spks_t, pf_t)
    return spks_sq + pf_sq - 2 * cross_term


def _angle_scores(spks_t: torch.Tensor, pf_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative cosine similarity scores. Lower = better (= smaller angle)."""
    cross = torch.einsum("ij,jkl->ikl", spks_t, pf_t)
    spks_norm = torch.sqrt((spks_t**2).sum(dim=1))[:, None, None]
    pf_norm = torch.sqrt((pf_t**2).sum(dim=0))[None, ...]
    return -(cross / (spks_norm * pf_norm + eps))


_OBJECTIVE_SCORES: dict[str, callable] = {
    "mse": _mse_scores,
    "angle": _angle_scores,
}


def _estep(spks: np.ndarray, placefield: Placefield, frame_behavior: FrameBehavior, objective: str = "mse") -> FrameBehavior:
    """E-Step: Use placefields to predict latent variables (mouse's internal estimate)

    Parameters
    ----------
    spks : np.ndarray
        The spike counts for the given neuron. (Frames x ROIs)
    placefield : Placefield
        The placefield object used for predicting latent variables.
    frame_behavior : FrameBehavior
        The frame behavior object with the true latent variables.
    objective : str
        Objective for finding the best position. One of "mse" or "angle".

    Returns
    -------
    frame_behavior : FrameBehavior
        The frame behavior object with the predicted latent variables.
    """
    if placefield.trials is not None:
        raise ValueError("Provided placefield object is not averaged over trials. Use average=True when getting placefield.")
    if objective not in _OBJECTIVE_SCORES:
        raise ValueError(f"Unknown objective '{objective}'. Must be one of {list(_OBJECTIVE_SCORES)}")

    num_frames = len(spks)
    num_positions = placefield.shape[1]
    dist_centers = edge2center(placefield.dist_edges)
    score_fn = _OBJECTIVE_SCORES[objective]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spks_t = torch.from_numpy(spks).to(dev)
    pf_t = torch.from_numpy(placefield.placefield).to(dev).permute(2, 0, 1)
    scores = score_fn(spks_t, pf_t)
    min_idx = torch.argmin(scores.reshape(num_frames, -1), dim=1)
    env_idx = (min_idx // num_positions).cpu().numpy()
    pos_idx = (min_idx % num_positions).cpu().numpy()
    del spks_t, pf_t, scores, min_idx

    frame_behavior = FrameBehavior(
        position=dist_centers[pos_idx],
        speed=frame_behavior.speed,
        environment=placefield.environment[env_idx],
        trial=frame_behavior.trial,
        reward_delivery=frame_behavior.reward_delivery,
        reward_omitted=frame_behavior.reward_omitted,
    )
    return frame_behavior


def _compute_ce_score(
    spks: np.ndarray,
    placefield: Placefield,
    true_bin_idx: np.ndarray,
    objective: str = "mse",
) -> tuple[np.ndarray, float]:
    """Compute cross-entropy between predicted position distribution and true position bins.

    Scores are converted to log-probabilities via log_softmax(-scores) over all (env, position)
    bins. CE = -mean(log_prob[true_bin]) over frames.

    Parameters
    ----------
    spks : np.ndarray
        Spike counts. (Frames x ROIs)
    placefield : Placefield
        Averaged placefield used to score positions.
    true_bin_idx : np.ndarray
        Flat bin index per frame: env_idx * num_positions + pos_idx. Shape (Frames,).
    objective : str
        One of "mse" or "angle".

    Returns
    -------
    pred_bin_idx : np.ndarray
        Predicted flat bin index (argmin of scores) per frame. Shape (Frames,).
    ce_score : float
        Mean cross-entropy over frames.
    """
    if placefield.trials is not None:
        raise ValueError("Placefield must be averaged over trials.")
    if objective not in _OBJECTIVE_SCORES:
        raise ValueError(f"Unknown objective '{objective}'. Must be one of {list(_OBJECTIVE_SCORES)}")

    num_frames = len(spks)
    num_positions = placefield.shape[1]
    score_fn = _OBJECTIVE_SCORES[objective]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spks_t = torch.from_numpy(spks).to(dev)
    pf_t = torch.from_numpy(placefield.placefield).to(dev).permute(2, 0, 1)
    scores_flat = score_fn(spks_t, pf_t).reshape(num_frames, -1)
    log_probs = torch.nn.functional.log_softmax(-scores_flat.float(), dim=1)
    min_idx = torch.argmin(scores_flat, dim=1)
    tb = torch.from_numpy(true_bin_idx).long().to(dev)
    ce = -log_probs[torch.arange(num_frames, device=dev), tb].mean().item()
    env_idx = (min_idx // num_positions).cpu().numpy()
    pos_idx = (min_idx % num_positions).cpu().numpy()

    # Clean up torch variables to free GPU memory (safe move)
    del spks_t, pf_t, scores_flat, log_probs, min_idx, tb

    pred_bin_idx = env_idx * num_positions + pos_idx
    return pred_bin_idx, ce


def _mstep(
    spks: np.ndarray,
    frame_behavior: FrameBehavior,
    dist_edges: np.ndarray,
    smooth_width: float,
    weights: np.ndarray | None = None,
) -> Placefield:
    """M-Step: Update placefields to minimize the error between the activity and the predicted activity.

    Parameters
    ----------
    spks : np.ndarray
        The spike counts for the given neuron. (Frames x ROIs)
    frame_behavior : FrameBehavior
        The frame behavior object with the predicted latent variables.
    dist_edges : np.ndarray
        The edges of the distance bins. (N_bins + 1)
    smooth_width : float
        The width of the Gaussian kernel to use for smoothing the place field.
    weights : np.ndarray | None
        Optional per-frame IRLS weights aligned to frame_behavior.

    Returns
    -------
    placefield : Placefield
        The updated placefield object.
    """
    placefield = get_placefield(
        spks,
        frame_behavior,
        dist_edges=dist_edges,
        average=True,
        use_fast_sampling=False,
        smooth_width=smooth_width,
        weights=weights,
    )
    return placefield


# EM Procedure for estimating mouse internal representation of environment and internal placefields
# Parameters
@dataclass
class ExpMaxConfig:
    spks_type: str = "oasis"
    norm_method: str = "zero-one"
    speed_threshold: float = 1.0
    num_bins: int = 100
    train_test_split: tuple[float, float] = (0.8, 0.2)
    smooth_width: float | None = 0.25
    num_steps: int = 10
    reliability_cutoff: float = 0.1
    use_huber_irls: bool = True
    huber_k: float = 2.5
    huber_eps: float = 1e-8


def process_session(
    session: B2Session,
    config: ExpMaxConfig = ExpMaxConfig(),
) -> tuple[dict[str, np.ndarray], dict[str, list[FrameBehavior | Placefield]]]:
    """
    Process a session using the EM procedure to estimate the mouse's internal representation of the environment and internal placefields.


    Parameters
    ----------
    session : B2Session
    config : ExpMaxConfig
        The configuration for the EM procedure.

    Returns
    -------
    results : dict[str, np.ndarray]
        A dictionary containing the results of the EM procedure, including performance metrics.
    models : dict[str, list[FrameBehavior | Placefield]]
        A dictionary containing the estimated models (frame behaviors and placefields) at each step of the EM procedure.
    """
    # Select reliable cells
    spkmap = SpkmapProcessor(session)
    reliability = spkmap.get_reliability(use_session_filters=False)
    idx_reliable = np.any(reliability.values > config.reliability_cutoff, axis=0)
    idx_keep_rois = session.idx_rois & idx_reliable

    frame_behavior = get_frame_behavior(session)
    spks = session.spks[:, idx_keep_rois]

    # Data for training
    if config.norm_method == "zero-one":
        norm_value = np.max(spks, axis=0)
        spks = spks / norm_value

    # Filter samples for good data (fast, in VR, in best environment)
    idx_valid = frame_behavior.valid_frames()
    idx_fast = frame_behavior.speed >= config.speed_threshold
    idx_filter = idx_valid & idx_fast
    frame_behavior = frame_behavior.filter(idx_filter)
    spks = spks[idx_filter]

    # Do train/test split
    trial_folds = cross_validate_trials(session.trial_environment, config.train_test_split)
    idx_train = np.isin(frame_behavior.trial, trial_folds[0])
    idx_test = np.isin(frame_behavior.trial, trial_folds[1])

    spks_tr = spks[idx_train]
    spks_te = spks[idx_test]
    frame_behavior_tr = frame_behavior.filter(idx_train)
    frame_behavior_te = frame_behavior.filter(idx_test)

    # Make training and testing placefields
    dist_edges = np.linspace(0, session.env_length[0], config.num_bins + 1)
    placefield_kwargs = dict(
        dist_edges=dist_edges,
        speed_threshold=config.speed_threshold,
        smooth_width=config.smooth_width,
        use_fast_sampling=True,
        session=session,
    )
    placefield_tr = get_placefield(spks_tr, frame_behavior_tr, average=True, **placefield_kwargs)
    placefield_te = get_placefield(spks_te, frame_behavior_te, average=False, **placefield_kwargs)

    num_steps = config.num_steps
    fb_e = [frame_behavior_tr]
    pf_m = [placefield_tr]
    for _ in tqdm(range(num_steps - 1)):
        fb_next = _estep(spks_tr, pf_m[-1], fb_e[-1])
        weights = None
        if config.use_huber_irls:
            pred_curr = get_placefield_prediction(pf_m[-1], fb_next)[0]
            weights = _huber_irls_weights_from_pred(spks_tr, pred_curr, k=config.huber_k, eps=config.huber_eps)
        pf_next = _mstep(spks_tr, fb_next, dist_edges, config.smooth_width, weights=weights)
        fb_e.append(fb_next)
        pf_m.append(pf_next)

    # Measure improvement in performance of EM model over iterations
    step_mse = []
    step_r2 = []
    step_rms = []
    for step in tqdm(range(num_steps)):
        _step_pred = get_placefield_prediction(pf_m[step], fb_e[step])[0]
        step_mse.append(mse(_step_pred, spks_tr, dim=0, reduce="mean"))
        step_r2.append(measure_r2(_step_pred, spks_tr, dim=0, reduce="mean"))
        step_rms.append(measure_rms(_step_pred, spks_tr, dim=0, reduce="mean"))

    best_step = np.argmin(step_mse)
    print(f"Best step: {best_step}")

    # Compare EM model to null model (empirical placefield) on testing timepoints
    _test_pred = get_placefield_prediction(pf_m[best_step], frame_behavior_te)[0]
    _null_pred = get_placefield_prediction(placefield_te, frame_behavior_te)[0]
    em_test_r2 = measure_r2(_test_pred, spks_te, dim=0, reduce="mean")
    em_null_r2 = measure_r2(_null_pred, spks_te, dim=0, reduce="mean")
    em_test_rms = measure_rms(_test_pred, spks_te, dim=0, reduce="mean")
    em_null_rms = measure_rms(_null_pred, spks_te, dim=0, reduce="mean")
    em_test_mse = mse(_test_pred, spks_te, dim=0, reduce="mean")
    em_null_mse = mse(_null_pred, spks_te, dim=0, reduce="mean")

    print(f"EM test R2: {100*em_test_r2:.2f}, EM test RMS: {1e3*em_test_rms:.2f}, EM test MSE: {1e3*em_test_mse:.2f}")
    print(f"EM null R2: {100*em_null_r2:.2f}, EM null RMS: {1e3*em_null_rms:.2f}, EM null MSE: {1e3*em_null_mse:.2f}")

    for step in range(num_steps):
        print(f"{step} MSE: {1e3*step_mse[step]:.2f}, R2: {100*step_r2[step]:.1f}, RMS: {1e3*step_rms[step]:.1f}")

    results = dict(
        em_test_r2=em_test_r2,
        em_null_r2=em_null_r2,
        em_test_rms=em_test_rms,
        em_null_rms=em_null_rms,
        em_test_mse=em_test_mse,
        em_null_mse=em_null_mse,
        step_mse=step_mse,
        step_r2=step_r2,
        step_rms=step_rms,
    )
    extras = dict(
        frame_behavior_est=fb_e,
        placefield_est=pf_m,
        idx_keep_rois=idx_keep_rois,
    )
    return results, extras
