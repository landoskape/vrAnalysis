import torch
import numpy as np
from tqdm import tqdm

# ============================================================
# Model
# ============================================================

PARAM_NAMES = ["x_pref", "baseline", "amplitude", "width_left", "width_right", "exponent"]
N_PARAMS = len(PARAM_NAMES)

GAUSSIAN_PARAM_NAMES = ["x_pref", "baseline", "amplitude", "width"]
N_GAUSSIAN_PARAMS = len(GAUSSIAN_PARAM_NAMES)


def neuron_model_1d(x, params):
    """
    x: [n_bins]
    params: [n_cells, 6]
    """
    x = x.unsqueeze(0)  # [1, n_bins]
    x_pref, baseline, A, wL, wR, p = [params[:, i].unsqueeze(-1) for i in range(6)]

    dx = x - x_pref
    # wL_safe = torch.clamp(wL, min=1e-8)
    # wR_safe = torch.clamp(wR, min=1e-8)
    w = torch.where(dx < 0, wL, wR)
    z = torch.abs(dx) / w
    z_safe = torch.clamp(z, min=1e-8)

    peak = A * torch.exp(-0.5 * z_safe**p)
    return baseline + peak  # [n_cells, n_bins]


def gaussian_model_1d(x, params):
    """
    Symmetric Gaussian place field model.

    Parameters
    ----------
    x : torch.Tensor
        [n_bins] spatial positions.
    params : torch.Tensor
        [n_cells, 4] with columns [x_pref, baseline, amplitude, width].

    Returns
    -------
    torch.Tensor
        [n_cells, n_bins] predicted firing rates.
    """
    x = x.unsqueeze(0)  # [1, n_bins]
    x_pref, baseline, A, w = [params[:, i].unsqueeze(-1) for i in range(4)]

    w_safe = torch.clamp(w, min=1e-8)
    z_sq = ((x - x_pref) / w_safe) ** 2
    peak = A * torch.exp(-0.5 * z_sq)
    return baseline + peak  # [n_cells, n_bins]


# ============================================================
# Torch parameter estimator (simple + robust)
# ============================================================


def parameter_estimator_torch(x, y):
    """
    x: [n_bins]
    y: [n_bins]
    returns [6]
    """
    baseline = torch.clamp(torch.min(y), min=0.0)
    peak_idx = torch.argmax(y)
    x_pref = x[peak_idx]
    A = torch.clamp(y[peak_idx] - baseline, min=0.0)

    # crude width estimate: std weighted by activity
    weights = torch.clamp(y - baseline, min=0.0)
    if torch.sum(weights) > 0:
        mean = torch.sum(x * weights) / torch.sum(weights)
        var = torch.sum(weights * (x - mean) ** 2) / torch.sum(weights)
        width = torch.sqrt(var + 1e-6)
    else:
        width = torch.tensor(1.0, device=x.device)

    wL = width
    wR = width
    p = torch.tensor(2.0, device=x.device)

    return torch.stack([x_pref, baseline, A, wL, wR, p])


def gaussian_parameter_estimator(x, y):
    """
    Crude initial parameter estimates for the Gaussian model.

    Parameters
    ----------
    x : torch.Tensor
        [n_bins] spatial positions.
    y : torch.Tensor
        [n_bins] firing rate or activity.

    Returns
    -------
    torch.Tensor
        [4] initial params [x_pref, baseline, amplitude, width].
    """
    baseline = torch.clamp(torch.min(y), min=0.0)
    peak_idx = torch.argmax(y)
    x_pref = x[peak_idx]
    A = torch.clamp(y[peak_idx] - baseline, min=0.0)

    weights = torch.clamp(y - baseline, min=0.0)
    if torch.sum(weights) > 0:
        mean = torch.sum(x * weights) / torch.sum(weights)
        var = torch.sum(weights * (x - mean) ** 2) / torch.sum(weights)
        width = torch.sqrt(var + 1e-6)
    else:
        width = torch.tensor(1.0, device=x.device)

    return torch.stack([x_pref, baseline, A, width])


# ============================================================
# Training
# ============================================================


def train_model(x, R, steps=3000, lr=1e-3):
    """
    Fit both nonlinear and Gaussian place field models to the same data.

    Parameters
    ----------
    x : array-like
        [n_bins] spatial positions.
    R : array-like
        [n_cells, n_bins] firing rates.
    steps : int
        Number of optimization steps per model.
    lr : float
        Learning rate.

    Returns
    -------
    params_nl : torch.Tensor
        [n_cells, 6] fitted nonlinear params.
    params_gauss : torch.Tensor
        [n_cells, 4] fitted Gaussian params.
    loss_nl : float
        Final nonlinear MSE.
    loss_gauss : float
        Final Gaussian MSE.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(x, dtype=torch.float32, device=device)
    R = torch.tensor(R, dtype=torch.float32, device=device)

    n_cells = R.shape[0]

    # Nonlinear model
    p0_nl = torch.stack([parameter_estimator_torch(x, R[i]) for i in range(n_cells)])
    params_nl = torch.nn.Parameter(p0_nl)
    opt_nl = torch.optim.Adam([params_nl], lr=lr)

    # Gaussian model
    p0_gauss = torch.stack([gaussian_parameter_estimator(x, R[i]) for i in range(n_cells)])
    params_gauss = torch.nn.Parameter(p0_gauss)
    opt_gauss = torch.optim.Adam([params_gauss], lr=lr)

    for _ in tqdm(range(steps)):
        # Nonlinear step
        opt_nl.zero_grad()
        pred_nl = neuron_model_1d(x, params_nl)
        loss_nl = torch.mean((pred_nl - R) ** 2)
        loss_nl.backward()
        opt_nl.step()

        if torch.any(torch.isnan(params_nl)):
            print("NaN params found (nonlinear)")
            break

        # Gaussian step
        opt_gauss.zero_grad()
        pred_gauss = gaussian_model_1d(x, params_gauss)
        loss_gauss = torch.mean((pred_gauss - R) ** 2)
        loss_gauss.backward()
        opt_gauss.step()

        if torch.any(torch.isnan(params_gauss)):
            print("NaN params found (Gaussian)")
            break

    return params_nl.detach(), params_gauss.detach(), loss_nl.item(), loss_gauss.item()


# ============================================================
# ==================  EXAMPLE CALL  ==========================
# ============================================================

if __name__ == "__main__":

    # fake data
    n_cells = 5
    n_bins = 100
    track_length = 200.0

    x = np.linspace(0, track_length, n_bins)

    true_centers = np.random.uniform(20, 180, n_cells)
    R = []
    for c in true_centers:
        rate = 5 + 20 * np.exp(-((x - c) ** 2) / (2 * 15**2))
        R.append(rate + np.random.randn(n_bins))
    R = np.stack(R)

    # train both models
    params_nl, params_gauss, loss_nl, loss_gauss = train_model(x, R)

    print("Nonlinear loss:", loss_nl)
    print("Gaussian loss:", loss_gauss)
    print("Nonlinear params (first cell):", dict(zip(PARAM_NAMES, params_nl[0].cpu().numpy())))
    print("Gaussian params (first cell):", dict(zip(GAUSSIAN_PARAM_NAMES, params_gauss[0].cpu().numpy())))

    # quick evaluation
    with torch.no_grad():
        device = params_nl.device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        pred_nl = neuron_model_1d(x_t, params_nl)
        pred_gauss = gaussian_model_1d(x_t, params_gauss)

    print("Prediction shapes:", pred_nl.shape, pred_gauss.shape)
