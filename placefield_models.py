"""
Synthetic placefield models.

Two models:

  Thresholded Gaussian Process (TGP)
      Samples from a 1-D GP with an RBF (or periodic RBF) kernel, then
      rectifies at a threshold.  Produces smooth, sparse fields with a
      single dominant peak; multiple peaks are possible but suppressed by
      the kernel length-scale.

  Pink noise + smoothing (PNS)
      Shapes white noise to a 1/f^alpha power spectrum (periodic via FFT),
      optionally applies a symmetric Gaussian kernel, then rectifies.
      Produces fields with a more irregular, multi-scale texture than TGP.

Both high-level samplers return (n_neurons, n_bins) float64 arrays.

Dependencies: numpy, scipy
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from syd import Viewer

from dimilibi.cvpca import CVPCA


@dataclass
class PlacefieldConfig:
    """Fixed configuration for PlacefieldViewer.

    Pass this directly in notebooks; ``main()`` builds it from argparse args.
    Key parameters (thresholds, length scales, alpha, etc.) are exposed as
    interactive controls inside the viewer — these are just the starting values
    and any fixed options that don't need a slider.
    """

    # Population
    n_neurons: int = 80
    n_bins: int = 100
    normalize: str = "peak"
    seed: int = 0

    # TGP fixed
    tgp_amplitude: float = 1.0
    tgp_length_scale: float = 0.12  # viewer default
    tgp_threshold: float = 0.6  # viewer default
    tgp_periodic: bool = True

    # PNS fixed
    pns_smooth_width: float | None = 3.0  # None = no smoothing
    pns_alpha: float = 1.5  # viewer default
    pns_mean: float = 0.0  # viewer default
    pns_threshold: float = 0.5  # viewer default

    # CV split fixed
    max_rate: float = 10.0


# ── Shared utilities ──────────────────────────────────────────────────────────


def rectify(signals: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Hard-rectify: max(0, signal - threshold).

    Parameters
    ----------
    signals : (n_neurons, n_bins)
    threshold : float
        Subtract this before clipping to zero.

    Returns
    -------
    (n_neurons, n_bins), non-negative.
    """
    return np.maximum(0.0, signals - threshold)


def normalize_placefields(
    fields: np.ndarray,
    mode: str = "peak",
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Normalize each row of a placefield array.

    Parameters
    ----------
    fields : (n_neurons, n_bins)
    mode : {"peak", "sum", "none"}
        "peak"  — divide by row max so each field peaks at 1.
        "sum"   — divide by row sum so each field integrates to 1.
        "none"  — return as-is.
    epsilon : float
        Added to denominator to avoid divide-by-zero for silent neurons.

    Returns
    -------
    (n_neurons, n_bins)
    """
    if mode == "none":
        return fields
    if mode == "peak":
        denom = fields.max(axis=1, keepdims=True)
    elif mode == "sum":
        denom = fields.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"mode must be 'peak', 'sum', or 'none', got {mode!r}")
    return fields / (denom + epsilon)


def bin_positions(n_bins: int, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Return bin centres for a 1-D track of length [vmin, vmax]."""
    edges = np.linspace(vmin, vmax, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


# ── Thresholded Gaussian Process ──────────────────────────────────────────────


def rbf_kernel(
    positions: np.ndarray,
    length_scale: float,
    amplitude: float = 1.0,
    periodic: bool = True,
) -> np.ndarray:
    """RBF covariance matrix for 1-D positions.

    Parameters
    ----------
    positions : (n_bins,)
        Spatial bin centres.
    length_scale : float
        Kernel width in the same units as positions.  Larger values give
        smoother functions; a good default is ~10–20 % of the track length.
    amplitude : float
        Marginal standard deviation of the GP.  GP draws will have values
        roughly in [-amplitude, +amplitude] before thresholding.
    periodic : bool
        If True, use the periodic (arc-sine) RBF, which wraps the track
        into a ring.  Recommended for circular environments.

    Returns
    -------
    K : (n_bins, n_bins) positive-semi-definite covariance matrix.
    """
    pos = np.asarray(positions)
    diff = pos[:, None] - pos[None, :]  # (n_bins, n_bins)

    if periodic:
        period = pos[-1] - pos[0] + (pos[1] - pos[0])  # full track length
        # Periodic RBF: k(d) = a² exp(-2 sin²(π d / L) / ℓ²)
        K = amplitude**2 * np.exp(-2.0 * np.sin(np.pi * diff / period) ** 2 / length_scale**2)
    else:
        K = amplitude**2 * np.exp(-0.5 * diff**2 / length_scale**2)

    return K


def sample_gp(
    n_neurons: int,
    positions: np.ndarray,
    length_scale: float,
    amplitude: float = 1.0,
    jitter: float = 1e-6,
    periodic: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample n_neurons independent draws from a zero-mean 1-D GP.

    Uses the Cholesky decomposition of the covariance matrix, so all
    samples share the same kernel but are statistically independent.

    Parameters
    ----------
    n_neurons : int
    positions : (n_bins,)
    length_scale : float
    amplitude : float
    jitter : float
        Small value added to the diagonal for numerical stability.
    periodic : bool
    rng : numpy Generator, optional

    Returns
    -------
    (n_neurons, n_bins) array of GP draws.
    """
    if rng is None:
        rng = np.random.default_rng()

    K = rbf_kernel(positions, length_scale, amplitude, periodic)
    K += jitter * np.eye(len(positions))
    L = np.linalg.cholesky(K)  # (n_bins, n_bins)

    Z = rng.standard_normal((len(positions), n_neurons))
    return (L @ Z).T  # (n_neurons, n_bins)


def sample_tgp_placefields(
    n_neurons: int,
    n_bins: int = 100,
    amplitude: float = 1.0,
    length_scale: float = 0.1,
    threshold: float = 0.5,
    periodic: bool = True,
    normalize: str = "peak",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample thresholded GP placefields.

    Generates GP draws and rectifies at ``threshold``.  The threshold
    controls sparsity: at threshold ≈ 0 roughly half the bins are active;
    at threshold ≈ amplitude most neurons have a single narrow field.

    Parameters
    ----------
    n_neurons : int
    n_bins : int
    amplitude : float
        GP marginal standard deviation.  Sets the overall scale of the
        latent function before thresholding.
    length_scale : float
        Kernel width as a fraction of track length.  Converted internally
        to position units so 0.1 → one-tenth of the track.
        Typical range: 0.05–0.30.
    threshold : float
        Rectification threshold in the same units as amplitude.
        Values near amplitude give sparse, single-peaked fields.
        Values near 0 give broad, multi-modal fields.
    periodic : bool
    normalize : {"peak", "sum", "none"}
    rng : numpy Generator, optional

    Returns
    -------
    (n_neurons, n_bins) non-negative array.
    """
    if rng is None:
        rng = np.random.default_rng()

    positions = bin_positions(n_bins)
    # Convert fractional length_scale to position units
    track_length = positions[-1] - positions[0] + (positions[1] - positions[0])
    ls_abs = length_scale * track_length

    draws = sample_gp(n_neurons, positions, ls_abs, amplitude, periodic=periodic, rng=rng)
    fields = rectify(draws, threshold)
    return normalize_placefields(fields, mode=normalize)


# ── Pink noise + smoothing ────────────────────────────────────────────────────


def pink_noise_1d(
    n_neurons: int,
    n_bins: int,
    alpha: float = 1.0,
    mean: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 1/f^alpha noise with periodic (circular) boundary conditions.

    White noise is generated in the frequency domain, then each frequency
    component is scaled by f^(-alpha/2) so the power spectrum goes as
    1/f^alpha.  The AC components are variance-normalised to unit std before
    adding the mean, so ``mean`` is directly interpretable as a z-score offset
    relative to the noise fluctuations regardless of alpha.

    Parameters
    ----------
    n_neurons : int
    n_bins : int
    alpha : float
        Spectral exponent.
        0.0 → white noise
        1.0 → pink noise (equal power per octave)
        2.0 → red / Brownian noise (very smooth)
    mean : float
        DC offset added after variance normalisation.
        0.0  → zero-mean signal; roughly half the bins are above zero.
        >0.0 → shifted positive; at mean ≈ 1 most bins are above zero,
               giving dense, smoothly elevated "placefields" before any
               thresholding.  mean ≈ -1 gives very sparse output.
    rng : numpy Generator, optional

    Returns
    -------
    (n_neurons, n_bins) real-valued array with the requested mean.
    """
    if rng is None:
        rng = np.random.default_rng()

    freqs = np.fft.rfftfreq(n_bins)  # (n_bins // 2 + 1,)

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(freqs == 0, 0.0, freqs ** (-alpha / 2.0))

    noise_real = rng.standard_normal((n_neurons, len(freqs)))
    noise_imag = rng.standard_normal((n_neurons, len(freqs)))
    spectrum = (noise_real + 1j * noise_imag) * weights[None, :]
    spectrum[:, 0] = 0.0  # zero DC before irfft

    signals = np.fft.irfft(spectrum, n=n_bins)  # (n_neurons, n_bins)

    # Normalise AC variance to 1, then add requested mean
    std = signals.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return signals / std + mean


def sample_pns_placefields(
    n_neurons: int,
    n_bins: int = 100,
    alpha: float = 1.0,
    mean: float = 0.0,
    smooth_width: float | None = None,
    threshold: float = 0.5,
    normalize: str = "peak",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample pink-noise placefields with optional Gaussian smoothing.

    The AC fluctuations have unit variance before smoothing, so both
    ``mean`` and ``threshold`` are interpretable as z-score offsets.

    Parameters
    ----------
    n_neurons : int
    n_bins : int
    alpha : float
        Spectral exponent (see ``pink_noise_1d``).  Higher alpha gives
        smoother, more correlated signals.
    mean : float
        DC offset of the underlying signal (see ``pink_noise_1d``).
        0.0  → zero-mean; ~50 % of bins active at threshold=0.
        >0.0 → positively biased; more bins survive rectification,
               producing denser fields.  Useful when you want the noise
               to look like a background firing rate with modulation on top
               rather than sparse place fields.
    smooth_width : float or None
        Standard deviation of the Gaussian smoothing kernel in bins.
        Applied after spectral shaping.  None skips smoothing.
    threshold : float
        Rectification threshold.  With mean=0 and no smoothing this is a
        z-score; 0.5 is a reasonable default for sparse fields.
    normalize : {"peak", "sum", "none"}
    rng : numpy Generator, optional

    Returns
    -------
    (n_neurons, n_bins) non-negative array.
    """
    if rng is None:
        rng = np.random.default_rng()

    signals = pink_noise_1d(n_neurons, n_bins, alpha=alpha, mean=0.0, rng=rng)

    if smooth_width is not None and smooth_width > 0:
        signals = gaussian_filter1d(signals, sigma=smooth_width, axis=1, mode="wrap")
        # Re-standardize AC variance after smoothing, then add mean
        std = signals.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        signals = signals / std

    signals = signals + mean

    fields = rectify(signals, threshold)
    return normalize_placefields(fields, mode=normalize)


# ── Covariance / population structure helpers ─────────────────────────────────


def population_covariance(fields: np.ndarray) -> np.ndarray:
    """Neuron × neuron covariance matrix of a placefield array.

    Parameters
    ----------
    fields : (n_neurons, n_bins)

    Returns
    -------
    (n_neurons, n_neurons)
    """
    f = fields - fields.mean(axis=1, keepdims=True)
    return (f @ f.T) / fields.shape[1]


def spatial_autocorr(fields: np.ndarray) -> np.ndarray:
    """Mean spatial autocorrelation across neurons (circular shift).

    Parameters
    ----------
    fields : (n_neurons, n_bins)

    Returns
    -------
    autocorr : (n_bins,) mean autocorrelation at each lag.
    """
    n_bins = fields.shape[1]
    f = fields - fields.mean(axis=1, keepdims=True)
    F = np.fft.rfft(f, axis=1)
    power = np.abs(F) ** 2
    ac = np.fft.irfft(power, n=n_bins, axis=1)
    ac = ac / ac[:, [0]]  # normalize so lag-0 = 1
    return ac.mean(axis=0)


def fraction_active(fields: np.ndarray) -> np.ndarray:
    """Fraction of bins where each neuron is non-zero.

    Parameters
    ----------
    fields : (n_neurons, n_bins)

    Returns
    -------
    (n_neurons,)
    """
    return (fields > 0).mean(axis=1)


# ── Train / test split ───────────────────────────────────────────────────────


def poisson_train_test(
    fields: np.ndarray,
    max_rate: float = 10.0,
    n_visits: int | np.ndarray = 50,
    n_samples: int = 3,
    normalize: str = "peak",
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Generate N independent placefield estimates via Poisson spike simulation.

    Treats each field as a firing rate map (spikes per visit), draws
    independent Poisson counts for each sample, then divides by visit count
    to recover an estimated rate.  Noise variance at bin x scales as
    ``max_rate * field[x] / n_visits``, so more visits → less noise.

    Silent bins (field = 0) always produce zero counts, preserving sparsity
    structure without any rectification step.

    Parameters
    ----------
    fields : (n_neurons, n_bins)
        Peak- or sum-normalised placefield array.
    max_rate : float
        Peak firing rate in spikes per visit.  Scales all fields uniformly.
    n_visits : int or (n_bins,) array
        Number of position visits per bin per sample.
        A scalar gives uniform sampling; an array allows non-uniform coverage.
    n_samples : int
        Number of independent samples to generate.
    normalize : {"peak", "sum", "none"}
        Normalization applied to each estimated field after averaging.
    rng : numpy Generator, optional

    Returns
    -------
    list of n_samples arrays, each (n_neurons, n_bins)
    """
    if rng is None:
        rng = np.random.default_rng()

    rates = fields * max_rate  # (n_neurons, n_bins)
    n_vis = np.broadcast_to(n_visits, rates.shape[1])
    lam = rates * n_vis[None, :]
    safe = np.where(n_vis > 0, n_vis, 1)

    samples = []
    for _ in range(n_samples):
        counts = rng.poisson(lam).astype(float)
        est = counts / safe[None, :]
        samples.append(normalize_placefields(est, mode=normalize))
    return samples


def gp_noise_train_test(
    fields: np.ndarray,
    n_bins: int | None = None,
    noise_length_scale: float = 0.1,
    noise_amplitude: float = 0.15,
    periodic: bool = True,
    n_samples: int = 3,
    normalize: str = "peak",
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Generate N independent samples by adding independent GP noise realizations.

    Each sample is ``fields + epsilon`` where ``epsilon`` is an independent
    zero-mean GP draw with a smooth RBF kernel.  The spatial correlation of
    the noise matches the original model, unlike Poisson noise which is
    bin-independent.  Negative values after addition are clipped to zero.

    Parameters
    ----------
    fields : (n_neurons, n_bins)
    n_bins : int or None
        Inferred from ``fields`` if None.
    noise_length_scale : float
        RBF length scale of the noise GP, as a fraction of track length.
        Smaller → rougher, more spatially independent noise.
    noise_amplitude : float
        Std of the noise GP marginals.  In peak-normalised units 0.1–0.2 is
        moderate; 0.3+ is large.
    periodic : bool
    n_samples : int
        Number of independent samples to generate.
    normalize : {"peak", "sum", "none"}
    rng : numpy Generator, optional

    Returns
    -------
    list of n_samples arrays, each (n_neurons, n_bins)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_neurons, nb = fields.shape
    if n_bins is None:
        n_bins = nb

    positions = bin_positions(n_bins)
    track_length = positions[-1] - positions[0] + (positions[1] - positions[0])
    ls_abs = noise_length_scale * track_length

    samples = []
    for _ in range(n_samples):
        eps = sample_gp(n_neurons, positions, ls_abs, noise_amplitude, periodic=periodic, rng=rng)
        samples.append(normalize_placefields(rectify(fields + eps), mode=normalize))
    return samples


def pink_noise_train_test(
    fields: np.ndarray,
    noise_alpha: float = 1.0,
    noise_scale: float = 0.15,
    n_samples: int = 3,
    normalize: str = "peak",
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Generate N independent samples by adding independent pink noise.

    Each sample is ``fields + noise_scale * epsilon`` where ``epsilon`` is an
    independent zero-mean, unit-variance pink noise draw with the requested
    spectral exponent.  Matches the noise model of the PNS placefield
    generator itself.  Negative values are clipped to zero.

    Parameters
    ----------
    fields : (n_neurons, n_bins)
    noise_alpha : float
        Spectral exponent of the additive noise (need not match the field's
        alpha; lower alpha → rougher, more high-frequency noise).
    noise_scale : float
        Std of the additive noise in the same units as ``fields``.
    n_samples : int
        Number of independent samples to generate.
    normalize : {"peak", "sum", "none"}
    rng : numpy Generator, optional

    Returns
    -------
    list of n_samples arrays, each (n_neurons, n_bins)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_neurons, n_bins = fields.shape

    samples = []
    for _ in range(n_samples):
        eps = noise_scale * pink_noise_1d(n_neurons, n_bins, alpha=noise_alpha, rng=rng)
        samples.append(normalize_placefields(rectify(fields + eps), mode=normalize))
    return samples


def sort_by_peak(fields: np.ndarray) -> np.ndarray:
    """Return fields reordered so peak positions increase top-to-bottom.

    Silent neurons (all-zero rows) are placed at the end.

    Parameters
    ----------
    fields : (n_neurons, n_bins)

    Returns
    -------
    (n_neurons, n_bins) sorted copy.
    """
    # Silent neurons get peak index n_bins (past the end) so they sort last
    peak_pos = np.where(
        fields.max(axis=1) > 0,
        fields.argmax(axis=1),
        fields.shape[1],
    )
    return fields[np.argsort(peak_pos)]


# ── Syd viewer ────────────────────────────────────────────────────────────────


class PlacefieldViewer(Viewer):
    """Interactive viewer for synthetic placefields with train/test splits.

    Controls exposed in the UI select the PF model (TGP / PNS) and the
    CV-split method (Poisson / GP noise / Pink noise), giving 6 combos.
    Fixed parameters (n_neurons, n_bins, normalize, etc.) come from the
    argparse namespace stored in ``self.args``.

    The three panels show source → train → test, all sorted by the source
    peak-position order.
    """

    def __init__(self, cfg: PlacefieldConfig | None = None):
        self.cfg = cfg or PlacefieldConfig()
        cfg = self.cfg

        # ── model & split selectors ───────────────────────────────────────
        self.add_selection("pf_model", options=["TGP", "PNS"], value="PNS")
        self.add_selection("cv_split", options=["Poisson", "GP noise", "Pink noise"], value="Pink noise")
        self.add_unbounded_integer("seed", value=cfg.seed)

        # ── TGP controls ──────────────────────────────────────────────────
        self.add_float("tgp_threshold", min=0.0, max=2.0, value=cfg.tgp_threshold, step=0.05)
        self.add_float("tgp_length_scale", min=0.02, max=0.5, value=cfg.tgp_length_scale, step=0.01)

        # ── PNS controls ──────────────────────────────────────────────────
        self.add_float("pns_alpha", min=0.0, max=3.0, value=cfg.pns_alpha, step=0.1)
        self.add_float("pns_mean", min=-2.0, max=2.0, value=cfg.pns_mean, step=0.1)
        self.add_float("pns_threshold", min=-1.0, max=2.0, value=cfg.pns_threshold, step=0.05)

        # ── CV-split controls ─────────────────────────────────────────────
        self.add_integer("n_visits", min=5, max=500, value=50)
        self.add_float("noise_scale", min=0.01, max=1.0, value=0.5, step=0.01)

    def plot(self, state):
        import matplotlib.pyplot as plt

        cfg = self.cfg
        rng = np.random.default_rng(state["seed"])

        # ── generate source ───────────────────────────────────────────────
        if state["pf_model"] == "TGP":
            source = sample_tgp_placefields(
                n_neurons=cfg.n_neurons,
                n_bins=cfg.n_bins,
                amplitude=cfg.tgp_amplitude,
                length_scale=state["tgp_length_scale"],
                threshold=state["tgp_threshold"],
                periodic=cfg.tgp_periodic,
                normalize=cfg.normalize,
                rng=rng,
            )
        else:
            source = sample_pns_placefields(
                n_neurons=cfg.n_neurons,
                n_bins=cfg.n_bins,
                alpha=state["pns_alpha"],
                mean=state["pns_mean"],
                smooth_width=cfg.pns_smooth_width,
                threshold=state["pns_threshold"],
                normalize=cfg.normalize,
                rng=rng,
            )

        # ── sort index from source (silent neurons last) ──────────────────
        peak_pos = np.where(
            source.max(axis=1) > 0,
            source.argmax(axis=1),
            source.shape[1],
        )
        sort_idx = np.argsort(peak_pos)

        # ── generate train / test ─────────────────────────────────────────
        if state["cv_split"] == "Poisson":
            train, test1, test2 = poisson_train_test(
                source,
                max_rate=cfg.max_rate,
                n_visits=state["n_visits"],
                n_samples=3,
                normalize=cfg.normalize,
                rng=rng,
            )
        elif state["cv_split"] == "GP noise":
            train, test1, test2 = gp_noise_train_test(
                source,
                noise_length_scale=state["tgp_length_scale"],
                noise_amplitude=state["noise_scale"],
                periodic=cfg.tgp_periodic,
                n_samples=3,
                normalize=cfg.normalize,
                rng=rng,
            )
        else:
            train, test1, test2 = pink_noise_train_test(
                source,
                noise_alpha=state["pns_alpha"],
                noise_scale=state["noise_scale"],
                n_samples=3,
                normalize=cfg.normalize,
                rng=rng,
            )

        n_components = min(cfg.n_neurons, cfg.n_bins) - 1
        cvpca_neurons = CVPCA(num_components=n_components, center=True, on_stimuli=False)
        cvpca_neurons.fit(torch.tensor(train))
        score_neurons = cvpca_neurons.score(torch.tensor(test1), torch.tensor(test2))

        cvpca_positions = CVPCA(num_components=n_components, center=True, on_stimuli=True)
        cvpca_positions.fit(torch.tensor(train))
        score_positions = cvpca_positions.score(torch.tensor(test1), torch.tensor(test2))

        # ── plot ──────────────────────────────────────────────────────────
        xvals = np.arange(n_components) + 1
        fig, ax = plt.subplots(1, 3, figsize=(13, 5))
        ax[0].imshow(test1[sort_idx], aspect="auto", interpolation="nearest", origin="upper", cmap="viridis")
        ax[1].imshow(test2[sort_idx], aspect="auto", interpolation="nearest", origin="upper", cmap="viridis")
        ax[2].plot(xvals, score_neurons, label="on neurons", color="k")
        ax[2].plot(xvals, score_positions, label="on positions", color="b")
        ax[2].legend(loc="upper right")
        ax[2].set_xscale("log")
        ax[2].set_yscale("log")

        fig.tight_layout()
        return fig


# ── Main ──────────────────────────────────────────────────────────────────────


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Interactive synthetic placefield viewer (source / train / test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared
    p.add_argument("--n-neurons", type=int, default=80, help="Number of neurons.")
    p.add_argument("--n-bins", type=int, default=100, help="Number of spatial bins.")
    p.add_argument("--normalize", choices=["peak", "sum", "none"], default="peak")
    p.add_argument("--seed", type=int, default=0, help="Initial RNG seed (adjustable in viewer).")

    # TGP (amplitude and periodicity are fixed; threshold/length_scale live in the viewer)
    tgp = p.add_argument_group("TGP — fixed config (key params are in the viewer)")
    tgp.add_argument("--tgp-amplitude", type=float, default=1.0, metavar="A")
    tgp.add_argument("--tgp-length-scale", type=float, default=0.12, metavar="L", help="Viewer default for kernel width (fraction of track).")
    tgp.add_argument("--tgp-threshold", type=float, default=0.6, metavar="T", help="Viewer default for rectification threshold.")
    tgp.add_argument("--tgp-no-periodic", action="store_true", help="Use non-periodic (linear) RBF kernel.")

    # PNS (smooth_width is fixed; alpha/mean/threshold live in the viewer)
    pns = p.add_argument_group("PNS — fixed config (key params are in the viewer)")
    pns.add_argument("--pns-smooth-width", type=float, default=3.0, metavar="SW", help="Gaussian smoothing std in bins (0 = none). Fixed.")
    pns.add_argument("--pns-alpha", type=float, default=1.5, metavar="ALPHA", help="Viewer default for spectral exponent.")
    pns.add_argument("--pns-mean", type=float, default=0.0, metavar="MEAN", help="Viewer default for DC offset.")
    pns.add_argument("--pns-threshold", type=float, default=0.5, metavar="T", help="Viewer default for rectification threshold.")

    # CV-split fixed config
    cv = p.add_argument_group("CV split — fixed config (n_visits / noise_scale are in the viewer)")
    cv.add_argument("--max-rate", type=float, default=10.0, metavar="R", help="Peak firing rate in spikes/visit for Poisson splits.")

    # Deployment
    p.add_argument("--notebook", action="store_true", help="Use viewer.show() for notebook embedding instead of viewer.share() (browser).")

    return p


def main():
    args = _build_parser().parse_args()
    viewer = PlacefieldViewer(args)
    if args.notebook:
        viewer.show()
    else:
        viewer.share()


if __name__ == "__main__":
    main()
