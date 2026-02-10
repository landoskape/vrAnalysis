from dataclasses import dataclass, replace
from typing import Callable, Optional, Mapping, Any
import numpy as np
import numpy.typing as npt
from .utilities import generate_orthonormal, get_orthogonal_direction


Transform = Callable[["CovarianceGenerator", np.random.Generator, dict[str, Any]], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]


@dataclass(frozen=True)
class CovarianceGenerator:
    """
    A flexible generator for covariance matrices with configurable eigenstructure.

    This class represents a covariance matrix through its eigendecomposition,
    allowing for easy generation of data and transformation of the covariance
    structure through various transform methods.

    Parameters
    ----------
    num_neurons : int
        Number of neurons/dimensions.
    rank : int
        Rank of the covariance matrix (number of non-zero eigenvalues).
    eigenvectors : npt.NDArray[np.floating]
        Orthonormal eigenvectors matrix of shape (num_neurons, rank).
    eigenvalues : npt.NDArray[np.floating]
        Non-negative eigenvalues of shape (rank,).

    Examples
    --------
    >>> gen = CovarianceGenerator.powerlaw(num_neurons=100, alpha=1.4, rank=50)
    >>> data = gen.generate(num_samples=1000)
    >>> rotated = gen.variant("rotate", angle=np.pi/4)
    """

    num_neurons: int
    rank: int
    eigenvectors: npt.NDArray[np.floating]  # (N, r), orthonormal columns
    eigenvalues: npt.NDArray[np.floating]  # (r,), nonnegative

    def __post_init__(self):
        """Validate the generator parameters."""
        if self.rank > self.num_neurons:
            raise ValueError(f"rank {self.rank} > num_neurons {self.num_neurons}")
        if self.eigenvectors.shape != (self.num_neurons, self.rank):
            raise ValueError(f"eigenvectors shape {self.eigenvectors.shape} != ({self.num_neurons}, {self.rank})")
        if self.eigenvalues.shape != (self.rank,):
            raise ValueError(f"eigenvalues shape {self.eigenvalues.shape} != ({self.rank},)")
        if np.any(self.eigenvalues < 0):
            raise ValueError("Eigenvalues must be nonnegative")
        # Check orthonormality (with some tolerance)
        Q = self.eigenvectors
        if not np.allclose(Q.T @ Q, np.eye(self.rank), atol=1e-10):
            raise ValueError("Eigenvectors must be orthonormal")

    @classmethod
    def powerlaw(cls, num_neurons: int, alpha: float, rank: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> "CovarianceGenerator":
        """
        Create a CovarianceGenerator with a power-law eigenvalue spectrum.

        Parameters
        ----------
        num_neurons : int
            Number of neurons/dimensions.
        alpha : float
            Power-law exponent. Eigenvalues are i^(-alpha) for i=1,...,rank.
        rank : int, optional
            Rank of the covariance matrix. If None, defaults to num_neurons.
        rng : np.random.Generator, optional
            Random number generator for eigenvector initialization.

        Returns
        -------
        CovarianceGenerator
            A generator with power-law eigenvalue spectrum.
        """
        r = num_neurons if rank is None else rank
        if r > num_neurons:
            raise ValueError(f"rank {r} > num_neurons {num_neurons}")
        rng = np.random.default_rng() if rng is None else rng

        # power-law spectrum for the *rank* only
        evals = (np.arange(1, r + 1, dtype=float) ** (-alpha)).astype(float)

        # random orthonormal basis
        evecs = generate_orthonormal(num_neurons, rank=r, rng=rng)
        return cls(num_neurons=num_neurons, rank=r, eigenvectors=evecs, eigenvalues=evals)

    def expected_covariance(self) -> npt.NDArray[np.floating]:
        """
        Compute the expected covariance matrix.

        Returns
        -------
        npt.NDArray[np.floating]
            Covariance matrix of shape (num_neurons, num_neurons).
        """
        return self.eigenvectors @ np.diag(self.eigenvalues) @ self.eigenvectors.T

    def generate(
        self,
        num_samples: int,
        noise_variance: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.floating]:
        """
        Generate synthetic data samples from this covariance structure.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        noise_variance : float, default=0.0
            Variance of additional independent noise to add. Must be non-negative.
            If 0, no noise is added. The noise is independent across neurons and samples.
        rng : np.random.Generator, optional
            Random number generator for sample generation.

        Returns
        -------
        npt.NDArray[np.floating]
            Generated data of shape (num_neurons, num_samples).
            Each column is a sample.

        Examples
        --------
        >>> gen = CovarianceGenerator.powerlaw(num_neurons=100, alpha=1.4, rank=50)
        >>> data = gen.generate(num_samples=1000)  # No noise
        >>> noisy_data = gen.generate(num_samples=1000, noise_variance=0.1)  # With noise
        """
        if noise_variance < 0:
            raise ValueError("noise_variance must be non-negative")

        rng = np.random.default_rng() if rng is None else rng
        z = rng.standard_normal((self.rank, num_samples))
        data = self.eigenvectors @ (np.sqrt(self.eigenvalues)[:, None] * z)

        # Add independent noise if requested
        if noise_variance > 0:
            noise = rng.normal(0, np.sqrt(noise_variance), size=(self.num_neurons, num_samples))
            data = data + noise

        return data

    def variant(self, method: str, *, rng: Optional[np.random.Generator] = None, **prms) -> "CovarianceGenerator":
        """
        Create a variant of this generator by applying a transform.

        Parameters
        ----------
        method : str
            Name of the transform method to apply. Must be a key in VARIANTS.
        rng : np.random.Generator, optional
            Random number generator for the transform.
        **prms
            Additional parameters passed to the transform function.

        Returns
        -------
        CovarianceGenerator
            A new generator with transformed eigenstructure.

        Raises
        ------
        ValueError
            If method is unknown or transform returns invalid shapes/values.
        """
        rng = np.random.default_rng() if rng is None else rng
        try:
            tfm = VARIANTS[method]
        except KeyError as e:
            raise ValueError(f"Unknown method '{method}'. Available: {sorted(VARIANTS)}") from e

        new_evecs, new_evals = tfm(self, rng, prms)

        if new_evecs.shape != (self.num_neurons, self.rank):
            raise ValueError("Transform returned eigenvectors with wrong shape.")
        if new_evals.shape != (self.rank,):
            raise ValueError("Transform returned eigenvalues with wrong shape.")
        if np.any(new_evals < 0):
            raise ValueError("Eigenvalues must be nonnegative.")

        return replace(self, eigenvectors=new_evecs, eigenvalues=new_evals)


class PowerlawDataGenerator:
    """
    Generate synthetic data with a power-law eigenspectrum.

    This generator creates data where the covariance matrix has eigenvalues
    that follow a power-law distribution: λ_i = i^(-alpha).

    Parameters
    ----------
    num_neurons : int
        Number of neurons/dimensions in the generated data.
    alpha : float
        Power-law exponent. Larger values give steeper eigenvalue decay.
    rank : int, optional
        Rank of the covariance matrix. If None, defaults to num_neurons.
        If rank < num_neurons, the data will lie in a lower-dimensional subspace.

    Examples
    --------
    >>> generator = PowerlawDataGenerator(num_neurons=100, alpha=1.4, rank=50)
    >>> data = generator.generate(num_samples=1000)  # Shape: (100, 1000)
    """

    def __init__(self, num_neurons: int, alpha: float, rank: Optional[int] = None):
        self.num_neurons = num_neurons
        self.alpha = alpha
        self.rank = rank or num_neurons
        if self.rank > num_neurons:
            raise ValueError(f"Rank cannot be greater than number of neurons: {self.rank} > {num_neurons}")

        # Generate eigenspectrum: λ_i = i^(-alpha)
        self.eigenvalues = np.arange(1, num_neurons + 1, dtype=float) ** -alpha

        # Generate orthonormal eigenvectors
        self.eigenvectors = generate_orthonormal(num_neurons, rank=self.rank)

    def expected_covariance(self) -> npt.NDArray[np.floating]:
        """
        Get the expected covariance matrix for this generator.

        Returns
        -------
        npt.NDArray[np.floating]
            Covariance matrix of shape (num_neurons, num_neurons).
        """
        # Use only the eigenvalues corresponding to the rank
        evals_used = self.eigenvalues[: self.rank]
        return self.eigenvectors @ np.diag(evals_used) @ self.eigenvectors.T

    def generate(self, num_samples: int) -> npt.NDArray[np.floating]:
        """
        Generate synthetic data samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        npt.NDArray[np.floating]
            Generated data of shape (num_neurons, num_samples).
            Each column is a sample.
        """
        # Use only the eigenvalues corresponding to the rank
        evals_used = self.eigenvalues[: self.rank]
        return self.eigenvectors @ np.diag(np.sqrt(evals_used)) @ np.random.randn(self.rank, num_samples)

    def clone(self) -> "PowerlawDataGenerator":
        """
        Clone this generator.
        """


class RotatedEigenbasisGenerator:
    """
    Generate synthetic 2D data with rotated eigenbases.

    This generator creates two covariance matrices (C1 and C2) with the same
    eigenvalues but rotated eigenbases. The rotation is controlled by offset_ratio.

    Parameters
    ----------
    offset_ratio : float
        Ratio controlling the rotation of the eigenbasis. Creates main directions
        [1, offset_ratio] and [offset_ratio, 1] which are normalized.
    evals1 : npt.NDArray[np.floating]
        Eigenvalues for the first covariance matrix C1. Shape should be (2,).
    evals2 : npt.NDArray[np.floating], optional
        Eigenvalues for the second covariance matrix C2. If None, uses evals1.
        Shape should be (2,).

    Examples
    --------
    >>> evals = np.array([1.0, 0.1])
    >>> generator = RotatedEigenbasisGenerator(offset_ratio=2.0, evals1=evals)
    >>> data = generator.generate(num_samples=1000)  # Shape: (2, 1000)
    >>> C1 = generator.expected_covariance(which=1)
    >>> C2 = generator.expected_covariance(which=2)
    """

    def __init__(
        self,
        offset_ratio: float,
        evals1: npt.NDArray[np.floating],
        evals2: Optional[npt.NDArray[np.floating]] = None,
    ):
        self.offset_ratio = offset_ratio
        self.evals1 = np.asarray(evals1, dtype=float)
        self.evals2 = np.asarray(evals2, dtype=float) if evals2 is not None else self.evals1.copy()

        if self.evals1.shape != (2,):
            raise ValueError(f"evals1 must have shape (2,), got {self.evals1.shape}")
        if self.evals2.shape != (2,):
            raise ValueError(f"evals2 must have shape (2,), got {self.evals2.shape}")

        # Create main directions
        main_dir1 = np.array([1.0, offset_ratio], dtype=float)
        main_dir2 = np.array([offset_ratio, 1.0], dtype=float)
        main_dir1 = main_dir1 / np.linalg.norm(main_dir1)
        main_dir2 = main_dir2 / np.linalg.norm(main_dir2)

        # Get orthogonal directions
        orth_dir1 = get_orthogonal_direction(main_dir1)
        orth_dir2 = get_orthogonal_direction(main_dir2)

        # Create eigenbases (orthonormal matrices)
        self.Q1 = np.column_stack([main_dir1, orth_dir1])
        self.Q2 = np.column_stack([main_dir2, orth_dir2])
        # Ensure columns are normalized (should already be, but be safe)
        self.Q1 = self.Q1 / np.linalg.norm(self.Q1, axis=0, keepdims=True)
        self.Q2 = self.Q2 / np.linalg.norm(self.Q2, axis=0, keepdims=True)

        # Store covariance matrices
        self.C1 = self.Q1 @ np.diag(self.evals1) @ self.Q1.T
        self.C2 = self.Q2 @ np.diag(self.evals2) @ self.Q2.T

    def expected_covariance(self, which: int = 1) -> npt.NDArray[np.floating]:
        """
        Get the expected covariance matrix for this generator.

        Parameters
        ----------
        which : int
            Which covariance matrix to return: 1 for C1, 2 for C2. Default is 1.

        Returns
        -------
        npt.NDArray[np.floating]
            Covariance matrix of shape (2, 2).
        """
        if which == 1:
            return self.C1
        elif which == 2:
            return self.C2
        else:
            raise ValueError(f"which must be 1 or 2, got {which}")

    def generate(self, num_samples: int, which: int = 1) -> npt.NDArray[np.floating]:
        """
        Generate synthetic data samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        which : int
            Which covariance matrix to use for generation: 1 for C1, 2 for C2. Default is 1.

        Returns
        -------
        npt.NDArray[np.floating]
            Generated data of shape (2, num_samples).
            Each column is a sample.
        """
        if which == 1:
            Q = self.Q1
            evals = self.evals1
        elif which == 2:
            Q = self.Q2
            evals = self.evals2
        else:
            raise ValueError(f"which must be 1 or 2, got {which}")

        return Q @ np.diag(np.sqrt(evals)) @ np.random.randn(2, num_samples)


def _transform_scale_spectrum(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Scale all eigenvalues by a constant factor (spectrum-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator (not used).
    prms : dict
        Required parameters:
        - scale : float
            Scaling factor to apply to eigenvalues.

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvectors are unchanged.
    """
    scale = prms.get("scale")
    if scale is None:
        raise ValueError("scale parameter is required")
    if scale < 0:
        raise ValueError("scale must be nonnegative")
    new_evals = gen.eigenvalues * scale
    return gen.eigenvectors.copy(), new_evals


def _transform_band_scale_spectrum(
    gen: CovarianceGenerator,
    rng: np.random.Generator,
    prms: dict[str, Any],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Scale head / middle / tail eigenvalues differently (spectrum-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator (not used).
    prms : dict
        Required parameters:
        - k_head : int
            Number of eigenvalues in the head (largest).
        - k_tail : int
            Number of eigenvalues in the tail (smallest).
        - scale_head : float
            Scaling factor for head eigenvalues.
        - scale_mid : float
            Scaling factor for middle eigenvalues.
        - scale_tail : float
            Scaling factor for tail eigenvalues.

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvectors are unchanged.
    """
    k_head = prms.get("k_head", 0)
    k_tail = prms.get("k_tail", 0)
    scale_head = prms.get("scale_head", 1.0)
    scale_mid = prms.get("scale_mid", 1.0)
    scale_tail = prms.get("scale_tail", 1.0)

    if k_head < 0 or k_tail < 0 or k_head + k_tail > gen.rank:
        raise ValueError(f"Invalid k_head={k_head}, k_tail={k_tail} for rank={gen.rank}")

    new_evals = gen.eigenvalues.copy()
    # Head (largest eigenvalues, indices 0 to k_head-1)
    if k_head > 0:
        new_evals[:k_head] *= scale_head
    # Tail (smallest eigenvalues, indices rank-k_tail to rank-1)
    if k_tail > 0:
        new_evals[-k_tail:] *= scale_tail
    # Middle (everything else)
    k_mid_start = k_head
    k_mid_end = gen.rank - k_tail
    if k_mid_end > k_mid_start:
        new_evals[k_mid_start:k_mid_end] *= scale_mid

    return gen.eigenvectors.copy(), new_evals


def _transform_powerlaw_resample(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Replace eigenvalues with a new power-law spectrum (spectrum-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator (not used).
    prms : dict
        Required parameters:
        - alpha_new : float
            New power-law exponent.
        - normalize_total_variance : bool, default=False
            If True, scale eigenvalues to preserve total variance.

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvectors are unchanged.
    """
    alpha_new = prms.get("alpha_new")
    if alpha_new is None:
        raise ValueError("alpha_new parameter is required")
    normalize_total_variance = prms.get("normalize_total_variance", False)

    # Generate new power-law spectrum
    new_evals = (np.arange(1, gen.rank + 1, dtype=float) ** (-alpha_new)).astype(float)

    if normalize_total_variance:
        # Scale to preserve total variance
        old_total = np.sum(gen.eigenvalues)
        new_total = np.sum(new_evals)
        if new_total > 0:
            new_evals = new_evals * (old_total / new_total)

    return gen.eigenvectors.copy(), new_evals


def _transform_perturb_eigenvectors(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Add small random perturbations to eigenvectors and re-orthonormalize (eigenvector-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        Optional parameters:
        - epsilon : float, default=0.1
            Scale of the random noise to add.
        - where : str, default="all"
            Which eigenvectors to perturb: "all" or "topk".

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvalues are unchanged.
    """
    epsilon = prms.get("epsilon", 0.1)
    where = prms.get("where", "all")

    new_evecs = gen.eigenvectors.copy()

    if where == "all":
        perturbed = new_evecs + epsilon * rng.standard_normal(new_evecs.shape)
    elif where == "topk":
        k = prms.get("k")
        if k is None:
            raise ValueError("k parameter required for topk mode")
        if k < 1 or k > gen.rank:
            raise ValueError(f"k must be between 1 and {gen.rank}")
        perturbed = new_evecs.copy()
        perturbed[:, :k] += epsilon * rng.standard_normal((gen.num_neurons, k))
    else:
        raise ValueError(f"Unknown where '{where}'. Must be one of: all, topk")

    # Re-orthonormalize using QR
    Q, _ = np.linalg.qr(perturbed)
    new_evecs = Q[:, : gen.rank]

    return new_evecs, gen.eigenvalues.copy()


def _transform_randomize_eigenvectors(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Replace all eigenvectors with a new random orthonormal basis (eigenvector-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        No additional parameters required.

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvalues are unchanged.
    """
    new_evecs = generate_orthonormal(gen.num_neurons, rank=gen.rank, rng=rng)
    return new_evecs, gen.eigenvalues.copy()


def _transform_rotate_plane(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Rotate eigenvectors in a selected 2-D subspace by angle θ (eigenvector-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator (not used if theta is provided).
    prms : dict
        Required parameters:
        - i : int
            Index of first eigenvector (0-indexed).
        - j : int
            Index of second eigenvector (0-indexed).
        - theta : float
            Rotation angle in radians.

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvalues are unchanged.
    """
    i = prms.get("i")
    j = prms.get("j")
    theta = prms.get("theta")

    if i is None or j is None:
        raise ValueError("i and j parameters are required")
    if theta is None:
        raise ValueError("theta parameter is required")
    if i < 0 or i >= gen.rank or j < 0 or j >= gen.rank or i == j:
        raise ValueError(f"Invalid indices i={i}, j={j} for rank={gen.rank}")

    new_evecs = gen.eigenvectors.copy()
    c, s = np.cos(theta), np.sin(theta)
    # Rotate in the plane spanned by eigenvectors i and j
    vi = new_evecs[:, i].copy()
    vj = new_evecs[:, j].copy()
    new_evecs[:, i] = c * vi + s * vj
    new_evecs[:, j] = -s * vi + c * vj

    return new_evecs, gen.eigenvalues.copy()


def _transform_rotate_block(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Apply a random or structured rotation within k eigenvectors (eigenvector-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        Required parameters:
        - k : int
            Number of eigenvectors to rotate.
        - where : str, default="top"
            Which eigenvectors to rotate: "top" or "tail".
        - strength : float, optional
            Strength of rotation (0-1). If provided, uses structured rotation.
        - theta : float, optional
            Explicit rotation angle. If provided, applies same rotation to all pairs.

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvalues are unchanged.
    """
    k = prms.get("k")
    if k is None:
        raise ValueError("k parameter is required")
    if k < 2 or k > gen.rank:
        raise ValueError(f"k must be between 2 and {gen.rank}")

    where = prms.get("where", "top")
    if where not in ("top", "tail"):
        raise ValueError(f"where must be 'top' or 'tail', got '{where}'")

    strength = prms.get("strength", None)
    theta = prms.get("theta", None)

    new_evecs = gen.eigenvectors.copy()

    # Determine indices to rotate
    if where == "top":
        indices = np.arange(k)
    else:  # where == "tail"
        indices = np.arange(gen.rank - k, gen.rank)

    if theta is not None:
        # Apply same rotation angle to all pairs
        c, s = np.cos(theta), np.sin(theta)
        for idx_i, i in enumerate(indices):
            for idx_j, j in enumerate(indices):
                if idx_j > idx_i:  # Only rotate each pair once
                    vi = new_evecs[:, i].copy()
                    vj = new_evecs[:, j].copy()
                    new_evecs[:, i] = c * vi + s * vj
                    new_evecs[:, j] = -s * vi + c * vj
    elif strength is not None:
        # Structured rotation with given strength
        if strength < 0 or strength > 1:
            raise ValueError("strength must be between 0 and 1")
        # Generate random rotation matrix for k-dimensional subspace
        A = rng.standard_normal((k, k))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        # Interpolate between identity and random rotation
        R = (1 - strength) * np.eye(k) + strength * Q
        # Apply to selected eigenvectors
        new_evecs[:, indices] = new_evecs[:, indices] @ R.T
    else:
        # Fully random rotation
        A = rng.standard_normal((k, k))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        new_evecs[:, indices] = new_evecs[:, indices] @ Q.T

    # Re-orthonormalize to ensure numerical stability
    Q, _ = np.linalg.qr(new_evecs)
    new_evecs = Q[:, : gen.rank]

    return new_evecs, gen.eigenvalues.copy()


def _transform_permute_eigenvalues(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Permute eigenvalues while keeping eigenvectors fixed (spectrum-only transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        Optional parameters:
        - mode : str, default="full"
            Permutation mode: "full", "partial".
        - k : int, optional
            For "partial" mode, number of eigenvalues to permute.
        - where : str, default="head"
            For "partial" mode, where to select eigenvalues: "head", "tail", or "random".
        - frac : float, optional
            Alternative to k: fraction of eigenvalues to permute (0-1).

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) where eigenvectors are unchanged.
    """
    mode = prms.get("mode", "full")
    k = prms.get("k", None)
    frac = prms.get("frac", None)
    where = prms.get("where", "head")

    new_evals = gen.eigenvalues.copy()

    if mode == "full":
        perm = rng.permutation(gen.rank)
        new_evals = new_evals[perm]
    elif mode == "partial":
        if k is None:
            if frac is not None:
                k = max(1, int(gen.rank * frac))
            else:
                raise ValueError("k or frac parameter required for partial mode")
        if k < 1 or k > gen.rank:
            raise ValueError(f"k must be between 1 and {gen.rank}")

        if where == "head":
            indices = np.arange(k)
            perm = rng.permutation(k)
            new_evals[indices] = new_evals[indices[perm]]
        elif where == "tail":
            indices = np.arange(gen.rank - k, gen.rank)
            perm = rng.permutation(k)
            new_evals[indices] = new_evals[indices[perm]]
        elif where == "random":
            indices = rng.choice(gen.rank, size=k, replace=False)
            perm = rng.permutation(k)
            new_evals[indices] = new_evals[indices[perm]]
        else:
            raise ValueError(f"Unknown where '{where}'. Must be one of: head, tail, random")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of: full, partial")

    return gen.eigenvectors.copy(), new_evals


def _transform_shared_subspace_rotated(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Share span of top-k eigenvectors but rotate within subspace (mixed transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        Required parameters:
        - k : int
            Number of top eigenvectors to preserve span of.
        - rotation_strength : float
            Strength of rotation within subspace (0-1).

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) with rotated subspace.
    """
    k = prms.get("k")
    if k is None:
        raise ValueError("k parameter is required")
    if k < 1 or k > gen.rank:
        raise ValueError(f"k must be between 1 and {gen.rank}")

    rotation_strength = prms.get("rotation_strength", 1.0)
    if rotation_strength < 0 or rotation_strength > 1:
        raise ValueError("rotation_strength must be between 0 and 1")

    new_evecs = gen.eigenvectors.copy()
    new_evals = gen.eigenvalues.copy()

    # Generate rotation matrix for top-k subspace
    A = rng.standard_normal((k, k))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    # Interpolate between identity and random rotation
    R = (1 - rotation_strength) * np.eye(k) + rotation_strength * Q

    # Apply rotation to top-k eigenvectors
    new_evecs[:, :k] = new_evecs[:, :k] @ R.T

    # Re-orthonormalize
    Q_full, _ = np.linalg.qr(new_evecs)
    new_evecs = Q_full[:, : gen.rank]

    return new_evecs, new_evals


def _transform_shared_subset_full_match(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Exactly copy selected eigenpairs; randomize others (mixed transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        Required parameters (one of):
        - indices : array-like
            Explicit indices of eigenpairs to preserve.
        - k : int
            Number of eigenpairs to preserve.
        - where : str, default="head"
            Where to preserve: "head", "tail", or "random" (used with k).

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) with preserved and randomized parts.
    """
    indices = prms.get("indices", None)
    k = prms.get("k", None)
    where = prms.get("where", "head")

    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= gen.rank):
            raise ValueError(f"indices must be between 0 and {gen.rank - 1}")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("indices must be unique")
        preserve_indices = np.sort(indices)
    elif k is not None:
        if k < 0 or k > gen.rank:
            raise ValueError(f"k must be between 0 and {gen.rank}")
        if where == "head":
            preserve_indices = np.arange(k)
        elif where == "tail":
            preserve_indices = np.arange(gen.rank - k, gen.rank)
        elif where == "random":
            preserve_indices = np.sort(rng.choice(gen.rank, size=k, replace=False))
        else:
            raise ValueError(f"Unknown where '{where}'. Must be one of: head, tail, random")
    else:
        raise ValueError("Must provide either indices or k parameter")

    new_evecs = gen.eigenvectors.copy()
    new_evals = gen.eigenvalues.copy()

    if len(preserve_indices) == 0:
        # Randomize all
        new_evecs = generate_orthonormal(gen.num_neurons, rank=gen.rank, rng=rng)
    elif len(preserve_indices) == gen.rank:
        # Keep all
        pass
    else:
        # Preserve selected eigenpairs, randomize rest
        other_indices = np.setdiff1d(np.arange(gen.rank), preserve_indices)
        preserved_evecs = new_evecs[:, preserve_indices].copy()
        preserved_evals = new_evals[preserve_indices].copy()

        # Generate random eigenvectors for the rest
        random_evecs = generate_orthonormal(gen.num_neurons, rank=len(other_indices), rng=rng)
        # Ensure orthogonal to preserved
        proj = preserved_evecs @ preserved_evecs.T
        random_evecs = random_evecs - proj @ random_evecs
        Q, _ = np.linalg.qr(random_evecs)
        random_evecs = Q[:, : len(other_indices)]

        # Reconstruct in original order
        combined_evecs = np.zeros((gen.num_neurons, gen.rank))
        combined_evecs[:, preserve_indices] = preserved_evecs
        combined_evecs[:, other_indices] = random_evecs
        new_evecs = combined_evecs

    return new_evecs, new_evals


def _transform_add_isotropic_noise(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Add σ²I to covariance (additive transform - recomputes eigendecomposition).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator (not used).
    prms : dict
        Required parameters:
        - sigma2 : float
            Variance of isotropic noise to add.

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) with added noise.
    """
    sigma2 = prms.get("sigma2")
    if sigma2 is None:
        raise ValueError("sigma2 parameter is required")
    if sigma2 < 0:
        raise ValueError("sigma2 must be nonnegative")

    # Compute current covariance
    C = gen.expected_covariance()
    # Add isotropic noise
    C_new = C + sigma2 * np.eye(gen.num_neurons)

    # Recompute eigendecomposition
    evals_full, evecs_full = np.linalg.eigh(C_new)
    # Sort by descending eigenvalues
    idx = np.argsort(evals_full)[::-1]
    evals_sorted = evals_full[idx]
    evecs_sorted = evecs_full[:, idx]

    # Keep top rank eigenvalues/eigenvectors
    new_evals = evals_sorted[: gen.rank]
    new_evecs = evecs_sorted[:, : gen.rank]

    return new_evecs, new_evals


def _transform_add_diagonal_noise(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Add neuron-specific variance (heteroskedastic noise) - recomputes eigendecomposition.

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        Required parameters:
        - scale : float
            Scale of diagonal noise.
        - distribution : str, default="uniform"
            Distribution for noise: "uniform", "exponential", or "gamma".

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) with added noise.
    """
    scale = prms.get("scale")
    if scale is None:
        raise ValueError("scale parameter is required")
    if scale < 0:
        raise ValueError("scale must be nonnegative")

    distribution = prms.get("distribution", "uniform")

    # Generate diagonal noise
    if distribution == "uniform":
        diag_noise = scale * rng.uniform(0, 1, gen.num_neurons)
    elif distribution == "exponential":
        diag_noise = scale * rng.exponential(1.0, gen.num_neurons)
    elif distribution == "gamma":
        diag_noise = scale * rng.gamma(2.0, 1.0, gen.num_neurons)
    else:
        raise ValueError(f"Unknown distribution '{distribution}'. Must be one of: uniform, exponential, gamma")

    # Compute current covariance
    C = gen.expected_covariance()
    # Add diagonal noise
    C_new = C + np.diag(diag_noise)

    # Recompute eigendecomposition
    evals_full, evecs_full = np.linalg.eigh(C_new)
    idx = np.argsort(evals_full)[::-1]
    evals_sorted = evals_full[idx]
    evecs_sorted = evecs_full[:, idx]

    # Keep top rank eigenvalues/eigenvectors
    new_evals = evals_sorted[: gen.rank]
    new_evecs = evecs_sorted[:, : gen.rank]

    return new_evecs, new_evals


def _transform_add_rank1_mode(
    gen: CovarianceGenerator, rng: np.random.Generator, prms: dict[str, Any]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Add a new low-rank covariance component vvᵀ (additive transform).

    Parameters
    ----------
    gen : CovarianceGenerator
        The generator to transform.
    rng : np.random.Generator
        Random number generator.
    prms : dict
        Required parameters:
        - strength : float
            Strength of the rank-1 component.
        - alignment : str, default="random"
            Alignment mode: "aligned", "orthogonal", or "random".

    Returns
    -------
    tuple
        (new_eigenvectors, new_eigenvalues) with added rank-1 component.
    """
    strength = prms.get("strength")
    if strength is None:
        raise ValueError("strength parameter is required")
    if strength < 0:
        raise ValueError("strength must be nonnegative")

    alignment = prms.get("alignment", "random")

    # Generate vector v
    if alignment == "aligned":
        # Align with first eigenvector
        v = gen.eigenvectors[:, 0].copy()
    elif alignment == "orthogonal":
        # Orthogonal to first eigenvector
        v = rng.standard_normal(gen.num_neurons)
        v = v - (v @ gen.eigenvectors[:, 0]) * gen.eigenvectors[:, 0]
        v = v / np.linalg.norm(v)
    elif alignment == "random":
        # Random direction
        v = rng.standard_normal(gen.num_neurons)
        v = v / np.linalg.norm(v)
    else:
        raise ValueError(f"Unknown alignment '{alignment}'. Must be one of: aligned, orthogonal, random")

    # Compute current covariance
    C = gen.expected_covariance()
    # Add rank-1 component
    C_new = C + strength * np.outer(v, v)

    # Recompute eigendecomposition
    evals_full, evecs_full = np.linalg.eigh(C_new)
    idx = np.argsort(evals_full)[::-1]
    evals_sorted = evals_full[idx]
    evecs_sorted = evecs_full[:, idx]

    # Keep top rank eigenvalues/eigenvectors
    new_evals = evals_sorted[: gen.rank]
    new_evecs = evecs_sorted[:, : gen.rank]

    return new_evecs, new_evals


# Registry of available transform methods
VARIANTS: Mapping[str, Transform] = {
    # Spectrum-only transforms
    "scale_spectrum": _transform_scale_spectrum,
    "band_scale_spectrum": _transform_band_scale_spectrum,
    "powerlaw_resample": _transform_powerlaw_resample,
    "permute_eigenvalues": _transform_permute_eigenvalues,
    # Eigenvector-only transforms
    "randomize_eigenvectors": _transform_randomize_eigenvectors,
    "rotate_plane": _transform_rotate_plane,
    "rotate_block": _transform_rotate_block,
    "perturb_eigenvectors": _transform_perturb_eigenvectors,
    # Mixed transforms
    "shared_subspace_rotated": _transform_shared_subspace_rotated,
    "shared_subset_full_match": _transform_shared_subset_full_match,
    # Additive transforms
    "add_isotropic_noise": _transform_add_isotropic_noise,
    "add_diagonal_noise": _transform_add_diagonal_noise,
    "add_rank1_mode": _transform_add_rank1_mode,
}
