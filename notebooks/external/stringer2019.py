"""Stringer et al. 2019 — two-photon calcium imaging dataset.

Registry, experiment objects, and visualization tools for the dataset hosted at:
https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images/6845348

Data directory resolved via ``vrAnalysis.files.literature_data_path() / "Stringer2019"``.

Usage
-----
>>> from notebooks.external.stringer2019 import StringerRegistry, plot_rastermap
>>> reg = StringerRegistry()
>>> reg.available()
>>> exp = reg.get("M160825_MP027", "2016-12-14")
>>> fig, ax, model = plot_rastermap(exp.spont)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from rastermap import Rastermap

from vrAnalysis import files
from vrAnalysis.helpers import beeswarm
from dimilibi import PCA, make_time_splits


_DATA_DIR: Path = files.literature_data_path() / "Stringer2019"
_FIGURE_DIR: Path = files.literature_data_path() / "Stringer2019" / "figures"
if not _FIGURE_DIR.exists():
    _FIGURE_DIR.mkdir(parents=True)

# Stimulus types that use a shared image file (same images shown to all mice)
_SHARED_IMAGE_STIMS = {"natimg2800", "natimg2800_white"}
# Stimulus types with per-session image files
_PER_SESSION_IMAGE_STIMS = {"natimg2800_8D", "natimg2800_4D", "natimg2800_small"}
# Stimulus types with no separate image files
_NO_IMAGE_STIMS = {"ori32", "natimg32"}


# ---------------------------------------------------------------------------
# RecordingInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingInfo:
    """Lightweight descriptor for one recording session.

    Parameters
    ----------
    mouse_name : str
        Mouse identifier, e.g. ``"M160825_MP027"``.
    date : str
        Recording date, e.g. ``"2016-12-14"``.
    stimset : str
        Stimulus type from ``{"natimg2800", "natimg2800_white", "natimg2800_8D",
        "natimg2800_4D", "natimg2800_small", "ori32", "natimg32"}``.
    mouse_type : str
        Genotype / reporter label, e.g. ``"EMX_G6s"``.
    nplanes : int
        Number of imaging planes.
    file_path : Path
        Absolute path to the ``.mat`` file for this session.
    """

    mouse_name: str
    date: str
    stimset: str
    mouse_type: str
    nplanes: int
    file_path: Path

    def __repr__(self) -> str:
        return f"RecordingInfo({self.mouse_name!r}, {self.date!r}, " f"stimset={self.stimset!r}, nplanes={self.nplanes})"


# ---------------------------------------------------------------------------
# StringerRegistry
# ---------------------------------------------------------------------------


class StringerRegistry:
    """Registry for all 32 Stringer 2019 recording sessions.

    Loads ``dbstims.mat`` on construction and provides methods for
    discovering and loading experiments.

    Examples
    --------
    >>> reg = StringerRegistry()
    >>> reg.available()
    >>> exp = reg.get("M160825_MP027", "2016-12-14")
    """

    def __init__(self, data_dir: Path | str = _DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._records: list[RecordingInfo] = self._load_records()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> list[RecordingInfo]:
        db_path = self._data_dir / "dbstims.mat"
        raw = sio.loadmat(str(db_path), simplify_cells=True)

        stimset_names: npt.NDArray = raw["stimset"]  # shape (7,)
        stype: npt.NDArray = raw["stype"]  # shape (32,) 1-indexed
        dbstims: list[dict] = raw["dbstims"]  # list of 32 dicts

        records = []
        for i, (entry, stype_idx) in enumerate(zip(dbstims, stype)):
            stimset = str(stimset_names[int(stype_idx) - 1])
            mouse_name = str(entry["mouse_name"])
            date = str(entry["date"])
            mouse_type = str(entry.get("mouse_type", ""))
            nplanes = int(entry.get("nplanes", 0))
            fname = f"{stimset}_{mouse_name}_{date}.mat"
            file_path = self._data_dir / fname
            records.append(
                RecordingInfo(
                    mouse_name=mouse_name,
                    date=date,
                    stimset=stimset,
                    mouse_type=mouse_type,
                    nplanes=nplanes,
                    file_path=file_path,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list(
        self,
        stimset: str | None = None,
        mouse: str | None = None,
    ) -> list[RecordingInfo]:
        """Return records filtered by stimulus type and/or mouse name.

        Parameters
        ----------
        stimset : str, optional
            Filter to sessions with this stimulus type.
        mouse : str, optional
            Filter to sessions from this mouse (substring match).

        Returns
        -------
        list[RecordingInfo]
        """
        records = self._records
        if stimset is not None:
            records = [r for r in records if r.stimset == stimset]
        if mouse is not None:
            records = [r for r in records if mouse in r.mouse_name]
        return records

    def available(self) -> None:
        """Print a formatted table of all available sessions."""
        col_w = [14, 12, 22, 18, 8]
        header = ["mouse_name", "date", "stimset", "mouse_type", "nplanes"]
        sep = "  ".join("-" * w for w in col_w)
        row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(row_fmt.format(*header))
        print(sep)
        for r in self._records:
            print(row_fmt.format(r.mouse_name, r.date, r.stimset, r.mouse_type, str(r.nplanes)))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def get(self, mouse_name: str, date: str) -> "StringerExperiment":
        """Load and return the experiment for the given mouse and date.

        Parameters
        ----------
        mouse_name : str
            E.g. ``"M160825_MP027"``.
        date : str
            E.g. ``"2016-12-14"``.

        Returns
        -------
        StringerExperiment
        """
        matches = [r for r in self._records if r.mouse_name == mouse_name and r.date == date]
        if not matches:
            raise KeyError(f"No recording found for mouse={mouse_name!r}, date={date!r}. " f"Use .available() to see all sessions.")
        return StringerExperiment(info=matches[0], _data_dir=self._data_dir)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        mice = len({r.mouse_name for r in self._records})
        stims = len({r.stimset for r in self._records})
        return f"StringerRegistry({len(self._records)} sessions, " f"{mice} mice, {stims} stimulus types)"


# ---------------------------------------------------------------------------
# StringerExperiment
# ---------------------------------------------------------------------------


@dataclass
class StringerExperiment:
    """Lazy-loading wrapper for one Stringer 2019 recording session.

    Instantiate via :meth:`StringerRegistry.get` rather than directly.

    Attributes
    ----------
    resp : ndarray, shape (n_presentations, n_cells)
        Average stimulus-evoked response per neuron.  In presentation order.
    spont : ndarray, shape (n_frames, n_cells)
        Spontaneous activity (includes gray-screen periods during stim blocks).
    istim : ndarray, shape (n_presentations,)
        Stimulus identity for each row of ``resp``. ``gray_screen_id`` encodes
        gray screen presentations.
    ori : ndarray or None
        For ``ori32`` sessions: direction of each grating (shape (33,)).
    reps : ndarray or None
        For ``ori32`` sessions: per-repeat responses (n_unique+1, n_cells, 2).
    med : ndarray, shape (n_cells, 3)
        Estimated 3-D position [y, x, plane] of each cell in tissue.
    stat : list[dict]
        Suite2p single-cell statistics, one dict per neuron.
    """

    info: RecordingInfo
    _data_dir: Path = field(default=_DATA_DIR, repr=False)
    _data: dict | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._data is not None:
            return
        self._data = sio.loadmat(str(self.info.file_path), simplify_cells=True)

    @property
    def _stim(self) -> dict:
        self._load()
        return self._data["stim"]

    # ------------------------------------------------------------------
    # Core neural data properties
    # ------------------------------------------------------------------

    @property
    def resp(self) -> npt.NDArray[np.float32]:
        """Stimulus-evoked responses, shape (n_presentations, n_cells)."""
        return self._stim["resp"]

    @property
    def spont(self) -> npt.NDArray[np.float32]:
        """Spontaneous activity, shape (n_frames, n_cells)."""
        return self._stim["spont"]

    @property
    def istim(self) -> npt.NDArray:
        """Stimulus identity for each presentation, shape (n_presentations,)."""
        return self._stim["istim"]

    @property
    def ori(self) -> npt.NDArray | None:
        """Grating direction per unique stimulus (ori32 only), shape (33,)."""
        stim = self._stim
        return stim.get("ori", None)

    @property
    def reps(self) -> npt.NDArray | None:
        """Per-repeat averaged responses (ori32 only), shape (33, n_cells, 2)."""
        stim = self._stim
        return stim.get("reps", None)

    # ------------------------------------------------------------------
    # Cell metadata properties
    # ------------------------------------------------------------------

    @property
    def med(self) -> npt.NDArray:
        """3-D cell positions [y, x, plane], shape (n_cells, 3)."""
        self._load()
        return self._data["med"]

    @property
    def stat(self) -> list[dict]:
        """Suite2p single-cell statistics, one dict per neuron."""
        self._load()
        return self._data["stat"]

    @property
    def redcell(self) -> npt.NDArray | None:
        """Boolean interneuron labels from tdTomato (if available)."""
        if not self.stat or "redcell" not in self.stat[0]:
            return None
        return np.array([s["redcell"] for s in self.stat], dtype=bool)

    @property
    def redprob(self) -> npt.NDArray | None:
        """Classifier probability of tdTomato expression (if available)."""
        if not self.stat or "redprob" not in self.stat[0]:
            return None
        return np.array([s["redprob"] for s in self.stat], dtype=float)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of neurons."""
        return self.resp.shape[1]

    @property
    def n_stim(self) -> int:
        """Number of stimulus presentations (including gray screens)."""
        return self.resp.shape[0]

    @property
    def stimset(self) -> str:
        """Stimulus type label."""
        return self.info.stimset

    @property
    def mouse_name(self) -> str:
        return self.info.mouse_name

    @property
    def date(self) -> str:
        return self.info.date

    @property
    def is_orientation(self) -> bool:
        """True for ``ori32`` sessions."""
        return self.stimset == "ori32"

    @property
    def gray_screen_id(self) -> int:
        """Stimulus ID that encodes gray-screen presentations (max of istim)."""
        return int(self.istim.max())

    @property
    def valid_istim_mask(self) -> npt.NDArray[np.bool_]:
        """Boolean mask selecting non-gray-screen stimulus presentations."""
        return self.istim < self.gray_screen_id

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def images(self) -> npt.NDArray | None:
        """Load and return the stimulus image array.

        Returns
        -------
        ndarray or None
            Shape ``(68, 270, n_images)`` in degrees of visual space, or
            ``None`` for stimulus types without separate image files
            (``ori32``, ``natimg32``).
        """
        stimset = self.stimset
        if stimset in _NO_IMAGE_STIMS:
            return None

        if stimset in _SHARED_IMAGE_STIMS:
            img_path = self._data_dir / f"images_{stimset}_all.mat"
        else:
            img_path = self._data_dir / f"images_{stimset}_{self.mouse_name}_{self.date}.mat"

        raw = sio.loadmat(str(img_path), simplify_cells=True)
        return raw["imgs"]

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def mean_responses(self) -> npt.NDArray:
        """Average responses over repeated presentations of each stimulus.

        Gray-screen presentations are excluded.

        Returns
        -------
        ndarray, shape (n_unique_stim, n_cells)
            Rows correspond to unique stimulus IDs in sorted order.
        """
        mask = self.valid_istim_mask
        resp_valid = self.resp[mask]
        istim_valid = self.istim[mask]

        unique_ids = np.unique(istim_valid)
        averaged = np.zeros((len(unique_ids), self.n_cells), dtype=np.float32)
        for i, sid in enumerate(unique_ids):
            averaged[i] = resp_valid[istim_valid == sid].mean(axis=0)
        return averaged

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = "loaded" if self._data is not None else "not loaded"
        return f"StringerExperiment({self.mouse_name!r}, {self.date!r}, " f"stimset={self.stimset!r}, [{loaded}])"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_rastermap(
    activity: npt.NDArray,
    ax: Axes | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "gray_r",
    **rastermap_kwargs,
) -> tuple[Figure, Axes, Rastermap]:
    """Sort neurons with Rastermap and display activity as an image.

    Parameters
    ----------
    activity : ndarray, shape (n_samples, n_cells)
        Either ``exp.resp`` or ``exp.spont``.
    ax : Axes, optional
        If provided, draw into this axes.
    title : str, optional
        Axes title.
    vmin, vmax : float, optional
        Color limits passed to ``imshow``.
    cmap : str
        Colormap (default: ``"gray_r"``).
    **rastermap_kwargs
        Additional keyword arguments forwarded to :class:`rastermap.Rastermap`.

    Returns
    -------
    fig : Figure
    ax : Axes
    model : Rastermap
        Fitted Rastermap model; ``model.isort`` gives the sorted neuron indices.
    """
    model = Rastermap(**rastermap_kwargs)
    # Rastermap expects (n_neurons, n_samples)
    model.fit(activity.T)
    isort = model.isort

    sorted_activity = activity[:, isort].T  # (n_cells, n_samples)

    # Normalise each neuron for display
    row_min = sorted_activity.min(axis=1, keepdims=True)
    row_max = sorted_activity.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min == 0, 1.0, row_max - row_min)
    display = (sorted_activity - row_min) / denom

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    ax.imshow(
        display,
        aspect="auto",
        interpolation="none",
        vmin=vmin if vmin is not None else 0.0,
        vmax=vmax if vmax is not None else 1.0,
        cmap=cmap,
        origin="upper",
    )
    ax.set_xlabel("Stimulus presentation / time frame")
    ax.set_ylabel("Neuron (Rastermap order)")
    if title is not None:
        ax.set_title(title)

    return fig, ax, model


def plot_stim_responses(
    exp: StringerExperiment,
    n_stim: int = 9,
    seed: int = 0,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot example stimuli next to their mean population response.

    For sessions with image files, each row shows the stimulus image (left)
    and the sorted mean response across cells (right).  For orientation
    sessions, tuning curves are shown instead.

    Parameters
    ----------
    exp : StringerExperiment
    n_stim : int
        Number of stimuli to display (default 9).
    seed : int
        Random seed for selecting stimuli.
    figsize : tuple, optional
        Figure size; auto-scaled if omitted.

    Returns
    -------
    Figure
    """
    rng = np.random.default_rng(seed)

    mean_resp = exp.mean_responses()  # (n_unique, n_cells)
    unique_ids = np.unique(exp.istim[exp.valid_istim_mask])
    n_stim = min(n_stim, len(unique_ids))
    chosen_idx = rng.choice(len(unique_ids), size=n_stim, replace=False)
    chosen_idx = np.sort(chosen_idx)

    imgs = exp.images()
    has_images = imgs is not None

    ncols = 2 if has_images else 1
    if figsize is None:
        figsize = (4 * ncols + 2, 2.5 * n_stim)

    fig, axes = plt.subplots(n_stim, ncols, figsize=figsize, squeeze=False)

    # Sort cells by mean response across all stimuli for consistent ordering
    cell_order = np.argsort(mean_resp.mean(axis=0))[::-1]

    for row, idx in enumerate(chosen_idx):
        stim_id = unique_ids[idx]
        response = mean_resp[idx, cell_order]

        if has_images:
            img_ax = axes[row, 0]
            resp_ax = axes[row, 1]
            # imgs shape: (height, width, n_images), 0-indexed stim IDs
            img_ax.imshow(imgs[:, :, stim_id - 1], cmap="gray", aspect="auto")
            img_ax.axis("off")
            if row == 0:
                img_ax.set_title("Stimulus")
        else:
            resp_ax = axes[row, 0]

        resp_ax.plot(response, linewidth=0.8, color="steelblue")
        resp_ax.set_xlim(0, len(response))
        resp_ax.set_ylabel(f"stim {stim_id}")
        if row < n_stim - 1:
            resp_ax.set_xticks([])
        else:
            resp_ax.set_xlabel("Neuron (sorted by mean activity)")
        if row == 0:
            resp_ax.set_title("Mean response")

    fig.suptitle(
        f"{exp.mouse_name}  {exp.date}  [{exp.stimset}]",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_tuning_curves(
    exp: StringerExperiment,
    n_cells: int = 12,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot orientation tuning curves for the most responsive cells.

    Only meaningful for ``ori32`` sessions.

    Parameters
    ----------
    exp : StringerExperiment
        Must have ``exp.is_orientation == True``.
    n_cells : int
        Number of cells to show (selected by peak response).
    ax : Axes, optional

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if not exp.is_orientation:
        raise ValueError(f"plot_tuning_curves requires an ori32 session; " f"got stimset={exp.stimset!r}")

    ori = exp.ori  # (33,) — last entry is gray screen direction
    reps = exp.reps  # (33, n_cells, 2)

    # Use mean across 2 repeats, exclude gray-screen row (last)
    mean_reps = reps[:-1].mean(axis=-1)  # (32, n_cells)
    ori_angles = ori[:-1]  # (32,)

    # Select top cells by peak response
    peak = mean_reps.max(axis=0)
    top_idx = np.argsort(peak)[::-1][:n_cells]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    sort_order = np.argsort(ori_angles)
    angles_sorted = ori_angles[sort_order]
    for ci in top_idx:
        tuning = mean_reps[sort_order, ci]
        # Normalise to [0, 1]
        tuning = (tuning - tuning.min()) / max(tuning.max() - tuning.min(), 1e-9)
        ax.plot(angles_sorted, tuning, alpha=0.6, linewidth=1)

    ax.set_xlabel("Grating direction (degrees)")
    ax.set_ylabel("Normalised response")
    ax.set_title(f"Orientation tuning — top {n_cells} cells\n" f"{exp.mouse_name}  {exp.date}")
    ax.set_xticks(np.arange(0, 361, 45))
    fig.tight_layout()
    return fig, ax


def do_cka(exp: StringerExperiment, var_threshold: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    resp = exp.mean_responses()
    spont = exp.spont
    stim_var = np.var(resp, axis=0)

    # Filter by neurons so cov is manageable
    idx_keep = stim_var > var_threshold
    resp = resp[:, idx_keep]
    spont = spont[:, idx_keep]

    # Cross-validate spontaneous covariance
    n_frames = spont.shape[0]
    time_splits = make_time_splits(n_frames, num_groups=2, relative_size=None, chunks_per_group=8, num_buffer=5, force_even=False)
    spont_train = spont[time_splits[0].numpy()]
    spont_test = spont[time_splits[1].numpy()]

    # Measure training covariance with PCA to make root-cov easy
    num_components_train = min(*resp.shape, *spont_train.shape)
    pca_stim = PCA(num_components=num_components_train, center=True).fit(resp.T)
    pca_spont_train = PCA(num_components=num_components_train, center=True).fit(spont_train.T)

    train_evecs_spont = pca_spont_train.get_components().numpy()
    train_eval_spont_root = np.diag(np.sqrt(pca_spont_train.get_eigenvalues().numpy()))
    train_evecs_stim = pca_stim.get_components().numpy()
    train_eval_stim_root = np.diag(np.sqrt(pca_stim.get_eigenvalues().numpy()))

    # Measure cross-covariance with test data (stim-spont, spont-spont)
    C_spont_test = np.cov(spont_test, rowvar=False)
    inner_block_spont = train_eval_spont_root @ train_evecs_spont.T @ C_spont_test @ train_evecs_spont @ train_eval_spont_root
    inner_block_stim = train_eval_stim_root @ train_evecs_stim.T @ C_spont_test @ train_evecs_stim @ train_eval_stim_root

    variance_spont = np.sqrt(np.maximum(np.flipud(np.linalg.eigvalsh(inner_block_spont)), 0.0))
    variance_stim = np.sqrt(np.maximum(np.flipud(np.linalg.eigvalsh(inner_block_stim)), 0.0))

    return variance_spont, variance_stim


def plot(
    variance_spont: list[np.ndarray],
    variance_stim: list[np.ndarray],
    idx_example: int,
    stimset: str,
    show: bool = True,
    save: bool = False,
) -> None:
    xvals = lambda x: range(1, len(x) + 1)
    max_len = min(len(v) for v in variance_stim + variance_spont)
    nan_padding = lambda v: np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
    variance_spont = np.stack([nan_padding(v) for v in variance_spont])
    variance_stim = np.stack([nan_padding(v) for v in variance_stim])
    total_spont = np.nansum(variance_spont, axis=1)
    cum_spont = np.cumsum(variance_spont, axis=1) / total_spont[:, None]
    cum_stim = np.cumsum(variance_stim, axis=1) / total_spont[:, None]
    ratio = cum_stim / cum_spont

    plt.close("all")
    fig, ax = plt.subplots(1, 4, figsize=(13, 4), layout="constrained", width_ratios=[1, 1, 1, 0.5])
    ax[0].plot(xvals(variance_stim[idx_example]), variance_stim[idx_example], label="Stim-Stim", color="Blue")
    ax[0].plot(xvals(variance_spont[idx_example]), variance_spont[idx_example], label="Spont-Spont", color="Black")
    ax[0].legend()
    ax[0].set_title(f"Example session: {idx_example}")
    ax[0].set_xlabel("Dimension")
    ax[0].set_ylabel("Variance")

    ax[1].plot(xvals(cum_stim[idx_example]), cum_stim[idx_example], label="Stim-Stim", color="Blue")
    ax[1].plot(xvals(cum_spont[idx_example]), cum_spont[idx_example], label="Spont-Spont", color="Black")
    ax[1].set_xlabel("Number of Dimensions")
    ax[1].set_ylabel("Cumulative Variance")
    ax[1].set_title(f"Example session: {idx_example}")

    ax[2].plot(xvals(ratio.T), ratio.T, color="Black")
    ax[2].set_xlabel("Number of Dimensions")
    ax[2].set_ylabel("Ratio of Cumulative Variance")
    ax[2].set_title("All Sessions")

    ax[3].plot(beeswarm(ratio[:, -1]), ratio[:, -1], "o", color="Black")
    ax[3].set_ylabel("Ratio of Cumulative Variance")
    ax[3].set_xticks([])

    fig.suptitle(f"CKA Analysis — Stimulus: {stimset}")

    if save:
        save_path = _FIGURE_DIR / f"cka_{stimset}_Ex{idx_example}.png"
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()


def handle_args() -> None:
    parser = argparse.ArgumentParser(description="Run CKA analysis on Stringer 2019 dataset.")
    parser.add_argument("--stimset", type=str, default="ori32", help="Stimulus type to analyze (default: ori32)")
    parser.add_argument("--var-threshold", type=float, default=1.0, help="Variance threshold for neuron filtering (default: 1.0)")
    parser.add_argument("--idx-example", type=int, default=0, help="Index of example session to plot (default: 0)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively instead of saving")
    parser.add_argument("--save", action="store_true", help="Save plots to disk instead of showing interactively")
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    if not args.show and not args.save:
        print("No output option specified; defaulting to --show.")
        args.show = True

    reg = StringerRegistry()

    variance_spont = []
    variance_stim = []
    exp_list = reg.list(stimset=args.stimset)
    for rec in exp_list:
        exp = reg.get(rec.mouse_name, rec.date)
        var_spont, var_stim = do_cka(exp, var_threshold=args.var_threshold)
        variance_spont.append(var_spont)
        variance_stim.append(var_stim)

    plot(variance_spont, variance_stim, args.idx_example, args.stimset, args.show, args.save)
