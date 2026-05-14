from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vrAnalysis.files import literature_data_path
from vrAnalysis.external import pixease

DATA_PATH = literature_data_path() / "CortexLab_ZebraNoise"

EXPERIMENT_TARGETS = ["natural_images", "resting_state", "video", "full_field_drifting_grating"]


def gather_by_experiment_type(fpath, targets=EXPERIMENT_TARGETS):
    files_by_target = {target: list(fpath.glob(f"expcache_{target}_*.npz")) for target in targets}
    return files_by_target


def print_data_structure(data):
    for key in data.files:
        if key.endswith("___TYPE"):
            continue
        val = data[key]
        type_key = key + "___TYPE"
        type_info = str(data[type_key]) if type_key in data.files else ""
        print(f"{key}: shape={val.shape}, dtype={val.dtype}  {type_info}")


def print_metadata(data):
    # Inspect any scalar/string fields (dtype=object, shape=())

    # Note one of these is literally the full code and other metadata to run - might need some filtering before printing
    # and putting everything in context!!!
    for key in data.files:
        if key.endswith("___TYPE"):
            continue
        val = data[key]
        if val.ndim == 0 or val.dtype == object:
            print(f"{key}: {val.item() if val.ndim == 0 else val}")


def load(fpath) -> pixease.Experiment:
    """path_format = 'D:/literatureData/CortexLab_ZebraNoise/expcache_natural_images_BZ014_2025-04-16_2.npz'"""
    mouse, date, expnum = fpath.stem.split("_")[-3:]
    exp = pixease.load(mouse, date, expnum, from_dir=DATA_PATH)
    return exp


# ---------------------------------------------------------------------------
# RecordingInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingInfo:
    """Lightweight descriptor for one CortexLab ZebraNoise recording session.

    Parameters
    ----------
    mouse : str
        Mouse identifier, e.g. ``"BZ012"``.
    date : str
        Recording date, e.g. ``"2024-11-19"``.
    expnum : int
        Experiment number within the session.
    stimtype : str
        Stimulus type from ``EXPERIMENT_TARGETS``.
    file_path : Path
        Absolute path to the ``.npz`` cache file for this session.
    """

    mouse: str
    date: str
    expnum: int
    stimtype: str
    file_path: Path

    def __repr__(self) -> str:
        return f"RecordingInfo({self.mouse!r}, {self.date!r}, " f"expnum={self.expnum}, stimtype={self.stimtype!r})"


# ---------------------------------------------------------------------------
# PixeaseRegistry
# ---------------------------------------------------------------------------


class PixeaseRegistry:
    """Registry for all CortexLab ZebraNoise recording sessions.

    Scans the data directory for ``.npz`` cache files and provides methods
    for discovering and loading experiments.

    Examples
    --------
    >>> reg = PixeaseRegistry()
    >>> reg.available()
    >>> exp = reg.get("BZ012", "2024-11-19", 7)
    """

    def __init__(self, data_dir: Path | str = DATA_PATH) -> None:
        self._data_dir = Path(data_dir)
        self._records: list[RecordingInfo] = self._load_records()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> list[RecordingInfo]:
        records = []
        for fpath in sorted(self._data_dir.glob("expcache_*.npz")):
            stem = fpath.stem  # e.g. expcache_natural_images_BZ012_2024-11-19_7
            # strip "expcache_" prefix
            remainder = stem[len("expcache_") :]
            # match against known stimulus types (longest first to avoid prefix clashes)
            stimtype = None
            for target in sorted(EXPERIMENT_TARGETS, key=len, reverse=True):
                if remainder.startswith(target + "_"):
                    stimtype = target
                    remainder = remainder[len(target) + 1 :]
                    break
            if stimtype is None:
                continue
            # remainder is now "{mouse}_{date}_{expnum}"
            # expnum has no underscores; date is "YYYY-MM-DD" (contains hyphens, not underscores)
            # so split on "_" gives [mouse, date, expnum]
            parts = remainder.split("_")
            if len(parts) != 3:
                continue
            mouse, date, expnum_str = parts
            try:
                expnum = int(expnum_str)
            except ValueError:
                continue
            records.append(
                RecordingInfo(
                    mouse=mouse,
                    date=date,
                    expnum=expnum,
                    stimtype=stimtype,
                    file_path=fpath,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list(
        self,
        stimtype: str | None = None,
        mouse: str | None = None,
    ) -> list[RecordingInfo]:
        """Return records filtered by stimulus type and/or mouse name.

        Parameters
        ----------
        stimtype : str, optional
            Filter to sessions with this stimulus type.
        mouse : str, optional
            Filter to sessions from this mouse (substring match).

        Returns
        -------
        list[RecordingInfo]
        """
        records = self._records
        if stimtype is not None:
            records = [r for r in records if r.stimtype == stimtype]
        if mouse is not None:
            records = [r for r in records if mouse in r.mouse]
        return records

    def available(self) -> None:
        """Print a formatted table of all available sessions."""
        col_w = [8, 12, 8, 32]
        header = ["mouse", "date", "expnum", "stimtype"]
        sep = "  ".join("-" * w for w in col_w)
        row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(row_fmt.format(*header))
        print(sep)
        for r in self._records:
            print(row_fmt.format(r.mouse, r.date, str(r.expnum), r.stimtype))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def get(self, mouse: str, date: str, expnum: int) -> "PixeaseExperiment":
        """Load and return the experiment for the given mouse, date, and expnum.

        Parameters
        ----------
        mouse : str
            E.g. ``"BZ012"``.
        date : str
            E.g. ``"2024-11-19"``.
        expnum : int
            Experiment number within the session.

        Returns
        -------
        PixeaseExperiment
        """
        matches = [r for r in self._records if r.mouse == mouse and r.date == date and r.expnum == expnum]
        if not matches:
            raise KeyError(f"No recording found for mouse={mouse!r}, date={date!r}, expnum={expnum}. " f"Use .available() to see all sessions.")
        return PixeaseExperiment(info=matches[0])

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        mice = len({r.mouse for r in self._records})
        stims = len({r.stimtype for r in self._records})
        return f"PixeaseRegistry({len(self._records)} sessions, " f"{mice} mice, {stims} stimulus types)"


# ---------------------------------------------------------------------------
# PixeaseExperiment
# ---------------------------------------------------------------------------


@dataclass
class PixeaseExperiment:
    """Lazy-loading wrapper for one CortexLab ZebraNoise recording session.

    Instantiate via :meth:`PixeaseRegistry.get` rather than directly.

    The underlying pixease ``Experiment`` object is loaded on first access and
    cached for subsequent calls.
    """

    info: RecordingInfo
    _exp: pixease.Experiment | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._exp is not None:
            return
        self._exp = pixease.load(
            self.info.mouse,
            self.info.date,
            str(self.info.expnum),
            from_dir=DATA_PATH,
        )

    # ------------------------------------------------------------------
    # Convenience properties (from info)
    # ------------------------------------------------------------------

    @property
    def mouse(self) -> str:
        return self.info.mouse

    @property
    def date(self) -> str:
        return self.info.date

    @property
    def expnum(self) -> int:
        return self.info.expnum

    @property
    def stimtype(self) -> str:
        return self.info.stimtype

    # ------------------------------------------------------------------
    # Cell metadata properties
    # ------------------------------------------------------------------

    @property
    def cellinfo(self):
        """Suite2p cell info object (loads data on first access)."""
        self._load()
        return self._exp.cellinfo

    @property
    def iscell(self) -> np.ndarray:
        """Boolean mask of manually curated cells, shape (n_cells,)."""
        return self.cellinfo.iscell

    @property
    def n_cells(self) -> int:
        """Total number of detected ROIs (including non-cells)."""
        return self.cellinfo.n_cells

    # ------------------------------------------------------------------
    # Passthrough methods
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise human-readable summary of the experiment."""
        self._load()
        self._exp.summary()

    def stimulus_timings(self):
        """Return stimulus timing information for this session.

        Returns a DataFrame with trial start/stop times and stimulus metadata.
        Exact columns depend on the stimulus type.
        """
        self._load()
        return self._exp.stimulus_timings()

    def start_end_time(self) -> tuple[float, float]:
        """Return the common valid (start_time, stop_time) across all loaded measurements."""
        self._load()
        return self._exp.start_end_time()

    def timeseries(self, intervals, measurement=None, **kwargs) -> np.ndarray:
        """Extract a regularly sampled timeseries.

        Parameters
        ----------
        intervals : tuple or list of tuples
            ``(start, stop)`` in seconds, or a list of such tuples.
        measurement : str, optional
            Signal to extract, e.g. ``"dspikes"``, ``"f"``, ``"running"``,
            ``"pupil_size"``.  See ``pixease_usage.md`` for the full list.
        **kwargs
            Forwarded to ``pixease.Experiment.timeseries`` (e.g. ``dt``,
            ``smooth``, ``cells``).

        Returns
        -------
        ndarray or list of ndarray
        """
        self._load()
        return self._exp.timeseries(intervals, measurement, **kwargs)

    def timeseries_at(self, times, measurement=None, **kwargs) -> np.ndarray:
        """Sample a measurement at explicit timestamps.

        Parameters
        ----------
        times : array-like
            Timestamps (seconds) at which to evaluate the signal.
        measurement : str, optional
            Signal to sample.
        **kwargs
            Forwarded to ``pixease.Experiment.timeseries_at``.

        Returns
        -------
        ndarray
        """
        self._load()
        return self._exp.timeseries_at(times, measurement, **kwargs)

    def interval_mean(self, intervals, measurement=None, **kwargs) -> np.ndarray:
        """Compute mean values over one or more time intervals.

        Parameters
        ----------
        intervals : tuple or list of tuples
            ``(start, stop)`` in seconds, or a list of such tuples.
        measurement : str, optional
            Signal to average, e.g. ``"dspikes"``.
        **kwargs
            Forwarded to ``pixease.Experiment.interval_mean`` (e.g. ``cells``).

        Returns
        -------
        ndarray, shape (n_intervals, n_cells) or (n_cells,) for a single interval
        """
        self._load()
        return self._exp.interval_mean(intervals, measurement, **kwargs)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = "loaded" if self._exp is not None else "not loaded"
        return f"PixeaseExperiment({self.mouse!r}, {self.date!r}, " f"expnum={self.expnum}, stimtype={self.stimtype!r}, [{loaded}])"
