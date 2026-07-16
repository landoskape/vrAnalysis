"""BehaviorSpeedEnvConfig — decode environment identity from pre-reward running speed.

Asks whether a mouse's running speed *before the reward zone* reveals which environment
it is in. The pre-reward window is purely visually driven: the mouse has not yet been
rewarded on that trial, so environment information in the speed trace reflects what it
sees and recognizes, not what it has just received.

Two decoding questions, each answered on a single pre-reward position window:

1. **Random split** — stratified 50/50 over trials. Is environment decodable at all?
2. **Block split** — train on non-first trials of each environment block, test on the
   first trial of each block. Environment blocks are randomized, so the first trial is
   the mouse's first encounter with that environment since a switch. This asks whether
   recognition is immediate rather than re-learned within the block.

The feature window is the first ``WINDOW_FRACTION`` of the track (see below). It ends at
the earliest reward zone across all environments — environment 4 rewards first, at
fraction 0.25 — so the decoder never sees post-reward speed for any environment.

Binning is by *fraction of track*: ``num_bins`` bins span ``[0, env_length]``. Track
length is not uniform across the colony, so a fixed bin count keeps bin index comparable
across sessions and keeps every result array the same width. Speed stays in physical
cm/s; only the position axis is normalized.

Every environment-indexed result is emitted on the canonical ``ENV_SLOTS`` axis (NaN for
curves, 0 for counts, where a session lacks an environment), so column j always means the
same environment and cross-session averaging is meaningful. Which environments a session
actually contributed is recorded per window in ``env_used_{window}``.

This is a **behavior-only** analysis — no neural data is loaded, and the ``registry``
argument to :meth:`process` is ignored (registry splits are *frame* splits; ours are
*trial* splits).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from vrAnalysis.helpers import cross_validate_trials, edge2center, environmentRewardZone
from vrAnalysis.processors.support import (
    convolve_toeplitz,
    correct_map,
    get_gauss_kernel,
    get_summation_map,
    replace_missing_data,
)
from vrAnalysis.sessions import B2Session

from ..pipeline.base import AnalysisConfigBase
from ..registry import PopulationRegistry

# The CR_* mice run a different track length (245 cm vs 200 cm) and a disjoint set of
# environments (1, 2 rather than 1, 3, 4). Their "environment 1" rewards at a different
# fraction of track than the ATL "environment 1", so the label does not carry across
# cohorts. They are excluded rather than silently pooled.
EXCLUDED_MOUSE_PREFIXES: tuple[str, ...] = ("CR_",)

# Reward zone *start* (position - halfwidth) in cm on the reference track, per
# environment. Verified across all imaging sessions by
# ``scripts/derive_prereward_cutoff.py``. These set how far a feature window may extend
# before it would touch reward.
REFERENCE_ENV_LENGTH_CM: float = 200.0
ENV_REWARD_MAP: dict[int, float] = {
    1: 150,
    3: 100,
    4: 50,
}

# Canonical environment axis. Every environment-indexed result is emitted on these slots,
# with NaN (curves) or 0 (counts) where a session lacks that environment. Sessions differ
# in which environments they ran, so indexing by the session's own ``np.unique`` would
# make column j mean env 1 in one session and env 3 in another -- averaging across
# sessions or mice would then silently mix environments together.
ENV_SLOTS: tuple[int, ...] = tuple(sorted(ENV_REWARD_MAP))

# Pre-reward feature window, as a fraction of track. Derived from ENV_REWARD_MAP rather
# than hard-coded: it stops at the earliest reward zone across all environments (env 4, at
# fraction 0.25), so the window is reward-free for every environment by construction.
WINDOW_FRACTION: float = min(ENV_REWARD_MAP.values()) / REFERENCE_ENV_LENGTH_CM

# The vrControl software always reports a trial's first sample at position 0, even when
# the rotary encoder has a built-up buffer, so position can jump at the second sample.
# The leading bins of a trial are therefore unreliable and are excluded from the feature
# window. This mirrors ``SpkmapParams.full_trial_flexibility`` (also 3.0 cm), which is why
# ``get_env_maps`` never requires the first 3 cm either. Without this, ~13% of trials are
# discarded for a NaN in bin 0 alone.
TRIAL_START_FLEXIBILITY_CM: float = 3.0


def _window_bins(dist_edges: npt.NDArray[np.floating], fraction: float) -> tuple[int, int]:
    """Half-open bin range ``[start, stop)`` for a pre-reward window.

    ``stop`` is the last bin ending at or before ``fraction`` of the track; ``start``
    skips the unreliable leading bins (see ``TRIAL_START_FLEXIBILITY_CM``).

    Parameters
    ----------
    dist_edges : np.ndarray
        Position bin edges in cm, shape ``(num_bins + 1,)``.
    fraction : float
        Window end as a fraction of track length.

    Returns
    -------
    tuple of int
        ``(start_bin, stop_bin)``.
    """
    num_bins = len(dist_edges) - 1
    start = int(np.searchsorted(dist_edges, TRIAL_START_FLEXIBILITY_CM))
    stop = int(np.floor(fraction * num_bins))
    return start, stop


def _speed_map(
    session: B2Session,
    dist_edges: npt.NDArray[np.floating],
    speed_threshold: float,
    smooth_width: float | None,
) -> npt.NDArray[np.floating]:
    """Time-weighted mean running speed per (trial, position bin).

    Mirrors :meth:`SpkmapProcessor.get_raw_maps` / :meth:`Maps.raw_to_processed` but
    computes occupancy and speed only, so no neural data is loaded. Unlike
    ``get_env_maps``, rows align 1:1 with ``session.trial_environment`` — trial identity
    is preserved, which is required to find the first trial of each block.

    Parameters
    ----------
    session : B2Session
    dist_edges : np.ndarray
        Position bin edges in cm, shape ``(num_bins + 1,)``.
    speed_threshold : float
        Samples at or below this speed are excluded. Pass ``-np.inf`` for no filtering
        (``SpkmapParams`` cannot express this: it validates ``speed_threshold > 0``).
    smooth_width : float or None
        Gaussian smoothing width in cm, applied along position before dividing by
        occupancy. None disables smoothing.

    Returns
    -------
    np.ndarray
        Shape ``(num_trials, num_bins)``. Bins the mouse never visited are NaN.
    """
    num_positions = len(dist_edges) - 1
    timestamps, positions, trial_numbers, idx_behave_to_frame = session.positions

    # Speed per behavioural sample. The last sample of each trial has undefined speed.
    within_trial_sample = np.append(np.diff(trial_numbers) == 0, True)
    sample_duration = np.append(np.diff(timestamps), 0)
    speeds = np.append(np.diff(positions) / sample_duration[:-1], 0)
    sample_duration = sample_duration * within_trial_sample
    speeds = speeds * within_trial_sample
    position_bin = np.digitize(positions, dist_edges) - 1

    # Reject behavioural samples too far from an imaging frame, matching the rest of the
    # pipeline's frame selection.
    frame_time_stamps = session.timestamps
    sampling_period = np.median(np.diff(frame_time_stamps))
    dist_cutoff = sampling_period / 2
    delay_position_to_imaging = frame_time_stamps[idx_behave_to_frame] - timestamps

    dtype = np.float32
    occmap = np.zeros((session.num_trials, num_positions), dtype=dtype)
    counts = np.zeros((session.num_trials, num_positions), dtype=dtype)
    speedmap = np.zeros((session.num_trials, num_positions), dtype=dtype)

    get_summation_map(
        sample_duration,
        trial_numbers,
        position_bin,
        occmap,
        counts,
        speeds,
        speed_threshold,
        np.inf,
        delay_position_to_imaging,
        dist_cutoff,
        sample_duration,
        scale_by_sample_duration=False,
        use_sample_to_value_idx=False,
        sample_to_value_idx=idx_behave_to_frame,
    )
    get_summation_map(
        speeds,
        trial_numbers,
        position_bin,
        speedmap,
        counts,
        speeds,
        speed_threshold,
        np.inf,
        delay_position_to_imaging,
        dist_cutoff,
        sample_duration,
        scale_by_sample_duration=True,
        use_sample_to_value_idx=False,
        sample_to_value_idx=idx_behave_to_frame,
    )

    # Bins outside the visited range are meaningless -> NaN. Offset by one sample because
    # vrControl always reports the first sample at position 0 even when the rotary encoder
    # has a built-up buffer.
    position_bin_per_trial = [position_bin[trial_numbers == tnum] for tnum in range(session.num_trials)]
    first_valid_bin = [np.min(bpb[1:] if len(bpb) > 1 else bpb) for bpb in position_bin_per_trial]
    last_valid_bin = [np.max(bpb) for bpb in position_bin_per_trial]
    occmap = replace_missing_data(occmap, first_valid_bin, last_valid_bin)
    speedmap = replace_missing_data(speedmap, first_valid_bin, last_valid_bin)

    if smooth_width is not None:
        kernel = get_gauss_kernel(edge2center(dist_edges), smooth_width)
        idxnan = np.isnan(occmap)
        occmap[idxnan] = 0
        speedmap[idxnan] = 0
        occmap = convolve_toeplitz(occmap, kernel, axis=1)
        speedmap = convolve_toeplitz(speedmap, kernel, axis=1)
        occmap[idxnan] = np.nan
        speedmap[idxnan] = np.nan

    return correct_map(occmap, speedmap)


def reward_pattern_mismatch(session: B2Session) -> str | None:
    """Check a session's reward geometry against the canonical ``ENV_REWARD_MAP``.

    The feature windows are sized from ``ENV_REWARD_MAP``, so a session that rewards
    somewhere else would break the analysis' central claim -- that the decoder only ever
    sees pre-reward speed. Such sessions are filtered out rather than silently analysed.
    At least one session is known to deviate (an ATL028 session rewards environment 3 at
    150 cm rather than the usual 100 cm).

    Because the windows are derived from this map, a session that matches is guaranteed
    reward-free by construction.

    Parameters
    ----------
    session : B2Session

    Returns
    -------
    str or None
        A human-readable reason for the mismatch, or None if the session conforms.
    """
    env_length = np.unique(np.asarray(session.env_length))
    if env_length.size != 1:
        return f"non-unique env_length {env_length.tolist()}"
    if not np.isclose(float(env_length[0]), REFERENCE_ENV_LENGTH_CM):
        return f"env_length {float(env_length[0]):.2f} != reference {REFERENCE_ENV_LENGTH_CM:.2f}"

    reward_positions, halfwidths = environmentRewardZone(session)
    for env, position, halfwidth in zip(session.environments, reward_positions, halfwidths):
        env = int(env)
        if env < 0:
            continue
        if env not in ENV_REWARD_MAP:
            return f"environment {env} is not in ENV_REWARD_MAP"
        reward_start = float(position) - float(halfwidth)
        expected = ENV_REWARD_MAP[env]
        if not np.isclose(reward_start, expected):
            return f"environment {env} rewards from {reward_start:.2f} cm, expected {expected:.2f} cm"
    return None


def _mean_curve(speedmap: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.floating]:
    """Trial-averaged speed curve over the masked trials.

    A position bin that no masked trial reached has no data, and averages to NaN. That is
    the correct answer, so numpy's "Mean of empty slice" warning is suppressed rather than
    surfaced -- it fires routinely (aborted trials are all-NaN, and an environment may
    have only a handful of first-of-block trials).

    Parameters
    ----------
    speedmap : np.ndarray
        Shape ``(trials, bins)``.
    mask : np.ndarray
        Boolean trial selector, shape ``(trials,)``.

    Returns
    -------
    np.ndarray
        Shape ``(bins,)``, NaN where the masked trials have no coverage.
    """
    if not np.any(mask):
        return np.full(speedmap.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(speedmap[mask], axis=0)


def _block_starts(trial_environment: npt.NDArray[np.integer]) -> npt.NDArray[np.bool_]:
    """Boolean mask, True at the first trial of each contiguous environment run.

    Blocks of length one count. Trial 0 is always a block start.

    Parameters
    ----------
    trial_environment : np.ndarray
        Environment label per trial, shape ``(num_trials,)``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(num_trials,)``.
    """
    mask = np.zeros(len(trial_environment), dtype=bool)
    if len(trial_environment) == 0:
        return mask
    mask[0] = True
    mask[1:] = np.diff(trial_environment) != 0
    return mask


def _decode(
    x_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.integer],
    x_test: npt.NDArray[np.floating],
    y_test: npt.NDArray[np.integer],
    regularization: float,
) -> dict:
    """Fit a multinomial logistic regression and score it on a held-out set.

    Features are z-scored using training statistics only. The confusion matrix is built on
    the canonical ``ENV_SLOTS`` axis, so it is comparable across splits and sessions
    regardless of which environments a session actually ran.

    Parameters
    ----------
    x_train, x_test : np.ndarray
        Speed features, shape ``(trials, bins)``.
    y_train, y_test : np.ndarray
        Environment label per trial.
    regularization : float
        Inverse regularization strength (sklearn's ``C``).

    Returns
    -------
    dict
        ``acc_train``, ``acc_test``, ``bal_acc_test``, ``chance``, ``confusion``.
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.maximum(std, 1e-12)
    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std

    model = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        max_iter=1000,
    ).fit(x_train_z, y_train)

    y_pred = model.predict(x_test_z)
    # Chance = always guessing the most common training class, evaluated on the test set.
    _, train_counts = np.unique(y_train, return_counts=True)
    majority = np.unique(y_train)[np.argmax(train_counts)]
    return dict(
        acc_train=float(model.score(x_train_z, y_train)),
        acc_test=float(model.score(x_test_z, y_test)),
        bal_acc_test=float(balanced_accuracy_score(y_test, y_pred)),
        chance=float(np.mean(y_test == majority)),
        confusion=confusion_matrix(y_test, y_pred, labels=list(ENV_SLOTS)).astype(float),
    )


@dataclass
class _DecodeResult:
    """Everything the feature window contributes to :meth:`BehaviorSpeedEnvConfig.process`.

    ``random`` and ``block`` hold the dict returned by :func:`_decode`, or None when that
    split could not be run (too few trials, or fewer than two environments in the training
    set). A window that produced neither split is represented by ``None`` rather than by an
    instance of this class, and contributes no keys at all.
    """

    env_used: npt.NDArray[np.floating]
    n_trials: float
    n_trials_per_env: npt.NDArray[np.floating]
    n_window_bins: float
    window_start_bin: float
    window_stop_bin: float
    window_fraction: float
    random: dict | None
    block: dict | None


def _meta(result: _DecodeResult | None, field: str):
    """Window-level value, or None if the window was unusable (key then omitted)."""
    return None if result is None else getattr(result, field)


def _metric(result: _DecodeResult | None, split: str, field: str):
    """Decoding metric for one split, or None if the window or split did not run."""
    if result is None:
        return None
    metrics = getattr(result, split)
    return None if metrics is None else metrics[field]


@dataclass(frozen=True)
class BehaviorSpeedEnvConfig(AnalysisConfigBase):
    """Decode environment identity from pre-reward running speed.

    Parameters
    ----------
    num_bins : int
        Number of position bins spanning the whole track. Bins are a fixed *fraction* of
        track, so bin index is comparable across sessions with different track lengths.
    speed_threshold : float
        Behavioural samples at or below this speed are excluded from the speed map.
        ``-np.inf`` (default) keeps everything, including stopping behaviour, which is
        itself likely to be environment-informative.
    smooth_width : float or None
        Gaussian smoothing width in cm applied along position.
    regularization : float
        Inverse regularization strength for the logistic regression (sklearn's ``C``).
    split_seed : int
        Seed applied before the random trial split. ``cross_validate_trials`` takes no
        seed and draws from global numpy state, so this is set externally.
    """

    # v2: environment-indexed results moved to the canonical ENV_SLOTS axis (v1 used the
    # session's own sorted environments, so column j meant different environments in
    # different sessions and any cross-session average mixed them).
    # v3: dropped the second ("w50") feature window; the single pre-reward window now emits
    # unsuffixed keys (e.g. acc_test_random rather than acc_test_w25_random).
    schema_version: str = "v3"
    data_config_name: str = "default"  # unused; required by AnalysisConfigBase

    num_bins: int = 100
    speed_threshold: float = -np.inf
    smooth_width: float | None = 5.0
    regularization: float = 1.0
    split_seed: int = 0

    display_name: ClassVar[str] = "behavior_speed_env"

    @staticmethod
    def _param_grid() -> dict:
        return {
            "num_bins": [50, 100],
            "regularization": [0.1, 1.0],
            "speed_threshold": [-np.inf, 1.0],
        }

    def validate(self):
        # env_length is session-specific; 200 cm is the colony's shortest track and so
        # gives the coarsest bins / fewest features, making this the strict check.
        edges = np.linspace(0, 200.0, self.num_bins + 1)
        start, stop = _window_bins(edges, WINDOW_FRACTION)
        if stop - start < 2:
            raise ValueError(
                f"num_bins={self.num_bins} leaves < 2 features for the feature window "
                f"(fraction={WINDOW_FRACTION}, bins [{start}, {stop})); decoding would be meaningless."
            )

    def summary(self) -> str:
        return "_".join(
            [
                self.display_name,
                f"num_bins={self.num_bins}",
                f"speed_threshold={self.speed_threshold}",
                f"smooth_width={self.smooth_width}",
                f"regularization={self.regularization}",
                f"split_seed={self.split_seed}",
                self.schema_version,
            ]
        )

    def _run_decode(
        self,
        speedmap: npt.NDArray[np.floating],
        dist_edges: npt.NDArray[np.floating],
        trial_env: npt.NDArray[np.integer],
        is_block_start: npt.NDArray[np.bool_],
    ) -> _DecodeResult | None:
        """Run both splits on the pre-reward feature window.

        Returns None (contributing no keys) if the window is unusable: too few trials
        covering it, or fewer than two environments.
        """
        start_bin, stop_bin = _window_bins(dist_edges, WINDOW_FRACTION)

        # A trial is usable if the mouse covered the whole feature window.
        features = speedmap[:, start_bin:stop_bin]
        keep = ~np.any(np.isnan(features), axis=1)
        if np.sum(keep) < 4:
            return None

        env_labels = np.unique(trial_env[keep])
        if len(env_labels) < 2:
            return None

        x = features[keep]
        y = trial_env[keep]
        block_start = is_block_start[keep]

        # Random split: stratified 50/50 by environment. cross_validate_trials shuffles
        # from global numpy state, so seed immediately before the call.
        np.random.seed(self.split_seed)
        folds = cross_validate_trials(y, [1, 1])
        idx_train = np.array(folds[0], dtype=int)
        idx_test = np.array(folds[1], dtype=int)
        random_metrics = None
        if len(np.unique(y[idx_train])) >= 2 and len(idx_test) > 0:
            random_metrics = _decode(x[idx_train], y[idx_train], x[idx_test], y[idx_test], self.regularization)

        # Block split: train on later trials of each block, test on first trials.
        idx_train = np.flatnonzero(~block_start)
        idx_test = np.flatnonzero(block_start)
        block_metrics = None
        if len(idx_test) > 0 and len(idx_train) > 0 and len(np.unique(y[idx_train])) >= 2:
            block_metrics = _decode(x[idx_train], y[idx_train], x[idx_test], y[idx_test], self.regularization)

        if random_metrics is None and block_metrics is None:
            return None

        return _DecodeResult(
            env_used=np.array([float(e in env_labels) for e in ENV_SLOTS]),
            n_trials=float(len(y)),
            n_trials_per_env=np.array([np.sum(y == e) for e in ENV_SLOTS], dtype=float),
            n_window_bins=float(stop_bin - start_bin),
            window_start_bin=float(start_bin),
            window_stop_bin=float(stop_bin),
            window_fraction=float(WINDOW_FRACTION),
            random=random_metrics,
            block=block_metrics,
        )

    def process(self, session: B2Session, registry: PopulationRegistry) -> dict:
        """Run the analysis on one session.

        ``registry`` is ignored: this analysis uses only behavioural data, and the
        registry's splits are over frames rather than trials.

        Returns
        -------
        dict
            Empty for excluded mice and for sessions whose reward geometry does not match
            ``ENV_REWARD_MAP`` (see :func:`reward_pattern_mismatch`). For
            single-environment sessions, only the speed curves and metadata (decoding keys
            are omitted, which the aggregator renders as NaN). Otherwise adds per-window,
            per-split decoding metrics.
        """
        if session.mouse_name.startswith(EXCLUDED_MOUSE_PREFIXES):
            return {}
        if reward_pattern_mismatch(session) is not None:
            return {}

        env_length = float(np.unique(np.asarray(session.env_length))[0])
        dist_edges = np.linspace(0, env_length, self.num_bins + 1)
        speedmap = _speed_map(session, dist_edges, self.speed_threshold, self.smooth_width)

        trial_env = np.asarray(session.trial_environment)
        is_block_start = _block_starts(trial_env)

        # env < 0 marks invalid trials.
        valid = trial_env >= 0
        speedmap = speedmap[valid]
        trial_env = trial_env[valid]
        is_block_start = is_block_start[valid]

        if len(trial_env) == 0:
            # A few sessions have no valid trials at all; there is nothing to report.
            return {}

        # Emit on the canonical ENV_SLOTS axis so column j always means the same
        # environment, and averaging across sessions or mice stays meaningful.
        speed_curve_all = np.stack([_mean_curve(speedmap, trial_env == e) for e in ENV_SLOTS])
        speed_curve_first = np.stack([_mean_curve(speedmap, (trial_env == e) & is_block_start) for e in ENV_SLOTS])

        decoded = self._run_decode(speedmap, dist_edges, trial_env, is_block_start)

        # Every key this analysis can emit is written out literally below. Keys whose value
        # is None are dropped (see the return): a window or split that could not run
        # contributes nothing, and the aggregator renders the gap as NaN.
        result: dict = {
            # -- session-level, always present --
            "speed_curve_all": speed_curve_all,  # (n_env_slots, num_bins) cm/s, all trials
            "speed_curve_first": speed_curve_first,  # (n_env_slots, num_bins) first-of-block trials only
            "environments": np.array(ENV_SLOTS, dtype=float),  # canonical env axis for every per-env key
            "n_trials_per_env": np.array([np.sum(trial_env == e) for e in ENV_SLOTS], dtype=float),
            "n_blocks_per_env": np.array([np.sum((trial_env == e) & is_block_start) for e in ENV_SLOTS], dtype=float),
            "dist_edges": dist_edges,  # (num_bins + 1,) cm
            "dist_fraction_centers": edge2center(dist_edges) / env_length,  # (num_bins,) shared across sessions
            "env_length": env_length,
            "num_bins": float(self.num_bins),
            # -- pre-reward feature window (all environments, stops at env 4's reward) --
            "env_used": _meta(decoded, "env_used"),
            "n_trials": _meta(decoded, "n_trials"),
            "n_trials_per_env_window": _meta(decoded, "n_trials_per_env"),
            "n_window_bins": _meta(decoded, "n_window_bins"),
            "window_start_bin": _meta(decoded, "window_start_bin"),
            "window_stop_bin": _meta(decoded, "window_stop_bin"),
            "window_fraction": _meta(decoded, "window_fraction"),
            "acc_train_random": _metric(decoded, "random", "acc_train"),
            "acc_test_random": _metric(decoded, "random", "acc_test"),
            "bal_acc_test_random": _metric(decoded, "random", "bal_acc_test"),
            "chance_random": _metric(decoded, "random", "chance"),
            "confusion_random": _metric(decoded, "random", "confusion"),
            "acc_train_block": _metric(decoded, "block", "acc_train"),
            "acc_test_block": _metric(decoded, "block", "acc_test"),
            "bal_acc_test_block": _metric(decoded, "block", "bal_acc_test"),
            "chance_block": _metric(decoded, "block", "chance"),
            "confusion_block": _metric(decoded, "block", "confusion"),
        }
        return {key: value for key, value in result.items() if value is not None}
