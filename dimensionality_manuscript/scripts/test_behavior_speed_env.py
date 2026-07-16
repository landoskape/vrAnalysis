"""Verify the behavior_speed_env analysis against existing, tested code.

The config reimplements speed-map construction (``_speed_map``) rather than calling
``SpkmapProcessor.get_env_maps``, because ``get_env_maps`` discards trial identity and so
cannot support first-of-block detection. That reimplementation must be faithful, and this
script is the proof.

Checks
------
1. **Speed map cross-check (decisive).** With matching params, ``_speed_map`` must
   reproduce ``get_env_maps().speedmap`` exactly on the trials ``get_env_maps`` keeps.
   This validates both the map maths and the trial alignment.
2. **Block detection.** Prints ``trial_environment`` alongside the derived first-trial
   mask so transitions can be eyeballed.
3. **Reward-free window.** Re-checks *in physical cm* that the last feature bin of each
   window ends before the earliest reward zone of every environment used.
4. **process() smoke test.** Runs the analysis and prints the result dict.
5. **Determinism.** Runs process() twice and confirms the random-split accuracy is stable.

Usage
-----
    conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.test_behavior_speed_env
    conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.test_behavior_speed_env \
        --session ATL027.2023-07-27.701
"""

import argparse
import sys

import numpy as np

from vrAnalysis.database import get_database
from vrAnalysis.helpers import environmentRewardZone
from vrAnalysis.processors import spkmaps as SMPs

from dimensionality_manuscript.configs.behavior_speed_env import (
    EXCLUDED_MOUSE_PREFIXES,
    WINDOW_FRACTION,
    BehaviorSpeedEnvConfig,
    _block_starts,
    _speed_map,
    _window_bins,
)


def check_speed_map(session, atol: float) -> bool:
    """_speed_map must reproduce get_env_maps().speedmap on the trials it keeps."""
    params = SMPs.SpkmapParams(dist_step=1.0, speed_threshold=1.0, smooth_width=1.0)
    smp = SMPs.SpkmapProcessor(session, params=params)
    dist_edges = smp.dist_edges

    mine = _speed_map(session, dist_edges, speed_threshold=1.0, smooth_width=1.0)
    env_maps = smp.get_env_maps()

    # Reproduce get_env_maps' own trial filtering so rows can be matched up.
    raw = smp.get_processed_maps()
    idx_required = smp._idx_required_position_bins()
    full_trials = np.where(np.all(~np.isnan(raw.occmap[:, idx_required]), axis=1))[0]

    ok = True
    for i, env in enumerate(env_maps.environments):
        theirs = env_maps.speedmap[i]
        idx_env = np.where(np.take(np.isin(session.trial_environment, [env]), full_trials, axis=0))[0]
        mine_env = mine[full_trials[idx_env]]
        if mine_env.shape != theirs.shape:
            print(f"    env {env}: SHAPE MISMATCH mine={mine_env.shape} theirs={theirs.shape}")
            ok = False
            continue
        both = ~np.isnan(mine_env) & ~np.isnan(theirs)
        nan_agree = np.array_equal(np.isnan(mine_env), np.isnan(theirs))
        maxdiff = float(np.max(np.abs(mine_env[both] - theirs[both]))) if both.any() else 0.0
        # The maps are float32 and speeds are ~10-100 cm/s, so absolute agreement bottoms
        # out around 1e-5. Judge on relative tolerance.
        close = np.allclose(mine_env[both], theirs[both], rtol=1e-5, atol=atol)
        good = close and nan_agree
        ok &= good
        print(
            f"    env {env}: trials={theirs.shape[0]:3d} maxdiff={maxdiff:.3e} "
            f"nan_pattern={'MATCH' if nan_agree else 'DIFFER'} {'PASS' if good else 'FAIL'}"
        )
    return ok


def check_blocks(session) -> bool:
    trial_env = np.asarray(session.trial_environment)
    mask = _block_starts(trial_env)
    print(f"    trial_environment: {trial_env.tolist()}")
    print(f"    first-of-block:    {mask.astype(int).tolist()}")
    print(f"    {mask.sum()} blocks across {len(trial_env)} trials")

    expected = np.concatenate([[True], np.diff(trial_env) != 0]) if len(trial_env) else np.zeros(0, bool)
    ok = np.array_equal(mask, expected)
    # Every block start must differ from the preceding trial, and no non-start may.
    for i in range(1, len(trial_env)):
        if mask[i] != (trial_env[i] != trial_env[i - 1]):
            ok = False
    print(f"    block detection: {'PASS' if ok else 'FAIL'}")
    return ok


def check_window_is_reward_free(session, config: BehaviorSpeedEnvConfig) -> bool:
    """In physical cm: last feature bin must end before the earliest reward zone."""
    env_length = float(np.unique(np.asarray(session.env_length))[0])
    dist_edges = np.linspace(0, env_length, config.num_bins + 1)
    rew_pos, rew_hw = environmentRewardZone(session)

    _, stop_bin = _window_bins(dist_edges, WINDOW_FRACTION)
    window_end_cm = float(dist_edges[stop_bin])

    ok = True
    for env, pos, hw in zip(session.environments, rew_pos, rew_hw):
        if env < 0:
            continue
        reward_start = float(pos) - float(hw)
        good = window_end_cm <= reward_start
        ok &= good
        print(
            f"    env {env}: window ends {window_end_cm:6.2f} cm, "
            f"reward starts {reward_start:6.2f} cm  {'PASS' if good else 'FAIL — OVERLAPS REWARD'}"
        )
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify behavior_speed_env.")
    parser.add_argument("--session", default=None)
    parser.add_argument("--n-sessions", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    sessiondb = get_database("vrSessions")
    sessions = list(sessiondb.iter_sessions(imaging=True))
    sessions = [s for s in sessions if not s.mouse_name.startswith(EXCLUDED_MOUSE_PREFIXES)]
    if args.session:
        sessions = [s for s in sessions if s.session_uid == args.session]
        if not sessions:
            print(f"session {args.session!r} not found", file=sys.stderr)
            sys.exit(1)
    else:
        # Prefer sessions with several environments so decoding actually runs.
        sessions = sorted(sessions, key=lambda s: -len(s.environments))[: args.n_sessions]

    config = BehaviorSpeedEnvConfig()
    all_ok = True
    for session in sessions:
        print(f"\n=== {session.session_uid} (envs={list(session.environments)}) ===")
        print("  [1] speed map vs get_env_maps:")
        all_ok &= check_speed_map(session, args.atol)
        print("  [2] block detection:")
        all_ok &= check_blocks(session)
        print("  [3] window is reward-free (physical cm):")
        all_ok &= check_window_is_reward_free(session, config)

        print("  [4] process():")
        result = config.process(session, None)
        for key in sorted(result):
            value = result[key]
            if isinstance(value, np.ndarray) and value.size > 6:
                print(f"    {key}: ndarray{value.shape}")
            else:
                print(f"    {key}: {value}")

        print("  [5] determinism:")
        result2 = config.process(session, None)
        keys = [k for k in result if k.startswith("acc_") and "random" in k]
        for key in keys:
            same = result[key] == result2[key]
            all_ok &= same
            print(f"    {key}: {result[key]:.6f} vs {result2[key]:.6f} {'PASS' if same else 'FAIL'}")
        if not keys:
            print("    (no random-split keys; single-env session?)")
        session.clear_cache()

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
