"""Survey reward-zone geometry across the colony for the behavior_speed_env analysis.

The analysis decodes environment identity from the speed profile *before* the reward zone,
so it needs to know where each environment rewards. This script is where
``ENV_REWARD_MAP`` (and the window fractions derived from it) comes from, and is what to
re-run when the colony changes or a session is suspected of deviating.

What it established, as of the last run over 149 imaging sessions:

- Two disjoint cohorts. The 12 ``ATL*`` mice run a 200 cm track with environments 1/3/4,
  rewarding from 150/100/50 cm (fractions 0.75/0.50/0.25). The 2 ``CR_*`` mice run a
  245 cm track with environments 1/2 at fractions 0.825/0.575.
- The cohorts share only the *label* "environment 1", at different fractions, so the label
  does not carry across them. The ``CR_*`` mice are excluded by the analysis.
- Environment 4 rewards earliest and therefore caps the all-environment window at 0.25;
  dropping it lets the window reach 0.50.
- Exactly one session deviates: ATL028.2023-08-04.701 rewards environment 3 at 150 cm
  rather than 100 cm. The analysis filters it out.

Usage
-----
    conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.derive_prereward_cutoff
    conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.derive_prereward_cutoff --experiment-type "Blender VR"
"""

import argparse
import traceback

import numpy as np
import pandas as pd

from vrAnalysis.database import get_database
from vrAnalysis.helpers import environmentRewardZone

from dimensionality_manuscript.configs.behavior_speed_env import REFERENCE_ENV_LENGTH_CM


def collect(experiment_type: str | None) -> pd.DataFrame:
    """One row per (session, environment) with reward geometry in cm and as a fraction."""
    sessiondb = get_database("vrSessions")
    kwargs = dict(imaging=True)
    if experiment_type is not None:
        kwargs["experimentType"] = experiment_type
    sessions = list(sessiondb.iter_sessions(**kwargs))
    print(f"Scanning {len(sessions)} sessions...\n")

    rows = []
    n_failed = 0
    for session in sessions:
        try:
            env_length = session.env_length
            env_length = np.unique(np.asarray(env_length))
            if env_length.size != 1:
                print(f"[skip] {session.session_uid}: non-unique env_length {env_length}")
                continue
            env_length = float(env_length[0])

            rew_pos, rew_hw = environmentRewardZone(session)
            for env, pos, hw in zip(session.environments, rew_pos, rew_hw):
                rows.append(
                    dict(
                        mouse=session.mouse_name,
                        session_uid=session.session_uid,
                        env=int(env),
                        env_length=env_length,
                        reward_position=float(pos),
                        halfwidth=float(hw),
                        reward_start=float(pos) - float(hw),
                        reward_start_fraction=(float(pos) - float(hw)) / env_length,
                    )
                )
        except Exception as e:
            n_failed += 1
            print(f"[fail] {session.session_uid}: {e}")
            traceback.print_exc()
        finally:
            session.clear_cache()

    if n_failed:
        print(f"\n{n_failed} sessions failed.\n")
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    valid = df[df["env"] >= 0]
    print("=" * 78)
    print(f"{len(valid)} (session, env) pairs across {valid['mouse'].nunique()} mice, " f"{valid['session_uid'].nunique()} sessions")

    print("\n--- track lengths ---")
    print(valid.groupby("env_length")["mouse"].agg(["nunique", "count"]))
    print("\nmice per track length:")
    for length, grp in valid.groupby("env_length"):
        print(f"  {length:g} cm: {sorted(grp['mouse'].unique())}")

    print("\n--- THE KEY CHECK: reward_start_fraction by (env, track length) ---")
    print("If fractional binning is valid, each env's fraction must agree across track lengths.")
    pivot = valid.pivot_table(
        index="env",
        columns="env_length",
        values="reward_start_fraction",
        aggfunc=["min", "max"],
    )
    print(pivot.to_string())

    lengths = sorted(valid["env_length"].unique())
    if len(lengths) > 1:
        print("\n  per-env spread across track lengths:")
        ok = True
        for env, grp in valid.groupby("env"):
            per_len = grp.groupby("env_length")["reward_start_fraction"].mean()
            if len(per_len) < 2:
                print(f"    env {env}: only present at {list(per_len.index)} — cannot compare")
                continue
            spread = float(per_len.max() - per_len.min())
            flag = "OK" if spread < 0.01 else "MISMATCH"
            if spread >= 0.01:
                ok = False
            print(f"    env {env}: spread={spread:.4f}  {flag}  ({dict(per_len.round(4))})")
        print(f"\n  => fractions {'AGREE' if ok else 'DO NOT AGREE'} across track lengths")
        if not ok:
            print("  !! Fractional binning does NOT rescue the cross-mouse comparison.")
            print("  !! Stop and reconsider before writing the config.")

    print("\n--- reward_start_fraction by env (pooled) ---")
    print(valid.groupby("env")["reward_start_fraction"].agg(["min", "mean", "max", "count"]).to_string())

    print("\n--- earliest-reward environment ---")
    per_env_min = valid.groupby("env")["reward_start_fraction"].min()
    earliest = per_env_min.idxmin()
    print(f"env with earliest reward: {earliest} (expected 4)")
    if earliest != 4:
        print("  NOTE: expectation was env 4 — worth a look.")

    print("\n--- ENV_REWARD_MAP (reward zone start, cm, on the reference track) ---")
    reference = valid[np.isclose(valid["env_length"], REFERENCE_ENV_LENGTH_CM)]
    print(f"reference track = {REFERENCE_ENV_LENGTH_CM:g} cm ({reference['session_uid'].nunique()} sessions)")
    modal = reference.groupby("env")["reward_start"].agg(lambda s: s.mode().iloc[0])
    print("\nENV_REWARD_MAP: dict[int, float] = {")
    for env, start in modal.items():
        print(f"    {env}: {start:g},")
    print("}")

    print("\n--- sessions deviating from that map (the analysis filters these out) ---")
    deviant = reference[~np.isclose(reference["reward_start"], reference["env"].map(modal))]
    if deviant.empty:
        print("  none")
    else:
        print(deviant[["mouse", "session_uid", "env", "reward_start"]].to_string(index=False))

    print("\n--- resulting windows ---")
    for excluded in ((), (4,)):
        included = modal.drop(labels=[e for e in excluded if e in modal.index])
        fraction = included.min() / REFERENCE_ENV_LENGTH_CM
        label = "all environments" if not excluded else f"excluding env {list(excluded)}"
        print(
            f"  {label:24s} -> fraction {fraction:.3f} " f"({fraction * REFERENCE_ENV_LENGTH_CM:.0f} cm, earliest reward at env {included.idxmin()})"
        )
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Survey reward-zone geometry and regenerate ENV_REWARD_MAP.")
    parser.add_argument("--experiment-type", default=None, help="filter sessions by experimentType")
    parser.add_argument("--csv", default=None, help="optional path to dump the per-(session, env) table")
    args = parser.parse_args()

    df = collect(args.experiment_type)
    if df.empty:
        print("No sessions collected.")
        return
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    report(df)


if __name__ == "__main__":
    main()
