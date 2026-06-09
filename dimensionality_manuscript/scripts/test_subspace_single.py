"""Debug script for CovCovSubspace linalg.svd crash on Myriad.

Reproduces the fit step-by-step with contiguity diagnostics,
then verifies the .contiguous() fix actually resolves it.

Usage
-----
    qrsh -l h_rt=1:00:00,mem=16G
    cd ~/vrAnalysis
    python -m dimensionality_manuscript.scripts.test_subspace_single \
        --sessions-file sessions.json \
        --session ATL076.2025-08-19.704

    # Or test first N sessions:
    python -m dimensionality_manuscript.scripts.test_subspace_single \
        --sessions-file sessions.json --n-sessions 3
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from dimilibi import PCA
from vrAnalysis.processors.placefields import get_placefield

from dimensionality_manuscript.registry import RegistryPaths, PopulationRegistry
from dimensionality_manuscript.regression_models.hyperparameters import PlaceFieldHyperparameters
from dimensionality_manuscript.scripts.run import collect_sessions_from_file
from dimensionality_manuscript.subspace_analysis.subspaces import CovCovSubspace
from dimensionality_manuscript import SubspaceConfig

REGISTRY_PATHS = RegistryPaths()


def _cov_or_gram(X: torch.Tensor, centered: bool) -> torch.Tensor:
    if centered:
        return torch.cov(X)
    return X @ X.T / (X.shape[1] - 1)


def _diagnose_tensor(name: str, t: torch.Tensor) -> None:
    print(
        f"  {name}: shape={tuple(t.shape)} contiguous={t.is_contiguous()} "
        f"nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} "
        f"dtype={t.dtype}"
    )


def run_step_by_step(session, registry, spks_type="oasis", num_bins=100, smooth_width=5.0, centered=False):
    """Reproduce CovCovSubspace.fit() step-by-step with diagnostics."""
    print(f"\n--- Step-by-step fit: {session.session_uid} ---")

    model = CovCovSubspace(registry, centered=centered, match_dimensions=True, autosave=False)
    hyps = PlaceFieldHyperparameters(num_bins=num_bins, smooth_width=smooth_width)

    train_data, frame_behavior_train, num_neurons = model.get_session_data(session, spks_type, "train", use_cell_split=False)
    _diagnose_tensor("train_data (raw)", train_data)

    dist_edges = model._get_placefield_dist_edges(session, hyps)
    placefield = get_placefield(
        train_data.T.numpy(),
        frame_behavior_train,
        dist_edges=dist_edges,
        average=True,
        smooth_width=hyps.smooth_width,
    )

    placefield_extended_raw = torch.tensor(placefield.flattened()).T
    _diagnose_tensor("placefield_extended (after .T, before filter)", placefield_extended_raw)

    placefield_extended, train_data_filtered = model._check_and_filter_nans(placefield_extended_raw, train_data)
    _diagnose_tensor("placefield_extended (after NaN filter)", placefield_extended)
    _diagnose_tensor("train_data (after NaN filter)", train_data_filtered)

    num_components = model._compute_num_components(model.max_components, train_data_filtered.shape, placefield_extended.shape)
    print(f"  num_components={num_components}")

    # Test 1: original (non-contiguous path)
    print("\n  [TEST 1] PCA.fit() on original tensors (may crash on Myriad)...")
    try:
        pca = PCA(num_components=num_components, center=centered).fit(placefield_extended)
        print("  PASS: PCA.fit() on original placefield_extended succeeded")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 2: with .contiguous()
    print("\n  [TEST 2] PCA.fit() on .contiguous() tensors (proposed fix)...")
    try:
        pca = PCA(num_components=num_components, center=centered).fit(placefield_extended.contiguous())
        print("  PASS: PCA.fit() on placefield_extended.contiguous() succeeded")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 3: full fit via model
    print("\n  [TEST 3] Full CovCovSubspace.fit() via model...")
    try:
        subspace = model.fit(session, spks_type=spks_type, hyperparameters=hyps)
        print("  PASS: model.fit() succeeded")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()


def run_via_config(session, registry, config):
    """Run via SubspaceConfig.process() — exactly what the batch job does."""
    label = f"{session.session_uid} / {config.summary()}"
    print(f"\n  [CONFIG] {label}")
    try:
        result = config.process(session, registry)
        print(f"  PASS: keys={list(result.keys())}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Debug CovCovSubspace SVD crash on Myriad")
    parser.add_argument("--sessions-file", type=Path, required=True)
    parser.add_argument("--session", default=None, help="session_uid to test")
    parser.add_argument("--n-sessions", type=int, default=1)
    parser.add_argument("--spks-type", default="oasis")
    parser.add_argument("--skip-stepwise", action="store_true", help="Skip step-by-step and only run configs")
    args = parser.parse_args()

    sessions = collect_sessions_from_file(args.sessions_file)
    if args.session:
        sessions = [s for s in sessions if s.session_uid == args.session]
        if not sessions:
            print(f"Session {args.session!r} not found in {args.sessions_file}", file=sys.stderr)
            sys.exit(1)
    else:
        sessions = sessions[: args.n_sessions]

    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version:   {np.__version__}")
    print(f"Sessions: {[s.session_uid for s in sessions]}")

    # One covcov config to test end-to-end
    test_config = SubspaceConfig(subspace_name="covcov_subspace", spks_type=args.spks_type, num_bins=100, smooth_width=5.0)
    data_config_name = test_config.data_config_name
    registry = PopulationRegistry(registry_params=test_config.data_config.to_registry_params())

    n_ok = n_fail = 0
    for session in sessions:
        if not args.skip_stepwise:
            run_step_by_step(session, registry, spks_type=args.spks_type)

        ok = run_via_config(session, registry, test_config)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{n_ok} passed, {n_fail} failed.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
