"""Real-session speed/accuracy check for ``StimspaceSVCAConfig`` (see ``configs/stimspace.py``).

This is a local/Myriad benchmark, not a correctness gate: it needs an actual mounted data
store (``paths.toml`` + the sessions below present in the local database), so it is skipped
outright wherever that isn't available -- e.g. CI, or a machine set up differently than the
one it was authored on. Windows dev boxes and the Linux/Myriad cluster both hit this same
skip check; whichever has the data configured will run it.

Purpose: benchmark ``dimilibi.SVCA``'s full ``np.linalg.svd`` fit against ``SVCA(truncated=True)``
(randomized SVD) on the same source/target gram matrices the real pipeline fits, to keep checking
that the truncated path -- which ``StimspaceSVCAConfig`` now uses for every SVCA fit -- stays both
accurate (top singular values/vectors close to the full solve) and faster.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if not (_REPO_ROOT / "paths.toml").exists():
    pytest.skip("paths.toml not configured on this machine; skipping real-data benchmark.", allow_module_level=True)

torch = pytest.importorskip("torch")

from dimilibi import SVCA  # noqa: E402
from vrAnalysis.sessions import B2Session  # noqa: E402
from dimensionality_manuscript.registry import PopulationRegistry, get_activity_parameters  # noqa: E402
from dimensionality_manuscript.configs.stimspace import StimspaceSVCAConfig  # noqa: E402

# Matches StimspaceSVCAConfig's default activity_parameters_name so the benchmark scales
# activity the same way the real pipeline does.
ACTIVITY_PARAMETERS_NAME = "std"

# A couple of sessions used as running examples elsewhere in the manuscript notebooks.
# Swap these for sessions known to exist in your local database if these aren't available.
SESSION_SPECS = [
    ("ATL022", "2023-04-12", "701"),
    ("CR_Hippocannula6", "2022-08-30", "701"),
]

MAX_COMPONENTS = 300

# Fast mode (default on) truncates the direct SVD benchmark to a small ROI/timepoint subset --
# the full-session SVD (both full and randomized) is the actual cost driver, so this alone gives
# most of the speedup. It also skips the full StimspaceSVCAConfig.process() smoke test, which
# can't be safely restricted to a subset without touching internal time-split/placefield
# bookkeeping. Set STIMSPACE_TEST_FAST=0 to run the full, slow, unrestricted versions instead.
FAST_MODE = os.environ.get("STIMSPACE_TEST_FAST", "1") != "0"
FAST_MAX_ROIS = 500
FAST_MAX_TIMEPOINTS = 600


def _load_sessions() -> list[B2Session]:
    sessions = []
    for mouse, date, session_id in SESSION_SPECS:
        try:
            sessions.append(B2Session.create(mouse, date, session_id))
        except Exception as exc:  # pragma: no cover - depends on local data availability
            pytest.skip(f"Could not load session {mouse}/{date}/{session_id} on this machine: {exc}")
    return sessions


@pytest.fixture(scope="module")
def real_sessions() -> list[B2Session]:
    return _load_sessions()


@pytest.fixture(scope="module")
def registry() -> PopulationRegistry:
    """Registry built from the config's own data config, not the bare default.

    ``StimspaceSVCAConfig`` declares ``data_config_name="even"``, whose four time groups are
    equal-sized. A bare ``PopulationRegistry()`` silently uses the ``"default"`` preset instead,
    whose ``time_split_relative_size`` is ``(4, 4, 1, 1)`` -- that makes the ``train``/``not_train``
    halves an 80/20 split rather than 50/50 and changes every fold size, so the benchmark would not
    be measuring what the pipeline actually runs.
    """
    return PopulationRegistry(registry_params=StimspaceSVCAConfig().data_config.to_registry_params())


def test_stimspace_svca_config_runs_and_is_finite(real_sessions, registry):
    """End-to-end smoke test: the config completes and returns sane spectra.

    Skipped in fast mode (default) since it runs the full production pipeline (placefields
    across 4 folds x however many environments, per session) and can't be safely shrunk to a
    ROI/timepoint subset. Set STIMSPACE_TEST_FAST=0 to run it.
    """
    if FAST_MODE:
        pytest.skip("Fast mode: skipping slow full-pipeline smoke test (set STIMSPACE_TEST_FAST=0 to run it).")
    cfg = StimspaceSVCAConfig()
    for session in real_sessions:
        t0 = time.perf_counter()
        result = cfg.process(session, registry)
        elapsed = time.perf_counter() - t0
        print(f"\n[stimspace_svca] {session.session_name}: process() took {elapsed:.2f}s")

        for key in ("ff", "ss", "ff_res", "ss_pred"):
            spectrum = np.asarray(result[key])
            assert spectrum.size > 0
            assert np.all(np.isfinite(spectrum)), f"{key} contains non-finite values"
            # These are *cross-validated* shared variances, not singular values, so they are not
            # monotonically decreasing and must not be asserted to be. SVCA orders its components
            # by the singular values of the fit fold's gram matrix; the held-out fold's shared
            # variance along those components need not follow that order. Measured on ATL022 the
            # averaged ff spectrum inverts at component 5 (11.21 -> 16.34) and individual fold
            # pairs invert as early as component 0 -- identically with truncated=True and with the
            # full LAPACK solve, so this is inherent to the estimator, not an SVD artifact. Past
            # the noise floor the tail straddles zero as well.
            #
            # What should hold is that the leading components carry more shared variance than the
            # tail, which is the property the spectra are actually read for.
            lead = spectrum[: max(1, spectrum.size // 20)]
            tail = spectrum[-max(1, spectrum.size // 20) :]
            assert lead.mean() > tail.mean(), f"{key} leading block does not dominate its tail"


def test_truncated_svd_matches_full_svd_and_is_not_slower(real_sessions, registry):
    """Benchmarks full vs. truncated (randomized) SVD on real source/target gram matrices.

    Uses each session's 'full' time-split source/target activity -- the same shape of
    problem ``StimspaceSVCAConfig`` fits internally many times per session -- to directly
    compare ``dimilibi.SVCA(truncated=False)`` (current default) against
    ``SVCA(truncated=True)``.
    """
    ap = get_activity_parameters(ACTIVITY_PARAMETERS_NAME)
    for session in real_sessions:
        population, _ = registry.get_population(session, spks_type="sigrebase")
        source_data, target_data = population.get_split_data(
            registry.time_split["full"],
            scale=ap.scale,
            scale_type=ap.scale_type,
            pre_split=ap.presplit,
        )

        if FAST_MODE:
            source_data = source_data[:FAST_MAX_ROIS]
            target_data = target_data[:FAST_MAX_ROIS]
            # Raw activity has genuine NaNs at dropped/invalid frames, so slicing the first N
            # columns outright can land on a NaN-heavy window (breaks sklearn's randomized_svd,
            # which checks finiteness unlike np.linalg.svd). Take the first N *finite* columns.
            valid_cols = torch.isfinite(source_data).all(dim=0) & torch.isfinite(target_data).all(dim=0)
            valid_idx = torch.nonzero(valid_cols, as_tuple=False).squeeze(1)[:FAST_MAX_TIMEPOINTS]
            source_data = source_data[:, valid_idx]
            target_data = target_data[:, valid_idx]

        num_components = min(MAX_COMPONENTS, source_data.shape[0], target_data.shape[0])
        if num_components < 5:
            pytest.skip(f"Too few neurons in {session.session_name} for a meaningful SVD benchmark.")

        t0 = time.perf_counter()
        full = SVCA(centered=True, num_components=num_components, truncated=False).fit(source_data, target_data)
        full_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        trunc = SVCA(centered=True, num_components=num_components, truncated=True).fit(source_data, target_data)
        trunc_time = time.perf_counter() - t0

        print(
            f"\n[svd benchmark{' (fast)' if FAST_MODE else ''}] {session.session_name}: "
            f"n_source={source_data.shape[0]}, n_target={target_data.shape[0]}, k={num_components}, "
            f"full={full_time:.3f}s, truncated={trunc_time:.3f}s"
        )

        # Accuracy: leading singular values from the randomized solve should closely track the
        # full solve. Randomized SVD is deliberately biased toward the dominant components --
        # trailing/near-noise-floor singular values naturally diverge in *relative* terms (a
        # tiny absolute gap against a near-zero denominator), which isn't a correctness issue
        # since only the leading spectrum is ever interpreted downstream. So we only check the
        # leading fraction of components, not a median over the full (near-full-rank) spectrum.
        full_s = full.S.numpy()
        trunc_s = trunc.S.numpy()
        rel_error = np.abs(full_s - trunc_s) / np.clip(np.abs(full_s), 1e-8, None)
        leading = max(1, num_components // 4)
        assert np.median(rel_error[:leading]) < 0.01, (
            f"median relative error over leading {leading} components too high: {np.median(rel_error[:leading])}"
        )
        top_k = min(20, num_components)
        assert rel_error[:top_k].max() < 0.05, f"leading singular values diverge: {rel_error[:top_k]}"

        # Speed: truncated/randomized SVD should not be meaningfully slower than the full
        # solve on realistically sized data (some slack for machine noise / small matrices).
        assert trunc_time <= full_time * 1.5 + 0.05
