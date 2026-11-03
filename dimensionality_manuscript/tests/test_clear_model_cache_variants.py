"""Scoped cache clearing must include trial/legacy keys and preserve other models."""

from types import SimpleNamespace

import pytest

from dimensionality_manuscript.scripts.clear_model_caches import clear_model_cache_variants, find_model_cache_variants


@pytest.fixture
def cache_files(tmp_path):
    paths = SimpleNamespace(hyperparameter_path=tmp_path / "hyperparameters", score_path=tmp_path / "scores")
    model = "internal_placefield_1d_structured_gain_residual"
    for folder in (paths.hyperparameter_path, paths.score_path):
        folder.mkdir()
        for name in (
            f"{model}_mouse.date.1_sigrebase_trialhash_train_validation_golden_scaling.joblib",
            f"{model}_mouse.date.1_sigrebase_cvhash_train_validation_golden_scaling_regularized_v2.joblib",
            "internal_placefield_1d_structured_gain_mouse.date.1.joblib",
            "internal_placefield_1d_structured_gain_sqrt_mouse.date.1.joblib",
            "external_placefield_1d_structured_gain_residual_mouse.date.1.joblib",
            "rrr_mouse.date.1.joblib",
            f"unrelated_{model}_mouse.date.1.joblib",
        ):
            (folder / name).write_bytes(b"test cache")
        (folder / "nested").mkdir()
        (folder / "nested" / f"{model}_nested.joblib").write_bytes(b"not scanned")
    return model, paths


def test_all_variants_preview_changes_nothing(cache_files):
    model, paths = cache_files
    before = sorted(paths.hyperparameter_path.parent.rglob("*.joblib"))
    counts = clear_model_cache_variants([model], registry_paths=paths, dry_run=True)
    assert counts == {"hyperparameters": 2, "scores": 2}
    assert sorted(paths.hyperparameter_path.parent.rglob("*.joblib")) == before


def test_all_variants_deletes_only_requested_model(cache_files):
    model, paths = cache_files
    planned = find_model_cache_variants([model], paths)
    selected = {p for files in planned[model].values() for p in files}
    before = set(paths.hyperparameter_path.parent.rglob("*.joblib"))
    counts = clear_model_cache_variants([model], registry_paths=paths)
    assert counts == {"hyperparameters": 2, "scores": 2}
    assert set(paths.hyperparameter_path.parent.rglob("*.joblib")) == before - selected


def test_base_model_does_not_match_registered_variants(cache_files):
    _, paths = cache_files
    base = "internal_placefield_1d_structured_gain"
    planned = find_model_cache_variants([base], paths)
    assert all(len(files) == 1 for files in planned[base].values())


def test_scores_only_preserves_hyperparameters(cache_files):
    model, paths = cache_files
    before = set(paths.hyperparameter_path.glob("*.joblib"))
    counts = clear_model_cache_variants([model], registry_paths=paths, clear_hyperparameters=False)
    assert counts == {"hyperparameters": 0, "scores": 2}
    assert set(paths.hyperparameter_path.glob("*.joblib")) == before


def test_unknown_models_are_rejected_before_deletion(cache_files):
    _, paths = cache_files
    with pytest.raises(ValueError, match="Unknown models"):
        clear_model_cache_variants(["typo"], registry_paths=paths)
