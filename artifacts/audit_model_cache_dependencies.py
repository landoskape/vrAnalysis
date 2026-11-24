"""Read-only production audit; cache writes are confined to a temporary directory."""
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import pandas as pd

from dimensionality_manuscript.regression_models.base import ActivityParameters, RegressionModel
from dimensionality_manuscript.regression_models.models import PlaceFieldModel, PlaceFieldStructuredGainModel
from dimensionality_manuscript.regression_models.gain_residual import PlaceFieldGainResidualModel
from dimensionality_manuscript.trial_registry import EnvironmentTrialRegistryParameters, EnvironmentTrialCVRegistryParameters
from dimensionality_manuscript.scripts.clear_model_caches import clear_model_cache_variants

SESSION = SimpleNamespace(session_name=("audit_mouse", "2026-11-05", "1"), params=SimpleNamespace(spks_type="sigrebase"))


def run():
    checks = []
    caveats = []
    with TemporaryDirectory(prefix="vr_gain_cache_audit_") as tmp:
        root = Path(tmp)
        paths = SimpleNamespace(hyperparameter_path=root / "hyperparameters", score_path=root / "scores")
        for folder in (paths.hyperparameter_path, paths.score_path):
            folder.mkdir()
        registry = SimpleNamespace(registry_paths=paths, registry_params=EnvironmentTrialRegistryParameters(
            environment=1, selected_roi_indices=(1, 2, 3, 4)))
        calls = Counter()

        def optimize(model, session, spks_type, train_split, validation_split, method, **kwargs):
            calls[(model._get_model_name(), method)] += 1
            if isinstance(model, PlaceFieldGainResidualModel):
                parent = model.inherited_gain_hyperparameters(session, spks_type, train_split, validation_split, method)
                params = dict(num_bins=parent.num_bins, smooth_width=parent.smooth_width,
                              alpha=parent.alpha, rank=parent.rank, alpha_residual=3.0, rank_residual=2)
            elif isinstance(model, PlaceFieldStructuredGainModel):
                parent = model.inherited_placefield_hyperparameters(session, spks_type, train_split, validation_split)
                params = dict(num_bins=parent.num_bins, smooth_width=parent.smooth_width, alpha=2.0, rank=1)
            else:
                if method == "golden":
                    raise NotImplementedError("Plain PF has no golden optimizer")
                params = dict(num_bins=40 if method == "grid" else 100, smooth_width=1.0)
            score = 0.1 if method == "grid" else 0.2
            return params, score, pd.DataFrame([dict(params, score=score)])

        def get(model, method="preferred", **kwargs):
            return model.get_best_hyperparameters(SESSION, "sigrebase", method=method, **kwargs)[0]

        with patch.object(RegressionModel, "optimize", optimize):
            for internal in (False, True):
                pf = PlaceFieldModel(registry, internal=internal)
                raw = PlaceFieldStructuredGainModel(registry, internal=internal)
                sqrt = PlaceFieldStructuredGainModel(registry, internal=internal, gain_transform="sqrt")
                residual = PlaceFieldGainResidualModel(registry, internal=internal)
                # Start with just the standard preferred PF optimization.
                assert get(pf).num_bins == 100
                before = calls.copy()
                assert get(pf, "optuna").num_bins == 100
                assert calls == before
                checks.append(f"{internal=}: preferred and concrete method share one disk entry")

                hp = get(residual)
                assert hp.num_bins == 100 and hp.alpha == 2 and hp.alpha_residual == 3
                assert calls[(raw._get_model_name(), "golden")] == 1
                before = calls.copy()
                assert get(raw).num_bins == hp.num_bins
                assert asdict(get(PlaceFieldGainResidualModel(registry, internal=internal))) == asdict(hp)
                assert calls == before
                checks.append(f"{internal=}: PF -> raw G -> G+E builds once; new instances hit disk")

                assert get(sqrt).num_bins == hp.num_bins
                assert calls[(sqrt._get_model_name(), "golden")] == 1
                assert calls[(pf._get_model_name(), "optuna")] == 1
                assert calls[(pf._get_model_name(), "grid")] == 0
                checks.append(f"{internal=}: sqrt shares PF but has separate gain optimization")

                before = calls.copy()
                clear_model_cache_variants([residual._get_model_name()], registry_paths=paths)
                assert raw.check_existing_hyperparameters(SESSION, "sigrebase", method="preferred")
                assert sqrt.check_existing_hyperparameters(SESSION, "sigrebase", method="preferred")
                get(residual)
                assert calls[(residual._get_model_name(), "golden")] == before[(residual._get_model_name(), "golden")] + 1
                assert calls[(raw._get_model_name(), "golden")] == before[(raw._get_model_name(), "golden")]
                checks.append(f"{internal=}: targeted child invalidation preserves and reuses both parents")

                before = calls.copy()
                get(residual, force_remake=True)
                assert calls[(residual._get_model_name(), "golden")] == before[(residual._get_model_name(), "golden")] + 1
                assert calls[(raw._get_model_name(), "golden")] == before[(raw._get_model_name(), "golden")]
                checks.append(f"{internal=}: force_remake reoptimizes only the requested model")

                # Characterize the historical 'best available cache' behavior.
                assert get(pf, "best").num_bins == 100
                get(pf, "grid")
                assert get(pf, "best").num_bins == 40
                assert get(raw).num_bins == 100
                raw.clear_cached_hyperparameter(SESSION, "sigrebase", method="preferred")
                assert get(raw).num_bins == 40
                assert get(residual).num_bins == 100
            caveats.append("PF method=best depends on which method caches exist; adding a better parent cache does not invalidate cached descendants.")

            raw = PlaceFieldStructuredGainModel(registry)
            session2 = SimpleNamespace(session_name=("other_mouse", "2026-11-05", "1"), params=SESSION.params)
            base = raw._get_hyperparameter_cache_key(SESSION, "sigrebase")
            variants = [
                raw._get_hyperparameter_cache_key(session2, "sigrebase"),
                raw._get_hyperparameter_cache_key(SESSION, "oasis"),
                raw._get_hyperparameter_cache_key(SESSION, "sigrebase", "train0", "validation"),
                raw._get_hyperparameter_cache_key(SESSION, "sigrebase", "train", "test"),
                raw._get_hyperparameter_cache_key(SESSION, "sigrebase", method="optuna"),
            ]
            for params in (
                replace(registry.registry_params, environment=2),
                replace(registry.registry_params, selected_roi_indices=(1, 2, 3, 5)),
                replace(registry.registry_params, split_seed=1),
                EnvironmentTrialCVRegistryParameters(environment=1, selected_roi_indices=(1, 2, 3, 4), cv_test_fold=0),
            ):
                other = SimpleNamespace(registry_paths=paths, registry_params=params)
                variants.append(PlaceFieldStructuredGainModel(other)._get_hyperparameter_cache_key(SESSION, "sigrebase"))
            variants.append(PlaceFieldStructuredGainModel(registry, activity_parameters=ActivityParameters(scale_type="std"))._get_hyperparameter_cache_key(SESSION, "sigrebase"))
            assert base not in variants and len(set(variants)) == len(variants)
            checks.append("Keys isolate session, spike type, train/validation split, method, environment, ROIs, seed, CV rotation, and scaling")
            assert raw._get_score_cache_key(SESSION, "sigrebase", test_split="test") != raw._get_score_cache_key(SESSION, "sigrebase", test_split="validation")
            checks.append("Score keys additionally isolate the scored split")

            for option in (dict(gain_regularization=0.5), dict(min_frames_per_unit=20)):
                assert PlaceFieldStructuredGainModel(registry, **option)._get_hyperparameter_cache_key(SESSION, "sigrebase") == base
            caveats.append("Custom gain_regularization/min_frames_per_unit are not included in cache keys; registered factory models use fixed defaults.")

            cold_session = SimpleNamespace(session_name=("cold", "date", "1"), params=SESSION.params)
            before = calls.copy()
            cold_hp = PlaceFieldGainResidualModel(registry).get_best_hyperparameters(cold_session, "sigrebase", method="preferred")[0]
            assert cold_hp.num_bins == 40
            for name, method in (("external_placefield_1d", "grid"), ("external_placefield_1d", "optuna"),
                                 ("external_placefield_1d_structured_gain", "golden"),
                                 ("external_placefield_1d_structured_gain_residual", "golden")):
                assert calls[(name, method)] == before[(name, method)] + 1
            before = calls.copy()
            PlaceFieldGainResidualModel(registry).get_best_hyperparameters(cold_session, "sigrebase", method="preferred")
            assert calls == before
            checks.append("Fully cold PF -> G -> G+E builds each dependency and subsequent reads hit cache")

            no_save = PlaceFieldModel(registry, internal=False, autosave=False)
            no_save_session = SimpleNamespace(session_name=("unsaved", "date", "1"), params=SESSION.params)
            before = set(paths.hyperparameter_path.glob("*.joblib"))
            no_save.get_best_hyperparameters(no_save_session, "sigrebase", method="preferred")
            assert set(paths.hyperparameter_path.glob("*.joblib")) == before
            checks.append("autosave=False does not create optimization files")

            score_calls = []
            def process(*args, **kwargs):
                score_calls.append(kwargs)
                return SimpleNamespace(predicted_data=np.array([[1., 2., 3.]]),
                                       target_data=np.array([[1., 2., 3.]]), metrics=dict(mse=0., r2=1.))
            with patch.object(raw, "process", process):
                raw.get_best_score(SESSION, "sigrebase", method="preferred")
                raw.get_best_score(SESSION, "sigrebase", method="golden")
                assert len(score_calls) == 1
                before = calls.copy()
                raw.get_best_score(SESSION, "sigrebase", method="preferred", force_remake=True)
                assert len(score_calls) == 2 and calls == before
                raw.get_best_score(SESSION, "sigrebase", method="preferred", force_reoptimize=True)
                assert len(score_calls) == 3
            checks.append("Score cache hits avoid refitting; force_remake refreshes scores and force_reoptimize refreshes the requested optimization")
    print(json.dumps(dict(passed_checks=checks, reproducibility_caveats=caveats), indent=2))


if __name__ == "__main__":
    run()

