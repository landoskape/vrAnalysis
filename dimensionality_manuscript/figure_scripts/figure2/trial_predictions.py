"""Replay the saved trial-regression fits for example prediction curves."""

import numpy as np

from dimensionality_manuscript.registry import PopulationRegistry, get_model
from dimensionality_manuscript.trial_registry import (
    EnvironmentTrialRegistry, TrialRegistryParameters, ROTATING_TEST_FOLDS,
)
from dimensionality_manuscript.figure_scripts import session_cache


def bin_prediction(values, position, edges):
    """Average temporal predictions within position bins, leaving unvisited bins missing."""
    bins = np.searchsorted(edges, position, side="right") - 1
    valid = np.isfinite(values) & np.isfinite(position) & (bins >= 0) & (bins < len(edges) - 1)
    counts = np.bincount(bins[valid], minlength=len(edges) - 1)
    sums = np.bincount(bins[valid], weights=values[valid], minlength=len(edges) - 1)
    return np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)


def replay_predictions(config, saved, session, edges, *, temporal=False, reference=False):
    """Refit with stored parameters and ROI identities on exactly the scored test folds.

    Return position-binned curves keyed by (raw ROI, absolute trial). Convert the
    model's affine activity normalization to the session map's sigma units before
    binning. Dropped prediction frames are removed from behavior using the
    report's original test-frame indices. Curves retain all bins from ``edges``;
    their x coordinates are its bin centers, independent of activity-map cropping.
    With ``temporal=True``, return (absolute imaging-frame indices, values)
    instead of position-binned curves. No validation or training predictions
    are substituted. With ``reference=True`` (for an internal placefield model),
    return (registry frame indices, training PF at true position, training PF at
    decoded position). Reference traces use the full registry test split,
    retaining each reference's undefined values independently.
    """
    registry = PopulationRegistry(registry_params=config.data_config.to_registry_params())
    curves = {}
    standardized = session_cache.get_spks(session)
    raw_rois = np.flatnonzero(session.idx_rois)
    lookup = {int(roi): row for row, roi in enumerate(raw_rois)}
    folds = ROTATING_TEST_FOLDS if config.schema_version == "v2" else (3,)
    for slot, environment in enumerate(saved["env_slot_ids"]):
        params = saved["selected_hyperparameters"][slot]
        if not np.isfinite(environment) or params is None:
            continue
        for fold_index, fold in enumerate(folds):
            env_registry = EnvironmentTrialRegistry(
                registry, int(environment), saved["selected_roi_indices"][slot],
                TrialRegistryParameters(
                    relative_size=(2, 1, 1), split_seed=config.trial_split_seed,
                    speed_threshold=registry.registry_params.speed_threshold,
                ),
                **({} if fold_index == 0 else {"cv_test_fold": fold}),
            )
            model = get_model(config.model_name, env_registry,
                              activity_parameters=config.activity_parameters_name)
            hyperparameters = model._model_hyperparameters.from_dict(params)
            report = model.process(session, config.spks_type, train_split="train",
                                   test_split="test", hyperparameters=hyperparameters)
            source, _, behavior = model.get_session_data(session, config.spks_type, "test")
            prediction = np.asarray(report.predicted_data)
            if reference:
                from vrAnalysis.processors.placefields import get_placefield_prediction

                target_field, source_field = report.trained_model[:2]
                prediction = get_placefield_prediction(target_field, behavior)[0].T
                decoded = model.decode_internal_frame_behavior(source, source_field, behavior)
                internal_prediction = get_placefield_prediction(target_field, decoded)[0].T
            elif report.extras.get("predictions_were_filtered", False):
                # The report columns have already been filtered by the model.
                # Apply its original test-frame indices to trial/position too.
                behavior = behavior.filter(report.extras["idx_valid_predictions"])
            target_rois = np.asarray(saved["target_roi_indices"][slot])
            population, _ = env_registry.get_population(session, config.spks_type)
            actual_rois = np.asarray(population.idx_neurons)[np.asarray(population.cell_split_indices[1])]
            if not np.array_equal(actual_rois, target_rois):
                raise ValueError("Replayed target ROI order differs from the stored result")
            if prediction.shape != (len(target_rois), len(behavior)):
                raise ValueError("Replayed predictions and test behavior are misaligned")
            normalization_split = "full" if model.activity_parameters.presplit else "test"
            _, full_target, full_behavior = model.get_session_data(session, config.spks_type, normalization_split)
            full_target = np.asarray(full_target)
            for row, roi in enumerate(target_rois):
                if int(roi) not in lookup:
                    continue
                reference_activity = standardized[np.asarray(full_behavior.idx, dtype=int), lookup[int(roi)]]
                normalized = full_target[row]
                scale = np.std(reference_activity) / np.std(normalized) if np.std(normalized) > 0 else 0.0
                offset = np.mean(reference_activity) - scale * np.mean(normalized)
                values = prediction[row] * scale + offset
                for trial in np.unique(behavior.trial):
                    keep = behavior.trial == trial
                    if reference:
                        curves[int(roi), int(trial)] = (
                            np.asarray(behavior.idx[keep], dtype=int), values[keep],
                            internal_prediction[row, keep] * scale + offset,
                        )
                    elif temporal:
                        curves[int(roi), int(trial)] = (
                            np.asarray(behavior.idx[keep], dtype=int), values[keep],
                        )
                    else:
                        curves[int(roi), int(trial)] = bin_prediction(values[keep], behavior.position[keep], edges)
    return curves


