"""Guards on analysis-config identity.

``ResultsStore`` identifies a result by ``result_uid = H(session_id : analysis_key)``
where ``analysis_key`` is :meth:`AnalysisConfigBase.key`. Nothing else takes part —
``analysis_type`` is a descriptive column, not part of identity. So two config classes
whose ``key()`` values coincide map to the same primary key *and* the same blob file,
and silently overwrite each other. These tests make that a build failure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

import pytest

from dimensionality_manuscript.configs import ANALYSIS_CONFIG_CLASS_LIST, ANALYSIS_CONFIG_CLASSES

# Keys are hashed per schema version, so a collision can hide in a version that no
# config currently defaults to (exactly how the placefield_prediction bug arose).
SCHEMA_VERSIONS = tuple(f"v{i}" for i in range(1, 11))


def test_display_names_are_unique():
    by_name = defaultdict(list)
    for cls in ANALYSIS_CONFIG_CLASS_LIST:
        by_name[cls.display_name].append(cls.__name__)
    duplicates = {name: names for name, names in by_name.items() if len(names) > 1}
    assert not duplicates, f"display_name must be unique across config classes: {duplicates}"


def test_display_names_are_usable_as_key_namespaces():
    """``key()`` namespacing joins on ':', so the name must not contain one."""
    bad = [cls.__name__ for cls in ANALYSIS_CONFIG_CLASS_LIST if ":" in cls.display_name]
    assert not bad, f"display_name must not contain ':': {bad}"


def test_every_class_declares_a_display_name():
    missing = [cls.__name__ for cls in ANALYSIS_CONFIG_CLASS_LIST if not getattr(cls, "display_name", None)]
    assert not missing, f"config classes must declare a non-empty display_name: {missing}"


def test_registry_is_keyed_by_display_name():
    assert ANALYSIS_CONFIG_CLASSES == {cls.display_name: cls for cls in ANALYSIS_CONFIG_CLASS_LIST}


def test_store_uses_the_shared_registry():
    """``ResultsStore`` must not keep its own hand-maintained class list."""
    from dimensionality_manuscript.pipeline.store import _analysis_config_classes

    assert _analysis_config_classes() == ANALYSIS_CONFIG_CLASSES


def test_run_script_can_reach_every_config():
    """Every registered analysis must be requestable via ``--analyses``."""
    from dimensionality_manuscript.scripts.run import _analysis_name_mapping

    mapping = _analysis_name_mapping()
    unreachable = sorted(set(ANALYSIS_CONFIG_CLASSES) - set(mapping))
    assert not unreachable, f"analyses not reachable from scripts.run: {unreachable}"


def test_run_script_default_analyses_are_valid():
    from dimensionality_manuscript.scripts.run import DEFAULT_ANALYSES, _analysis_name_mapping

    mapping = _analysis_name_mapping()
    unknown = sorted(set(DEFAULT_ANALYSES) - set(mapping))
    assert not unknown, f"DEFAULT_ANALYSES names not in the registry: {unknown}"


def _all_keys() -> dict[str, list[tuple[str, str, dict]]]:
    """Map ``key()`` -> [(class name, schema version, fields)] over every variation."""
    keys: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for cls in ANALYSIS_CONFIG_CLASS_LIST:
        for schema_version in SCHEMA_VERSIONS:
            for cfg in cls.generate_variations(schema_version=schema_version):
                keys[cfg.key()].append((cls.__name__, schema_version, asdict(cfg)))
    return keys


def test_no_cross_class_key_collisions():
    """No two config *classes* may ever produce the same analysis_key."""
    collisions = []
    for key, entries in _all_keys().items():
        classes = {name for name, _, _ in entries}
        if len(classes) > 1:
            collisions.append((key, sorted(classes), entries[0][1]))

    if collisions:
        lines = [f"{len(collisions)} cross-class analysis_key collision(s):"]
        for key, classes, schema_version in sorted(collisions, key=lambda c: c[2]):
            lines.append(f"  key={key} schema={schema_version} classes={classes}")
        pytest.fail("\n".join(lines))


def test_no_duplicate_keys_within_a_class():
    """A class's own variations must be distinguishable from one another."""
    for cls in ANALYSIS_CONFIG_CLASS_LIST:
        variations = cls.generate_variations()
        keys = [cfg.key() for cfg in variations]
        assert len(set(keys)) == len(keys), f"{cls.__name__} generates duplicate keys across its param grid"
