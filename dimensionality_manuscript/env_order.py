"""Per-mouse environment experience order.

Assigns each environment a mouse experienced a *slot*: slot 0 is the first
environment the mouse ever saw, slot 1 the second, and so on (capped at 4).
Order is defined by walking a mouse's sessions in chronological order and
recording the first appearance of each environment index.

This is the manuscript-pipeline analogue of ``MultiSessionSpkmaps.env_selector``
(``vrAnalysis/multisession.py``), refactored to take a plain session list and to
sort by real dates rather than trusting the incoming list order. It reads only
``session.date``, ``session.session_id`` and ``session.environments`` — all
available offline — so it runs identically on MYRIAD (from ``sessions.json``)
and locally (from the Access database).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from vrAnalysis.helpers.typing import PrettyDatetime
from vrAnalysis.sessions import B2Session

MAX_ENV_SLOTS = 4


def _session_sort_key(session: B2Session) -> tuple:
    """Chronological sort key ``(date, session_id)`` for one session."""
    try:
        session_id = int(session.session_id)
    except (TypeError, ValueError):
        session_id = session.session_id
    return (PrettyDatetime.make_pretty(session.date), session_id)


def build_env_order(sessions: list[B2Session]) -> dict[str, list[int]]:
    """Build the per-mouse environment experience order.

    Parameters
    ----------
    sessions : list of B2Session
        Sessions to consider. Grouped by ``mouse_name``; order within the input
        list is irrelevant (each mouse's sessions are sorted chronologically).

    Returns
    -------
    dict[str, list[int]]
        ``{mouse_name: [env_id_slot0, env_id_slot1, ...]}``, chronological
        first-appearance order, truncated to at most ``MAX_ENV_SLOTS`` entries.
    """
    by_mouse: dict[str, list[B2Session]] = {}
    for session in sessions:
        by_mouse.setdefault(session.mouse_name, []).append(session)

    order: dict[str, list[int]] = {}
    for mouse, mouse_sessions in by_mouse.items():
        seen: list[int] = []
        for session in sorted(mouse_sessions, key=_session_sort_key):
            for env in session.environments:
                env = int(env)
                if env < 0:  # negative environmentIndex is the invalid/unlabeled sentinel
                    continue
                if env not in seen:
                    seen.append(env)
        order[mouse] = seen[:MAX_ENV_SLOTS]
    return order


@lru_cache(maxsize=None)
def load_env_order(path: str | None = None) -> dict[str, list[int]]:
    """Load a precomputed environment order map from JSON.

    Parameters
    ----------
    path : str or None
        Path to the JSON file written by ``export_env_order.py``. When None, the
        default ``RegistryPaths().env_order_path`` is used.

    Returns
    -------
    dict[str, list[int]]
        ``{mouse_name: [env_id, ...]}``.
    """
    if path is None:
        from .registry import RegistryPaths

        path = RegistryPaths().env_order_path
    return json.loads(Path(path).read_text())


def env_slot(mouse: str, env: int, order: dict[str, list[int]] | None = None) -> int:
    """Return the experience-order slot of ``env`` for ``mouse``.

    Parameters
    ----------
    mouse : str
        Mouse name.
    env : int
        Environment index.
    order : dict[str, list[int]] or None
        Precomputed order map. When None, loads the default via ``load_env_order``.

    Returns
    -------
    int
        Slot index in ``[0, MAX_ENV_SLOTS)``.
    """
    if order is None:
        order = load_env_order()
    return order[mouse].index(int(env))
