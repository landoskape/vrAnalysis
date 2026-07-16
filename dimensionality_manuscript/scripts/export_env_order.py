"""Export the per-mouse environment experience order to a JSON file.

Slot 0 is the first environment a mouse ever experienced, slot 1 the second,
etc. (capped at 4). ``StimSpaceSpectraConfig`` reads this map to place each
environment's per-env spectra into a stable slot so results align across
sessions and mice.

Run locally (Access database available) or on MYRIAD from an exported session
list — ``session.environments`` reads oneData, which is present on both.

Usage
-----
    python -m dimensionality_manuscript.scripts.export_env_order [--output PATH] [--sessions PATH]

On MYRIAD the session list lives at ``~/vrAnalysis/sessions.json`` (see MYRIAD_SETUP.md;
also the ``DIM_MANUSCRIPT_SESSIONS_FILE`` env var), so run:

    python -m dimensionality_manuscript.scripts.export_env_order --sessions ~/vrAnalysis/sessions.json
"""

import argparse
import json
from pathlib import Path

from dimensionality_manuscript.env_order import build_env_order
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.scripts.run import collect_sessions, collect_sessions_from_file


def export_env_order(output: Path, sessions_file: Path | None = None) -> None:
    """Write the per-mouse environment order map to JSON.

    Parameters
    ----------
    output : Path
        Destination file. Parent directories are created if needed.
    sessions_file : Path or None
        When given, load sessions from this JSON (offline path, e.g. MYRIAD).
        Otherwise read the vrSessions database.
    """
    sessions = collect_sessions_from_file(sessions_file) if sessions_file else collect_sessions()
    order = build_env_order(sessions)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(order, indent=2))
    print(f"Exported environment order for {len(order)} mice → {output}")


def main():
    parser = argparse.ArgumentParser(description="Export per-mouse environment order to JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=RegistryPaths().env_order_path,
        help="Output JSON file (default: RegistryPaths().env_order_path)",
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help="Optional sessions.json to load sessions from (offline path). Default: use the database.",
    )
    args = parser.parse_args()
    export_env_order(args.output, sessions_file=args.sessions)


if __name__ == "__main__":
    main()
