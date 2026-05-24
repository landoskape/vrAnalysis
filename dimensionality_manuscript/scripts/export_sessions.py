"""Export the session list to a JSON file for use on MYRIAD.

The vrSessions database is a Microsoft Access .accdb file, which cannot be
read on Linux. Run this script locally before transferring to MYRIAD. The
resulting JSON is the sole input needed to reconstruct B2Session objects on
the server.

Usage
-----
    python -m dimensionality_manuscript.scripts.export_sessions [--output sessions.json]
"""

import argparse
import json
from pathlib import Path

from dimensionality_manuscript.scripts.run import collect_sessions


def export_sessions(output: Path) -> None:
    """Write all imaging sessions to a JSON file.

    Parameters
    ----------
    output : Path
        Destination file. Parent directories are created if needed.
    """
    sessions = collect_sessions()
    records = [
        {
            "mouse_name": s.mouse_name,
            "date": str(s.date),
            "session_id": s.session_id,
        }
        for s in sessions
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))
    print(f"Exported {len(records)} sessions → {output}")


def main():
    parser = argparse.ArgumentParser(description="Export session list to JSON for MYRIAD")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sessions.json"),
        help="Output JSON file (default: sessions.json)",
    )
    args = parser.parse_args()
    export_sessions(args.output)


if __name__ == "__main__":
    main()
