"""Render stills of the VR environments by driving Blender headlessly.

The mice navigate four-room virtual corridors authored in Blender
(``vrEnvironment_001/003/004.blend``, one per environment of the ATL cohort). This module
places a camera at the entrance of each room and renders a still, so the environments can
be shown in a figure exactly as the mouse sees them.

Blender is not importable from the project environment, so rendering shells out to
``blender.exe -b <env.blend> -P render_vr_rooms.py -- <spec.json>``. That is slow enough
(a few seconds for a cold shader compile) that results are cached on disk under
``RegistryPaths.cache_path / "vr_renders"``, keyed by a hash of the render parameters and
the .blend file's modification time. Repeat calls with the same parameters just read PNGs,
which is what makes the Syd viewer in ``figure_scripts/figure1.py`` usable.

Locations are discovered in this order, so a different machine only needs the env vars:

- Blender executable: ``$VRANALYSIS_BLENDER_EXE``, else the newest
  ``C:/Program Files/Blender Foundation/Blender */blender.exe``, else ``blender`` on PATH.
- .blend directory: ``$VRANALYSIS_BLEND_DIR``, else ``~/Documents/Blender/BlenderCreations``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import image as mpimg

from ..registry import RegistryPaths

_RENDER_SCRIPT = Path(__file__).parent / "render_vr_rooms.py"
_MANIFEST_PREFIX = "__RENDER_MANIFEST__ "

# One .blend per environment of the ATL cohort. The CR_* mice run a different track length
# and a disjoint environment set (1, 2), so their environments are not in this map.
VR_ENV_BLENDS: dict[int, str] = {
    1: "vrEnvironment_001.blend",
    3: "vrEnvironment_003.blend",
    4: "vrEnvironment_004.blend",
}

# The .blend files place the full track in [0, 0.2] Blender units for a 200 cm rig track.
UNITS_PER_CM: float = 0.001

# Horizontal FOV of the actual rig camera (1 mm lens on an 8 mm sensor). Rendering at this
# value reproduces what the mouse sees; anything narrower is a crop for legibility.
RIG_HFOV_DEG: float = 151.93


@dataclass(frozen=True)
class RenderParams:
    """Camera, optics, and lighting knobs for a room render.

    Every field participates in the cache key, so changing any of them triggers a re-render
    while layout-only changes in the figure do not.

    Parameters
    ----------
    entrance_offset_cm : float
        Camera position along the track, measured in cm from the room's doorway plane.
        0 puts the camera exactly in the doorway; negative values sit in the previous room
        looking through it.
    hfov_deg : float
        Horizontal field of view. ``RIG_HFOV_DEG`` (~152 deg) is the rig's true optics and
        looks strongly fisheyed; ~90 deg reads better as a figure panel.
    panel_aspect : float
        Rendered width / height. The vertical FOV follows from this and ``hfov_deg``.
    panel_width_px : int
        Rendered width in pixels; height is derived from ``panel_aspect``.
    camera_height_cm : float
        Eye height above the floor. The corridor walls are 15 cm tall; the rig camera sits
        at 7.5 cm.
    yaw_deg : float
        Rotation off the track axis, in degrees. 0 looks straight down the corridor.
    use_dof : bool
        Enable the camera's depth of field. The .blend files ship with it on, which softens
        the far end of the corridor.
    light_scale : float
        Multiplier on every light's stored energy. Changes shading contrast, unlike
        ``exposure``.
    exposure : float
        Color-management exposure in stops. A pure brightness offset that leaves shading
        alone; the right knob for matching panel brightness across environments.
    samples : int
        EEVEE render samples. 32 is clean enough for a figure panel.
    """

    entrance_offset_cm: float = 2.0
    hfov_deg: float = 90.0
    panel_aspect: float = 1.6
    panel_width_px: int = 480
    camera_height_cm: float = 7.5
    yaw_deg: float = 0.0
    use_dof: bool = False
    light_scale: float = 1.0
    exposure: float = 0.0
    samples: int = 32


def blender_executable() -> Path:
    """Path to the Blender executable, from the env var, a standard install, or PATH."""
    override = os.environ.get("VRANALYSIS_BLENDER_EXE")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"VRANALYSIS_BLENDER_EXE points at a missing file: {path}")
        return path

    installs = sorted(Path("C:/Program Files/Blender Foundation").glob("Blender */blender.exe"))
    if installs:
        return installs[-1]

    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)

    raise FileNotFoundError("Blender not found. Set VRANALYSIS_BLENDER_EXE to the blender executable.")


def blend_directory() -> Path:
    """Directory holding the ``vrEnvironment_*.blend`` files."""
    override = os.environ.get("VRANALYSIS_BLEND_DIR")
    directory = Path(override) if override else Path.home() / "Documents" / "Blender" / "BlenderCreations"
    if not directory.is_dir():
        raise FileNotFoundError(f"Blend directory not found: {directory}. Set VRANALYSIS_BLEND_DIR.")
    return directory


def blend_path(env: int) -> Path:
    """Path to the .blend file for one environment."""
    if env not in VR_ENV_BLENDS:
        raise KeyError(f"No .blend registered for environment {env}. Known: {sorted(VR_ENV_BLENDS)}")
    path = blend_directory() / VR_ENV_BLENDS[env]
    if not path.exists():
        raise FileNotFoundError(f"Missing .blend for environment {env}: {path}")
    return path


def _cache_directory(env: int, params: RenderParams, blend: Path) -> Path:
    """Cache directory for one (environment, parameter set) pair.

    The modification times of both the .blend and the render script are folded into the key,
    so re-authoring an environment or changing how it is rendered invalidates the cache
    instead of silently serving stale images.
    """
    payload = dict(
        asdict(params),
        env=env,
        blend=blend.name,
        blend_mtime=blend.stat().st_mtime_ns,
        script_mtime=_RENDER_SCRIPT.stat().st_mtime_ns,
    )
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return RegistryPaths().cache_path / "vr_renders" / f"env{env}" / digest


def render_vr_rooms(env: int, params: RenderParams | None = None, force: bool = False) -> list[Path]:
    """Render one still per room of an environment, returning the PNG paths in track order.

    Parameters
    ----------
    env : int
        Environment identifier (a key of :data:`VR_ENV_BLENDS`).
    params : RenderParams or None
        Render settings. Defaults to ``RenderParams()``.
    force : bool
        Re-render even if a cached result exists.

    Returns
    -------
    list of pathlib.Path
        One PNG per room, ordered along the track (room 1 first).
    """
    params = params or RenderParams()
    blend = blend_path(env)
    cache_dir = _cache_directory(env, params, blend)
    manifest_path = cache_dir / "manifest.json"

    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        paths = [Path(room["path"]) for room in manifest["rooms"]]
        if all(path.exists() for path in paths):
            return paths

    cache_dir.mkdir(parents=True, exist_ok=True)
    spec_path = cache_dir / "spec.json"
    spec = dict(asdict(params), out_dir=str(cache_dir), units_per_cm=UNITS_PER_CM)
    spec_path.write_text(json.dumps(spec, indent=2))

    command = [str(blender_executable()), "-b", str(blend), "-P", str(_RENDER_SCRIPT), "--", str(spec_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Blender failed for environment {env} (exit {result.returncode}):\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")

    manifest_lines = [line for line in result.stdout.splitlines() if line.startswith(_MANIFEST_PREFIX)]
    if not manifest_lines:
        raise RuntimeError(f"Blender produced no manifest for environment {env}:\n{result.stdout[-4000:]}")
    manifest = json.loads(manifest_lines[-1][len(_MANIFEST_PREFIX) :])
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return [Path(room["path"]) for room in manifest["rooms"]]


def load_vr_room_images(env: int, params: RenderParams | None = None, force: bool = False) -> list[npt.NDArray[np.float32]]:
    """Rendered room stills for one environment as RGB float arrays in track order."""
    return [mpimg.imread(path)[..., :3] for path in render_vr_rooms(env, params=params, force=force)]
