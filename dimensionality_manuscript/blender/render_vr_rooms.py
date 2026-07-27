"""Render one still per room of a VR environment .blend file.

**This module runs inside Blender's Python, not the project environment.** It is invoked as::

    blender.exe -b <env.blend> -P render_vr_rooms.py -- <spec.json>

and only imports ``bpy`` plus the standard library. The host-side driver that writes the
spec, shells out, and caches the results is :mod:`dimensionality_manuscript.blender`.

Scene conventions (verified against ``vrEnvironment_001/003/004.blend``)
-----------------------------------------------------------------------
The mouse's track runs along **+X** starting at x = 0, the corridor is centered on y = 0,
and the floor is at z = 0. The four room shells live in a ``Rooms`` collection, each a box
open at the top with a full-height doorway slot cut into both end walls. Blender units are
metric-ish but not meters: the whole track spans 0.2 units for a 200 cm rig track, so
``units_per_cm`` is 0.001.

The camera is unparented and unconstrained; its only animation is a two-key linear
``location.x`` ramp from 0 to 0.2 over the scene's frame range. That F-curve is *cleared*
here — we place the camera by position rather than by frame, so the animation would
otherwise override every x we set.

The spec is a flat JSON dict; see ``RenderParams`` on the host side for the fields.
"""

import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# Rooms are matched by name over the scene's *linked* objects, deliberately not via the
# "Rooms" collection: in vrEnvironment_003 that collection holds only 3 of the 4 shells (the
# fourth, "Room 2.001", sits directly in the scene collection) and an orphaned duplicate
# "Room 1" is linked to no collection at all, so it never renders. Iterating scene.objects
# picks up exactly what the renderer sees. Names carry a room label and a Blender ".001"
# duplicate suffix, neither of which tracks position -- in vrEnvironment_004 the shells run
# Room 2, Room 1, Room 4, Room 3 along the corridor -- so ordering always comes from +X.
_ROOM_NAME_RE = re.compile(r"^room[\s_]*\d+(\.\d+)?$", re.IGNORECASE)

# Two shells whose x-spans overlap by more than this fraction of the shorter span occupy the
# same slot, which means the name-matching picked up a duplicate that would render on top of
# its twin. Real neighbours share only a ~1 cm doorway gap, so they never come close.
_MAX_ROOM_OVERLAP_FRACTION = 0.1

# Camera orientation for "standing in the corridor looking down the track": +90 deg about X
# puts the view direction in the horizontal plane, +270 deg about Z aims it along +X.
_CAMERA_PITCH = math.radians(90.0)
_CAMERA_YAW = math.radians(270.0)

MANIFEST_PREFIX = "__RENDER_MANIFEST__ "


def _room_objects(scene):
    """Room shell objects, ordered by their start along +X (i.e. track order)."""
    rooms = [ob for ob in scene.objects if ob.type == "MESH" and not ob.hide_render and _ROOM_NAME_RE.match(ob.name)]
    if not rooms:
        raise RuntimeError("No room meshes found (looked for renderable scene objects named /^Room ?\\d+$/)")

    rooms.sort(key=_room_x_start)
    for previous, current in zip(rooms, rooms[1:]):
        p0, p1 = _room_x_span(previous)
        c0, c1 = _room_x_span(current)
        overlap = min(p1, c1) - max(p0, c0)
        if overlap > _MAX_ROOM_OVERLAP_FRACTION * min(p1 - p0, c1 - c0):
            raise RuntimeError(
                f"Room shells {previous.name!r} x[{p0:.4f}, {p1:.4f}] and {current.name!r} x[{c0:.4f}, {c1:.4f}] "
                "occupy the same stretch of track; the scene has a duplicate."
            )
    return rooms


def _room_x_span(room):
    """World-space (min x, max x) of a room shell."""
    xs = [(room.matrix_world @ Vector(corner)).x for corner in room.bound_box]
    return min(xs), max(xs)


def _room_x_start(room):
    """World-space x of the room's near end (its doorway plane)."""
    return _room_x_span(room)[0]


def _configure_camera(scene, spec):
    """Point the camera down +X with the requested FOV, and strip its position animation."""
    camera = scene.camera
    if camera is None:
        raise RuntimeError("Scene has no active camera")

    # The stored F-curve drives location.x; without clearing it every frame_set / render
    # would snap the camera back onto the movie track.
    camera.animation_data_clear()
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = (_CAMERA_PITCH, 0.0, _CAMERA_YAW + math.radians(spec["yaw_deg"]))

    data = camera.data
    # Fit the FOV horizontally so hfov_deg is exact regardless of the render aspect ratio;
    # the vertical FOV then follows from panel_aspect.
    data.sensor_fit = "HORIZONTAL"
    data.lens = data.sensor_width / (2.0 * math.tan(math.radians(spec["hfov_deg"]) / 2.0))
    data.dof.use_dof = bool(spec["use_dof"])
    return camera


def _configure_render(scene, spec):
    width = int(spec["panel_width_px"])
    scene.render.resolution_x = width
    scene.render.resolution_y = max(int(round(width / spec["panel_aspect"])), 1)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = int(spec["samples"])
    scene.view_settings.exposure = float(spec["exposure"])


def _scale_lights(spec):
    """Scale every light's stored energy, leaving relative placement and balance intact."""
    scale = float(spec["light_scale"])
    if scale == 1.0:
        return
    for light in bpy.data.lights:
        light.energy *= scale


def main():
    spec_path = Path(sys.argv[sys.argv.index("--") + 1])
    spec = json.loads(spec_path.read_text())

    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    camera = _configure_camera(scene, spec)
    _configure_render(scene, spec)
    _scale_lights(spec)

    units_per_cm = float(spec["units_per_cm"])
    offset = float(spec["entrance_offset_cm"]) * units_per_cm
    height = float(spec["camera_height_cm"]) * units_per_cm

    rooms = _room_objects(scene)
    rendered = []
    for index, room in enumerate(rooms, start=1):
        x_start = _room_x_start(room)
        camera.location = (x_start + offset, 0.0, height)
        path = out_dir / f"room{index}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append(dict(room=index, name=room.name, x_start=x_start, camera_x=x_start + offset, path=str(path)))

    manifest = dict(
        blend=bpy.data.filepath,
        resolution=[scene.render.resolution_x, scene.render.resolution_y],
        lens_mm=camera.data.lens,
        rooms=rendered,
    )
    print(MANIFEST_PREFIX + json.dumps(manifest))


if __name__ == "__main__":
    main()
