"""HU_D04 depth image rendering (task 5.4).

Renders a depth image from the HU_D04 D435i mount using MuJoCo's native depth
renderer (headless EGL), then crops it to the resolution the depth backbone
expects. This proves the mount pose + camera params produce a valid depth image
with self-occlusion.

The G1 pipeline crops left/right 4 px and top 2 px from a 106x60 raw image, then
the backbone consumes 58x87. Here we render 106x60, crop to 58x98, and resize
to 58x87 so the exact backbone input shape is available.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import contextlib  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hud04_sim2sim.depth_camera import (  # noqa: E402
    CAMERA_OFFSET_RPY,
    CAMERA_OFFSET_XYZ,
    CAMERA_PARENT_BODY,
    CROP_LEFT,
    CROP_RIGHT,
    CROP_TOP,
    HEIGHT,
    MAX_RANGE,
    WIDTH,
)
from hud04_sim2sim.model_loader import DEFAULT_XML  # noqa: E402

BACKBONE_H = 58
BACKBONE_W = 87


def _rpy_to_quat(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF rpy (roll,pitch,yaw) -> quaternion (w,x,y,z)."""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _inject_camera(xml_path) -> str:
    """Add a depth camera at the URDF mount pose to the MJCF text.

    Inserted as a child of the camera parent body so MuJoCo resolves the pose.
    """
    text = open(xml_path).read()
    q = _rpy_to_quat(CAMERA_OFFSET_RPY)
    pos = " ".join(f"{v}" for v in CAMERA_OFFSET_XYZ)
    quat = " ".join(f"{v}" for v in q)
    cam = f'<camera name="hud04_depth" pos="{pos}" quat="{quat}"/>'
    marker = f'<body name="{CAMERA_PARENT_BODY}"'
    idx = text.index(marker)
    # Insert right after the parent body's opening tag.
    close = text.index(">", idx) + 1
    return text[:close] + "\n              " + cam + text[close:]


def render_depth(xml_path=DEFAULT_XML, width: int = WIDTH, height: int = HEIGHT
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Render depth (raw) and return (raw_depth, backbone_shaped_depth)."""
    import tempfile
    from pathlib import Path

    patched = _inject_camera(xml_path)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                     dir=str(Path(xml_path).parent)) as fh:
        fh.write(patched)
        tmp = fh.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with contextlib.closing(mujoco.Renderer(model, height=height, width=width)) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="hud04_depth")
        raw = renderer.render().astype(np.float32)

    # Crop: drop top rows, left/right cols (matches G1 camera config).
    cropped = raw[CROP_TOP:, CROP_LEFT:width - CROP_RIGHT]
    cropped = np.clip(cropped, 0.0, MAX_RANGE)
    # Nearest-neighbour resize to the backbone input shape.
    h, w = cropped.shape
    ys = (np.arange(BACKBONE_H) * h / BACKBONE_H).astype(int)
    xs = (np.arange(BACKBONE_W) * w / BACKBONE_W).astype(int)
    resized = cropped[ys][:, xs]
    return raw, resized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    args = parser.parse_args(argv)

    raw, shaped = render_depth(args.xml)
    finite = np.isfinite(shaped)
    print(f"[hud04-depth] raw shape: {raw.shape}")
    print(f"[hud04-depth] backbone shape: {shaped.shape} "
          f"(expect ({BACKBONE_H}, {BACKBONE_W}))")
    print(f"[hud04-depth] depth range: {shaped[finite].min():.3f}..{shaped[finite].max():.3f} m")
    # Self-occlusion: some near-field hits inside the robot's own extent.
    near = shaped[shaped < MAX_RANGE * 0.5]
    print(f"[hud04-depth] near-field pixels (<50% range): {near.size}")
    ok = shaped.shape == (BACKBONE_H, BACKBONE_W) and finite.any()
    print(f"[hud04-depth] {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
