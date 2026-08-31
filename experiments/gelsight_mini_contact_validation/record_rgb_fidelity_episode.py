#!/usr/bin/env python3
"""Re-run one campaign episode and record simulator RGB plus tactile diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from campaign_specs import build_campaign
from fidelity_campaign import collect_cohort, episode_metrics
from scfields_assets import load_manifest
from tactile_grid import render_shear_field


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


class RGBFrames:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.viewer = None
        self.frames: list[np.ndarray] = []

    def setup(self, scene: Any, envs: list[Any]) -> None:
        from superdex.physics.utils.coordinate_systems import CoordinateSystem
        from superdex.physics.viewer import Viewer, ViewerCfg

        self.viewer = Viewer(ViewerCfg(
            backend="openGL3_egl", size=(self.width, self.height), offscreen=True,
            coordinate_system=CoordinateSystem("-Y", "+Z", "+X"),
        ))
        self.viewer.set_scene(scene)
        target = envs[0].tool_position
        direction = target - np.array([1.05, -1.00, 0.88])
        self.viewer.frame_scene(direction / np.linalg.norm(direction))

    def __call__(self, _step: int, _scene: Any, _envs: list[Any], _data: Any) -> None:
        frame = self.viewer.render()
        if frame is None:
            raise RuntimeError("offscreen renderer returned no RGB frame")
        self.frames.append(np.asarray(frame)[..., :3].copy())

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


def _chart(
    draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int],
    truth: np.ndarray, estimate: np.ndarray, step: int, title: str, units: str,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=(248, 248, 248), outline=(70, 70, 70), width=1)
    draw.text((x0 + 5, y0 + 3), f"{title} [{units}]", fill=(15, 15, 15))
    top = y0 + 20
    maximum = max(float(np.max(truth)), float(np.max(estimate)), 1e-6)
    def points(values: np.ndarray) -> list[tuple[int, int]]:
        return [
            (
                int(x0 + 3 + index * (x1 - x0 - 6) / max(len(values) - 1, 1)),
                int(y1 - 4 - value * (y1 - top - 8) / maximum),
            )
            for index, value in enumerate(values[:step + 1])
        ]
    for values, color in ((truth, (214, 39, 40)), (estimate, (31, 119, 180))):
        line = points(values)
        if len(line) > 1:
            draw.line(line, fill=color, width=2)
    cursor = int(x0 + 3 + step * (x1 - x0 - 6) / max(len(truth) - 1, 1))
    draw.line((cursor, top, cursor, y1 - 3), fill=(30, 30, 30), width=1)
    draw.text((x1 - 95, y0 + 3), f"max {maximum:.3g}", fill=(40, 40, 40))


def compose_video(
    frames: list[np.ndarray], data: dict[str, np.ndarray], spec: Any,
    metrics: dict[str, Any], output: Path, fps: int,
) -> None:
    truth = data["extrinsic_contact_wrench"][0]
    estimate = data["dynamic_inferred_extrinsic_wrench"][0]
    force_truth = np.linalg.norm(truth[:, :3], axis=1)
    force_estimate = np.linalg.norm(estimate[:, :3], axis=1)
    torque_truth = np.linalg.norm(truth[:, 3:], axis=1)
    torque_estimate = np.linalg.norm(estimate[:, 3:], axis=1)
    ee, tool = data["ee_pose"][0], data["tool_pose"][0]
    relative = np.asarray([
        Rotation.from_quat(e[3:]).inv().apply(t[:3] - e[:3]) for e, t in zip(ee, tool)
    ])
    approach = np.flatnonzero(data["contact_phase"][0] == 2)
    reference = np.median(relative[approach[-10:]], axis=0)
    slip = np.linalg.norm(relative - reference, axis=1) * 1000.0
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for step, rgb in enumerate(frames):
            canvas = Image.new("RGB", (1280, 720), (24, 24, 28))
            rgb_image = Image.fromarray(rgb).resize((896, 720), Image.Resampling.LANCZOS)
            canvas.paste(rgb_image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((10, 10, 600, 104), fill=(255, 255, 255))
            draw.text((20, 18), f"{spec.tool_name} — {spec.protocol}", fill=(10, 10, 10))
            draw.text(
                (20, 43),
                f"step {step:03d}  friction {spec.friction_scale:.2f}  grip 28 N/finger",
                fill=(10, 10, 10),
            )
            retained_color = (20, 125, 45) if slip[step] <= 12.0 else (205, 35, 35)
            draw.text(
                (20, 68), f"tool/EE slip {slip[step]:.2f} mm  "
                f"final retained={metrics['grasp_retained']}", fill=retained_color,
            )
            for index, side in enumerate(("left", "right")):
                panel = Image.fromarray(render_shear_field(data[f"tactile_force_field_{side}"][0, step]))
                panel = panel.resize((176, 234), Image.Resampling.NEAREST)
                x = 906 + index * 184
                canvas.paste(panel, (x, 32))
                draw.text((x + 48, 8), f"{side} gel", fill=(245, 245, 245))
            draw.text((913, 272), "red: direct   blue: tactile-field inferred", fill=(240, 240, 240))
            _chart(
                draw, (910, 298, 1265, 492), force_truth, force_estimate,
                step, "environment force magnitude", "N",
            )
            _chart(
                draw, (910, 510, 1265, 704), torque_truth, torque_estimate,
                step, "environment torque magnitude", "N m",
            )
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--friction", type=float, default=1.4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    os.environ.pop("DISPLAY", None)
    os.environ.pop("SUPERDEX_PRECISION", None)
    os.environ["SUPERDEX_ASSETS_PATH"] = str(PROJECT_ROOT / "assets")
    manifest = load_manifest()
    records = {record["name"]: record for record in manifest["tools"]}
    campaign = build_campaign(manifest["tools"])
    original = campaign[args.episode_id]
    spec = replace(original, friction_scale=args.friction)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from soft_gripper import GelMaterial
    import superdex.physics as physics

    recorder = RGBFrames(960, 640)
    physics.initialize(num_worker_threads=0)
    try:
        data, timing = collect_cohort(
            [spec], records, 200, 0.01,
            GelMaterial(friction_coefficient=args.friction), 28.0, recorder,
        )
    finally:
        recorder.close()
        physics.shutdown()
    metrics = episode_metrics(data, 0, spec, 0.01)
    if not metrics["grasp_retained"]:
        raise RuntimeError(
            f"refusing to publish dropped-tool video: max slip "
            f"{metrics['grasp_slip_max_m']*1000:.1f} mm, bilateral "
            f"{metrics['bilateral_grasp_fraction']:.1%}"
        )
    video = output / f"episode_{args.episode_id:06d}_{spec.tool_name}_{spec.protocol}_rgb.mp4"
    compose_video(recorder.frames, data, spec, metrics, video, args.fps)
    np.savez_compressed(
        output / "episode_data.npz",
        tactile_force_field_left=data["tactile_force_field_left"][0],
        tactile_force_field_right=data["tactile_force_field_right"][0],
        extrinsic_contact_wrench=data["extrinsic_contact_wrench"][0],
        dynamic_inferred_extrinsic_wrench=data["dynamic_inferred_extrinsic_wrench"][0],
        ee_pose=data["ee_pose"][0], tool_pose=data["tool_pose"][0],
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({
        "episode": spec.to_dict(), "grip_force_per_finger_n": 28.0,
        "timing": timing, "video": str(video),
    }, indent=2) + "\n")
    print(f"wrote retained-tool RGB review video to {video}", flush=True)


if __name__ == "__main__":
    main()
