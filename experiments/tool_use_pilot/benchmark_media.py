"""Fast EGL RGB review of recorded rigid poses, synchronized with recorded tactile.

This is explicitly a recorded-state visualization, not a fresh physics replay.
Rigid geometry/poses are exact; deformable gel interiors are not reconstructed.
Use benchmark.py --replay for an independent physics-and-render replay.
"""

from pathlib import Path
import argparse
import json
import os
import h5py
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from benchmark import HERE
from tactile_grid import render_shear_field


def render(path, output):
    os.environ.pop("DISPLAY", None)
    import superdex.physics as p
    from superdex.physics.viewer import Viewer, ViewerCfg
    from superdex.physics.utils.coordinate_systems import CoordinateSystem

    p.initialize(num_worker_threads=0)
    scene = p.create_scene("recorded_pose_review")
    viewer = None
    try:
        output.mkdir(parents=True, exist_ok=True)
        with h5py.File(path) as f:
            spec = json.loads(f.attrs["config_json"])
            d = f["observations"]
            actors = []
            for key in sorted(f["geometry"], key=int):
                g = f["geometry"][key]
                shape = p.create_tri_mesh_shape(
                    g["vertices"][:].astype(np.float32).ravel(),
                    g["faces"][:].astype(np.int32).ravel(),
                )
                actors.append(
                    scene.create_rigid_actor(
                        name=g.attrs["name"],
                        shape=shape,
                        mass=0.1,
                        has_gravity=False,
                        collider_type=p.ColliderType.MESH,
                    )
                )
            viewer = Viewer(
                ViewerCfg(
                    backend="openGL3_egl",
                    size=(800, 600),
                    offscreen=True,
                    coordinate_system=CoordinateSystem("-Y", "+Z", "+X"),
                )
            )
            viewer.set_scene(scene)
            target = f["camera/target_world"][:]
            if spec["family"] == "surface":
                vertices = f["geometry/1/vertices"][:]
                pose = d["body_root_poses"][0, 1]
                center = (vertices.min(0) + vertices.max(0)) / 2
                center[2] = vertices[:, 2].max()
                target = Rotation.from_quat(pose[3:]).apply(center) + pose[:3]
            rotation = np.asarray(f.attrs["action_frame_rotation_world"]).reshape(3, 3)
            eye = target + rotation @ np.array([0.22, -0.25, 0.13])
            viewer.set_camera_view(eye, target, [0, 0, 1])
            every = max(1, round(0.05 / spec["dt"]))
            indices = list(range(0, len(d["timestamps"]), every))
            chosen = set(np.linspace(0, len(indices) - 1, 6, dtype=int))
            movie = output / (path.stem + "_recorded_rgb_tactile.mp4")
            snapshots = []
            with imageio.get_writer(
                movie, fps=20, codec="libx264", quality=7
            ) as writer:
                for frame, i in enumerate(indices):
                    for actor, pose in zip(actors, d["body_root_poses"][i]):
                        actor.set_root_transform(
                            p.TransformRT(
                                translation=pose[:3],
                                rotation=p.Quaternion.from_rotation_vector(
                                    Rotation.from_quat(pose[3:]).as_rotvec()
                                ),
                            )
                        )
                    rgb = np.asarray(viewer.render())[..., :3].copy()
                    canvas = Image.new("RGB", (1440, 608), (24, 24, 28))
                    canvas.paste(Image.fromarray(rgb), (0, 8))
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((8, 15, 785, 82), fill="white")
                    draw.text(
                        (15, 20),
                        f"{spec['family']} / {spec['variant']} | {d['phase'].asstr()[i]} | t={d['timestamps'][i]:.2f}s",
                        fill="black",
                    )
                    draw.text(
                        (15, 42),
                        "Recorded-state EGL RGB; not a new physics replay; gel interior deformation omitted",
                        fill="black",
                    )
                    draw.text(
                        (15, 62),
                        f"Grasp drift {d['slip_m'][i] * 1000:.2f} mm / {d['angular_slip_deg'][i]:.2f} deg",
                        fill="black",
                    )
                    ref = f["tactile_reference"][:]
                    for j, side in enumerate(("left", "right")):
                        field = d["tactile_force_field_" + side][i]
                        for col, values in enumerate((field, field - ref[j])):
                            x = 810 + col * 310 + j * 150
                            panel = Image.fromarray(
                                render_shear_field(
                                    values,
                                    shear_scale_n=0.25 if col == 0 else 0.1,
                                    normal_scale_n=1.5 if col == 0 else 0.25,
                                )
                            ).resize((145, 230))
                            canvas.paste(panel, (x, 45))
                            draw.text(
                                (x, 22),
                                side + (" raw" if col == 0 else " tared"),
                                fill="white",
                            )
                    for j, key in enumerate(
                        (
                            "extrinsic_contact_wrench",
                            "dynamic_inferred_extrinsic_wrench",
                        )
                    ):
                        w = d[key][i]
                        draw.text(
                            (815, 305 + j * 100),
                            "Contact" if j == 0 else "Tactile inferred",
                            fill="white",
                        )
                        draw.text(
                            (815, 330 + j * 100),
                            "F [N] " + np.array2string(w[:3], precision=3),
                            fill="white",
                        )
                        draw.text(
                            (815, 355 + j * 100),
                            "T [Nm] " + np.array2string(w[3:], precision=4),
                            fill="white",
                        )
                    writer.append_data(np.asarray(canvas))
                    if frame in chosen:
                        canvas.save(output / f"{path.stem}_review_{frame:04d}.png")
                        snapshots.append(np.asarray(canvas.resize((720, 304))))
            contact = Image.fromarray(np.concatenate(snapshots, axis=0))
            contact.save(output / (path.stem + "_contact_sheet.png"))
            audit = dict(
                source=str(path.resolve()),
                frames=len(indices),
                fps=20,
                rgb_source="EGL rendering of recorded rigid mesh poses",
                physics_replay=False,
                gel_interior_reconstructed=False,
                source_implementation=f.attrs.get(
                    "benchmark_implementation_hash", "legacy"
                ),
            )
            (output / (path.stem + "_media.json")).write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            print(movie, flush=True)
    finally:
        if viewer:
            viewer.close()
        p.destroy_scene(scene)
        p.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--output", type=Path, default=HERE / "output/diversity_pilot/media"
    )
    a = parser.parse_args()
    render(a.episode, a.output)
