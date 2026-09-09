"""Actual EGL RGB review videos and quantitative plots; no synthetic scene imagery."""

from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from tactile_grid import render_shear_field


class Recorder:
    def __init__(self, world, path):
        import os

        os.environ.pop("DISPLAY", None)
        from superdex.physics.viewer import Viewer, ViewerCfg
        from superdex.physics.utils.coordinate_systems import CoordinateSystem

        self.viewer = Viewer(
            ViewerCfg(
                backend="openGL3_egl",
                size=(800, 600),
                offscreen=True,
                coordinate_system=CoordinateSystem("-Y", "+Z", "+X"),
            )
        )
        self.viewer.set_scene(world.scene)
        self.world = world
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.target = world.origin + world.rot.apply(
            [0.015, 0, -0.075 if getattr(world.spec, "kind", None) == "key" else -0.025]
        )
        self.target = getattr(world, "camera_target_world", self.target)
        self.viewer.set_camera_view(
            self.target + world.rot.apply([0.22, -0.25, 0.13]), self.target, [0, 0, 1]
        )
        self.writer = imageio.get_writer(
            str(self.path) + "_rgb_tactile.mp4", fps=20, codec="libx264", quality=8
        )
        self.frames = 0

    def frame(self, row):
        rgb = np.asarray(self.viewer.render())[..., :3].copy()
        if self.frames == 0:
            self.viewer.set_camera_view(
                self.target + [0.75, -0.75, 0.45], [0.25, 0, 0.30], [0, 0, 1]
            )
            Image.fromarray(np.asarray(self.viewer.render())[..., :3].copy()).save(
                str(self.path) + "_wide.png"
            )
            self.viewer.set_camera_view(
                self.target + self.world.rot.apply([0.22, -0.25, 0.13]),
                self.target,
                [0, 0, 1],
            )
        decision = hasattr(self.world.spec, "kind")
        canvas = Image.new("RGB", (1440 if decision else 1120, 608), (24, 24, 28))
        canvas.paste(Image.fromarray(rgb), (0, 8))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((8, 16, 570, 83), fill="white")
        draw.text(
            (16, 23),
            f"{getattr(self.world.spec, 'family', getattr(self.world.spec, 'kind', self.world.spec.task))} {getattr(self.world.spec, 'variant', getattr(self.world.spec, 'stage', ''))}: {row['phase']}  t={row['timestamps']:.2f}s",
            fill="black",
        )
        draw.text(
            (16, 44),
            (
                f"insertion {row['insertion_depth_m'] * 1000:.1f} mm"
                if getattr(self.world.spec, "stage", None) == "insertion" and getattr(self.world.spec, "family", "insertion") in ("insertion", "composite")
                else f"rotor {np.rad2deg(row['task_progress']):.1f} deg"
                if getattr(self.world.spec, "kind", None) == "key"
                else f"working tip X {row['working_tip_task_m'][0] * 1000:.1f} mm"
                if 'working_tip_task_m' in row
                else f"progress {row['task_progress'] * 1000:.1f} mm"
            )
            + f" | slip {row['slip_m'] * 1000:.2f} mm",
            fill="black",
        )
        draw.text(
            (16, 63),
            "Actual simulator RGB; prepared grasp, freely held after 1 s",
            fill="black",
        )
        if 'blade_contact_force_n' in row:
            draw.text((1130,420),f"Blade contact: {row['blade_contact_force_n']:.2f} N",fill='white')
            draw.text((1130,442),f"Holder contact: {row['holder_contact_force_n']:.2f} N",fill='white')
        for j, s in enumerate(("left", "right")):
            panel = Image.fromarray(
                render_shear_field(
                    row[f"tactile_force_field_{s}"],
                    shear_scale_n=0.25,
                    normal_scale_n=1.5,
                )
            ).resize((148, 230))
            canvas.paste(panel, (810 + j * 154, 40))
            draw.text((818 + j * 154, 18), s + " gel", fill="white")
            if decision:
                reference = (
                    self.world.reference[j]
                    if self.world.reference is not None
                    else np.zeros((7, 9, 3))
                )
                delta = row[f"tactile_force_field_{s}"] - reference
                panel = Image.fromarray(
                    render_shear_field(delta, shear_scale_n=0.10, normal_scale_n=0.25)
                ).resize((148, 230))
                canvas.paste(panel, (1130 + j * 154, 40))
                draw.text((1130 + j * 154, 18), s + " minus reference", fill="white")
        if decision:
            draw.text((1130, 300), "Delta: shear 0.10 N/node", fill="white")
            draw.text((1130, 323), "normal 0.25 N/node; fixed scales", fill="white")
            draw.text(
                (1130, 356),
                f"Reference valid: {self.world.reference is not None}",
                fill="white",
            )
            draw.text(
                (1130, 389),
                "Grasp load [N]: "
                + np.array2string(row["fingertip_load_n"], precision=1),
                fill="white",
            )
        draw.text((814, 290), "World wrench about tool COM", fill="white")
        draw.text((814, 540), "Fixed tactile scale: shear 0.25 N", fill="white")
        draw.text((814, 560), "normal 1.5 N/node; arrows clipped", fill="white")
        for k, (name, key) in enumerate(
            (
                ("Contact", "extrinsic_contact_wrench"),
                ("Tactile inferred", "dynamic_inferred_extrinsic_wrench"),
            )
        ):
            w = row[key]
            draw.text((814, 325 + k * 95), name, fill="white")
            draw.text(
                (814, 348 + k * 95),
                "F [N] " + np.array2string(w[:3], precision=2),
                fill="white",
            )
            draw.text(
                (814, 371 + k * 95),
                "T [Nm] " + np.array2string(w[3:], precision=3),
                fill="white",
            )
        self.writer.append_data(np.asarray(canvas))
        if self.frames in (
            0,
            30,
            70,
            110,
            160,
            210,
            round(0.75 * self.world.spec.duration * 20),
            round(0.96 * self.world.spec.duration * 20),
        ):
            canvas.save(str(self.path) + f"_frame_{self.frames:03d}.png")
            # Orthogonal side view exposes hook mouth and table clearance.
            self.viewer.set_camera_view(
                self.target + self.world.rot.apply([0, -0.30, 0]),
                self.target,
                [0, 0, 1],
            )
            Image.fromarray(np.asarray(self.viewer.render())[..., :3].copy()).save(
                str(self.path) + f"_side_{self.frames:03d}.png"
            )
            self.viewer.set_camera_view(
                self.target + self.world.rot.apply([0.22, -0.25, 0.13]),
                self.target,
                [0, 0, 1],
            )
        self.frames += 1

    def close(self, data):
        self.writer.close()
        self.viewer.close()
        plot_episode(data, Path(str(self.path) + "_wrenches.png"),
                     task_kind=getattr(self.world.spec, "kind", None),
                     task_stage=getattr(self.world.spec, "stage", None))


def plot_episode(d, path, task_kind=None, task_stage=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
    t = d["timestamps"]
    for j, axis in enumerate("xyz"):
        for r, (offset, unit) in enumerate(((0, "N"), (3, "Nm"))):
            axs[r, j].plot(
                t,
                d["extrinsic_contact_wrench"][:, j + offset],
                label="direct environment",
                lw=1,
            )
            axs[r, j].plot(
                t,
                d["dynamic_inferred_extrinsic_wrench"][:, j + offset],
                label="dense gel + dynamics",
                lw=1,
                alpha=0.8,
            )
            axs[r, j].set_ylabel(f"{'F' if r == 0 else 'T'}{axis} [{unit}]")
            axs[r, j].axvspan(0, 1.5, color="gray", alpha=0.15)
        axs[2, j].plot(t, d["field_gel_wrench"][:, j], label="field")
        axs[2, j].plot(t, d["direct_gel_wrench"][:, j], label="direct gel", alpha=0.7)
        axs[2, j].set_ylabel(f"gel F{axis} [N]")
    if task_stage == "insertion":
        axs[3, 0].plot(t, d["insertion_depth_m"] * 1000)
        axs[3, 0].set_ylabel("blade insertion depth [mm]")
    elif task_kind == "key":
        axs[3, 0].plot(t, np.rad2deg(d["rotor_angle_rad"]))
        axs[3, 0].set_ylabel("socket rotor angle [deg]")
    else:
        axs[3, 0].plot(t, d["task_progress"] * 1000)
        axs[3, 0].set_ylabel("fixture travel [mm]")
    axs[3, 1].plot(t, d["slip_m"] * 1000)
    axs[3, 1].set_ylabel("tool/EE slip [mm]")
    axs[3, 2].plot(t, d["rigid_penetration_m"] * 1000)
    axs[3, 2].set_ylabel("rigid penetration [mm]")
    for ax in axs.ravel():
        ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=8)
    axs[2, 0].legend(fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
