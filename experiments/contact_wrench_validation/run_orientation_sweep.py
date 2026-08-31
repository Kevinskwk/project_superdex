#!/usr/bin/env python3
"""Run and compare FP32 contact-wrench validation across plug/gripper orientations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
DEFAULT_OUTPUT = HERE / "output" / "orientation_sweep"
BASELINE = HERE / "output" / "fp32"

CASES = {
    "roll_pos": {
        "gripper_roll": 12.0,
        "gripper_pitch": 0.0,
        "plug_roll": 6.0,
        "plug_pitch": 0.0,
    },
    "roll_neg": {
        "gripper_roll": -12.0,
        "gripper_pitch": 0.0,
        "plug_roll": -6.0,
        "plug_pitch": 0.0,
    },
    "pitch_pos": {
        "gripper_roll": 0.0,
        "gripper_pitch": 12.0,
        "plug_roll": 0.0,
        "plug_pitch": 8.0,
    },
    "pitch_neg": {
        "gripper_roll": 0.0,
        "gripper_pitch": -12.0,
        "plug_roll": 0.0,
        "plug_pitch": -8.0,
    },
    "compound": {
        "gripper_roll": 10.0,
        "gripper_pitch": -10.0,
        "plug_roll": 5.0,
        "plug_pitch": -6.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun complete cases")
    return parser.parse_args()


def run_case(name: str, values: dict[str, float], output: Path, args: argparse.Namespace) -> None:
    result = output / name / "fp32"
    required = [result / "metrics.json", result / "contact_wrenches.csv"]
    if not args.no_video:
        required.append(result / "simulation.mp4")
    if not args.force and all(path.is_file() for path in required):
        print(f"Skipping complete case {name}", flush=True)
        return
    command = [
        sys.executable,
        str(RUNNER),
        "--precision", "single",
        "--output", str(output / name),
        "--case-name", name,
        "--gripper-roll-deg", str(values["gripper_roll"]),
        "--gripper-pitch-deg", str(values["gripper_pitch"]),
        "--plug-roll-deg", str(values["plug_roll"]),
        "--plug-pitch-deg", str(values["plug_pitch"]),
    ]
    if args.no_video:
        command.append("--no-video")
    subprocess.run(command, cwd=HERE.parents[1], check=True)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def phase_force_balance(rows: list[dict[str, str]], phase: str) -> dict[str, float]:
    selected = [row for row in rows if row["phase"] == phase]
    table = np.array([float(row["table_measured_force_world_z"]) for row in selected])
    transmitted = np.array(
        [
            float(row["left_measured_force_world_z"])
            + float(row["right_measured_force_world_z"])
            + 0.08 * 9.81
            for row in selected
        ]
    )
    error = table - transmitted
    return {
        "rmse_n": float(np.sqrt(np.mean(error**2))),
        "bias_n": float(np.mean(error)),
        "correlation": float(np.corrcoef(table, transmitted)[0, 1]),
    }


def load_window_causality(rows: list[dict[str, str]]) -> dict[str, float | int]:
    phases = {
        "ramp_2N", "hold_2N", "ramp_5N", "hold_5N", "ramp_10N", "hold_10N"
    }
    selected = [row for row in rows if row["phase"] in phases]
    table = np.array([float(row["table_measured_force_world_z"]) for row in selected])
    transmitted = np.array(
        [
            float(row["left_measured_force_world_z"])
            + float(row["right_measured_force_world_z"])
            + 0.08 * 9.81
            for row in selected
        ]
    )
    table_delta = np.diff(table)
    transmitted_delta = np.diff(transmitted)
    table_delta -= table_delta.mean()
    transmitted_delta -= transmitted_delta.mean()
    lag_values = range(-10, 11)
    correlations = []
    for lag in lag_values:
        if lag < 0:
            first, second = table_delta[-lag:], transmitted_delta[:lag]
        elif lag > 0:
            first, second = table_delta[:-lag], transmitted_delta[lag:]
        else:
            first, second = table_delta, transmitted_delta
        correlations.append(float(np.corrcoef(first, second)[0, 1]))
    best_index = int(np.nanargmax(correlations))
    best_lag = list(lag_values)[best_index]
    return {
        "force_correlation": float(np.corrcoef(table, transmitted)[0, 1]),
        "best_derivative_lag_samples": best_lag,
        "best_derivative_lag_s": best_lag * 0.005,
        "best_derivative_correlation": correlations[best_index],
    }


def achieved_orientation(rows: list[dict[str, str]]) -> dict[str, list[float]] | None:
    if "ee_rotation_vector_world_x" not in rows[0]:
        return None
    initial = Rotation.from_rotvec(
        [float(rows[0][f"ee_rotation_vector_world_{axis}"]) for axis in "xyz"]
    )
    selected = [row for row in rows if row["phase"] == "hold_5N"]
    row = selected[len(selected) // 2]
    end_effector = Rotation.from_rotvec(
        [float(row[f"ee_rotation_vector_world_{axis}"]) for axis in "xyz"]
    )
    plug = Rotation.from_rotvec(
        [float(row[f"plug_rotation_vector_world_{axis}"]) for axis in "xyz"]
    )
    return {
        "gripper_world_delta_xyz_deg": (end_effector * initial.inv()).as_euler(
            "xyz", degrees=True
        ).tolist(),
        "plug_world_xyz_deg": plug.as_euler("xyz", degrees=True).tolist(),
    }


def summarize_case(name: str, directory: Path, commanded: dict[str, float]) -> dict:
    metrics = json.loads((directory / "metrics.json").read_text())
    rows = load_rows(directory / "contact_wrenches.csv")
    phases = ("hold_2N", "hold_5N", "hold_10N")
    return {
        "commanded_orientation_deg": commanded,
        "achieved_orientation": achieved_orientation(rows),
        "incidence_time_s": metrics["incidence_time_s"],
        "plateaus": metrics["plateaus"],
        "force_balance": {phase: phase_force_balance(rows, phase) for phase in phases},
        "causality": load_window_causality(rows),
        "interfaces": metrics["interfaces"],
        "execution_checks": metrics["execution_checks"],
    }


def make_plot(summary: dict, output: Path) -> None:
    cases = list(summary["cases"])
    phases = ("hold_2N", "hold_5N", "hold_10N")
    targets = np.array([2.0, 5.0, 10.0])
    x = np.arange(len(phases))
    width = 0.8 / len(cases)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    for index, name in enumerate(cases):
        case = summary["cases"][name]
        means = [case["plateaus"][phase]["mean_n"] for phase in phases]
        axes[0].bar(x + (index - (len(cases) - 1) / 2) * width, means, width, label=name)
        axes[1].plot(
            targets,
            [case["force_balance"][phase]["rmse_n"] for phase in phases],
            marker="o",
            label=name,
        )
        axes[2].plot(
            [0.0, 1.0],
            [
                case["interfaces"][side]["torque"]["relative_rmse"] * 100.0
                for side in ("left", "right")
            ],
            marker="o",
            label=f"{name} fingertips",
        )
    axes[0].plot(x, targets, "ko", label="target")
    axes[0].set_xticks(x, ["2 N", "5 N", "10 N"])
    axes[0].set_ylabel("Mean table normal force [N]")
    axes[1].set_xlabel("Target normal force [N]")
    axes[1].set_ylabel("Indirect force-balance RMSE [N]")
    axes[2].set_xticks([0.0, 1.0], ["left", "right"])
    axes[2].set_ylabel("Fingertip torque relative RMSE [%]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=3)
    fig.suptitle("FP32 contact sensing across plug/gripper orientations")
    fig.tight_layout()
    fig.savefig(output / "orientation_sweep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        run_case(name, CASES[name], output, args)

    summary: dict = {
        "precision": "fp32",
        "recommendation": "Use FP32 for routine SuperDex contact collection; reserve FP64 for numerical audits.",
        "cases": {},
    }
    if (BASELINE / "metrics.json").is_file():
        summary["cases"]["baseline"] = summarize_case(
            "baseline",
            BASELINE,
            {"gripper_roll": 0.0, "gripper_pitch": 0.0, "plug_roll": 0.0, "plug_pitch": 0.0},
        )
    for name in args.cases:
        summary["cases"][name] = summarize_case(name, output / name / "fp32", CASES[name])
    (output / "orientation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plot(summary, output)
    print(f"Orientation sweep results written to {output}")


if __name__ == "__main__":
    main()
