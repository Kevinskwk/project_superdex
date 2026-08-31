#!/usr/bin/env python3
"""Select an efficient stable uncalibrated gel material in FP32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
DEFAULT_OUTPUT = HERE / "output/material_sweep"
MODULI = (50_000.0, 100_000.0, 200_000.0, 400_000.0)
DAMPING = (0.001, 0.003, 0.010)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_case(output: Path, modulus: float, damping: float, force: bool) -> dict:
    name = f"e{int(modulus/1000)}k_d{damping:g}".replace(".", "p")
    directory = output / name
    metrics_path = directory / "metrics.json"
    elapsed = None
    if force or not metrics_path.exists():
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, str(RUNNER), "--no-video", "--output", str(directory),
             "--case-name", name, "--youngs-modulus-pa", str(modulus),
             "--stiffness-damping", str(damping)],
            cwd=HERE.parents[1], check=True,
        )
        elapsed = time.perf_counter() - started
    metrics = json.loads(metrics_path.read_text())
    if elapsed is None:
        elapsed = float(metrics.get("wall_time_s", float("inf")))
    plateaus = metrics["plateaus"]
    stable = (
        metrics["causality"]["max_indentation_m"] < 0.0032
        and all(value["both_gels_contact_fraction"] >= 0.95 for value in plateaus.values())
        and all(abs(value["mean_tracking_error_n"]) <= 1.0 for value in plateaus.values())
    )
    correlations = [value["indirect_correlation"] for value in plateaus.values()]
    rmses = [value["indirect_rmse_n"] for value in plateaus.values()]
    return {
        "name": name, "youngs_modulus_pa": modulus,
        "stiffness_damping_s": damping, "wall_time_s": elapsed,
        "stable": stable, "minimum_correlation": min(correlations),
        "mean_indirect_rmse_n": sum(rmses)/len(rmses), "metrics": metrics,
    }


def score(case: dict) -> tuple:
    return (
        not case["stable"], -case["minimum_correlation"],
        case["mean_indirect_rmse_n"], case["wall_time_s"],
    )


def main() -> None:
    args = parse_args(); output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    stage_one = [run_case(output, modulus, 0.003, args.force) for modulus in MODULI]
    selected_modulus = min(stage_one, key=score)["youngs_modulus_pa"]
    stage_two = [
        run_case(output, selected_modulus, damping, args.force)
        for damping in DAMPING if damping != 0.003
    ]
    cases = stage_one + stage_two
    selected = min(cases, key=score)
    summary = {
        "selection_rule": "stable, highest minimum indirect correlation, lowest indirect RMSE, shortest runtime",
        "selected": {key:selected[key] for key in ("name","youngs_modulus_pa","stiffness_damping_s")},
        "cases": cases,
    }
    (output/"material_sweep.json").write_text(json.dumps(summary,indent=2)+"\n")
    (HERE/"selected_material.json").write_text(json.dumps({
        "youngs_modulus_pa": selected["youngs_modulus_pa"],
        "poisson_ratio": 0.49, "mass_damping_s_inv": 5.0,
        "stiffness_damping_s": selected["stiffness_damping_s"],
        "calibration_status": "uncalibrated numerical selection",
    },indent=2)+"\n")
    print(json.dumps(summary["selected"],indent=2))


if __name__ == "__main__":
    main()
