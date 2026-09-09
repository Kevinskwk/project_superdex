#!/usr/bin/env python3
"""Benchmark gel/tool friction against grasp retention for two difficult tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from campaign_specs import build_campaign
from run_fidelity_campaign import (
    assemble_master, launch_workers, reserved_core_layout,
)
from scfields_assets import load_manifest


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "output" / "friction_benchmark"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    campaign = build_campaign(load_manifest()["tools"])
    bases = [
        next(spec for spec in campaign if spec.tool_name == "tool_cylinder_45" and spec.protocol == "tip_stroke"),
        next(spec for spec in campaign if spec.tool_name == "peeler_7" and spec.protocol == "peel"),
    ]
    scales = (0.6, 1.0, 1.4, 2.0)
    specs = []
    for base in bases:
        for scale in scales:
            for replicate in range(args.replicates):
                specs.append(replace(
                    base, episode_id=len(specs), seed=base.seed + replicate + int(scale * 1000),
                    campaign_block="friction_benchmark", replicate=replicate,
                    friction_scale=scale,
                ))
    usable, _reserved = reserved_core_layout(0.25)
    runs = []
    for scale in scales:
        selected = [spec for spec in specs if spec.friction_scale == scale]
        runs.append(launch_workers(
            selected, output, processes=1, parallel_envs=4, worker_threads=2,
            steps=args.steps, dt=0.01, compression="lzf", core_pool=usable,
            label=f"friction_mu_{str(scale).replace('.', 'p')}", gel_friction=scale,
        ))
    shard_paths = [Path(path) for run in runs for path in run["shards"]]
    result = {
        "runs": runs,
        "frames": sum(run["frames"] for run in runs),
        "wall_seconds": sum(run["wall_seconds"] for run in runs),
    }
    result["aggregate_environment_fps"] = result["frames"] / result["wall_seconds"]
    master = assemble_master(output, shard_paths, specs)
    metrics = {}
    for path in shard_paths:
        metrics.update({
            int(key): value
            for key, value in json.loads(path.with_suffix(".json").read_text())["metrics"].items()
        })
    rows = []
    for base in bases:
        for scale in scales:
            selected = [
                metrics[spec.episode_id] for spec in specs
                if spec.tool_name == base.tool_name and spec.friction_scale == scale
            ]
            rows.append({
                "tool_name": base.tool_name, "protocol": base.protocol,
                "tool_friction": scale, "episodes": len(selected),
                "retention_rate": float(np.mean([m["grasp_retained"] for m in selected])),
                "slip_median_m": float(np.median([m["grasp_slip_max_m"] for m in selected])),
                "slip_max_m": float(np.max([m["grasp_slip_max_m"] for m in selected])),
                "bilateral_fraction_mean": float(np.mean([m["bilateral_grasp_fraction"] for m in selected])),
                "dense_pass_rate": float(np.mean([m["dense_pass"] for m in selected])),
            })
    recommended = max(scales)
    for scale in scales:
        # Keep 2 mm margin below the 12 mm publication cutoff rather than
        # selecting the first coefficient that only barely passes.
        if all(
            row["retention_rate"] == 1.0 and row["slip_max_m"] <= 0.010
            for row in rows if row["tool_friction"] == scale
        ):
            recommended = scale
            break
    report = {
        "recommended_gel_and_tool_friction": recommended,
        "grip_force_per_finger_n": 28.0,
        "collection": result,
        "rows": rows,
        "master_hdf5": str(master),
    }
    (output / "friction_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    table = "\n".join(
        f"| {row['tool_name']} | {row['protocol']} | {row['tool_friction']:.1f} | "
        f"{row['retention_rate']:.0%} | {row['slip_median_m']*1000:.2f} mm | "
        f"{row['slip_max_m']*1000:.2f} mm | {row['bilateral_fraction_mean']:.1%} |"
        for row in rows
    )
    (output / "friction_benchmark.md").write_text(
        "# Gel/tool friction and grasp-retention benchmark\n\n"
        f"Recommended gel/tool coefficient: **{recommended:.1f}** with 28 N per finger.\n\n"
        "| Tool | Motion | Gel/tool friction | Retained | Median slip | Max slip | Bilateral contact |\n"
        "|---|---|---:|---:|---:|---:|---:|\n" + table + "\n"
    )
    print(f"wrote friction benchmark to {output}", flush=True)


if __name__ == "__main__":
    main()
