#!/usr/bin/env python3
"""Autotune CPU sharding, run the SCFields campaign, and assemble its report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import h5py
import numpy as np
import psutil

from campaign_specs import EpisodeSpec, build_campaign, write_specs
from fidelity_campaign import SCHEMA_VERSION
from scfields_assets import DEFAULT_ROOT, load_manifest, prepare


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "output" / "scfields_fidelity_999x200"
LEGACY_BASELINE_FPS = 32.61
DATASET_TARGETS = {
    "Isaac Gym accepted": 23_678,
    "MuJoCo accepted": 23_749,
    "combined accepted": 47_427,
}


def physical_cores() -> list[list[int]]:
    """Return allowed logical CPU siblings grouped by physical core."""
    allowed = set(os.sched_getaffinity(0))
    result = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], check=True,
        capture_output=True, text=True,
    ).stdout
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for line in result.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, _node = map(int, line.split(","))
        if cpu in allowed:
            groups[(socket, core)].append(cpu)
    return [sorted(groups[key]) for key in sorted(groups)]


def reserved_core_layout(reserve_fraction: float) -> tuple[list[list[int]], list[list[int]]]:
    cores = physical_cores()
    reserve = max(1, int(math.ceil(len(cores) * reserve_fraction)))
    if len(cores) - reserve < 1:
        raise RuntimeError("not enough physical cores after reservation")
    return cores[:-reserve], cores[-reserve:]


def partition_cores(cores: list[list[int]], processes: int) -> list[list[int]]:
    result = [[] for _ in range(processes)]
    for index, siblings in enumerate(cores):
        result[index % processes].extend(siblings)
    return result


def process_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "SUPERDEX_ASSETS_PATH": str(PROJECT_ROOT / "assets"),
    })
    env.pop("SUPERDEX_PRECISION", None)
    return env


def launch_workers(
    specs: list[EpisodeSpec], output: Path, processes: int, parallel_envs: int,
    worker_threads: int, steps: int, dt: float, compression: str,
    core_pool: list[list[int]], label: str, gel_friction: float = 1.4,
    grip_force_n: float = 28.0,
) -> dict[str, Any]:
    spec_root, shard_root = output / f"{label}_specs", output / f"{label}_shards"
    spec_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    partitions = [specs[index::processes] for index in range(processes)]
    affinities = partition_cores(core_pool, processes)
    pending: list[tuple[subprocess.Popen[str], Path, Path, list[int]]] = []
    launched = 0
    for index, partition in enumerate(partitions):
        if not partition:
            continue
        spec_path = spec_root / f"spec_{index:03d}.json"
        shard_path = shard_root / f"shard_{index:03d}.hdf5"
        write_specs(spec_path, partition)
        # Completed shards are resumable only when their exact episode IDs match.
        if shard_path.exists() and shard_path.with_suffix(".json").exists():
            with h5py.File(shard_path, "r") as stream:
                found = stream["index/episode_id"][...].tolist()
            if found == [item.episode_id for item in partition]:
                continue
            raise RuntimeError(f"existing shard does not match requested specs: {shard_path}")
        cpu_list = ",".join(str(cpu) for cpu in affinities[index])
        command = [
            "taskset", "-c", cpu_list, sys.executable, str(HERE / "fidelity_campaign.py"),
            "--spec-file", str(spec_path), "--manifest", str(DEFAULT_ROOT / "manifest.json"),
            "--output", str(shard_path), "--steps", str(steps),
            "--parallel-envs", str(min(parallel_envs, len(partition))),
            "--worker-threads", str(worker_threads), "--dt", str(dt),
            "--compression", compression, "--gel-friction", str(gel_friction),
            "--grip-force-n", str(grip_force_n),
        ]
        log_path = shard_root / f"shard_{index:03d}.log"
        log = log_path.open("w")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, text=True,
            env=process_environment(), start_new_session=True,
        )
        process._campaign_log = log  # type: ignore[attr-defined]
        pending.append((process, shard_path, log_path, affinities[index]))
        launched += 1
    start = time.perf_counter()
    cpu_samples, rss_samples = [], []
    while any(process.poll() is None for process, *_ in pending):
        cpu_samples.append(psutil.cpu_percent(interval=0.25))
        rss = 0
        for process, *_ in pending:
            if process.poll() is None:
                try:
                    rss += psutil.Process(process.pid).memory_info().rss
                except psutil.Error:
                    pass
        rss_samples.append(rss)
    wall = time.perf_counter() - start
    failures = []
    for process, _shard, log_path, _affinity in pending:
        process._campaign_log.close()  # type: ignore[attr-defined]
        if process.returncode:
            failures.append((process.returncode, log_path))
    if failures:
        details = "\n".join(
            f"{path} (exit {code}):\n{path.read_text()[-4000:]}" for code, path in failures
        )
        raise RuntimeError(f"campaign worker failure\n{details}")
    shard_paths = sorted(shard_root.glob("shard_*.hdf5"))
    sidecars = [json.loads(path.with_suffix(".json").read_text()) for path in shard_paths]
    frames = sum(item["benchmark"]["environment_frames"] for item in sidecars)
    # If all shards were resumed, use their summed durations as a conservative record.
    effective_wall = wall if launched else sum(item["benchmark"]["total_seconds"] for item in sidecars)
    return {
        "label": label, "processes": processes, "parallel_envs": parallel_envs,
        "worker_threads": worker_threads, "steps": steps, "frames": frames,
        "gel_friction": gel_friction, "grip_force_per_finger_n": grip_force_n,
        "wall_seconds": effective_wall, "aggregate_environment_fps": frames / max(effective_wall, 1e-9),
        "cpu_percent_mean": float(np.mean(cpu_samples)) if cpu_samples else float("nan"),
        "cpu_percent_peak": float(np.max(cpu_samples)) if cpu_samples else float("nan"),
        "peak_rss_gib": float(np.max(rss_samples) / 2**30) if rss_samples else float("nan"),
        "core_affinities": affinities, "shards": [str(path) for path in shard_paths],
        "launched_workers": launched,
    }


def autotune(
    specs: list[EpisodeSpec], output: Path, cores: list[list[int]], steps: int,
    dt: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    physical = len(cores)
    process_choices = sorted(set(value for value in (1, 2, 4, 6) if value <= physical))
    results = []
    for processes in process_choices:
        cores_each = max(1, physical // processes)
        parallel_envs = 8 if processes <= 2 else (4 if processes <= 4 else 2)
        worker_threads = -1 if processes == 1 else max(0, cores_each - 1)
        count = processes * parallel_envs
        # Use standardized contacts across several shape families. Eighty steps
        # includes grasp, release, and table contact while keeping tuning bounded.
        probe = [specs[index * 18 % 486] for index in range(count)]
        probe = [EpisodeSpec(**{**item.to_dict(), "episode_id": index}) for index, item in enumerate(probe)]
        result = launch_workers(
            probe, output / "autotune", processes, parallel_envs, worker_threads,
            min(80, steps), dt, "none", cores, f"p{processes}_e{parallel_envs}_w{worker_threads}",
        )
        results.append(result)
        print(
            f"autotune {processes} process(es), {parallel_envs} env/process, "
            f"workers={worker_threads}: {result['aggregate_environment_fps']:.1f} env-fps",
            flush=True,
        )
    valid = [item for item in results if item["aggregate_environment_fps"] > 0]
    if not valid:
        raise RuntimeError("no autotune configuration completed")
    return max(valid, key=lambda item: item["aggregate_environment_fps"]), results


def assemble_master(output: Path, shard_paths: list[Path], specs: list[EpisodeSpec]) -> Path:
    master = output / "scfields_fidelity_master.hdf5"
    partial = master.with_suffix(master.suffix + ".partial")
    text = h5py.string_dtype("utf-8")
    with h5py.File(shard_paths[0], "r") as first_shard:
        steps_per_episode = int(first_shard.attrs["steps_per_episode"])
    with h5py.File(partial, "w", libver="latest") as stream:
        stream.attrs.update({
            "schema_version": SCHEMA_VERSION, "domain": "superdex",
            "simulator": "superdex_mochi", "episode_count": len(specs),
            "steps_per_episode": steps_per_episode,
            "external_link_shards": len(shard_paths),
        })
        index = stream.create_group("index")
        index.create_dataset("episode_id", data=np.asarray([s.episode_id for s in specs], dtype=np.int32))
        for key in ("tool_name", "tool_family", "campaign_block", "protocol"):
            index.create_dataset(key, data=np.asarray([getattr(s, key) for s in specs], dtype=text))
        episodes = stream.create_group("episodes", track_order=True)
        for shard in shard_paths:
            relative = os.path.relpath(shard, master.parent)
            with h5py.File(shard, "r") as source:
                names = list(source["episodes"])
            for name in names:
                if name in episodes:
                    raise RuntimeError(f"duplicate episode while assembling master: {name}")
                episodes[name] = h5py.ExternalLink(relative, f"/episodes/{name}")
    partial.replace(master)
    with h5py.File(master, "r") as stream:
        if len(stream["episodes"]) != len(specs):
            raise RuntimeError("master HDF5 episode count mismatch")
        for spec in specs:
            group = stream[f"episodes/episode_{spec.episode_id:06d}"]
            if group["actions"].shape[0] != stream.attrs["steps_per_episode"]:
                raise RuntimeError(f"invalid episode link for {spec.episode_id}")
    return master


def aggregate_report(
    output: Path, result: dict[str, Any], tuning: list[dict[str, Any]], specs: list[EpisodeSpec],
    master: Path,
) -> dict[str, Any]:
    metrics: dict[int, dict[str, Any]] = {}
    hdf5_bytes = master.stat().st_size
    for path_string in result["shards"]:
        path = Path(path_string)
        hdf5_bytes += path.stat().st_size
        sidecar = json.loads(path.with_suffix(".json").read_text())
        metrics.update({int(key): value for key, value in sidecar["metrics"].items()})
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        groups[(spec.tool_family, spec.protocol)].append(metrics[spec.episode_id])
    group_rows = []
    for (family, protocol), values in sorted(groups.items()):
        group_rows.append({
            "tool_family": family, "protocol": protocol, "episodes": len(values),
            "physical_valid_rate": float(np.mean([v["physical_valid"] for v in values])),
            "dense_pass_rate": float(np.mean([v["dense_pass"] for v in values])),
            "environment_pass_rate": float(np.mean([v["environment_pass"] for v in values])),
            "force_nrmse_median": float(np.nanmedian([v["environment_force_nrmse"] for v in values])),
            "torque_nrmse_median": float(np.nanmedian([v["environment_torque_nrmse"] for v in values])),
        })
    def dimension_summary(attribute: str) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spec in specs:
            buckets[str(getattr(spec, attribute))].append(metrics[spec.episode_id])
        rows = []
        for name, values in sorted(buckets.items()):
            valid = [value for value in values if value["physical_valid"]]
            rows.append({
                attribute: name, "episodes": len(values),
                "physical_valid_rate": float(np.mean([v["physical_valid"] for v in values])),
                "dense_pass_rate": float(np.mean([v["dense_pass"] for v in values])),
                "environment_pass_rate_all": float(np.mean([v["environment_pass"] for v in values])),
                "environment_pass_rate_valid": float(
                    np.mean([v["environment_pass"] for v in valid]) if valid else np.nan
                ),
                "force_nrmse_median_valid": float(
                    np.nanmedian([v["environment_force_nrmse"] for v in valid]) if valid else np.nan
                ),
                "torque_nrmse_median_valid": float(
                    np.nanmedian([v["environment_torque_nrmse"] for v in valid]) if valid else np.nan
                ),
                "force_correlation_median_valid": float(
                    np.nanmedian([v["environment_force_correlation"] for v in valid]) if valid else np.nan
                ),
                "cop_p95_median_m_valid": float(
                    np.nanmedian([v["cop_p95_m"] for v in valid]) if valid else np.nan
                ),
            })
        return rows

    valid_metrics = [value for value in metrics.values() if value["physical_valid"]]

    def metric_stats(key: str, values: list[dict[str, Any]]) -> dict[str, float]:
        samples = np.asarray([value[key] for value in values], dtype=float)
        samples = samples[np.isfinite(samples)]
        return {
            "median": float(np.median(samples)),
            "p05": float(np.percentile(samples, 5)),
            "p95": float(np.percentile(samples, 95)),
        }

    wrench_fidelity = {
        "dense_field_vs_direct_gel_contact": {
            "pass_rate": float(np.mean([value["dense_pass"] for value in metrics.values()])),
            "force_relative_p95": metric_stats("dense_force_rel_p95", valid_metrics),
            "force_absolute_p95_n": metric_stats("dense_force_abs_p95_n", valid_metrics),
            "torque_absolute_p95_nm": metric_stats("dense_torque_abs_p95_nm", valid_metrics),
            "force_cosine_p05": metric_stats("dense_force_cosine_p05", valid_metrics),
            "centroid_p95_m": metric_stats("dense_centroid_p95_m", valid_metrics),
        },
        "field_inferred_vs_tool_table_contact": {
            "pass_rate": float(np.mean([value["environment_pass"] for value in valid_metrics])),
            "force_nrmse": metric_stats("environment_force_nrmse", valid_metrics),
            "torque_nrmse": metric_stats("environment_torque_nrmse", valid_metrics),
            "force_correlation": metric_stats("environment_force_correlation", valid_metrics),
            "cop_p95_m": metric_stats("cop_p95_m", valid_metrics),
        },
    }
    repeat_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for spec in specs:
        if spec.campaign_block == "standard":
            repeat_groups[(spec.tool_name, spec.protocol)].append(
                metrics[spec.episode_id]["peak_environment_force_n"]
            )
    cvs = [float(np.std(values) / max(np.mean(values), 1e-9))
           for values in repeat_groups.values() if len(values) >= 2]
    total_seconds = result["wall_seconds"]
    seconds_episode = total_seconds / len(specs)
    bytes_episode = hdf5_bytes / len(specs)
    summary = {
        "schema_version": SCHEMA_VERSION, "episodes": len(specs),
        "steps_per_episode": result["steps"], "environment_frames": result["frames"],
        "collection": result, "autotune": tuning,
        "master_hdf5": str(master), "hdf5_bytes_including_shards": hdf5_bytes,
        "hdf5_gib_including_shards": hdf5_bytes / 2**30,
        "physical_valid_rate": float(np.mean([v["physical_valid"] for v in metrics.values()])),
        "grasp_retention_rate": float(np.mean([v["grasp_retained"] for v in metrics.values()])),
        "grasp_slip_max_mm": float(max(v["grasp_slip_max_m"] for v in metrics.values()) * 1000),
        "grasp_slip_p99_mm": float(np.percentile(
            [v["grasp_slip_max_m"] for v in metrics.values()], 99
        ) * 1000),
        "grasp_rotation_max_deg": float(max(
            v["grasp_rotation_max_deg"] for v in metrics.values()
        )),
        "bilateral_grasp_fraction_min": float(min(
            v["bilateral_grasp_fraction"] for v in metrics.values()
        )),
        "episodes_without_environment_contact": int(sum(
            v["environment_contact_frames"] == 0 for v in metrics.values()
        )),
        "dense_pass_rate": float(np.mean([v["dense_pass"] for v in metrics.values()])),
        "environment_pass_rate": float(np.mean([v["environment_pass"] for v in metrics.values()])),
        "environment_pass_rate_on_valid": float(
            np.mean([v["environment_pass"] for v in valid_metrics])
        ),
        "wrench_fidelity": wrench_fidelity,
        "repeatability_cv_median": float(np.median(cvs)) if cvs else float("nan"),
        "repeatability_cv_p95": float(np.percentile(cvs, 95)) if cvs else float("nan"),
        "speedup_vs_previous_32_61_env_fps": result["aggregate_environment_fps"] / LEGACY_BASELINE_FPS,
        "group_results": group_rows,
        "campaign_block_results": dimension_summary("campaign_block"),
        "tool_family_results": dimension_summary("tool_family"),
        "dataset_size_estimates": {
            name: {
                "episodes": count, "hours": seconds_episode * count / 3600,
                "gib": bytes_episode * count / 2**30,
            } for name, count in DATASET_TARGETS.items()
        },
    }
    (output / "campaign_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    failed = [row for row in group_rows if row["physical_valid_rate"] < 0.95
              or row["dense_pass_rate"] < 0.90 or row["environment_pass_rate"] < 0.90]
    estimates = "\n".join(
        f"| {name} | {value['episodes']:,} | {value['hours']:.2f} h | {value['gib']:.2f} GiB |"
        for name, value in summary["dataset_size_estimates"].items()
    )
    block_rows = "\n".join(
        f"| {row['campaign_block']} | {row['episodes']} | {row['physical_valid_rate']:.1%} | "
        f"{row['dense_pass_rate']:.1%} | {row['environment_pass_rate_valid']:.1%} | "
        f"{row['force_nrmse_median_valid']:.3f} | {row['torque_nrmse_median_valid']:.3f} | "
        f"{row['force_correlation_median_valid']:.3f} | {row['cop_p95_median_m_valid']*1000:.2f} mm |"
        for row in summary["campaign_block_results"]
    )
    family_rows = "\n".join(
        f"| {row['tool_family']} | {row['episodes']} | {row['physical_valid_rate']:.1%} | "
        f"{row['dense_pass_rate']:.1%} | {row['environment_pass_rate_valid']:.1%} | "
        f"{row['force_nrmse_median_valid']:.3f} | {row['torque_nrmse_median_valid']:.3f} |"
        for row in summary["tool_family_results"]
    )
    failure_rows = "\n".join(
        f"| {row['tool_family']} | {row['protocol']} | {row['episodes']} | "
        f"{row['physical_valid_rate']:.1%} | {row['dense_pass_rate']:.1%} | "
        f"{row['environment_pass_rate']:.1%} | {row['force_nrmse_median']:.3f} | "
        f"{row['torque_nrmse_median']:.3f} |" for row in failed
    ) or "| none | — | — | — | — | — | — | — |"
    dense_wrench = wrench_fidelity["dense_field_vs_direct_gel_contact"]
    external_wrench = wrench_fidelity["field_inferred_vs_tool_table_contact"]
    (output / "campaign_report.md").write_text(
        "# SCFields tool-use tactile fidelity campaign\n\n"
        f"Collected **{len(specs):,} episodes × {result['steps']} steps = "
        f"{result['frames']:,} environment-frames** in {total_seconds:.2f} s.\n\n"
        f"- Aggregate collection speed: **{result['aggregate_environment_fps']:.2f} env-frames/s**\n"
        f"- Speedup over the previous single-process collector: **{summary['speedup_vs_previous_32_61_env_fps']:.2f}×**\n"
        f"- Mean/peak host CPU: **{result['cpu_percent_mean']:.1f}% / {result['cpu_percent_peak']:.1f}%**\n"
        f"- Peak worker RSS: **{result['peak_rss_gib']:.2f} GiB**\n"
        f"- Physical validity: **{summary['physical_valid_rate']:.1%}**\n"
        f"- Grasp retention: **{summary['grasp_retention_rate']:.1%}** "
        f"(max/p99 slip {summary['grasp_slip_max_mm']:.2f}/{summary['grasp_slip_p99_mm']:.2f} mm; "
        f"max relative rotation {summary['grasp_rotation_max_deg']:.2f} deg; "
        f"minimum bilateral contact {summary['bilateral_grasp_fraction_min']:.1%})\n"
        f"- Episodes without tool/table contact: "
        f"**{summary['episodes_without_environment_contact']}**\n"
        f"- Dense gel-field fidelity pass: **{summary['dense_pass_rate']:.1%}**\n"
        f"- Dynamic environment-wrench pass: **{summary['environment_pass_rate']:.1%}**\n"
        f"- Dynamic environment-wrench pass among physically valid grasps: "
        f"**{summary['environment_pass_rate_on_valid']:.1%}**\n"
        f"- Repeatability peak-force CV median/p95: **{summary['repeatability_cv_median']:.1%} / "
        f"{summary['repeatability_cv_p95']:.1%}**\n\n"
        "The thresholds test simulator-internal consistency and causality against Mochi contact "
        "ground truth. They are not a calibration claim for a physical GelSight Mini.\n\n"
        "## Wrench comparison\n\n"
        f"- Dense field vs direct gel contact: median episode force relative-p95 "
        f"**{dense_wrench['force_relative_p95']['median']:.3f}**, force absolute-p95 "
        f"**{dense_wrench['force_absolute_p95_n']['median']:.3f} N**, torque absolute-p95 "
        f"**{dense_wrench['torque_absolute_p95_nm']['median']:.4f} N m**, and force-direction "
        f"cosine-p05 **{dense_wrench['force_cosine_p05']['median']:.3f}**.\n"
        f"- Field/inertia-inferred vs direct tool/table contact: median force NRMSE "
        f"**{external_wrench['force_nrmse']['median']:.3f}**, torque NRMSE "
        f"**{external_wrench['torque_nrmse']['median']:.3f}**, force correlation "
        f"**{external_wrench['force_correlation']['median']:.3f}**, and CoP-p95 "
        f"**{external_wrench['cop_p95_m']['median']*1000:.2f} mm**.\n\n"
        "## Results by campaign block\n\n"
        "| Block | N | Valid | Dense pass | Environment pass (valid) | Force NRMSE | Torque NRMSE | Force corr. | CoP p95 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n" + block_rows + "\n\n"
        "## Results by tool family\n\n"
        "| Family | N | Valid | Dense pass | Environment pass (valid) | Force NRMSE | Torque NRMSE |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n" + family_rows + "\n\n"
        "## Groups below a 90% fidelity or 95% validity target\n\n"
        "| Family | Protocol | N | Valid | Dense pass | Environment pass | Force NRMSE | Torque NRMSE |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n" + failure_rows + "\n\n"
        "## Dataset-size projections\n\n"
        "| Target | Episodes | Estimated collection | Estimated storage |\n"
        "|---|---:|---:|---:|\n" + estimates + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=999)
    parser.add_argument(
        "--episode-ids",
        help="Comma-separated campaign episode IDs (overrides --episodes; useful for targeted validation).",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--reserve-physical-cores", type=float, default=0.25)
    parser.add_argument("--skip-autotune", action="store_true")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Rebuild the master links and reports from completed campaign shards.",
    )
    parser.add_argument("--processes", type=int)
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument("--worker-threads", type=int)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--gel-friction", type=float, default=1.4)
    parser.add_argument(
        "--grip-force-n", type=float, default=35.0,
        help="Inward effort per finger; use at most 35 N for the 70 N Franka Hand limit.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.episodes, args.steps, args.skip_autotune = 8, 80, True
        args.processes, args.parallel_envs, args.worker_threads = 2, 4, 2
    if not 0.0 <= args.reserve_physical_cores < 1.0:
        parser.error("--reserve-physical-cores must be in [0, 1)")
    if not 0.0 < args.grip_force_n <= 35.0:
        parser.error("--grip-force-n must be in (0, 35] N per finger")
    return args


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (DEFAULT_ROOT / "manifest.json").exists():
        prepare(DEFAULT_ROOT)
    manifest = load_manifest(DEFAULT_ROOT / "manifest.json")
    full_specs = build_campaign(manifest["tools"])
    if args.episode_ids:
        requested = [int(value) for value in args.episode_ids.split(",")]
        by_id = {spec.episode_id: spec for spec in full_specs}
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise ValueError(f"unknown campaign episode IDs: {missing}")
        specs = [by_id[value] for value in requested]
    else:
        specs = full_specs[: min(args.episodes, len(full_specs))]
    write_specs(output / "campaign_specs.json", specs)
    usable, reserved = reserved_core_layout(args.reserve_physical_cores)
    topology = {
        "physical_cores_total": len(usable) + len(reserved),
        "physical_cores_used": len(usable), "physical_cores_reserved": len(reserved),
        "logical_cpus_used": sum(len(group) for group in usable),
        "logical_cpus_reserved": sum(len(group) for group in reserved),
        "used_core_siblings": usable, "reserved_core_siblings": reserved,
    }
    (output / "cpu_topology.json").write_text(json.dumps(topology, indent=2) + "\n")
    if args.report_only:
        old_report = json.loads((output / "campaign_report.json").read_text())
        result = old_report["collection"]
        shard_paths = sorted((output / "campaign_shards").glob("shard_*.hdf5"))
        result["shards"] = [str(path) for path in shard_paths]
        master = assemble_master(output, shard_paths, specs)
        aggregate_report(output, result, old_report.get("autotune", []), specs, master)
        print(f"rebuilt report and master links at {output}", flush=True)
        return
    tuning: list[dict[str, Any]] = []
    if args.skip_autotune:
        processes = args.processes or min(4, len(usable))
        parallel_envs = args.parallel_envs or 4
        worker_threads = args.worker_threads if args.worker_threads is not None else max(
            0, len(usable) // processes - 1
        )
    else:
        best, tuning = autotune(full_specs, output, usable, args.steps, args.dt)
        processes = args.processes or best["processes"]
        parallel_envs = args.parallel_envs or best["parallel_envs"]
        worker_threads = args.worker_threads if args.worker_threads is not None else best["worker_threads"]
    print(
        f"collection config: {processes} processes × {parallel_envs} envs, "
        f"{worker_threads} Mochi worker threads/process; using {len(usable)} physical cores "
        f"and reserving {len(reserved)}",
        flush=True,
    )
    result = launch_workers(
        specs, output, processes, parallel_envs, worker_threads, args.steps, args.dt,
        args.compression, usable, "campaign", args.gel_friction, args.grip_force_n,
    )
    shard_paths = [Path(path) for path in result["shards"]]
    master = assemble_master(output, shard_paths, specs)
    summary = aggregate_report(output, result, tuning, specs, master)
    print(
        f"completed {summary['episodes']} episodes at "
        f"{result['aggregate_environment_fps']:.1f} env-fps; master={master}", flush=True,
    )


if __name__ == "__main__":
    main()
