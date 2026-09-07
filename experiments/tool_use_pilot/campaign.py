#!/usr/bin/env python3
"""Bounded pilot collection in independent CPU worker processes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import time
from pilot import Spec


def specifications():
    result = []

    def add(**kw):
        result.append(Spec(episode_id=len(result), seed=20260907 + len(result), **kw))

    for i in range(48):
        add(
            task="hook",
            stiffness=30.0,
            offset=[0.0, 0.008, -0.008, 0.055][i % 4],
            angle_deg=[0.0, -3.0, 3.0][i // 4 % 3],
            group=f"hook-{i // 3}",
            duration=18.5,
        )
    for i in range(24):
        add(
            task="push",
            friction=[0.3, 0.5, 0.8][i % 3],
            offset=[0.0, 0.006, -0.006, 0.032][i // 3 % 4],
            angle_deg=[-3.0, 3.0][i // 12],
            group=f"push-{i // 3}",
        )
    for i in range(32):
        for stiffness in (30.0, 60.0, 90.0):
            add(
                task="probe",
                stiffness=stiffness,
                pause=[0.25, 0.75, 1.5][i % 3],
                offset=[0.0, 0.006, -0.006][i // 3 % 3],
                angle_deg=[-2.0, 0.0, 2.0][i // 9 % 3],
                group=f"probe-{i % 27}",
                duration=14.0,
                branch_time=8.7 + [0.25, 0.75, 1.5][i % 3],
            )
    for i in range(12):
        add(
            task="calibration",
            angle_deg=[-3.0, 0.0, 3.0][i % 3],
            friction=[0.3, 0.5, 0.8][i // 3 % 3],
            group=f"calibration-{i // 3}",
        )
    return result


def worker(spec, output, branch):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import superdex.physics as p
    from pilot import run

    p.initialize(num_worker_threads=0)
    try:
        return run(spec, Path(output), branches=branch)
    finally:
        p.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--task", choices=["hook", "push", "probe", "calibration"])
    parser.add_argument("--no-branches", action="store_true")
    args = parser.parse_args()
    specs = specifications()
    if args.qualification:
        specs = [specs[i] for i in [0, 1, 2, 3, 48, 49, 50, 72, 73, 74, 168, 169]]
    if args.task:
        specs = [s for s in specs if s.task == args.task]
    if args.limit:
        specs = specs[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "specs.json").write_text(
        json.dumps([asdict(s) for s in specs], indent=2)
    )
    start = time.perf_counter()
    results = []
    # Twelve anchors each have four continuations, in addition to 180 main episodes.
    anchors = {0, 4, 8, 12, 48, 51, 54, 57, 72, 75, 78, 81}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(
                worker,
                s,
                str(args.output),
                s.episode_id in anchors and not args.no_branches,
            ): s
            for s in specs
        }
        for future in as_completed(pending):
            spec = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(episode_id=spec.episode_id, error=repr(exc))
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    files = list(args.output.glob("*.h5"))
    import h5py

    frames = 0
    for path in files:
        with h5py.File(path) as f:
            frames += len(f["observations/timestamps"])
    # Portable master index: relative external links retain one copy per episode.
    with h5py.File(args.output / "dataset.h5", "w") as f:
        f.attrs.update(
            schema_version="vt_acwm_superdex_tool_pilot_v1",
            episode_count=len(files),
            force_units="N",
            torque_units="N m",
            position_units="m",
            quaternion_order="xyzw",
            sample_period_s=0.01,
            tactile_grid_shape=[7, 9, 3],
        )
        group = f.create_group("episodes")
        for path in sorted(files):
            group[path.stem] = h5py.ExternalLink(path.name, "/")
    timing = dict(
        wall_seconds=time.perf_counter() - start,
        workers=args.workers,
        episodes=len(files),
        recorded_frames=frames,
        physics="CPU FP32",
        rgb_in_timing=False,
    )
    timing["recorded_frames_per_second"] = frames / timing["wall_seconds"]
    (args.output / "timing.json").write_text(json.dumps(timing, indent=2))
    print(timing)


if __name__ == "__main__":
    main()
