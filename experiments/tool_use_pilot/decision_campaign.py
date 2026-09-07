"""Reproducible bounded qualification and decision-anchor manifests."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time
from decisions import DecisionSpec, worker


def specs_for(stage, kind, grip=35, modulus=200000):
    records = []

    def add(**kw):
        base = dict(
            kind=kind,
            grip_force_n=grip,
            gel_modulus_pa=modulus,
            revision=2,
            case="captured" if kind == "hook" else "seated",
            stiffness=30,
            dt=0.01 if kind == "hook" else 0.005,
            correction_m=0.012 if kind == "hook" else 0.0015,
        )
        base.update(kw)
        i = len(records)
        records.append(
            DecisionSpec(
                episode_id=i, seed=20260907 + i, group=f"{kind}-{stage}-{i}", **base
            )
        )

    if stage == "grip":
        for g in (15, 25, 35):
            for e in (100000, 200000, 400000):
                add(grip_force_n=g, gel_modulus_pa=e)
    elif stage == "resistance":
        for k in (30, 60, 90) if kind == "hook" else (0.01, 0.02, 0.04):
            add(**({"stiffness": k} if kind == "hook" else {"torsion_nm_rad": k}))
        for mu in (0.8, 1.1, 1.4):
            add(gel_friction=mu)
        for mu in (0.3, 0.5, 0.8):
            add(friction=mu)
    elif stage == "boundary":
        if kind == "hook":
            for offset in (
                -0.048,
                -0.042,
                -0.036,
                -0.030,
                -0.024,
                -0.018,
                -0.012,
                0,
                0.012,
                0.018,
                0.024,
                0.030,
                0.036,
                0.042,
                0.048,
            ):
                add(offset=offset)
        else:
            for x in (-0.002, -0.001, 0, 0.001, 0.002):
                add(lateral_m=x)
            for yaw in (-12, -8, -4, 4, 8, 12):
                add(yaw_deg=yaw)
    elif stage == "decision_screen":
        if kind == "hook":
            for i in range(3):
                for mirror in (False, True):
                    for case, offset in (("captured", 0.038), ("near_miss", 0.043)):
                        add(
                            case=case,
                            offset=(offset + (i - 1) * 0.0005) * (-1 if mirror else 1),
                            mirror_fixture=mirror,
                        )
        else:
            for case, x, yaw in (
                ("seated", 0, 0),
                ("unseated", 0, 0),
                ("left_rim", -0.0015, 0),
                ("right_rim", 0.0015, 0),
                ("positive_yaw", 0, 8),
                ("negative_yaw", 0, -8),
            ):
                add(case=case, lateral_m=x, yaw_deg=yaw)
    elif stage == "evaluation":
        if kind == "hook":
            for i in range(6):
                for mirror in (False, True):
                    for case, offset in (("captured", 0.038), ("near_miss", 0.043)):
                        add(
                            case=case,
                            offset=(offset + (i - 2.5) * 0.0003)
                            * (-1 if mirror else 1),
                            mirror_fixture=mirror,
                        )
        else:
            for i in range(4):
                for case, x, yaw in (
                    ("seated", 0, 0),
                    ("unseated", 0, 0),
                    ("left_rim", -0.0015, 0),
                    ("right_rim", 0.0015, 0),
                    ("positive_yaw", 0, 8),
                    ("negative_yaw", 0, -8),
                ):
                    add(
                        case=case,
                        lateral_m=x + (i - 1.5) * 0.0001,
                        yaw_deg=yaw,
                        torsion_nm_rad=(0.01, 0.02)[i % 2],
                    )
    elif stage == "timestep":
        for dt in (0.01, 0.005):
            add(dt=dt)
    else:
        raise ValueError(stage)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=[
            "grip",
            "resistance",
            "boundary",
            "decision_screen",
            "evaluation",
            "timestep",
        ],
        required=True,
    )
    p.add_argument("--kind", choices=["hook", "key"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--grip", type=float, default=35)
    p.add_argument("--modulus", type=float, default=200000)
    p.add_argument("--branches", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = specs_for(args.stage, args.kind, args.grip, args.modulus)
    manifest = args.output / "specs.json"
    if manifest.exists():
        raise FileExistsError(
            "Use a fresh output directory; do not overwrite qualification evidence"
        )
    manifest.write_text(json.dumps([asdict(s) for s in specs], indent=2))
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(worker, asdict(s), str(args.output), args.branches): s
            for s in specs
        }
        for future in as_completed(pending):
            try:
                r = future.result()
            except Exception as e:
                r = dict(spec=asdict(pending[future]), error=repr(e))
            results.append(r)
            print(json.dumps(r), flush=True)
    elapsed = time.perf_counter() - start
    import h5py

    frames = 0
    for path in args.output.glob("*.h5"):
        with h5py.File(path) as f:
            frames += len(f["timestamps"])
    report = dict(
        stage=args.stage,
        kind=args.kind,
        seconds=elapsed,
        workers=args.workers,
        raw_frames=frames,
        frames_per_second=frames / elapsed,
        results=results,
    )
    (args.output / "campaign.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
