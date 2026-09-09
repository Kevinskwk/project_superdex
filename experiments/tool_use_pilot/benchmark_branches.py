"""Exact-state action interventions on mechanically qualified pilot anchors."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing
import time
import h5py
import numpy as np
from benchmark import BenchmarkSpec, make_world, audit
from decisions import stop_reason, write_episode
from pilot import stack


def branch_worker(payload, output, first_id):
    import superdex.physics as p

    p.initialize(num_worker_threads=0)
    spec = BenchmarkSpec(**payload)
    world = None
    results = []
    start = time.perf_counter()
    try:
        world = make_world(spec)
        prefix = []
        n = round(spec.branch_time / spec.dt)
        for i in range(n):
            row = world.step(
                i * spec.dt, "pull" if spec.family == "hook" else "nominal"
            )
            prefix.append(row)
            if i * spec.dt >= 1.5 and stop_reason(row):
                raise RuntimeError("invalid branch prefix")
            if time.perf_counter() - start > spec.max_wall_s:
                raise RuntimeError("branch prefix runtime budget exceeded")
        state = world.checkpoint()
        prefix_data = stack(prefix)
        digest = hashlib.sha256()
        for name in sorted(prefix_data):
            digest.update(name.encode())
            digest.update(prefix_data[name].tobytes())
        prefix_hash = digest.hexdigest()
        branches = (
            ("pull", "left", "right", "hold")
            if spec.family == "hook"
            else ("hold", "advance", "retract", "reverse")
        )
        # Repeated nominal continuation verifies physics AND collector restore.
        replay = []
        for _ in range(2):
            world.restore(state)
            replay.append(
                stack(
                    [
                        world.step(
                            (n + i) * spec.dt,
                            "pull" if spec.family == "hook" else "nominal",
                        )
                        for i in range(20)
                    ]
                )
            )
        for key in (
            "tool_pose",
            "actions",
            "tactile_force_field_left",
            "tactile_force_field_right",
        ):
            if not np.array_equal(replay[0][key], replay[1][key]):
                raise RuntimeError("checkpoint replay mismatch: " + key)
        for j, branch in enumerate(branches):
            branch_start = time.perf_counter()
            world.restore(state)
            world.initial_hash = hashlib.sha256(state[0]).hexdigest()
            branch_spec = replace(
                spec,
                episode_id=first_id + j,
                branch=branch,
                case="branch_" + branch,
                block="branches",
            )
            rows = list(prefix)
            abort = None
            for i in range(n, round(spec.duration / spec.dt)):
                row = world.step(i * spec.dt, branch)
                rows.append(row)
                abort = stop_reason(row)
                if time.perf_counter() - branch_start > spec.max_wall_s:
                    abort = "runtime_budget_exceeded"
                if abort:
                    break
            data = stack(rows)
            m = audit(data, branch_spec, abort)
            path = (
                Path(output)
                / f"{first_id + j:04d}_{spec.family}_{spec.variant}_{branch}.h5"
            )
            write_episode(path, data, branch_spec, world, m, snapshot=state)
            with h5py.File(path, "a") as f:
                f.attrs.update(
                    schema_version="vt_acwm_superdex_benchmark_v1",
                    task_family=spec.family,
                    task_variant=spec.variant,
                    task_kind=spec.family,
                    task_stage=spec.stage,
                    common_prefix_hash=prefix_hash,
                    restore_replay_exact=True,
                    branch_anchor_id=spec.episode_id,
                    branch_frame=n,
                    collector_pose_feedback=spec.tool_pose_feedback,
                    collector_force_feedback=spec.family == "surface",
                )
                f["checkpoint"].attrs["collector_state_json"] = json.dumps(
                    state[1],
                    default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v,
                )
            record = dict(
                spec=asdict(branch_spec),
                metrics=m,
                file=path.name,
                common_prefix_hash=prefix_hash,
                restore_replay_exact=True,
            )
            path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
            results.append(record)
        return dict(seconds=time.perf_counter() - start, episodes=results)
    finally:
        if world:
            world.close()
        p.shutdown()


def finalize_cancelled(root, phase, workers, reason):
    """Retain completed files and account for reservations after worker shutdown."""
    folder = root / phase
    records = []
    resolved = set()
    for path in folder.glob("*.json"):
        r = json.loads(path.read_text())
        if isinstance(r, dict) and "spec" in r and ("metrics" in r or r.get("error")):
            records.append(r)
            resolved.add(r["spec"]["episode_id"])
    attempts = [
        a
        for a in json.loads((root / "attempts.json").read_text())
        if a["phase"] == phase
    ]
    for a in attempts:
        if a["episode_id"] in resolved:
            continue
        s = dict(a["anchor"])
        branches = (
            ("pull", "left", "right", "hold")
            if s["family"] == "hook"
            else ("hold", "advance", "retract", "reverse")
        )
        siblings = [x for x in attempts if x["anchor"]["episode_id"] == s["episode_id"]]
        branch = branches[[x["episode_id"] for x in siblings].index(a["episode_id"])]
        s.update(
            episode_id=a["episode_id"],
            branch=branch,
            case="branch_" + branch,
            block="branches",
        )
        record = dict(
            spec=s,
            error=reason,
            status="cancelled_incomplete",
            note="Completed episodes/checkpoints retained; an interrupted in-memory continuation was not saved and is not training data.",
        )
        (folder / f"cancelled_{a['episode_id']:04d}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        records.append(record)
    wall = time.time() - (folder / "anchors.json").stat().st_mtime
    (folder / "summary.json").write_text(
        json.dumps(
            dict(
                seconds=wall,
                workers=workers,
                episodes=records,
                timing_method="Approximate wall time from anchor manifest timestamp",
                cancelled=True,
            ),
            indent=2,
        )
        + "\n"
    )


def run(
    root,
    workers=4,
    anchors_per_family=1,
    phase="branches",
    families=("surface", "hook", "insertion", "turning"),
    continuation_s=6.0,
):
    rows = []
    report_path = root / "report.json"
    surface_valid = set()
    if report_path.exists():
        from benchmark_geometry import episode_eligibility

        rows = [
            r
            for r in json.loads(report_path.read_text())["episodes"]
            if episode_eligibility(r["metrics"])["imitation_eligible"]
            and r["spec"]["block"] != "branches"
        ]
        surface_valid = {
            r["spec"]["episode_id"]
            for r in rows
            if r["spec"]["family"] == "surface"
            and r["metrics"]["physical_valid"]
            and r["metrics"]["task_success"]
            and r.get("independent_surface_geometry")
            and not r["independent_surface_geometry"]["failed"]
        }
    else:
        raise ValueError(
            "Generate benchmark_report.py output before selecting audited branch anchors"
        )
    anchors = []
    for family in families:
        candidates = sorted(
            [
                r
                for r in rows
                if r["spec"]["family"] == family
                and r["spec"]["profile_shape"] != "legacy"
                and r["metrics"]["physical_valid"]
                and r["metrics"]["task_success"]
                and (family != "surface" or r["spec"]["episode_id"] in surface_valid)
            ],
            key=lambda r: r["spec"]["episode_id"],
        )
        # Select distinct variants when possible; do not pretend repeated nominal seeds are independent.
        used = set()
        for r in candidates:
            if r["spec"]["variant"] in used:
                continue
            used.add(r["spec"]["variant"])
            s = BenchmarkSpec(**r["spec"])
            anchors.append(
                asdict(
                    replace(
                        s, duration=s.branch_time + continuation_s, max_wall_s=1800.0
                    )
                )
            )
            if len(used) == anchors_per_family:
                break
    if not anchors:
        raise ValueError("No qualified branch anchors")
    folder = root / phase
    if folder.exists():
        raise FileExistsError(folder)
    import fcntl

    with (root / ".budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = root / "attempts.json"
        attempts = json.loads(ledger.read_text())
        start_id = len(attempts)
        if start_id + 4 * len(anchors) > 256:
            raise ValueError("trajectory budget exceeded")
        folder.mkdir()
        for i, s in enumerate(anchors):
            for j in range(4):
                attempts.append(
                    dict(episode_id=start_id + 4 * i + j, phase=phase, anchor=s)
                )
        temp = ledger.with_suffix(".tmp")
        temp.write_text(json.dumps(attempts, indent=2) + "\n")
        temp.replace(ledger)
    (folder / "anchors.json").write_text(json.dumps(anchors, indent=2) + "\n")
    import shutil

    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    for name in (
        "benchmark.py",
        "benchmark_geometry.py",
        "benchmark_branches.py",
        "pilot.py",
        "decisions.py",
        "key_stages.py",
    ):
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(branch_worker, s, str(folder), start_id + 4 * i): s
            for i, s in enumerate(anchors)
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                results.extend(r["episodes"])
                print(
                    json.dumps(
                        {
                            "anchor": futures[future]["episode_id"],
                            "completed": len(r["episodes"]),
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                (
                    folder / f"anchor_{futures[future]['episode_id']}_error.json"
                ).write_text(
                    json.dumps(dict(spec=futures[future], error=repr(error)), indent=2)
                )
    (folder / "summary.json").write_text(
        json.dumps(
            dict(
                seconds=time.perf_counter() - start, workers=workers, episodes=results
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--anchors-per-family", type=int, choices=(1, 2), default=1)
    parser.add_argument("--phase", default="branches")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("surface", "hook", "insertion", "turning"),
        default=["surface", "hook", "insertion", "turning"],
    )
    parser.add_argument("--continuation-s", type=float, default=6.0)
    parser.add_argument(
        "--finalize-cancelled",
        help="After stopping workers, account for unfinished reservations with this reason",
    )
    a = parser.parse_args()
    if a.finalize_cancelled:
        finalize_cancelled(a.folder, a.phase, a.workers, a.finalize_cancelled)
    else:
        if not 0 < a.continuation_s <= 60:
            parser.error("continuation must be in (0,60] seconds")
        run(
            a.folder,
            a.workers,
            a.anchors_per_family,
            a.phase,
            a.families,
            a.continuation_s,
        )
