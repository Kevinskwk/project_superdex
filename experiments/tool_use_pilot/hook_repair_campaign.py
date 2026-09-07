"""Parallel small branch audit with byte-identical prefix verification."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
from contextlib import nullcontext
import time
import h5py
import numpy as np
from decisions import worker, candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--specs", type=Path, default=Path(__file__).with_name("hook_repair_specs.json")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    specs = json.loads(a.specs.read_text())
    (a.output / "manifest.json").write_text(json.dumps(specs, indent=2))
    start = time.perf_counter()
    reports = []
    # Keep each worker's evidence even if a subsequent merge check fails.
    with nullcontext(a.output / "_branches") as staging:
        with ProcessPoolExecutor(
            max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            jobs = {}
            for spec in specs:
                for branch in candidates("hook"):
                    folder = Path(staging) / f"{spec['episode_id']}-{branch}"
                    jobs[
                        pool.submit(worker, spec, str(folder), False, False, branch)
                    ] = (folder, spec, branch)
            for future in as_completed(jobs):
                folder, spec, branch = jobs[future]
                report = future.result()
                stem = f"episode_{spec['episode_id']:04d}"
                src, dst = folder / f"{stem}_prefix.h5", a.output / f"{stem}_prefix.h5"
                if dst.exists():
                    with h5py.File(src) as x, h5py.File(dst) as y:
                        for key in x["observations"]:
                            np.testing.assert_array_equal(
                                x["observations"][key][:],
                                y["observations"][key][:],
                                err_msg=f"prefix mismatch: {key}",
                            )
                        assert (
                            x["checkpoint"].attrs["controller_state_hash"]
                            == y["checkpoint"].attrs["controller_state_hash"]
                        )
                else:
                    shutil.copy2(src, dst)
                for name in (f"{stem}_{branch}.h5",):
                    if (folder / name).exists():
                        shutil.copy2(folder / name, a.output / name)
                (a.output / f"{stem}_{branch}.json").write_text(
                    json.dumps(report, indent=2)
                )
                reports.append(report)
                print(
                    json.dumps(
                        dict(
                            episode_id=spec["episode_id"],
                            branch=branch,
                            metrics=report["branches"],
                        )
                    ),
                    flush=True,
                )
    (a.output / "timing.json").write_text(
        json.dumps(
            dict(
                seconds=time.perf_counter() - start,
                workers=a.workers,
                jobs=len(reports),
                all_prefixes_identical=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
