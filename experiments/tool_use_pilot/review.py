#!/usr/bin/env python3
"""Re-run eight selected configurations as actual RGB+tactile review videos."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--ids", default="0,1,3,49,57,73,80,169")
    args = parser.parse_args()
    out = args.folder / "review"
    out.mkdir(parents=True, exist_ok=True)
    records = {
        s["episode_id"]: s for s in json.loads((args.folder / "specs.json").read_text())
    }
    selected = [int(i) for i in args.ids.split(",")]

    def render(i):
        spec_path = out / f"spec_{i:04d}.json"
        spec_path.write_text(json.dumps([records[i]], indent=2))
        with (out / f"render_{i:04d}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("pilot.py")),
                    "--specs",
                    str(spec_path),
                    "--render",
                    "--output",
                    str(out),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"Recorded episode {i}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render, selected))
    (out / "selection.json").write_text(
        json.dumps({i: records[i] for i in selected}, indent=2)
    )


if __name__ == "__main__":
    main()
