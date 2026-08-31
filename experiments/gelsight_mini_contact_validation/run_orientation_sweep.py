#!/usr/bin/env python3
"""Run baseline and roll/pitch cases with the selected FP32 gel material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
DEFAULT_OUTPUT = HERE / "output/orientation_sweep"
CASES = {
    "baseline": (0,0,0,0), "roll_pos": (12,0,6,0), "roll_neg": (-12,0,-6,0),
    "pitch_pos": (0,12,0,8), "pitch_neg": (0,-12,0,-8),
    "compound": (10,-10,5,-6),
}


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--cases",nargs="+",choices=tuple(CASES),default=list(CASES))
    parser.add_argument("--no-video",action="store_true"); parser.add_argument("--force",action="store_true")
    return parser.parse_args()


def main() -> None:
    args=parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    material_path=HERE/"selected_material.json"
    material=json.loads(material_path.read_text()) if material_path.exists() else {
        "youngs_modulus_pa":200000.0,"stiffness_damping_s":0.003,
    }
    summary={"material":material,"cases":{}}
    for name in args.cases:
        target=output/name; required=[target/"metrics.json",target/"contact_wrenches.csv"]
        if not args.no_video: required.append(target/"simulation.mp4")
        if args.force or not all(path.exists() for path in required):
            gr, gp, pr, pp=CASES[name]
            command=[sys.executable,str(RUNNER),"--output",str(target),"--case-name",name,
                     "--youngs-modulus-pa",str(material["youngs_modulus_pa"]),
                     "--stiffness-damping",str(material["stiffness_damping_s"]),
                     "--gripper-roll-deg",str(gr),"--gripper-pitch-deg",str(gp),
                     "--plug-roll-deg",str(pr),"--plug-pitch-deg",str(pp)]
            if args.no_video: command.append("--no-video")
            subprocess.run(command,cwd=HERE.parents[1],check=True)
        summary["cases"][name]=json.loads((target/"metrics.json").read_text())
    (output/"orientation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    names=list(summary["cases"])
    fig,axes=plt.subplots(3,1,figsize=(12,11))
    for index,name in enumerate(names):
        metrics=summary["cases"][name]
        plateaus=metrics["plateaus"]
        targets=np.array([2.0,5.0,10.0])
        means=np.array([plateaus[f"hold_{int(v)}N"]["mean_n"] for v in targets])
        axes[0].plot(targets,means,marker="o",label=name)
        axes[1].plot(targets,[plateaus[f"hold_{int(v)}N"]["indirect_rmse_n"] for v in targets],marker="o",label=name)
        axes[2].bar(index,100*max(metrics["interfaces"][side]["torque"]["relative_rmse"] for side in ("left","right")))
    axes[0].plot([2,5,10],[2,5,10],"k--",label="target")
    axes[0].set_ylabel("Measured table force [N]"); axes[1].set_ylabel("Indirect force RMSE [N]")
    axes[2].set_ylabel("Worst tactile torque relative RMSE [%]"); axes[2].set_xticks(range(len(names)),names,rotation=20)
    axes[1].set_xlabel("Target table force [N]")
    for axis in axes: axis.grid(alpha=.3); axis.legend(fontsize=8,ncol=3) if axis is not axes[2] else None
    fig.suptitle("Soft GelSight sensing across plug/gripper orientations"); fig.tight_layout()
    fig.savefig(output/"orientation_comparison.png",dpi=180); plt.close(fig)
    print(f"wrote {len(summary['cases'])} cases to {output}")


if __name__ == "__main__":
    main()
