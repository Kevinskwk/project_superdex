"""Report the small held-out hook sensing experiment, including negative controls."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hook_observability import CONDITIONS, Reconstructor, augmentation_seed, corrupt


LABELS = ["Visible clean", "0.5 mm + 10% patch", "1 mm + 25% patch",
          "2 mm + 50% patch", "1 mm + contact mask",
          "2 mm + contact mask + 25% patch", "1 mm + large working-region mask"]


def render_clouds(root, records):
    """Show actual probe inputs, with fixed camera axes and limits per figure."""
    selected = [0, 2, 3, 4, 6]
    audit = []
    for time_s in (5, 8):
        fig, axes = plt.subplots(2, len(selected), figsize=(17, 9), sharex=True, sharey=True)
        all_uv = []
        for row, record in enumerate(records[:2]):
            path = record["path"] if time_s == 5 else record["path"].replace("_prefix", "_pull")
            with h5py.File(path) as f:
                d = f["observations"]
                i = int(np.argmin(abs(d["timestamps"][:] - time_s)))
                recon = Reconstructor(f)
                raw = recon.visible(d["body_root_poses"][i], 0,
                                    [d[f"tactile_coord_{s}"][i] for s in ("left", "right")])
                eye = recon.eyes[0]
                forward = recon.target - eye
                forward /= np.linalg.norm(forward)
                right = np.cross(forward, [0, 0, 1])
                right /= np.linalg.norm(right)
                up = np.cross(right, forward)
                for col, c in enumerate(selected):
                    cloud, counts = corrupt(raw, eye, recon.target, CONDITIONS[c],
                                            augmentation_seed(record["spec"]["episode_id"], 0), 9)
                    ax = axes[row, col]
                    for stream, color in enumerate(("#d66a25", "#287ab4")):
                        if counts[stream]:
                            rays = cloud[stream] - eye
                            uv = np.c_[rays @ right, rays @ up] / (rays @ forward)[:, None]
                            all_uv.append(uv)
                            ax.scatter(*uv.T, s=2, alpha=0.7, c=color, rasterized=True)
                    ax.set_title(LABELS[c] + f"\nretained tool/bar: {counts[0]}/{counts[1]}", fontsize=9)
                    if col == 0:
                        ax.set_ylabel(("Captured" if row == 0 else "Near miss") + "\ncamera v")
                    if row == 1:
                        ax.set_xlabel("camera u")
                    audit.append(dict(time_s=time_s, episode=record["spec"]["episode_id"],
                                      condition=CONDITIONS[c].name, retained=counts.tolist()))
        uv = np.concatenate(all_uv)
        low, high = uv.min(0) - 0.01, uv.max(0) + 0.01
        for ax in axes.flat:
            ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1])
            ax.set_aspect("equal"); ax.grid(alpha=0.15)
        fig.suptitle(f"Actual camera-visible probe PCD at {time_s} s | orange: hook, blue: bar\n"
                     "Fixed axes; hidden surfaces are not filled. Counts precede resampling to 1024 points.")
        fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3)
        fig.savefig(root / f"pcd_corruptions_{time_s}s.png", dpi=160)
        plt.close(fig)
    return audit


def main(root):
    stages = {s: json.loads((root / s / "probes.json").read_text())
              for s in ("before_probe", "after_probe")}
    features = {s: json.loads((root / s / "features.json").read_text()) for s in stages}
    comparisons = [dict(stage=s, **c) for s, data in stages.items() for c in data["paired_comparisons"]]
    ordered = sorted(comparisons, key=lambda c: c["paired_signflip_p"])
    running = 0.0
    for rank, c in enumerate(ordered):
        running = max(running, min(1.0, c["paired_signflip_p"] * (len(ordered) - rank)))
        c["holm_global_14_p"] = running
    physical = []
    for record in stages["before_probe"]["records"]:
        row = dict(episode=record["spec"]["episode_id"], **record["pull_metrics"])
        row["prefix_valid"] = record["prefix_valid"]
        for time_s in (1.45, 5, 8):
            path = record["path"] if time_s <= 5 else record["path"].replace("_prefix", "_pull")
            with h5py.File(path) as f:
                d = f["observations"]
                times = d["timestamps"][:]
                selected = (times >= time_s - (0.4 if time_s < 2 else 0.9)) & (times <= time_s + 1e-6)
                row[f"force_mean_{time_s}s_n"] = float(np.linalg.norm(d["extrinsic_contact_wrench"][:][selected, :3], axis=1).mean())
                row[f"max_progress_until_{time_s}s_mm"] = float(d["task_progress"][:][times <= time_s + 1e-6].max() * 1000)
        physical.append(row)
    assert all(r["physical_valid"] and r["prefix_valid"] and r["grasp_retained"] for r in physical)
    assert all(r["max_progress_until_8s_mm"] < 40 for r in physical)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    lines = ["#287ab4", "#d66a25", "#499467", "#888888"]
    modes = ["vision", "vision_tactile", "vision_wrench", "vision_tactile_shuffled_test"]
    names = ["Vision", "Vision + dense tactile", "Vision + tactile wrench", "VT: mismatched tactile at test"]
    text = ["# Hook: does tactile help under noisy / occluded PCD?", "",
            "Local positive evidence, not a general policy or sim-to-real result. Target: will a straight pull reach 40 mm for 0.3 s? This does not predict left/right recovery direction.", "",
            "## Protocol", "",
            "12 independent physical starting states (6 successful pulls, 6 failures), three offset levels, both mirrored sides. Eight new states supplement the four repaired states. Four corruption draws per state are repeated observations, NOT independent episodes. Leave one complete offset level out: 8 train / 4 test states per fold; unseen corruption seeds and no shared physical anchor across train/test.", "",
            "Frozen pretrained Point-M2AE on GPU, plus observed-cloud metric/relative geometry. Fixed C=1 logistic heads, train-only scaling; fixed RBF-SVM sensitivity check. Histories: 4.1–5.0 s at the initial decision, and 7.1–8.0 s after a common gentle pull. No future controller commands, tool/fixture pose features, extrinsic GT wrench, case labels, or privileged corrections. Geometry poses are used only to generate camera-visible observations; robot poses define coordinate frames.", "",
            "PCD: 1024 tool + 1024 environment points; mesh raycasting with robot, housings and gels as occluders, ideal segmentation, no hidden-surface filling. Noise is independent XYZ Gaussian sigma as named PLUS persistent per-stream registration bias sigma/2. Patch removal deletes a contiguous camera-plane region. Fixed contact mask 45 × 28 mm; large stress mask 140 × 90 mm, dimensions at working depth. These are stress augmentations, not a calibrated real camera model.", "",
            "![Accuracy](accuracy.png)", ""]
    for ax, (stage, data) in zip(axes, stages.items()):
        lookup = {(r["condition"], r["mode"]): r["balanced_accuracy"] * 100 for r in data["results"]}
        for mode, name, color in zip(modes, names, lines):
            ax.plot(range(7), [lookup[c.name, mode] for c in CONDITIONS],
                    "s--" if mode == "vision_wrench" else "o-", label=name, color=color,
                    markerfacecolor="none" if mode == "vision_wrench" else color)
        ax.axhline(50, ls=":", color="black", alpha=.5)
        ax.set_xticks(range(7), LABELS, rotation=35, ha="right", fontsize=8)
        ax.set_title("Initial decision (5 s)" if stage == "before_probe" else "After probing pull (8 s)")
        ax.set_ylim(0, 105); ax.grid(alpha=.2)
        text += [f"## {ax.get_title()}", "", "Balanced accuracy (%). V = vision; T = dense tactile; W = tactile-derived net wrench; pre = precontact tactile. All conditions shown.", "",
                 "| PCD | V | T | V+T | V+W | V+pre | VT mismatched | V RBF | VT RBF |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c, label in zip(CONDITIONS, LABELS):
            cols = ["vision", "tactile", "vision_tactile", "vision_wrench", "vision_precontact_tactile", "vision_tactile_shuffled_test", "vision_rbf", "vision_tactile_rbf"]
            text.append("| " + label + " | " + " | ".join(f"{lookup[c.name, m]:.1f}" for m in cols) + " |")
        text += ["", f"![PCD inputs](pcd_corruptions_{5 if stage == 'before_probe' else 8}s.png)", ""]
    axes[0].set_ylabel("Held-out balanced accuracy (%)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Hook pull-success prediction | 12 physical states, 3 held-out offset folds")
    fig.tight_layout(); fig.savefig(root / "accuracy.png", dpi=180); plt.close(fig)
    text += ["## Interpretation and controls", "",
             "- Tactile adds engagement information under noisy/patch-removed PCD. At 5 s, 2 mm noise + 50% patch: 50.0% → 87.5%; paired gain 37.5 pp, approximate anchor-bootstrap 95% interval [22.9, 50.0] pp.",
             "- Shuffling tactile across held-out states destroys the gain, and precontact tactile does not reproduce it. Test shuffling uses forced derangements of four states (8 permutations), so below-chance performance is possible; it is an association-breaking control, not an unbiased 50% baseline.",
             "- At 8 s, V+T reaches 100% on this small test. V+net tactile wrench also reaches 100%: this task currently provides NO evidence that a dense field is necessary over aggregate wrench sensing.",
             "- Clean vision reaches 100% after the pull. The large working-region mask also leaves vision at 100% at both stages: this particular mask does not create visual ambiguity. Visual inspection confirms the upper hook shank and parts of the bar/support remain visible. The smaller nominal contact-region mask only partially hides the interaction geometry, not every contact surface. Residual geometry and missingness remain available; greater occlusion is not monotonically harder.",
             "- RBF checks also improve under corruption, but initial clean V+T RBF is worse than V RBF (85.4% vs 89.6%). Tactile is not universally beneficial for every estimator/condition.",
             "- Statistics are exploratory. Correcting across all 14 stage × corruption comparisons leaves only initial 2 mm + 50% patch below p=0.05 (Holm p=0.0273); the initial 1 mm + 25% patch is p=0.0508. Most positive point estimates are not conclusive with N=12.",
             "- Simulated tactile is ideal force data, not noisy optical GelSight reconstruction. Collector motion uses privileged tool-pose feedback (excluded as probe features). No learned recovery controller, cross-object transfer, real-sensor calibration, or sim-to-real validation is established.", "",
             "## Physics / observation timing", "",
             f"All {len(physical)} prefixes and pull continuations passed existing physical-validity gates and retained their grasp. Max reported penetration: {max(r['max_penetration_mm'] for r in physical):.3f} mm. Max fixture travel by 8 s: {max(r['max_progress_until_8s_mm'] for r in physical):.2f} mm, below the 40 mm label threshold. Failure 103 brushes the bar but does not complete the pull; failures are retained, not filtered away.", "",
             "| Pull outcome | Mean GT extrinsic force, precontact (N) | Initial history (N) | After-probe history (N) |", "|---|---:|---:|---:|"]
    for success in (True, False):
        rows = [r for r in physical if r["task_success"] == success]
        values = [np.mean([r[f"force_mean_{t}s_n"] for r in rows]) for t in (1.45, 5, 8)]
        text.append(f"| {'Success' if success else 'Failure'} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    text += ["", "GT forces above are mechanism diagnostics only, never probe inputs.", "", "## Compute / artifacts", ""]
    for stage, f in features.items():
        text.append(f"- {stage}: frozen-encoder extraction {f['seconds']:.1f} s on {f['gpu']}; peak PyTorch allocated/reserved VRAM {f['peak_allocated_vram_mb']:.1f}/{f['peak_reserved_vram_mb']:.1f} MiB (not total device usage). Classifier heads run on CPU.")
    for side in ("rightward", "leftward"):
        name = f"near_miss_{side}_recovery"
        text.append(f"- [Near-miss → {side} recovery](../hook_repair_final/verification/{name}/replay/{name}_rgb_tactile.mp4)")
    text += ["- Final results: before_probe/probes.json and after_probe/probes.json; exact extractor snapshots and checkpoint hashes in each stage. shared_seed_* folders are superseded development runs, not report inputs.", "",
             "## Project relevance / next useful step", "",
             "This verifies a local sensing/affordance question relevant to ACWM: contact history can resolve uncertain pull outcomes. Do not optimize masks simply to make tactile win. Next use physically matched ambiguous observations, vary hidden load/compliance, and test prediction of action-conditioned displacement/load; compare dense tactile against net wrench. Left/right recovery needs additional directional evidence, and should be evaluated separately from engagement detection.", "",
             "## Paired uncertainty (all 14 comparisons)", "", "| Stage | Condition | Gain (pp) | 95% anchor interval (pp) | Global Holm p |", "|---|---|---:|---|---:|"]
    for c in comparisons:
        text.append(f"| {c['stage']} | {c['condition']} | {100*c['accuracy_gain']:.1f} | [{100*c['ci95'][0]:.1f}, {100*c['ci95'][1]:.1f}] | {c['holm_global_14_p']:.4f} |")
    audit = render_clouds(root, stages["before_probe"]["records"])
    (root / "REPORT.md").write_text("\n".join(text) + "\n")
    (root / "audit.json").write_text(json.dumps(dict(physical=physical, comparisons=comparisons, pcd_views=audit), indent=2))
    print(root / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output/hook_tactile_observability")
    main(parser.parse_args().output)
