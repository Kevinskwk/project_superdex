"""Report every key sensing condition, controls, uncertainty and limitations."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from key_observability import conditions, WINDOWS
from hook_observability import Reconstructor, corrupt, augmentation_seed
from hook_observability_report import LABELS


def turning_regression_plot(root):
    folder = root/"turning_after"
    d = json.loads((folder/"probes.json").read_text())
    if d["status"] != "complete":
        return
    p = np.load(folder/"predictions.npz")
    fig, axes = plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    anchors = p["anchor_ids"]
    ids = np.unique(anchors)
    truth = np.array([p["torque_target_nm"][anchors == a].mean() for a in ids])
    groups = np.array([p["groups"][anchors == a][0] for a in ids])
    for row,c in enumerate((0,5)):
        for col,mode in enumerate(("vision","vision_tactile","axial_wrench")):
            prediction = p[f"reg_{c}_{mode}"]
            average = np.array([prediction[anchors == a].mean() for a in ids])
            deviation = np.array([prediction[anchors == a].std() for a in ids])
            ax = axes[row,col]
            for group in np.unique(groups):
                mask = groups == group
                ax.errorbar(truth[mask],average[mask],yerr=deviation[mask],fmt="o",capsize=3,label=f"held-out {group:g} N")
            metric = next(r for r in d["regression"] if r["condition"] == conditions("turning")[c].name and r["mode"] == mode)
            ax.plot([0,.065],[0,.065],"k:")
            ax.set_xlim(0,.065);ax.set_ylim(0,.065);ax.grid(alpha=.2)
            ax.set_title(f"{mode}\n{LABELS[c]}\nMAE {metric['mae_nm']:.4f} Nm",fontsize=10)
            ax.set_xlabel("Measured future resistance [Nm]")
            if col == 0:
                ax.set_ylabel("Predicted future resistance [Nm]")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Future turning moment: observation at 10 s → target at 16–17 s\nDots: physical conditions; bars: variation across PCD corruption draws (not confidence intervals)")
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(root/"turning_torque_prediction.png",dpi=180);plt.close(fig)


def pcd_review(root, stage, records):
    selected = [next(r for r in records if r["label"] == label) for label in (0,1)]
    cfg_indices = [0,2,3,4,6]
    fig, axes = plt.subplots(2,5,figsize=(17,9),sharex=True,sharey=True)
    projections = []
    counts_audit = []
    t = WINDOWS[stage]["after"]
    for row, record in enumerate(selected):
        with h5py.File(record["path"]) as f:
            d = f["observations"]
            i = np.argmin(abs(d["timestamps"][:]-t))
            recon = Reconstructor(f)
            raw = recon.visible(d["body_root_poses"][i],0,[d[f"tactile_coord_{s}"][i] for s in ("left","right")])
            eye = recon.eyes[0]
            forward = recon.target-eye; forward /= np.linalg.norm(forward)
            right = np.cross(forward,[0,0,1]); right /= np.linalg.norm(right)
            up = np.cross(right,forward)
            for col,c in enumerate(cfg_indices):
                clouds, counts = corrupt(raw,eye,recon.target,conditions(stage)[c],
                                         augmentation_seed(record["spec"]["episode_id"],0),9)
                ax = axes[row,col]
                for stream,color in enumerate(("#d66a25","#287ab4")):
                    if counts[stream]:
                        rays = clouds[stream]-eye
                        uv = np.c_[rays@right,rays@up]/(rays@forward)[:,None]
                        projections.append(uv)
                        ax.scatter(*uv.T,c=color,s=2,alpha=.7)
                ax.set_title(LABELS[c]+f"\nretained key/env: {counts[0]}/{counts[1]}",fontsize=9)
                if col == 0:
                    ax.set_ylabel(("Will fail" if row == 0 else "Will seat") if stage == "insertion" else ("Low future load" if row == 0 else "High future load"))
                if row == 1:
                    ax.set_xlabel("camera u")
                counts_audit.append(dict(stage=stage,episode=record["spec"]["episode_id"],condition=conditions(stage)[c].name,retained=counts.tolist()))
    uv = np.concatenate(projections)
    for ax in axes.flat:
        ax.set_xlim(uv[:,0].min()-.01,uv[:,0].max()+.01)
        ax.set_ylim(uv[:,1].min()-.01,uv[:,1].max()+.01)
        ax.set_aspect("equal"); ax.grid(alpha=.15)
    fig.suptitle(f"Actual {stage} probe inputs at {t}s | orange: key; blue: fixture\nFixed camera masks, no hidden-surface filling; counts before resampling")
    fig.tight_layout(rect=(0,0,1,.93),h_pad=3)
    fig.savefig(root/f"{stage}_pcd_review.png",dpi=160);plt.close(fig)
    return counts_audit


def main(root):
    data = {}
    manifest = {}
    for stage in WINDOWS:
        for window in ("before","after"):
            folder = root/f"{stage}_{window}"
            data[stage,window] = json.loads((folder/"probes.json").read_text())
            manifest[stage,window] = json.loads((folder/"features.json").read_text())
    comparisons = [dict(stage=stage,window=window,**c) for (stage,window),d in data.items()
                   for c in d.get("paired_comparisons",[])]
    running = 0
    for rank,c in enumerate(sorted(comparisons,key=lambda r:r["paired_signflip_p"])):
        running = max(running,min(1,(len(comparisons)-rank)*c["paired_signflip_p"]))
        c["holm_global_p"] = running
    fig, axes = plt.subplots(2,2,figsize=(16,11),sharey=True)
    text = ["# Key tasks: tactile effectiveness under noisy / occluded PCD", "",
            "Bounded local sensing verification for separate insertion and turning. No main ACWM training, learned controller, cross-domain transfer or real-hardware performance is established.", "",
            "## Main findings", "",
            "- Turning is the stronger local sensing case. After the common turn probe, clean-PCD resistance classification rises from 52.1% (V) to 75.0% (V+T); with 1 mm noise +25% patch removal, 64.6% →87.5%. T alone reaches 91.7%. Before the turn, tactile does not provide a reliable benefit.",
            "- Insertion after probing is largely visually obvious: clean V and V+T both reach 100%; 1 mm +25% patch is 97.9% →100%; 2 mm +50% patch is 89.6% →100%. Before contact, T alone is 50% and adding it leaves V unchanged. Large-mask insertion also stays visually solvable: occlusion is not automatically ambiguity.",
            "- Tactile information is useful, but dense-field necessity is NOT established. For future turning torque, dense T-only Ridge MAE is 0.00536 Nm; an axis-aware aggregate-wrench baseline is better at 0.00379 Nm. Clean V+T fusion is also worse than T alone, showing estimator limitations.",
            "- These are exploratory point estimates from 12 conditions per stage. No positive V+T-versus-V gain survives the global 28-comparison Holm correction (smallest adjusted p≈0.109). Directional consistency and negative controls support further study, not a broad statistical efficacy claim.", "",
            "## Questions and protocol", "",
            "| Stage | Prediction target | Observation windows | Held-out variation |", "|---|---|---|---|",
            "| Insertion | Measured eventual seating success (≥14 mm, actual blade seated ≥0.3 s) | Before action: 3.1–4.0 s; after contact probe: 6.6–7.5 s | One complete socket-yaw group (−2°, 0°, +2°); positive offsets +0.05/+0.15 mm aligned and +2.5/+3 mm blocked |",
            "| Turning | Future resisting contact moment about rotor axis, mean 16–17 s; high/low threshold 0.027 Nm, plus continuous regression | Before action: 3.1–4.0 s; after common turning probe: 9.1–10.0 s | One complete grip-force group (30, 35, 40 N/finger); each contains spring stiffness 0.006, 0.010, 0.026, 0.034 Nm/rad |", "",
            "12 physical conditions per stage, four camera corruption draws each, three held-out groups (8 conditions train / 4 test). Draws are repeated observations, not independent episodes. Turning uses the SAME 100° tool command for every load, not a turn-versus-hold shortcut; geometry is identical across spring loads. The spring values themselves are not held out: this tests transfer across grip effort, not extrapolation to unseen stiffness.", "",
            "Protocol qualification changes, made before fitting: the original signed insertion sweep failed mechanics on negative offsets. A six-run follow-up plus the original six positive-offset cases forms the bounded ONE-SIDED insertion study. The failed signed sweep is retained below; do not infer bidirectional robustness. For turning, all 12 cases are audited uniformly through 18 s (turn and hold), including the complete 16–17 s target window. One full rollout aborted in return motion at 18.9 s; turn-and-hold qualification does not assert safe return. No physics thresholds were relaxed and no failing condition was individually deleted from either fitted study.", "",
            "Frozen pretrained Point-M2AE encoder on GPU, observed-cloud centroid/extent/covariance/scale and relative geometry. Fixed C=1 logistic classifiers; fixed RBF-SVM sensitivity check. Turning regression uses fixed alpha=1 Ridge. Training-only scaling, no tuning on held-out conditions. GT poses generate raycast clouds but are not direct features. Future commands, controller corrections, case/material labels, extrinsic wrenches and success labels are excluded from features. Robot/sensor poses define calibrated coordinate frames.", "",
            "PCD: camera-visible object/environment meshes, robot/housing/deformed gels as occluders, ideal segmentation, 1024 points per stream, no filling hidden surfaces. XYZ Gaussian sigma as named plus persistent stream registration bias sigma/2. Patch deletion removes a contiguous image-plane region. Fixed nominal-region masks are not recentered on true contact or outcome. These are stress corruptions, not calibrated real-camera noise.", "",
            "T = dense 2 × 7 × 9 × 3 simulated gel force history, preload-subtracted. W = net wrench integrated from the same fields about the EE, not a GT contact wrench or separate wrist sensor. `pre` uses precontact tactile at 1.05–1.45 s. Test mismatch deranges tactile between four held-out conditions (8 permutations), preserving PCD; below-chance scores are possible because this deliberately breaks the association.", "",
            "Additional aggregate controls were added after the initial turning fit: W-tare subtracts the same episode's preload-reference wrench; W-axis projects calibrated force/moment onto the known vertical interaction axis and tares it. These are diagnostic baselines, not additional tuned/confirmatory comparisons. They use robot calibration, never object/rotor GT pose or the target contact wrench. An axis-aware aggregate baseline is especially appropriate for a single-axis turning task.", "",
            "![Classification comparison](accuracy.png)", ""]
    pcd_audit = []
    for row,stage in enumerate(WINDOWS):
        for col,window in enumerate(("before","after")):
            d = data[stage,window]
            ax = axes[row,col]
            ax.set_title(f"{stage}: {window}, t={WINDOWS[stage][window]} s")
            text += [f"## {stage.title()} — {window} probe", ""]
            if d["status"] != "complete":
                text += [f"Qualification failed: {d['gate']['reasons']}. No fitted benefit claim.", ""]
                continue
            lookup = {(r["condition"],r["mode"]):r["balanced_accuracy"]*100 for r in d["results"]}
            for mode,label,color in (("vision","Vision","#287ab4"),("vision_tactile","Vision + dense tactile","#d66a25"),
                                     ("vision_wrench","Vision + net wrench","#499467"),("vision_tactile_shuffled_test","VT mismatched tactile","#888888")):
                ax.plot(range(7),[lookup[c.name,mode] for c in conditions(stage)],
                        "s--" if mode == "vision_wrench" else "o-",label=label,color=color,
                        markerfacecolor="none" if mode == "vision_wrench" else color)
            ax.axhline(50,color="black",ls=":",alpha=.4)
            ax.set_ylim(0,105);ax.grid(alpha=.2)
            ax.set_xticks(range(7),LABELS,rotation=30,ha="right",fontsize=7)
            ax.set_ylabel("Held-out balanced accuracy (%)")
            text += ["Balanced accuracy (%), all conditions and controls shown.", "",
                     "| PCD | V | T | W | V+T | V+W | V+pre | VT mismatch | V RBF | VT RBF |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            modes = ["vision","tactile","wrench","vision_tactile","vision_wrench","vision_precontact_tactile","vision_tactile_shuffled_test","vision_rbf","vision_tactile_rbf"]
            for cfg,label in zip(conditions(stage),LABELS):
                text.append("| "+label+" | "+" | ".join(f"{lookup[cfg.name,m]:.1f}" for m in modes)+" |")
            text += ["", "Aggregate calibration / known-axis controls (%):", "",
                     "| PCD | W-tare | W-axis | V+W-tare | V+W-axis |", "|---|---:|---:|---:|---:|"]
            for cfg,label in zip(conditions(stage),LABELS):
                text.append("| "+label+" | "+" | ".join(f"{lookup[cfg.name,m]:.1f}" for m in ("wrench_tared","axial_wrench","vision_wrench_tared","vision_axial_wrench"))+" |")
            if stage == "turning":
                reg = {(r["condition"],r["mode"]):r for r in d["regression"]}
                text += ["", f"Future-axis-moment regression: train-fold-mean baseline MAE {d['regression_train_mean_baseline']['mae_nm']:.5f} Nm. NRMSE is RMSE divided by target standard deviation, not by the mean load.", "",
                         "| PCD | V MAE (Nm) | V+T MAE | T MAE | W-tare MAE | W-axis MAE | V+W-axis MAE | VT mismatch MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
                for cfg,label in zip(conditions(stage),LABELS):
                    modes_reg = ["vision","vision_tactile","tactile","wrench_tared","axial_wrench","vision_axial_wrench","vision_tactile_shuffled_test"]
                    text.append("| "+label+" | "+" | ".join(f"{reg[cfg.name,m]['mae_nm']:.5f}" for m in modes_reg)+" |")
            text += [""]
        if data[stage,"after"]["status"] == "complete":
            pcd_audit += pcd_review(root,stage,data[stage,"after"]["records"])
            text += [f"![{stage} PCD]({stage}_pcd_review.png)", ""]
    axes[0,0].legend(fontsize=8,loc="lower left")
    fig.tight_layout();fig.savefig(root/"accuracy.png",dpi=180);plt.close(fig)
    turning_regression_plot(root)
    text += ["![Turning regression](turning_torque_prediction.png)", ""]
    text += ["## Physical and timing checks", "", "| Stage | Collected / valid | Actual task successes | Max reported penetration (mm) | Max observation depth / angle |", "|---|---|---:|---:|---|"]
    for stage in WINDOWS:
        records = data[stage,"after"]["records"]
        limit = max(r.get("max_depth_until_observation_mm",0) for r in records) if stage == "insertion" else max(r.get("rotor_at_observation_deg",0) for r in records)
        text.append(f"| {stage} | {len(records)} / {sum(r['metrics']['physical_valid'] for r in records)} | {sum(r['metrics']['task_success'] for r in records)} | {max(r['metrics']['max_penetration_mm'] for r in records):.3f} | {limit:.2f} {'mm' if stage == 'insertion' else 'deg'} |")
    text += ["", "All authored conditions are retained. Physically invalid conditions block a stage's fit rather than being silently discarded. Task success, mechanics validity and the existing tactile-fidelity flag remain different labels. Turning high/low labels are measured future axis moments, not whether the task succeeds.", "",
             "### Broader / full-rollout limitations", "",
             "| Original rollout | Full physical valid | Task success | Abort / max angular slip |", "|---|---|---|---|"]
    for path in sorted((root/"physics").glob("key_*.json")):
        original = json.loads(path.read_text())
        spec,m = original["spec"],original["metrics"]
        if not m["physical_valid"]:
            text.append(f"| {spec['stage']} #{spec['episode_id']} | {m['physical_valid']} | {m['task_success']} | {m['abort_reason']}; {m['max_angular_slip_deg']:.2f}° |")
    text += ["", "These failed original rollouts are preserved in physics/. The one-sided insertion follow-up is in physics_insertion_onesided/. No controller/material changes were made to rescue probe accuracy.", "",
             "## Paired uncertainty", "",
             f"Approximate class-stratified physical-condition bootstrap and paired sign-flip tests. Holm correction across all {len(comparisons)} stage × window × corruption comparisons. Only 12 conditions and 3 nuisance groups per stage; correlated setups and exploratory task design limit population-level inference. RBF/regression comparisons are descriptive, not additional confirmatory tests.", "",
             "| Stage / window | PCD | VT−V gain (pp) | 95% interval (pp) | Global Holm p |", "|---|---|---:|---|---:|"]
    for c in comparisons:
        text.append(f"| {c['stage']} / {c['window']} | {c['condition']} | {100*c['balanced_accuracy_gain']:.1f} | [{100*c['ci95'][0]:.1f}, {100*c['ci95'][1]:.1f}] | {c['holm_global_p']:.4f} |")
    text += ["", "## Scope and project relevance", "",
             "- Insertion tests a local affordance under imperfect geometry observations; large visible depth differences can make late prediction easy without tactile.",
             "- Turning tests hidden mechanical resistance with a common action and nearly identical visible geometry, directly relevant to action-conditioned physical prediction. It is not evidence for successful closed-loop effort adaptation or a necessary dense spatial force representation.",
             "- Judge tactile against both clean vision and net-wrench controls. A clean-vision ceiling, negative fusion result, or a net-wrench tie is informative and must not be hidden. Do not optimize masks simply to make tactile win.",
             "- Force fields are ideal simulator outputs; no optical GelSight reconstruction/noise/calibration is modelled. The collector uses privileged tool-pose feedback and a GT insertion safety guard, excluded as learned features. Real sensor fidelity, cross-object/domain transfer and main ACWM performance remain untested.",
             "- Existing nominal key episodes narrowly missed the whole-episode 0.2 N absolute tactile RMSE gate. An observability gain does not by itself validate force calibration or justify large-scale collection.", "",
             "## Compute and reproducibility", ""]
    for (stage,window),m in manifest.items():
        if "seconds" in m:
            text.append(f"- {stage}/{window}: {m['seconds']:.1f} s frozen-encoder extraction on {m['gpu']}; peak PyTorch allocated/reserved {m['peak_allocated_vram_mb']:.1f}/{m['peak_reserved_vram_mb']:.1f} MiB (not total device VRAM). Heads run on CPU.")
    text += ["- Raw data: physics/*.h5 preserves the original sweep; physics_insertion_onesided/*.h5 contains the qualified insertion set (six reused positive-offset episodes plus six new follow-ups). Features, fixed heads' held-out predictions and metrics are in each stage_window folder; extractor/checkpoint hashes are recorded.",
             "- Previously verified task videos: [insertion](../key_stages_split/key_insertion_aligned_0000_rgb_tactile.mp4), [turning](../key_stages_split/key_turning_aligned_0002_rgb_tactile.mp4). These are task demonstrations, not videos of every new condition."]
    snapshot = root/"source_snapshot";snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in ("key_observability.py","key_observability_report.py","key_observability_specs.json","key_observability_insertion_followup_specs.json","key_stages.py","hook_observability.py","decision_vision.py","point_m2ae_probe.py","test_key_observability.py","DECISION_PILOT.md"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source,snapshot/name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshot/"sha256.json").write_text(json.dumps(hashes,indent=2))
    (root/"REPORT.md").write_text("\n".join(text)+"\n")
    (root/"report_audit.json").write_text(json.dumps(dict(comparisons=comparisons,pcd_review=pcd_audit),indent=2))
    print(root/"REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    main(parser.parse_args().folder)
