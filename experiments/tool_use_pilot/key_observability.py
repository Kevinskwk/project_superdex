"""Bounded key sensing probes: future seating and future turning resistance.

No GT pose/force, controller correction, future action or authored condition is
a learned input. Simulated geometry poses generate only camera-visible clouds.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC
import torch

from hook_observability import (
    CONDITIONS, Reconstructor, normalize_and_metric, FrozenPointM2AE,
    summarize_history, augmentation_seed, corrupt, tactile_features,
    wrench_features, design, observed_relations,
)

WINDOWS = {"insertion": {"before": 4.0, "after": 7.5},
           "turning": {"before": 4.0, "after": 10.0}}
LOAD_THRESHOLD_NM = .027


def task_horizon_metrics(f, end_s):
    """Uniform turn-and-hold audit; never hides full-rollout return failures."""
    from key_stages import KeyStageSpec, stage_audit
    spec = KeyStageSpec(**json.loads(f.attrs["config_json"]))
    d = f["observations"]
    times = d["timestamps"][:]
    if times[-1] < end_s-1e-6:
        raise ValueError("Episode ended before the common task evaluation horizon")
    indices = np.flatnonzero(times <= end_s+1e-6)
    data = {k:v[:][indices] for k,v in d.items()}
    data["phase"] = data["phase"].astype(str)
    return stage_audit(data,spec,None)


def conditions(stage):
    # Fixed nominal apparatus masks, not moved to GT contact points. The
    # prepared turning fixture is 19 mm higher than the insertion fixture.
    z = .002 if stage == "insertion" else .021
    return [replace(c, window_center_z_m=z) if c.contact_window else c for c in CONDITIONS]


def target_record(f):
    spec = json.loads(f.attrs["config_json"])
    m = json.loads(f.attrs["metrics_json"])
    d = f["observations"]
    stage = spec["stage"]
    torque = None
    if stage == "turning":
        times = d["timestamps"][:]
        selected = np.flatnonzero((times >= 16) & (times <= 17))
        if len(selected) < round(.9/spec["dt"]):
            raise ValueError("Missing future 16–17 s torque target window")
        measured = []
        for i in selected:
            rotor = d["body_root_poses"][i, 2]
            axis = Rotation.from_quat(rotor[3:]).apply([0, 0, 1])
            w = d["extrinsic_contact_wrench"][i]
            lever = d["tool_pose"][i, :3] - rotor[:3]
            about_rotor = w[3:] + np.cross(lever, w[:3])
            measured.append(-float(about_rotor @ axis))
        torque = float(np.mean(measured))
    return dict(spec=spec, metrics=m,
                label=int(m["task_success"]) if stage == "insertion" else int(torque >= LOAD_THRESHOLD_NM),
                future_resistance_nm=torque,
                group=spec["yaw_deg"] if stage == "insertion" else spec["grip_force_n"])


def qualify(records):
    reasons = []
    if len(records) != 12:
        reasons.append("requires all 12 planned physical conditions; no outcome-based filtering")
    if any(not r["metrics"]["physical_valid"] for r in records):
        reasons.append("one or more complete episodes fail the mechanics gate")
    groups = {r["group"] for r in records}
    if len(groups) != 3 or any(sum(r["group"] == g for r in records) != 4 for g in groups):
        reasons.append("requires three complete groups of four")
    for group in groups:
        if {r["label"] for r in records if r["group"] != group} != {0, 1}:
            reasons.append("a training fold lacks both measured outcomes")
    return dict(passed=not reasons, reasons=reasons)


def extract(folder, output, stage, window, views=4):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "features"
    cache.mkdir(exist_ok=True)
    shutil.copy2(__file__, output / "extractor_source.py")
    records = []
    for path in sorted(folder.glob(f"key_{stage}_*.h5")):
        with h5py.File(path) as f:
            record = target_record(f)
            record["full_rollout_metrics"] = record["metrics"]
            if stage == "turning":
                record["metrics"] = task_horizon_metrics(f,18.0)
                record["evaluation_horizon_s"] = 18.0
            else:
                record["evaluation_horizon_s"] = record["spec"]["duration"]
        record["path"] = str(path.resolve())
        records.append(record)
    gate = qualify(records)
    manifest = dict(stage=stage, window=window, observation_time_s=WINDOWS[stage][window],
                    views=views, gate=gate, records=records,
                    conditions=[asdict(c) for c in conditions(stage)])
    if not gate["passed"]:
        (output / "features.json").write_text(json.dumps(manifest, indent=2))
        return manifest
    torch.set_num_threads(2)
    model = FrozenPointM2AE().cuda().eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    manifest.update(encoder_checkpoint=model.checkpoint_sha256,
                    extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    shared_extractor_sha256=hashlib.sha256(Path(__file__).with_name("hook_observability.py").read_bytes()).hexdigest())
    start = time.perf_counter()
    for record in records:
        path = Path(record["path"])
        key = path.stem
        file = cache / f"{key}.npz"
        record["file"] = str(file.resolve())
        with h5py.File(path) as f:
            d = f["observations"]
            times = d["timestamps"][:]
            observation_time = WINDOWS[stage][window]
            indices = np.array([np.argmin(abs(times-t)) for t in np.linspace(observation_time-.9, observation_time, 10)])
            before = np.array([np.argmin(abs(times-t)) for t in np.linspace(1.05, 1.45, 10)])
            record["max_depth_until_observation_mm"] = float(d["insertion_depth_m"][:][times <= observation_time+1e-6].max()*1000)
            record["rotor_at_observation_deg"] = float(np.rad2deg(d["rotor_angle_rad"][indices[-1]]))
            digest = hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k != "records"}, sort_keys=True).encode()
                + d["body_root_poses"][:].tobytes() + d["tactile_force_field_left"][:].tobytes()
                + d["tactile_force_field_right"][:].tobytes()).hexdigest()
            record["feature_hash"] = digest
            meta = cache / f"{key}.json"
            if meta.exists() and file.exists() and json.loads(meta.read_text()).get("feature_hash") == digest:
                print(f"Reused {stage}/{window}/{key}", flush=True)
                continue
            recon = Reconstructor(f)
            raw = [recon.visible(d["body_root_poses"][i], 0,
                   [d[f"tactile_coord_{s}"][i] for s in ("left", "right")]) for i in indices]
            neural, metric, coverage = [], [], []
            for cfg in conditions(stage):
                c_neural, c_metric, c_counts = [], [], []
                for view in range(views):
                    pcs, geometry, counts = [], [], []
                    for frame, i in enumerate(indices):
                        cloud, n = corrupt(raw[frame], recon.eyes[0], recon.target, cfg,
                                           augmentation_seed(record["spec"]["episode_id"], view), frame)
                        normalized, m = normalize_and_metric(cloud, n, d["ee_pose"][i])
                        pcs.extend(normalized); geometry.append(m); counts.append(n)
                    encoded = []
                    for j in range(0, len(pcs), 8):
                        encoded.append(model(torch.from_numpy(np.asarray(pcs[j:j+8])).cuda()).cpu().numpy())
                    e = np.concatenate(encoded).reshape(10, 2, -1)
                    e[np.array(counts) == 0] = 0
                    c_neural.append(summarize_history(e.reshape(10, -1)))
                    c_metric.append(summarize_history(np.array(geometry)))
                    c_counts.append(counts)
                neural.append(c_neural); metric.append(c_metric); coverage.append(c_counts)
            np.savez_compressed(file, neural=neural, metric=metric, coverage=coverage,
                                tactile=tactile_features(f, indices), precontact=tactile_features(f, before),
                                wrench=wrench_features(f, indices))
            meta.write_text(json.dumps(record, indent=2))
            print(f"Encoded {stage}/{window}/{key}", flush=True)
    assert all(torch.equal(v, original[k]) for k,v in model.state_dict().items())
    manifest.update(seconds=time.perf_counter()-start, gpu=torch.cuda.get_device_name(), encoder_frozen=True,
        peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_vram_mb=torch.cuda.max_memory_reserved()/2**20)
    (output / "features.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def summary(y, probability, anchor_ids):
    anchors = np.unique(anchor_ids)
    truth = np.array([y[anchor_ids == a][0] for a in anchors])
    correct = (probability >= .5) == y
    per_anchor = np.array([correct[anchor_ids == a].mean() for a in anchors])
    rng = np.random.default_rng(701)
    # Class-stratified physical-condition bootstrap, never augmentation bootstrap.
    draws = [rng.choice(np.flatnonzero(truth == label), (4000, sum(truth == label)), replace=True) for label in (0, 1)]
    boot = (per_anchor[draws[0]].mean(1) + per_anchor[draws[1]].mean(1))/2
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, probability >= .5)),
                auc=float(roc_auc_score(y, probability)), ci95=np.quantile(boot, [.025,.975]).tolist(),
                per_anchor_accuracy=per_anchor.tolist())


def regression_summary(y, prediction, anchor_ids):
    error = prediction-y
    anchors = np.unique(anchor_ids)
    return dict(mae_nm=float(abs(error).mean()), rmse_nm=float(np.sqrt(np.mean(error**2))),
                nrmse_target_std=float(np.sqrt(np.mean(error**2))/max(y.std(), 1e-8)),
                per_anchor_mae_nm=[float(abs(error[anchor_ids == a]).mean()) for a in anchors])


def tare_wrench_features(raw, baseline):
    """Apply the same episode-local preload calibration available to dense T."""
    result = raw.copy()
    result[:6] -= baseline
    result[6:12] -= baseline
    return result


def axial_wrench_features(f, indices, reference_index):
    """Strong task-aware aggregate baseline: calibrated vertical force/moment.

    Both tasks have a known vertical interaction axis. Uses robot calibration,
    not object/rotor GT poses, and moments remain about the EE origin.
    """
    def axial(i):
        w = wrench_features(f,[int(i)])[:6]
        rotation = Rotation.from_quat(f["observations/ee_pose"][i,3:])
        return np.array([rotation.apply(w[:3])[2],rotation.apply(w[3:])[2]])
    baseline = axial(reference_index)
    return summarize_history(np.array([axial(i)-baseline for i in indices]))


def paired_gain(y_anchor, vision, tactile):
    delta = np.array(tactile["per_anchor_accuracy"])-vision["per_anchor_accuracy"]
    weights = np.array([len(delta)/(2*sum(y_anchor == label)) for label in y_anchor])
    weighted = delta*weights
    signs = 2*((np.arange(2**len(delta))[:, None] >> np.arange(len(delta))) & 1)-1
    p = float(np.mean(abs((signs*weighted).mean(1)) >= abs(weighted.mean())-1e-12))
    rng = np.random.default_rng(801)
    draws = [rng.choice(np.flatnonzero(y_anchor == label), (4000, sum(y_anchor == label)), replace=True) for label in (0,1)]
    boot = (delta[draws[0]].mean(1)+delta[draws[1]].mean(1))/2
    return dict(balanced_accuracy_gain=float(weighted.mean()), ci95=np.quantile(boot,[.025,.975]).tolist(), paired_signflip_p=p)


def fit(output):
    manifest = json.loads((output / "features.json").read_text())
    if not manifest["gate"]["passed"]:
        result = dict(status="qualification_failed", **manifest)
        (output / "probes.json").write_text(json.dumps(result, indent=2))
        return result
    records = manifest["records"]
    data = [np.load(r["file"]) for r in records]
    tared_wrenches, axial_wrenches = [], []
    for r,d in zip(records,data):
        with h5py.File(r["path"]) as f:
            ref_indices = np.flatnonzero(f["observations/reference_reset"][:])
            if len(ref_indices) != 1:
                raise ValueError("Expected one recorded preload-reference event")
            baseline = wrench_features(f,[int(ref_indices[0])])[:6]
            times = f["observations/timestamps"][:]
            t = manifest["observation_time_s"]
            indices = [np.argmin(abs(times-x)) for x in np.linspace(t-.9,t,10)]
            axial_wrenches.append(axial_wrench_features(f,indices,int(ref_indices[0])))
        tared_wrenches.append(tare_wrench_features(d["wrench"],baseline))
    views = manifest["views"]
    anchor_ids = np.repeat(np.arange(len(records)), views)
    y_anchor = np.array([r["label"] for r in records])
    y = y_anchor[anchor_ids]
    groups = np.array([r["group"] for r in records])[anchor_ids]
    target = np.array([r["future_resistance_nm"] for r in records])[anchor_ids] if manifest["stage"] == "turning" else None
    results, comparisons, regressions, predictions = [], [], [], {}
    for c, cfg in enumerate(manifest["conditions"]):
        neural = np.stack([d["neural"][c] for d in data]).reshape(len(y), -1)
        metric = np.stack([d["metric"][c] for d in data]).reshape(len(y), -1)
        relations = observed_relations(metric)
        tactile = np.stack([d["tactile"] for d in data])[anchor_ids]
        before = np.stack([d["precontact"] for d in data])[anchor_ids]
        wrench = np.stack([d["wrench"] for d in data])[anchor_ids]
        tared = np.stack(tared_wrenches)[anchor_ids]
        axial = np.stack(axial_wrenches)[anchor_ids]
        modes = dict(vision=[neural,metric,relations], vision_metric=[metric,relations],
                     tactile=[tactile], wrench=[wrench], vision_tactile=[neural,metric,relations,tactile],
                     vision_wrench=[neural,metric,relations,wrench],
                     wrench_tared=[tared], vision_wrench_tared=[neural,metric,relations,tared],
                     axial_wrench=[axial], vision_axial_wrench=[neural,metric,relations,axial],
                     vision_precontact_tactile=[neural,metric,relations,before],
                     vision_rbf=[neural,metric,relations], vision_tactile_rbf=[neural,metric,relations,tactile])
        for mode, blocks in modes.items():
            probability = np.zeros(len(y)); reg_prediction = np.zeros(len(y)); shuffled = np.zeros((8,len(y)))
            shuffled_reg = np.zeros((8,len(y)))
            for group in np.unique(groups):
                train, test = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
                assert not set(anchor_ids[train]) & set(anchor_ids[test])
                x = design(blocks, train)
                model = SVC(C=1, kernel="rbf", gamma="scale") if mode.endswith("rbf") else LogisticRegression(C=1, solver="liblinear", max_iter=2000, random_state=0)
                model.fit(x[train], y[train])
                probability[test] = 1/(1+np.exp(-model.decision_function(x[test]))) if mode.endswith("rbf") else model.predict_proba(x[test])[:,1]
                reg = None
                if target is not None and not mode.endswith("rbf"):
                    reg = Ridge(alpha=1.0).fit(x[train], target[train])
                    reg_prediction[test] = reg.predict(x[test])
                if mode == "vision_tactile":
                    for seed in range(8):
                        donor = np.random.default_rng(100+seed).permutation(np.unique(anchor_ids[test]))
                        mapping = dict(zip(donor, np.roll(donor, 1)))
                        altered = tactile.copy()
                        for a in donor:
                            altered[anchor_ids == a] = tactile[anchor_ids == mapping[a]]
                        sx = design([neural,metric,relations,altered], train)
                        shuffled[seed,test] = model.predict_proba(sx[test])[:,1]
                        if reg is not None:
                            shuffled_reg[seed,test] = reg.predict(sx[test])
            results.append(dict(condition=cfg["name"], mode=mode, **summary(y,probability,anchor_ids)))
            predictions[f"{c}_{mode}"] = probability
            if target is not None and not mode.endswith("rbf"):
                regressions.append(dict(condition=cfg["name"],mode=mode,**regression_summary(target,reg_prediction,anchor_ids)))
                predictions[f"reg_{c}_{mode}"] = reg_prediction
            if mode == "vision_tactile":
                controls = [summary(y,p,anchor_ids) for p in shuffled]
                results.append(dict(condition=cfg["name"],mode="vision_tactile_shuffled_test",
                    balanced_accuracy=float(np.mean([r["balanced_accuracy"] for r in controls])),
                    per_anchor_accuracy=np.mean([r["per_anchor_accuracy"] for r in controls],axis=0).tolist()))
                if target is not None:
                    controls_reg = [regression_summary(target,p,anchor_ids) for p in shuffled_reg]
                    regressions.append(dict(condition=cfg["name"],mode="vision_tactile_shuffled_test",
                        **{k:float(np.mean([r[k] for r in controls_reg])) for k in ("mae_nm","rmse_nm","nrmse_target_std")}))
        v = next(r for r in results if r["condition"] == cfg["name"] and r["mode"] == "vision")
        vt = next(r for r in results if r["condition"] == cfg["name"] and r["mode"] == "vision_tactile")
        comparisons.append(dict(condition=cfg["name"],**paired_gain(y_anchor,v,vt)))
    result = dict(status="complete",stage=manifest["stage"],window=manifest["window"],
                  fit_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  observation_time_s=manifest["observation_time_s"],physical_conditions=len(records),
                  views=views, gate=manifest["gate"], records=records, results=results,
                  paired_comparisons=comparisons, regression=regressions,
                  target="future seating success" if target is None else "measured resisting moment at rotor axis, mean 16–17 s; high if >=0.027 Nm",
                  split="leave-one-socket-yaw-out" if target is None else "leave-one-grip-force-out; same commanded action for all loads",
                  inference_note="12 local conditions, 3 nuisance groups; approximate condition-bootstrap/sign-flip uncertainty, not broad population inference")
    if target is not None:
        baseline = np.zeros(len(target))
        for group in np.unique(groups):
            baseline[groups == group] = target[groups != group].mean()
        result["regression_train_mean_baseline"] = regression_summary(target,baseline,anchor_ids)
    (output / "probes.json").write_text(json.dumps(result,indent=2))
    np.savez_compressed(output / "predictions.npz",labels=y,anchor_ids=anchor_ids,groups=groups,
                        **({"torque_target_nm":target} if target is not None else {}),**predictions)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=list(WINDOWS),required=True)
    parser.add_argument("--window",choices=["before","after"],required=True)
    parser.add_argument("--views",type=int,default=4)
    parser.add_argument("--fit-only",action="store_true")
    args = parser.parse_args()
    if not args.fit_only:
        extract(args.physics,args.output,args.stage,args.window,args.views)
    print(json.dumps({k:v for k,v in fit(args.output).items() if k not in ("records","results","regression")},indent=2))
