"""Versioned, bounded diversification manifests; later stages require reviewed evidence.

Export is cheap and does not simulate. Run one stage at a time; no auto-promotion.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark import BenchmarkSpec, SURFACES, specifications, run

VERSION = 21
REPRESENTATIVES = (
    ("surface", "flat_scrape"),
    ("surface", "cylindrical_peel"),
    ("surface", "curved_slot"),
    ("hook", "friction_normal"),
    ("insertion", "round"),
    ("insertion", "key"),
    ("turning", "friction"),
    ("pushing", "pose"),
)
# Nominal feasibility does not establish a useful robustness margin. Keep these
# out of factor/combinations runs until the endpoint/contact repairs are reviewed.
REPAIR_HOLDBACK = {
    ("surface", "straight_wall"),
    ("surface", "straight_slot"),
    ("surface", "rounded_wall"),
    ("surface", "curved_slot"),
    ("pushing", "pose"),
}
STAGES = (
    "baseline",
    "prerequisites",
    "composites",
    "geometry",
    "physics",
    "alignment",
    "combined",
)
GEOMETRY_KEYS = (
    "family",
    "variant",
    "profile_shape",
    "scale",
    "tip_scale",
    "tip_length_m",
    "shaft_diameter_m",
    "handle_width_m",
    "handle_depth_m",
    "follower_diameter_m",
    "working_width_scale",
    "clearance_m",
    "chamfer_m",
    "radius_m",
    "peeler_radius_m",
    "hook_working_drop_m",
    "hook_heading_deg",
    "pusher_bottom_m",
    "rail_clearance_m",
)
PHYSICS_KEYS = (
    "gel_geometry",
    "gel_modulus_pa",
    "gel_friction",
    "friction",
    "support_friction",
    "grip_force_n",
    "sled_mass_kg",
    "object_mass_kg",
    "load_mode",
    "friction_torque_nm",
    "torsion_nm_rad",
    "detent_torque_nm",
    "rotor_stop_deg",
    "rotor_stop_stiffness",
    "rotor_stop_damping",
    "surface_force_n",
    "speed_scale",
    "arm_stiffness_n_m",
    "arm_damping_ns_m",
    "arm_stiffness_nm_rad",
    "arm_damping_nms_rad",
    "dt",
    "solver_iterations",
    "solver_abs_tolerance",
)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def baseline():
    result = [
        replace(s, revision=VERSION)
        for s in specifications("transfer")
        if s.family != "levering"
    ]
    present = {(s.family, s.variant) for s in result}
    for variant in SURFACES:
        if ("surface", variant) not in present:
            result.append(
                BenchmarkSpec(
                    revision=VERSION,
                    family="surface",
                    variant=variant,
                    solver_iterations=8,
                    effective_friction=True,
                    max_wall_s=5400,
                )
            )
    return result


def geometry_change(s, factor):
    if s.family in ("insertion", "turning", "composite", "hook"):
        return dict(scale=s.scale * factor)
    if s.family == "pushing" or s.variant in ("flat_scrape", "flat_draw"):
        return dict(working_width_scale=factor)
    if s.variant == "cylindrical_peel":
        return dict(peeler_radius_m=s.peeler_radius_m * factor)
    return dict(follower_diameter_m=s.follower_diameter_m * factor)


def load_change(s, factor):
    if s.family == "insertion":
        return dict(speed_scale=s.speed_scale * factor, duration=0)
    if s.family == "surface":
        return dict(surface_force_n=s.surface_force_n * factor)
    if s.family == "hook":
        return dict(sled_mass_kg=s.sled_mass_kg * factor)
    if s.family == "pushing":
        return dict(object_mass_kg=s.object_mass_kg * factor)
    name = {
        "spring": "torsion_nm_rad",
        "friction": "friction_torque_nm",
        "detent": "detent_torque_nm",
    }[s.load_mode]
    return {name: getattr(s, name) * factor}


def external_friction_change(s, factor):
    # Transfer supports use shared actor coefficients. Changing them affects
    # tool/object AND object/support; the manifest does not call this isolated.
    key = "support_friction" if s.family in ("hook", "pushing") else "friction"
    # With a fixed tool actor coefficient, sqrt(tool * support) needs factor²
    # on support to achieve factor on effective tool/object friction.
    exponent = 2 if key == "support_friction" else 1
    return {key: getattr(s, key) * factor**exponent}


def stamp(parent, stage, condition, changes, index, outcome="nominal_goal"):
    spec = replace(parent, **changes)
    payload = asdict(spec)
    geometry_payload = {k: payload[k] for k in GEOMETRY_KEYS}
    if spec.level_task_frame:
        # These independent grasps bake wrist orientation into tool vertices.
        # Do not alias them to the old co-rotated fixture geometry IDs.
        geometry_payload.update({k: payload[k] for k in (
            "level_task_frame", "surface_lean_deg", "robot_roll_deg", "robot_pitch_deg", "robot_pose_in_task_frame", "box_loop_height_m")})
    geometry = digest(geometry_payload)
    physics = digest({k: payload[k] for k in PHYSICS_KEYS})
    seed = int(
        digest([VERSION, stage, condition, parent.family, parent.variant])[:8], 16
    )
    return replace(
        spec,
        episode_id=index,
        diversification_stage=stage,
        diversification_condition=condition,
        parent_config_id=digest(asdict(parent)),
        geometry_id=geometry,
        physics_id=physics,
        split_group=digest([geometry, physics]),
        intended_outcome=outcome,
        group=f"{parent.family}/{parent.variant}/{condition}",
        physical_seed=seed,
        geometry_seed=seed ^ 12345,
        camera_seed=seed ^ 54321,
        tactile_seed=seed ^ 98765,
    )


def manifest(stage, all_variants=False):
    bases = baseline()
    by = {(s.family, s.variant): s for s in bases}
    rows = []

    def add(s, name, changes=None, outcome="nominal_goal"):
        rows.append(stamp(s, stage, name, changes or {}, len(rows), outcome))

    if stage == "baseline":
        for s in bases:
            add(s, "nominal")
    elif stage == "prerequisites":
        for key in (
            ("insertion", "key"),
            ("turning", "friction"),
            ("turning", "spring"),
        ):
            for i in (1, 2):
                add(by[key], f"repeat_{i}")
    elif stage == "composites":
        for mode in ("friction", "spring"):
            add(
                by["turning", mode],
                "insert_turn_hold",
                dict(family="composite", variant="key_sequence_" + mode, duration=0),
            )
    elif stage == "geometry":
        for key in tuple(by) if all_variants else REPRESENTATIVES:
            s = by[key]
            for factor in (0.9, 1.1):
                add(
                    s,
                    f"handle_{factor}",
                    dict(handle_width_m=s.handle_width_m * factor),
                )
                add(s, f"working_{factor}", geometry_change(s, factor))
            add(s, "roll_plus5", dict(robot_roll_deg=5))
            add(s, "pitch_minus5", dict(robot_pitch_deg=-5))
    elif stage == "physics":
        for key in tuple(by) if all_variants else REPRESENTATIVES:
            s = by[key]
            for factor in (0.75, 1.25):
                add(
                    s,
                    f"external_friction_{factor}",
                    external_friction_change(s, factor),
                )
                add(s, f"load_{factor}", load_change(s, factor))
            # Preserve the existing 35 N safety cap; do not raise it for a sweep.
            for grip in (25.0, 30.0):
                add(s, f"grip_{grip}", dict(grip_force_n=grip))
            for modulus in (150000.0, 250000.0):
                add(s, f"gel_{modulus}", dict(gel_modulus_pa=modulus))
    elif stage == "alignment":
        for key, offsets in (
            (("insertion", "key"), (0.0003, 0.0009, 0.002)),
            (("hook", "friction_normal"), (0.008, 0.018, 0.024)),
            (("surface", "straight_slot"), (0.0005, 0.0015, 0.003)),
            (("pushing", "pose"), (0.005, 0.015, 0.025)),
        ):
            s = by[key]
            for name, offset in zip(
                ("marginal", "near_miss", "incorrect_contact"), offsets
            ):
                field = "fixture_y_m" if s.family == "surface" else "lateral_m"
                add(s, name, {field: offset}, outcome="measure_not_assume_" + name)
    elif stage == "combined":
        for s in bases:
            for i in range(4):
                # New working sizes in geometry holdout; new load in physics
                # holdout. These are conditional OOD axes, not universal splits.
                factor = (0.9, 1.1, 0.95, 1.0)[i]
                changes = geometry_change(s, factor)
                changes.update(load_change(s, (0.75, 1.25, 1.0, 1.1)[i]))
                changes.update(
                    robot_roll_deg=(-5, 5, -5, 5)[i],
                    robot_pitch_deg=(5, -5, -5, 5)[i],
                    grasp_depth_m=s.grasp_depth_m + (-0.002, 0.002, 0, 0)[i],
                )
                add(
                    s,
                    (
                        "within_low",
                        "within_high",
                        "geometry_holdout",
                        "physics_holdout",
                    )[i],
                    changes,
                )
    else:
        raise ValueError(stage)
    return rows


def reviewed_variants(path, stage):
    """Explicit human review plus recorded independent metrics; fail closed."""
    if path is None:
        raise ValueError("Later stages require --review with checked episode evidence")
    review = json.loads(Path(path).read_text())
    conditions = {}
    expected_specs = {
        (s.family, s.variant, s.diversification_condition): asdict(s)
        for s in manifest(stage, all_variants=True)
    }
    from benchmark_report import summarize, geometry_audit

    for entry in review.get("episodes", []):
        if not entry.get("visual_review_passed") or not entry.get(
            "housing_clearance_passed"
        ):
            continue
        p = (Path(path).parent / entry["episode"]).resolve()
        if not p.with_suffix(".json").exists() or json.loads(
            p.with_suffix(".json").read_text()
        ).get("error"):
            continue
        if entry.get("sha256") != hashlib.sha256(p.read_bytes()).hexdigest():
            continue
        row = summarize(p)
        s, m = row["spec"], row["metrics"]
        # Newly added optional controls retain historical-safe defaults. Compare
        # reconstructed behavior, not a changed hash of default metadata fields.
        actual_spec = asdict(BenchmarkSpec(**s))
        expected_spec = expected_specs.get(
            (s["family"], s["variant"], s.get("diversification_condition"))
        )
        if expected_spec is None or any(
            actual_spec.get(k) != value
            for k, value in expected_spec.items()
            if k
            not in (
                "episode_id",
                "block",
                "max_wall_s",
                "parent_config_id",
                "geometry_id",
                "physics_id",
                "split_group",
            )
        ):
            continue  # A pass from another configuration cannot qualify this one.
        if s.get("revision") != VERSION or not m.get("imitation_eligible"):
            continue
        if not geometry_audit(p)["passed"]:
            continue
        if s.get("diversification_stage") == stage:
            conditions.setdefault((s["family"], s["variant"]), set()).add(
                s["diversification_condition"]
            )
    expected = {}
    for s in manifest(stage, all_variants=True):
        expected.setdefault((s.family, s.variant), set()).add(
            s.diversification_condition
        )
    return {
        key for key, names in conditions.items() if expected.get(key, set()) <= names
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--run", action="store_true")
    p.add_argument("--review", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--families", nargs="+")
    p.add_argument(
        "--variants",
        nargs="+",
        help="Select a bounded subset without promoting weaker variants",
    )
    p.add_argument(
        "--all-variants",
        action="store_true",
        help="Expand factor checks beyond the eight representatives; consumes extra budget",
    )
    p.add_argument("--phase")
    args = p.parse_args()
    specs = manifest(args.stage, args.all_variants)
    if args.families:
        specs = [s for s in specs if s.family in args.families]
    if args.variants:
        specs = [s for s in specs if s.variant in args.variants]
    requested = len(specs)
    held_back = 0
    if args.run and args.stage in ("geometry", "physics", "alignment", "combined"):
        held_back = sum((s.family, s.variant) in REPAIR_HOLDBACK for s in specs)
        specs = [s for s in specs if (s.family, s.variant) not in REPAIR_HOLDBACK]
    if args.run and args.stage != "baseline":
        allowed = reviewed_variants(args.review, "baseline")
        if args.stage == "combined":
            allowed &= reviewed_variants(args.review, "geometry")
            allowed &= reviewed_variants(args.review, "physics")
        specs = [
            s
            for s in specs
            if (s.family, s.variant) in allowed
            or s.family == "composite"
            and ("turning", s.load_mode) in allowed
            and ("insertion", "key") in allowed
        ]
    if not specs:
        raise ValueError("No qualified specifications; review baseline first")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / (args.phase or args.stage)
    manifest_path = destination.with_suffix(".manifest.json")
    payload = [asdict(s) for s in specs]
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != payload:
            raise FileExistsError("Existing manifest differs; choose a new --phase")
    else:
        with manifest_path.open("x") as f:
            json.dump(payload, f, indent=2)
    print(
        json.dumps(
            dict(
                manifest=str(manifest_path),
                episodes=len(specs),
                run=args.run,
                blocked_by_review=requested - len(specs) - held_back,
                repair_holdback=held_back,
            )
        ),
        flush=True,
    )
    if args.run:
        run(
            SimpleNamespace(
                output=str(args.output),
                phase=destination.name,
                specs=str(manifest_path),
                families=None,
                variants=None,
                max_wall_s=None,
                limit=None,
                prerequisite_output=None,
                workers=args.workers,
                render=True,
            )
        )


if __name__ == "__main__":
    main()
