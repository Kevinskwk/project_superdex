"""Bounded nominal repairs, kept separate from geometry/physics randomization."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from diversification import baseline, stamp


def preferred_pose_specs():
    """Current small real-matching candidates; qualification is in the report.

    Keep prototypes in numbered repair rounds; never emit invalid oblique
    sideways initializations in the recommended entry point.
    """
    rows = repair_specs(7) + repair_specs(6) + repair_specs(4)[-1:]
    return [stamp(s, "verification", s.case, {}, i) for i, s in enumerate(rows)]


def repair_specs(round_number=1):
    if round_number not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("Unknown bounded repair round")
    by = {(s.family, s.variant): s for s in baseline()}
    rows = []
    if round_number in (4, 5, 6, 7):
        hook = by["hook", "friction_normal"]
        for name, roll, pitch, mass in (
            ("top_down", 0, 0, 0.2),
            ("roll_10", 10, 0, 0.2),
            ("sideways", 0, -90, 0.2),
            ("sideways_loaded", 0, -90, 0.4),
        ):
            if round_number == 6 and pitch != -90:
                continue
            if round_number == 7 and pitch == -90:
                continue
            rows.append(
                stamp(
                    hook,
                    "repair",
                    name,
                    dict(
                        revision=22,
                        variant="loaded_box",
                        case=name,
                        level_task_frame=True,
                        hook_heading_deg=0,
                        robot_roll_deg=roll,
                        robot_pitch_deg=pitch,
                        sled_mass_kg=mass,
                        transfer_support_collider="analytic_box",
                        box_loop_height_m=0.055 if round_number >= 5 else None,
                        robot_pose_in_task_frame=round_number == 6,
                        tool_pose_feedback=round_number == 7,
                        max_wall_s=3600,
                    ),
                    len(rows),
                )
            )
        if round_number >= 5:
            return rows
        for lean in (10, 20):
            rows.append(
                stamp(
                    by["surface", "rounded_wall"],
                    "repair",
                    f"lean_{lean}",
                    dict(
                        revision=22,
                        case=f"lean_{lean}",
                        level_task_frame=True,
                        surface_lean_deg=lean,
                        tool_pose_feedback=True,
                        max_wall_s=3600,
                    ),
                    len(rows),
                )
            )
        return rows
    if round_number == 3:
        for variant in ("rounded_wall", "curved_slot"):
            rows.append(
                stamp(
                    by["surface", variant],
                    "repair",
                    "tip_tracking_lateral14",
                    dict(tool_pose_feedback=True, guide_max_offset_m=0.014),
                    len(rows),
                )
            )
        return rows
    for variant in (
        ("straight_wall", "straight_slot", "rounded_wall", "curved_slot")
        if round_number == 1
        else ()
    ):
        s = by["surface", variant]
        rows.append(
            stamp(s, "repair", "tip_tracking", dict(tool_pose_feedback=True), len(rows))
        )
    candidates = (
        (("path_a", -0.019, -0.006, 0.011), ("path_b", -0.018, -0.010, 0.012))
        if round_number == 1
        else (("path_c", -0.010, -0.010, 0.011), ("path_d", -0.012, -0.010, 0.011))
    )
    for name, offset, curve, extra in candidates:
        s = by["pushing", "pose"]
        rows.append(
            stamp(
                s,
                "repair",
                name,
                dict(
                    case=name,
                    push_contact_y_m=offset,
                    push_end_lateral_m=curve,
                    push_overtravel_m=extra,
                ),
                len(rows),
            )
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3, 4, 5, 6, 7), default=1)
    args = parser.parse_args()
    with args.output.open("x") as f:
        json.dump([asdict(s) for s in repair_specs(args.round)], f, indent=2)
