from collections import Counter
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from campaign_specs import build_campaign  # noqa: E402
from fidelity_campaign import motion_command, wrench_cop  # noqa: E402


def tool_records():
    records = []
    for family in (
        "cylinder", "rectangle", "hex_prism", "scraper",
        "cylinder_pen", "hex_pen", "square_pen",
    ):
        for size in ("small", "median", "large"):
            records.append(
                {"name": f"{family}_{size}", "family": family, "size_quantile": size}
            )
    for index in range(6):
        records.append(
            {"name": f"peeler_{index}", "family": "peeler", "size_quantile": "peeler"}
        )
    return records


def test_balanced_campaign_has_exact_agreed_blocks():
    specs = build_campaign(tool_records())
    assert len(specs) == 999
    assert [spec.episode_id for spec in specs] == list(range(999))
    assert Counter(spec.campaign_block for spec in specs) == {
        "standard": 486,
        "robustness": 216,
        "orientation": 108,
        "task_specific": 189,
    }
    assert len({spec.initial_state_hash for spec in specs}) == 999


def test_orientation_block_varies_both_tool_and_gripper():
    specs = build_campaign(tool_records())
    orientation = [spec for spec in specs if spec.campaign_block == "orientation"]
    assert all(abs(spec.gripper_roll_deg) + abs(spec.gripper_pitch_deg) > 0 for spec in orientation)
    assert all(abs(spec.tool_roll_deg) + abs(spec.tool_pitch_deg) > 0 for spec in orientation)


def test_wrench_cop_recovers_point_on_horizontal_plane():
    com = np.array([0.12, -0.04, 0.31])
    point = np.array([0.15, 0.02, 0.20])
    force = np.array([1.2, -0.8, 7.0])
    torque = np.cross(point - com, force)
    recovered = wrench_cop(np.r_[force, torque], com, point[2])
    np.testing.assert_allclose(recovered, point, atol=1e-12)


def test_task_motion_commands_are_distinct_and_finite():
    specs = build_campaign(tool_records())
    by_protocol = {}
    for spec in specs:
        if spec.protocol not in by_protocol:
            by_protocol[spec.protocol] = motion_command(spec, 0.625)
    for translation, rotation in by_protocol.values():
        assert np.isfinite(translation).all()
        assert np.isfinite(rotation).all()
    assert not np.allclose(by_protocol["drag_x"][0], by_protocol["drag_y"][0])
    assert not np.allclose(by_protocol["torsion"][1], by_protocol["rocking"][1])


def test_task_commands_have_no_onset_or_exit_discontinuity():
    specs = build_campaign(tool_records())
    task_specs = [spec for spec in specs if spec.campaign_block == "task_specific"]
    for spec in task_specs:
        start_translation, start_rotation = motion_command(spec, 0.48)
        end_translation, end_rotation = motion_command(spec, 0.90)
        np.testing.assert_allclose(start_translation, 0.0, atol=1e-12)
        np.testing.assert_allclose(start_rotation, 0.0, atol=1e-12)
        np.testing.assert_allclose(end_translation, 0.0, atol=1e-12)
        np.testing.assert_allclose(end_rotation, 0.0, atol=1e-12)


def test_peel_and_tip_stroke_are_bounded():
    specs = build_campaign(tool_records())
    for protocol, translation_limit, rotation_limit in (
        ("peel", 0.00301, np.deg2rad(8.01)),
        ("tip_stroke", 0.00801, np.deg2rad(10.01)),
    ):
        spec = next(item for item in specs if item.protocol == protocol)
        commands = [motion_command(spec, value) for value in np.linspace(0.48, 0.90, 101)]
        assert max(np.linalg.norm(value[0]) for value in commands) <= translation_limit
        assert max(np.linalg.norm(value[1]) for value in commands) <= rotation_limit
