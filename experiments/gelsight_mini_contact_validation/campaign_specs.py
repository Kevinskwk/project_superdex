#!/usr/bin/env python3
"""Deterministic episode matrix for the SCFields tactile-fidelity campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np


STANDARD_PROTOCOLS = (
    "normal_load_unload",
    "drag_x",
    "drag_y",
    "torsion",
    "rocking",
    "dynamic_tap",
)
ORIENTATION_PAIRS_DEG = (
    (-25.0, 0.0, 20.0, 0.0),
    (25.0, 0.0, -20.0, 0.0),
    (0.0, -22.0, 0.0, 28.0),
    (0.0, 22.0, 0.0, -28.0),
)
ROBUSTNESS_PERTURBATIONS = (
    ("friction_low", 0.60, 1.0),
    ("friction_high", 2.00, 1.0),
    ("mass_half", 1.40, 0.5),
    ("mass_double", 1.40, 2.0),
)


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    seed: int
    tool_name: str
    tool_family: str
    size_quantile: str
    campaign_block: str
    protocol: str
    replicate: int
    target_force_n: float
    friction_scale: float = 1.4
    mass_scale: float = 1.0
    gripper_roll_deg: float = 0.0
    gripper_pitch_deg: float = 0.0
    tool_roll_deg: float = 0.0
    tool_pitch_deg: float = 0.0
    attack_angle_deg: float = 0.0

    @property
    def branch_group_id(self) -> str:
        return f"scfields-{self.tool_name}-{self.protocol}-{self.replicate:02d}"

    @property
    def initial_state_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeSpec":
        return cls(**value)


def _target_force(seed: int) -> float:
    return float(np.random.default_rng(seed).uniform(3.0, 9.0))


def build_campaign(tool_records: Iterable[dict[str, Any]], seed: int = 20260831) -> list[EpisodeSpec]:
    """Build the balanced 999-episode matrix agreed for the fidelity study."""
    tools = list(tool_records)
    if len(tools) != 27:
        raise ValueError(f"the balanced campaign requires exactly 27 tools, got {len(tools)}")
    names = [str(tool["name"]) for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("tool names must be unique")
    specs: list[EpisodeSpec] = []
    rng = np.random.default_rng(seed)

    def add(
        tool: dict[str, Any], block: str, protocol: str, replicate: int, **kwargs: Any
    ) -> None:
        episode_seed = int(rng.integers(0, 2**31 - 1))
        specs.append(
            EpisodeSpec(
                episode_id=len(specs),
                seed=episode_seed,
                tool_name=str(tool["name"]),
                tool_family=str(tool["family"]),
                size_quantile=str(tool["size_quantile"]),
                campaign_block=block,
                protocol=protocol,
                replicate=replicate,
                target_force_n=_target_force(episode_seed),
                **kwargs,
            )
        )

    # 27 tools x 6 standardized motions x 3 replicates = 486.
    for tool in tools:
        for protocol in STANDARD_PROTOCOLS:
            for replicate in range(3):
                add(tool, "standard", protocol, replicate)

    # 27 x 4 physics perturbations x 2 replicates = 216.
    for tool in tools:
        for perturbation, friction_scale, mass_scale in ROBUSTNESS_PERTURBATIONS:
            for replicate in range(2):
                add(
                    tool,
                    "robustness",
                    f"composite_{perturbation}",
                    replicate,
                    friction_scale=friction_scale,
                    mass_scale=mass_scale,
                )

    # 27 x 4 paired, deliberately large gripper/tool orientations = 108.
    for tool in tools:
        for replicate, values in enumerate(ORIENTATION_PAIRS_DEG):
            gr, gp, tr, tp = values
            add(
                tool,
                "orientation",
                "composite_extreme_orientation",
                replicate,
                gripper_roll_deg=gr,
                gripper_pitch_deg=gp,
                tool_roll_deg=tr,
                tool_pitch_deg=tp,
            )

    # Shape-specific interactions = 189.
    shaft_families = {"cylinder", "hex_prism", "cylinder_pen", "hex_pen", "square_pen"}
    for tool in tools:
        family = str(tool["family"])
        if family in shaft_families:
            for protocol in ("rolling", "tip_stroke"):
                for replicate in range(3):
                    add(tool, "task_specific", protocol, replicate)
        elif family == "scraper":
            for attack in (15.0, 30.0, 45.0):
                for replicate in range(3):
                    add(
                        tool, "task_specific", "scrape", replicate,
                        attack_angle_deg=attack,
                    )
        elif family == "rectangle":
            for protocol in ("corner_push", "lever"):
                for replicate in range(3):
                    add(tool, "task_specific", protocol, replicate)
        elif family == "peeler":
            for attack in (15.0, 30.0, 45.0):
                for replicate in range(3):
                    add(
                        tool, "task_specific", "peel", replicate,
                        attack_angle_deg=attack,
                    )

    if len(specs) != 999:
        raise AssertionError(f"campaign construction bug: expected 999, got {len(specs)}")
    return specs


def write_specs(path: Any, specs: Iterable[EpisodeSpec]) -> None:
    from pathlib import Path

    Path(path).write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2) + "\n"
    )


def read_specs(path: Any) -> list[EpisodeSpec]:
    from pathlib import Path

    return [EpisodeSpec.from_dict(value) for value in json.loads(Path(path).read_text())]
