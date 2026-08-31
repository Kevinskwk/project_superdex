"""Contact-wrench utilities shared by the soft fingertip experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Wrench:
    force: np.ndarray
    torque: np.ndarray
    count: int = 0


def vec(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3)


def aggregate_contact_points(
    contacts: Iterable[Any], receiver: Any, other: Any, origin: Any
) -> Wrench:
    force_total = np.zeros(3)
    torque_total = np.zeros(3)
    origin = vec(origin)
    count = 0
    for point in contacts:
        if point.actor_a == receiver and point.actor_b == other:
            force, position = vec(point.force), vec(point.pos_a)
        elif point.actor_b == receiver and point.actor_a == other:
            force, position = -vec(point.force), vec(point.pos_b)
        else:
            continue
        force_total += force
        torque_total += np.cross(position - origin, force)
        count += 1
    return Wrench(force_total, torque_total, count)


def opposite_at(wrench: Wrench, from_origin: Any, to_origin: Any) -> Wrench:
    force = -wrench.force
    torque = -wrench.torque + np.cross(vec(from_origin) - vec(to_origin), force)
    return Wrench(force, torque, wrench.count)


def integrate_surface_force_field(
    force_field_sensor: Any,
    positions_sensor: Any,
    world_from_sensor: Any,
    origin_world: Any,
    negate: bool = False,
) -> Wrench:
    """Integrate nodal forces and moments about ``origin_world``.

    ``force_field_sensor`` and ``positions_sensor`` must have the same spatial
    dimensions and a final XYZ dimension. Set ``negate=True`` to convert forces
    applied to the sensor into the equal-and-opposite forces applied to the tool.
    """
    forces_sensor = np.asarray(force_field_sensor, dtype=np.float64)
    positions_sensor = np.asarray(positions_sensor, dtype=np.float64)
    if forces_sensor.shape != positions_sensor.shape or forces_sensor.shape[-1] != 3:
        raise ValueError(
            "force field and surface positions must have matching [..., 3] shapes; "
            f"got {forces_sensor.shape} and {positions_sensor.shape}"
        )
    forces_sensor = forces_sensor.reshape(-1, 3)
    positions_sensor = positions_sensor.reshape(-1, 3)
    if negate:
        forces_sensor = -forces_sensor
    rotation = Rotation.from_rotvec(
        np.asarray(world_from_sensor.rotation.to_rotation_vector(), dtype=float)
    )
    forces_world = rotation.apply(forces_sensor)
    positions_world = rotation.apply(positions_sensor) + vec(world_from_sensor.translation)
    force = forces_world.sum(axis=0)
    torque = np.cross(positions_world - vec(origin_world), forces_world).sum(axis=0)
    count = int(np.count_nonzero(np.linalg.norm(forces_sensor, axis=1) > 0.0))
    return Wrench(force, torque, count)


def append_vector(row: dict[str, Any], name: str, value: Any) -> None:
    for axis, component in zip("xyz", vec(value)):
        row[f"{name}_{axis}"] = float(component)


def inertia_matrix(packed: Any) -> np.ndarray:
    ixx, ixy, ixz, iyy, iyz, izz = np.asarray(packed, dtype=float)
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
