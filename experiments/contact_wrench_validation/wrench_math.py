"""Frame and contact-wrench utilities for the validation experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt


Vector = npt.NDArray[np.float64]


@dataclass
class Wrench:
    """A force and torque expressed at a documented world-space origin."""

    force: Vector
    torque: Vector
    count: int = 0


def as_vector(value: Any) -> Vector:
    return np.asarray(value, dtype=np.float64).reshape(3)


def shift_wrench(
    force: npt.ArrayLike,
    torque: npt.ArrayLike,
    from_origin: npt.ArrayLike,
    to_origin: npt.ArrayLike,
) -> Wrench:
    """Shift a wrench between origins without changing its coordinate frame."""

    force_array = as_vector(force)
    torque_array = as_vector(torque)
    lever = as_vector(from_origin) - as_vector(to_origin)
    return Wrench(force_array, torque_array + np.cross(lever, force_array))


def opposite_wrench(
    force: npt.ArrayLike,
    torque: npt.ArrayLike,
    from_origin: npt.ArrayLike,
    to_origin: npt.ArrayLike,
) -> Wrench:
    """Return the Newton-pair wrench on the other body at another origin."""

    return shift_wrench(-as_vector(force), -as_vector(torque), from_origin, to_origin)


def rotate_wrench_to_local(
    force_world: npt.ArrayLike,
    torque_world: npt.ArrayLike,
    rotation_world_from_local: npt.ArrayLike,
) -> Wrench:
    """Rotate world-frame wrench vectors into a body's local axes."""

    rotation = np.asarray(rotation_world_from_local, dtype=np.float64).reshape(3, 3)
    return Wrench(rotation.T @ as_vector(force_world), rotation.T @ as_vector(torque_world))


def inertia_matrix(packed: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Expand [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] into a symmetric matrix."""

    ixx, ixy, ixz, iyy, iyz, izz = np.asarray(packed, dtype=np.float64)
    return np.array(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
        dtype=np.float64,
    )


def aggregate_contact_points(
    contacts: Iterable[Any],
    receiver_handle: Any,
    other_handle: Any,
    origin_world: npt.ArrayLike,
) -> Wrench:
    """Integrate contacts into the wrench on ``receiver`` from ``other``.

    SuperDex may report the queried actor as either actor A or actor B. Contact
    force is defined as the force on actor A, so B-side contacts require a sign
    flip and use pos_b as their point of application.
    """

    origin = as_vector(origin_world)
    force_total = np.zeros(3, dtype=np.float64)
    torque_total = np.zeros(3, dtype=np.float64)
    count = 0
    for contact in contacts:
        if contact.actor_a == receiver_handle and contact.actor_b == other_handle:
            force = as_vector(contact.force)
            position = as_vector(contact.pos_a)
        elif contact.actor_b == receiver_handle and contact.actor_a == other_handle:
            force = -as_vector(contact.force)
            position = as_vector(contact.pos_b)
        else:
            continue
        force_total += force
        torque_total += np.cross(position - origin, force)
        count += 1
    return Wrench(force_total, torque_total, count)


def append_vector(row: dict[str, Any], prefix: str, value: npt.ArrayLike) -> None:
    """Append a 3-vector to a flat output row with x/y/z column suffixes."""

    vector = as_vector(value)
    for axis, component in zip("xyz", vector):
        row[f"{prefix}_{axis}"] = float(component)
