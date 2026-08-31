"""Convert soft-gel node queries into dense sensor-frame tactile arrays."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


GRID_X = 7
GRID_Y = 9


def render_shear_field(
    force_field_sensor: np.ndarray,
    size: tuple[int, int] = (144, 192),
    shear_scale_n: float | None = None,
    normal_scale_n: float | None = None,
) -> np.ndarray:
    """Render a HydroShear-style colored arrow field on black.

    Arrow direction and length encode sensor-frame ``Fx, Fy``. Arrow color
    transitions from green to blue with the magnitude of ``Fz``. Scales are the
    force represented by approximately one grid spacing; if omitted they adapt
    to the current frame for legibility.
    """
    field = np.asarray(force_field_sensor, dtype=float)
    if field.shape != (GRID_X, GRID_Y, 3):
        raise ValueError(f"expected {(GRID_X, GRID_Y, 3)}, got {field.shape}")
    width, height = size
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    padding_x, padding_y = 12.0, 12.0
    step_x = (width - 2.0 * padding_x) / (GRID_X - 1)
    step_y = (height - 2.0 * padding_y) / (GRID_Y - 1)
    grid_spacing = min(step_x, step_y)
    shear_magnitude = np.linalg.norm(field[..., :2], axis=2)
    normal_magnitude = np.abs(field[..., 2])
    shear_scale = max(
        1e-9,
        float(shear_scale_n) if shear_scale_n is not None else float(shear_magnitude.max()),
    )
    normal_scale = max(
        1e-9,
        float(normal_scale_n) if normal_scale_n is not None else float(normal_magnitude.max()),
    )

    for x_index in range(GRID_X):
        for y_index in range(GRID_Y):
            start = np.array(
                [
                    padding_x + x_index * step_x,
                    height - padding_y - y_index * step_y,
                ]
            )
            shear = field[x_index, y_index, :2]
            delta = np.array([shear[0], -shear[1]]) / shear_scale * grid_spacing
            length = float(np.linalg.norm(delta))
            if length > 0.85 * grid_spacing:
                delta *= 0.85 * grid_spacing / length
                length = 0.85 * grid_spacing
            end = start + delta
            normal_ratio = float(
                np.clip(normal_magnitude[x_index, y_index] / normal_scale, 0.0, 1.0)
            )
            color = (0, int(round(255 * (1.0 - normal_ratio))), int(round(255 * normal_ratio)))
            if length < 0.75:
                draw.ellipse(
                    (start[0] - 1, start[1] - 1, start[0] + 1, start[1] + 1),
                    fill=color,
                )
                continue
            draw.line((*start, *end), fill=color, width=2)
            direction = delta / length
            perpendicular = np.array([-direction[1], direction[0]])
            head = max(3.0, min(5.0, 0.4 * length))
            base = end - direction * head
            draw.polygon(
                [tuple(end), tuple(base + perpendicular * head * 0.45), tuple(base - perpendicular * head * 0.45)],
                fill=color,
            )
    return np.asarray(image)


def world_to_local(transform: Any, values: np.ndarray, vectors: bool = False) -> np.ndarray:
    rotation = Rotation.from_rotvec(
        np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
    )
    array = np.asarray(values, dtype=float)
    if not vectors:
        array = array - np.asarray(transform.translation, dtype=float)
    return rotation.inv().apply(array)


def dense_surface_force_field(
    node_contact_forces: Iterable[Any],
    surface_grid_indices: np.ndarray,
    sensor_transform: Any,
) -> np.ndarray:
    """Return a dense ``[sensor_x, sensor_y, xyz]`` force field.

    SuperDex reports only nodes with non-zero contact force. This function
    zero-fills the complete gel surface and rotates every reported world-frame
    vector into the authored GelSight sensor frame.
    """
    indices = np.asarray(surface_grid_indices, dtype=np.int32)
    if indices.shape != (GRID_X, GRID_Y):
        raise ValueError(f"expected a {(GRID_X, GRID_Y)} surface grid, got {indices.shape}")
    node_to_cell = {
        int(node_index): (x_index, y_index)
        for (x_index, y_index), node_index in np.ndenumerate(indices)
    }
    field = np.zeros((GRID_X, GRID_Y, 3), dtype=np.float32)
    for sample in node_contact_forces:
        cell = node_to_cell.get(int(sample.index))
        if cell is not None:
            field[cell] += world_to_local(
                sensor_transform, np.asarray(sample.force, dtype=float), vectors=True
            ).astype(np.float32)
    return field


def surface_displacement(
    node_positions_root: Any,
    surface_grid_indices: np.ndarray,
    housing_transform: Any,
    world_from_root: Any,
    rest_sensor: np.ndarray,
) -> np.ndarray:
    positions_root = np.asarray(node_positions_root, dtype=float).reshape(-1, 3)
    root_rotation = Rotation.from_rotvec(
        np.asarray(world_from_root.rotation.to_rotation_vector(), dtype=float)
    )
    positions_world = root_rotation.apply(positions_root) + np.asarray(
        world_from_root.translation, dtype=float
    )
    current_sensor = world_to_local(housing_transform, positions_world)
    indices = np.asarray(surface_grid_indices, dtype=np.int32)
    return (current_sensor[indices] - rest_sensor).astype(np.float32)
