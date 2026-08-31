"""Fast structural and tactile-grid tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import superdex.physics as physics

from soft_gripper import GEL_SHAPE, GelMaterial, build_soft_gripper
from tactile_grid import GRID_X, GRID_Y, dense_surface_force_field, render_shear_field

_SPEC = importlib.util.spec_from_file_location(
    "gelsight_wrench_math", Path(__file__).with_name("wrench_math.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_WRENCH_MATH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _WRENCH_MATH
_SPEC.loader.exec_module(_WRENCH_MATH)
integrate_surface_force_field = _WRENCH_MATH.integrate_surface_force_field


def test_generated_tetrahedra_are_positive_and_compact() -> None:
    payload = json.loads(Path(GEL_SHAPE).read_text())
    vertices = np.asarray(payload["mesh"]["coordinates"]).reshape(-1, 3)
    tets = np.asarray(payload["mesh"]["connectivity"]).reshape(-1, 4)
    signed = np.einsum(
        "ij,ij->i", vertices[tets[:,1]]-vertices[tets[:,0]],
        np.cross(vertices[tets[:,2]]-vertices[tets[:,0]], vertices[tets[:,3]]-vertices[tets[:,0]]),
    ) / 6.0
    assert np.all(signed > 0)
    assert len(np.unique(tets)) == len(vertices)
    constrained = np.asarray(payload["constrainedNodes"])
    assert len(constrained) > 0
    assert np.allclose(vertices[constrained,2], vertices[:,2].max())


def test_grid_shape() -> None:
    assert (GRID_X, GRID_Y) == (7, 9)


def test_dense_grid_zero_fills_and_rotates() -> None:
    class Sample:
        index = 17
        force = [1.0, 2.0, 3.0]

    indices = np.arange(GRID_X * GRID_Y, dtype=np.int32).reshape(GRID_X, GRID_Y)
    transform = physics.TransformRT(
        rotation=physics.Quaternion.from_rotation_vector([0.0, 0.0, np.pi / 2.0])
    )
    field = dense_surface_force_field([Sample()], indices, transform)
    assert field.shape == (7, 9, 3)
    assert np.count_nonzero(np.linalg.norm(field, axis=2)) == 1
    assert np.allclose(field.reshape(-1, 3)[17], [2.0, -1.0, 3.0])


def test_hydroshear_style_renderer_uses_full_grid() -> None:
    field = np.zeros((GRID_X, GRID_Y, 3), dtype=np.float32)
    field[..., 0] = 0.1
    field[..., 1] = np.linspace(-0.1, 0.1, GRID_Y)[None, :]
    field[..., 2] = np.linspace(0.0, 0.2, GRID_X)[:, None]
    image = render_shear_field(field)
    assert image.shape == (192, 144, 3)
    assert np.count_nonzero(image) > GRID_X * GRID_Y
    assert np.count_nonzero(image[:96]) > 0
    assert np.count_nonzero(image[96:]) > 0


def test_surface_force_field_wrench() -> None:
    forces = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    positions = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    wrench = integrate_surface_force_field(
        forces, positions, physics.TransformRT(), np.zeros(3)
    )
    assert np.allclose(wrench.force, [0.0, 1.0, 1.0])
    assert np.allclose(wrench.torque, [1.0, 0.0, 1.0])
    reaction = integrate_surface_force_field(
        forces, positions, physics.TransformRT(), np.zeros(3), negate=True
    )
    assert np.allclose(reaction.force, -wrench.force)
    assert np.allclose(reaction.torque, -wrench.torque)


def test_soft_gripper_topology() -> None:
    physics.initialize(num_worker_threads=0)
    try:
        scene = physics.create_scene("soft gels topology")
        robot = build_soft_gripper(scene, GelMaterial(poisson_ratio=0.45))
        assert robot.actor.get_num_dofs() == 9
        assert set(robot.gels) == {"left", "right"}
        assert set(robot.finger_dofs) == {"left", "right"}
        assert all(grid.shape == (7, 9) for grid in robot.gel_surface_grid_indices.values())
        assert all(
            gel.is_query_supported(physics.QueryType.CONTACT_POINTS)
            for gel in robot.gels.values()
        )
        assert all(
            gel.is_query_supported(physics.QueryType.NODE_CONTACT_FORCES)
            for gel in robot.gels.values()
        )
        for gel in robot.gels.values():
            gel.register_query(physics.QueryType.NODE_CONTACT_FORCES)
        scene.step(0.001)
        assert robot.get_surface_force_field("left").shape == (7, 9, 3)
        assert robot.get_surface_force_field("right").shape == (7, 9, 3)
    finally:
        physics.shutdown()
