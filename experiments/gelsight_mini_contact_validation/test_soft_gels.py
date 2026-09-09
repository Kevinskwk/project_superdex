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


def test_cell_moments_preserve_couples_and_report_unmapped_contacts():
    from tactile_grid import SurfaceGrid, surface_cell_moments
    from types import SimpleNamespace
    indices = np.arange(63).reshape(7,9)
    grid = SurfaceGrid(indices, {**{int(i):tuple(np.unravel_index(i,(7,9))) for i in range(63)},63:(0,0)})
    positions = np.zeros((65,3)); positions[63]=[.004,0,0]; positions[64]=[0,.002,0]
    forces = np.zeros_like(positions);forces[0]=[0,-2,0];forces[63]=[0,2,0];forces[64]=[0,0,3]
    tf=physics.TransformRT()
    field=dense_surface_force_field([SimpleNamespace(index=i,force=f) for i,f in enumerate(forces)],grid,tf)
    moments,remainder=surface_cell_moments(positions,forces,grid,positions[indices],tf)
    np.testing.assert_allclose(field.sum((0,1)),0,atol=1e-9)
    np.testing.assert_allclose(moments.sum((0,1)),[0,0,.008],atol=1e-9)
    np.testing.assert_allclose(remainder,[0,0,3,.006,0,0],atol=1e-9)
    np.testing.assert_allclose(moments.sum((0,1))+remainder[3:],np.cross(positions,forces).sum(0))


def test_cell_moments_zero_for_original_one_node_per_marker():
    from tactile_grid import surface_cell_moments
    rng=np.random.default_rng(23);p=rng.normal(size=(63,3));f=rng.normal(size=(63,3))
    grid=np.arange(63).reshape(7,9)
    m,u=surface_cell_moments(p,f,grid,p[grid],physics.TransformRT())
    np.testing.assert_array_equal(m,0);np.testing.assert_array_equal(u,0)

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


def test_ideal_arm_compensation_and_urdf_effort_limits():
    physics.initialize(num_worker_threads=0)
    try:
        scene=physics.create_scene('ideal-arm-compensation')
        gripper=build_soft_gripper(scene,GelMaterial(),ideal_arm_friction_compensation=True)
        for i in range(1,8):
            joint=next(j for j in gripper.prefab.joints if j.name==f'fr3_joint{i}')
            assert joint.friction.coulomb==joint.friction.viscous==0
            assert joint.effort_limit==(87 if i<=4 else 12)
    finally:physics.shutdown()


def test_finger_transmission_is_bilateral_and_keeps_common_mode_free():
    physics.initialize(num_worker_threads=0)
    try:
        scene=physics.create_scene('coupled-fingers')
        g=build_soft_gripper(scene,GelMaterial(),finger_coupling_stiffness_n_m=1e6)
        dofs=np.arange(g.actor.get_num_dofs(),dtype=np.int32)
        forces=np.zeros(len(dofs),dtype=np.float32)
        fingers=list(g.finger_dofs.values()); forces[fingers]=[2.,1.]
        for _ in range(80):
            g.actor.set_external_forces_on_dofs(dofs,forces);scene.step(.001)
        q=np.zeros(len(dofs),dtype=np.float32);g.actor.get_dof_values(dofs,q)
        assert abs(q[fingers[0]]-q[fingers[1]])<1e-5
        assert np.mean(q[fingers])>1e-5  # the gearbox did not lock jaw opening
    finally:physics.shutdown()


def test_source_surface_and_two_mm_markers() -> None:
    from soft_gripper import GELSIGHT_ROOT
    for name in ('source_surface', 'matched_box'):
        path = GELSIGHT_ROOT / 'generated' / f'gel_{name}.mochi.json'
        payload = json.loads(path.read_text())
        metadata = json.loads(path.with_suffix('.metadata.json').read_text())
        vertices = np.asarray(payload['mesh']['coordinates']).reshape(-1,3)
        tets = np.asarray(payload['mesh']['connectivity']).reshape(-1,4)
        assert np.all(np.linalg.det(vertices[tets[:,1:]] - vertices[tets[:,:1]]) > 0)
        markers = vertices[np.asarray(metadata['marker_indices'])]
        assert np.allclose(np.diff(markers[...,0],axis=0), .002)
        assert np.allclose(np.diff(markers[...,1],axis=1), .002)
        assert np.allclose(np.ptp(markers.reshape(-1,3)[:,:2],axis=0), [.012,.016])
        assert np.allclose(vertices[payload['constrainedNodes'],2], vertices[:,2].max())
        if name == 'source_surface':
            assert np.ptp(vertices[:99,2]) > .0005
            assert np.ptp(markers[...,2]) > 1e-6


def test_force_bins_include_shoulder_and_conserve_force() -> None:
    from tactile_grid import SurfaceGrid
    from types import SimpleNamespace
    grid = SurfaceGrid(np.arange(63).reshape(7,9), {100:(0,0),101:(0,0),102:(6,8)})
    samples = [SimpleNamespace(index=n,force=[1,2,3]) for n in [100,101,102]]
    field = dense_surface_force_field(samples,grid,physics.TransformRT())
    assert np.allclose(field.sum(axis=(0,1)),[3,6,9])
    assert np.allclose(field[0,0],[2,4,6])


def test_source_mounts_and_dense_api() -> None:
    from scipy.spatial.transform import Rotation
    physics.initialize(num_worker_threads=0)
    try:
        scene = physics.create_scene('source gel mounting')
        robot = build_soft_gripper(scene,GelMaterial(geometry='source_surface'))
        centers = {side:rest.mean(axis=(0,1)) for side,rest in robot.gel_rest_sensor_surface.items()}
        transforms = {side:robot.links[f'franka_{side}_gelsight_housing'].get_root_transform() for side in centers}
        rotations = {side:Rotation.from_rotvec(np.asarray(t.rotation.to_rotation_vector())) for side,t in transforms.items()}
        world_centers = {side:rotations[side].apply(c)+np.asarray(transforms[side].translation) for side,c in centers.items()}
        for side, other in [('left','right'),('right','left')]:
            normal = rotations[side].apply([0,0,-1])
            assert np.dot(normal,world_centers[other]-world_centers[side]) > 0
            rest=robot.gel_rest_sensor_surface[side]
            assert np.allclose(np.diff(rest[...,0],axis=0),.002,atol=1e-7)
            assert np.allclose(np.diff(rest[...,1],axis=1),.002,atol=1e-7)
            for query in [physics.QueryType.NODE_CONTACT_FORCES,physics.QueryType.NODE_POSITIONS]:
                robot.gels[side].register_query(query)
        scene.step(.001)
        for side in centers:
            positions,forces=robot.get_dense_contact_field(side)
            assert positions.shape == forces.shape == (297,3)
            assert np.isfinite(positions).all()
            assert robot.get_surface_force_field(side).shape == (7,9,3)
    finally:
        physics.shutdown()


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
