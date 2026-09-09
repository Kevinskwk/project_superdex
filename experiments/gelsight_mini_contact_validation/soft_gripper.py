"""Build an FR3 with a Franka gripper and two soft GelSight Mini gels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
from scipy.spatial.transform import Rotation
import superdex.physics as physics
import superdex.robotics as robotics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets"
GELSIGHT_ROOT = ASSET_ROOT / "bots/grippers/franka_gelsight_mini"
FR3_PATH = ASSET_ROOT / "bots/arms/fr3_v2/fr3_v2.superdex_bot"
HAND_SHAPE = (
    ASSET_ROOT
    / "test/urdf/fr3v2_1_urdf/meshes/robot_ee/franka_hand_white/collision/hand.stl"
)
HOUSING_SHAPE = GELSIGHT_ROOT / "generated/housing_collision.mochi.json"
LEGACY_GEL_SHAPE = GELSIGHT_ROOT / "generated/gel_tet.mochi.json"
GEL_SHAPE = GELSIGHT_ROOT / "generated/gel_source_surface.mochi.json"
# The HydroShear housings protrude roughly 13 mm from their joint frames. A
# 15.5 mm half-spacing lets the stock 0--40 mm Franka jaw travel close the gel
# faces to approximately 8.5 mm while the 16 mm initial joint position leaves
# roughly 40 mm for inserting the widest selected SCFields tool.
MOUNT_HALF_SEPARATION_M = 0.0155
DEFAULT_FINGER_POSITION_M = 0.016


@dataclass(frozen=True)
class GelMaterial:
    youngs_modulus_pa: float = 200_000.0
    poisson_ratio: float = 0.49
    density_kg_m3: float = 1000.0
    mass_damping_s_inv: float = 5.0
    stiffness_damping_s: float = 0.003
    friction_coefficient: float = 1.4
    geometry: str = "source_surface"


@dataclass
class SoftGripper:
    actor: Any
    prefab: Any
    links: dict[str, Any]
    gels: dict[str, Any]
    gel_rest_root: dict[str, np.ndarray]
    gel_rest_sensor_surface: dict[str, np.ndarray]
    gel_contact_nodes: dict[str, np.ndarray]
    gel_surface_grid_indices: dict[str, np.ndarray]
    finger_dofs: dict[str, int]

    def get_dense_contact_field(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Current world positions and contact forces for all FEM nodes (N,3).

        Includes shoulder contacts omitted by point-only marker sampling.
        Register NODE_POSITIONS and NODE_CONTACT_FORCES before stepping. Sum cross(r, f) for the
        dense nodal wrench; the 7x9 binned field is a separate representation.
        """
        gel = self.gels[side]
        root = self.actor.get_root_transform()
        rotation = Rotation.from_rotvec(np.asarray(root.rotation.to_rotation_vector()))
        positions = rotation.apply(np.asarray(gel.get_node_positions_local()).reshape(-1, 3)) + np.asarray(root.translation)
        forces = np.zeros_like(positions)
        for sample in gel.get_node_contact_forces_world():
            forces[int(sample.index)] += np.asarray(sample.force)
        return positions, forces

    def get_surface_force_field(self, side: str) -> np.ndarray:
        """Read one gel's 7x9x3 contact-force bins in its sensor frame.

        Register ``physics.QueryType.NODE_CONTACT_FORCES`` on the selected gel
        before stepping the scene. The output axes are sensor X, sensor Y, and
        the three sensor-frame vector components. Curved gels aggregate exposed
        FEM loads into nearest-marker cells; this is not optical force inference.
        """
        if side not in self.gels:
            raise KeyError(f"unknown gel side {side!r}; expected 'left' or 'right'")
        from tactile_grid import dense_surface_force_field

        return dense_surface_force_field(
            self.gels[side].get_node_contact_forces_world(),
            self.gel_surface_grid_indices[side],
            self.links[f"franka_{side}_gelsight_housing"].get_root_transform(),
        )

    def get_surface_cell_moments(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """7x9x3 intrinsic moments [Nm], plus the unmapped 6D wrench.

        Both are on the gel in housing/sensor axes. Moments are about CURRENT
        marker positions; the unmapped wrench is about the housing origin.
        These are extra FEM diagnostics, not quantities measured by optical
        markers. See tactile_grid.surface_cell_moments for reconstruction.
        """
        from tactile_grid import surface_cell_moments
        positions, forces = self.get_dense_contact_field(side)
        grid = self.gel_surface_grid_indices[side]
        return surface_cell_moments(positions, forces, grid, positions[np.asarray(grid)],
                                    self.links[f'franka_{side}_gelsight_housing'].get_root_transform())


def quaternion_from_rpy(rpy: tuple[float, float, float]) -> Any:
    return physics.Quaternion.from_rotation_vector(Rotation.from_euler("xyz", rpy).as_rotvec())


def _copy_joint(joint: Any) -> Any:
    return physics.ArticulatedJointParams(
        name=joint.name,
        type=joint.type,
        parent_link_from_joint=joint.parent_link_from_joint,
        axis=joint.axis,
        friction=joint.friction,
        inertia=joint.inertia,
        min_limit=joint.min_limit,
        max_limit=joint.max_limit,
        limit_stiffness=joint.limit_stiffness,
        limit_damping=joint.limit_damping,
    )


def _load_link_shape(link: Any) -> Any:
    if not link.shape_file:
        return physics.ShapeHandle()
    return physics.load_shape_from_file(
        link.shape_file,
        bake_scale=link.shape_scale,
        bake_transform=physics.TransformRT(
            rotation=link.shape_rotation, translation=link.shape_translation
        ),
    )


def _copy_link(link: Any) -> Any:
    return physics.ArticulatedLinkParams(
        name=link.name,
        parent_link=link.parent_link,
        parent_joint_from_link=link.parent_joint_from_link,
        shape=_load_link_shape(link),
        layer=link.layer,
        collider_type=link.collider_type,
        contact=link.contact,
        has_gravity=False,
        density=link.density,
        mass=link.mass,
        center_of_mass=link.center_of_mass,
        moment_of_inertia=link.moment_of_inertia,
        boundary_element_type=link.boundary_element_type,
        boundary_subsampling=link.boundary_subsampling,
    )


def _append_gripper(prefab: Any, name_suffix: str = "") -> None:
    hand_index = len(prefab.links)
    hand_joint = robotics.BotJointPrefab(
        name="franka_hand_joint",
        type=physics.ArticulatedJointType.HARD,
        parent_link_from_joint=physics.TransformRT(
            rotation=quaternion_from_rpy((0.0, 0.0, -np.pi / 4.0))
        ),
    )
    hand_link = robotics.BotLinkPrefab(
        name="franka_hand",
        parent_link=9,
        shape_file=str(HAND_SHAPE),
        mass=0.6544,
        center_of_mass=[-0.0000376, 0.0119128, 0.0207260],
        moment_of_inertia=[0.00186, 0.0, 0.0, 0.0003, -2e-5, 0.00174],
        has_gravity=False,
        layer="GripperHousing",
    )
    prefab.joints.append(hand_joint)
    prefab.links.append(hand_link)

    # Keep collision/gel meshes in HydroShear's authored sensor frame. The
    # mirrored mounts point both exposed -Z gel faces inward; visual-only mesh
    # correction is baked separately by generate_assets.py.
    mounts = (
        ("left", -MOUNT_HALF_SEPARATION_M, (-np.pi / 2.0, 0.0, np.pi)),
        ("right", MOUNT_HALF_SEPARATION_M, (-np.pi / 2.0, 0.0, 0.0)),
    )
    for side, y, rpy in mounts:
        prefab.joints.append(
            robotics.BotJointPrefab(
                name=f"franka_{side}_finger_joint",
                type=physics.ArticulatedJointType.PRISMATIC,
                parent_link_from_joint=physics.TransformRT(
                    rotation=quaternion_from_rpy(rpy),
                    translation=[0.0, y, 0.0834],
                ),
                axis=[0.0, 0.0, 1.0],
                min_limit=[0.0, 0.0, 0.0],
                max_limit=[0.0, 0.0, 0.04],
                limit_stiffness=20_000.0,
                limit_damping=20.0,
                effort_limit=100.0,
            )
        )
        prefab.links.append(
            robotics.BotLinkPrefab(
                name=f"franka_{side}_gelsight_housing",
                parent_link=hand_index,
                shape_file=str(HOUSING_SHAPE),
                mass=0.060,
                center_of_mass=[0.0, -0.04, 0.0],
                moment_of_inertia=[5.8e-5, 0.0, 0.0, 1.25e-5, 0.0, 5.45e-5],
                has_gravity=False,
                layer="GripperHousing",
            )
        )
    prefab.default_pose = [
        *list(prefab.default_pose),
        DEFAULT_FINGER_POSITION_M,
        DEFAULT_FINGER_POSITION_M,
    ]
    prefab.name = f"fr3_v2_franka_dual_gelsight_mini{name_suffix}"


def _bake_default_pose(prefab: Any) -> None:
    """Make the authored default configuration the articulation's zero pose.

    Nested soft shapes are authored in the skeleton reference configuration. If
    the joints were posed only after creation, their constrained nodes would move
    while the rest of each gel stayed behind. Baking avoids that startup strain.
    """
    values = list(prefab.default_pose)
    cursor = 0
    for index in range(len(prefab.joints)):
        joint = prefab.joints[index]
        if joint.type not in (
            physics.ArticulatedJointType.REVOLUTE,
            physics.ArticulatedJointType.PRISMATIC,
        ):
            continue
        value = float(values[cursor])
        cursor += 1
        axis = np.asarray(joint.axis, dtype=float)
        if joint.type == physics.ArticulatedJointType.REVOLUTE:
            motion = physics.TransformRT(
                rotation=physics.Quaternion.from_rotation_vector(axis * value)
            )
        else:
            motion = physics.TransformRT(translation=axis * value)
        joint.parent_link_from_joint = joint.parent_link_from_joint * motion
        if joint.min_limit is not None:
            joint.min_limit = np.asarray(joint.min_limit, dtype=float) - axis * value
        if joint.max_limit is not None:
            joint.max_limit = np.asarray(joint.max_limit, dtype=float) - axis * value
    if cursor != len(values):
        raise RuntimeError(f"default pose has {len(values)} values but consumed {cursor}")
    prefab.default_pose = [0.0] * len(values)


def _root_from_links(joints: list[Any], links: list[Any]) -> list[Any]:
    result: list[Any] = []
    for joint, link in zip(joints, links):
        parent_from_link = joint.parent_link_from_joint * link.parent_joint_from_link
        result.append(
            parent_from_link
            if link.parent_link < 0
            else result[link.parent_link] * parent_from_link
        )
    return result


def _gel_params(name: str, shape: Any, material: GelMaterial) -> Any:
    soft_material = physics.SoftMaterialParams(
        type=physics.SoftMaterialType.NEO_HOOKEAN,
        density=material.density_kg_m3,
        mass_damping_coefficient=material.mass_damping_s_inv,
        stiffness_damping_coefficient=material.stiffness_damping_s,
    )
    soft_material.neo_hookean.youngs_modulus = material.youngs_modulus_pa
    soft_material.neo_hookean.poisson_ratio = material.poisson_ratio
    contact = physics.ContactParams()
    contact.coulomb_friction_coefficient = material.friction_coefficient
    return physics.SoftActorParams(
        name=name,
        shape=shape,
        layer="TactileGel",
        contact=contact,
        has_gravity=False,
        has_inertia=True,
        has_stress=True,
        material=soft_material,
    )


def _resolve_nested(scene: Any, actor: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    info = actor.get_articulated_shape_info()
    links = {
        name: scene.get_actor(handle)
        for name, handle in zip(info.link_names, actor.get_nested_link_actors())
    }
    gels = {
        scene.get_actor(handle).get_name().rsplit("/", 1)[-1]: scene.get_actor(handle)
        for handle in actor.get_nested_soft_actors()
    }
    return links, gels


def _finger_dofs(actor: Any) -> dict[str, int]:
    info = actor.get_articulated_shape_info()
    result: dict[str, int] = {}
    for side in ("left", "right"):
        index = list(info.joint_names).index(f"franka_{side}_finger_joint")
        result[side] = int(info.dof_info[index].offset)
    return result


def build_soft_gripper(
    scene: Any,
    material: GelMaterial,
    *,
    name_suffix: str = "",
    world_from_root: Any | None = None,
    ideal_arm_friction_compensation: bool = False,
    finger_coupling_stiffness_n_m: float = 0.,
) -> SoftGripper:
    """Create and initialize the custom soft-skinned robot."""
    prefab = robotics.load_bot_prefab_from_file(str(FR3_PATH))
    if ideal_arm_friction_compensation:
        # FCI externally commanded torques are augmented by motor-friction and
        # gravity compensation. Arm gravity is already disabled in _copy_link.
        # Zero residual joint friction models ideal cancellation without an
        # explicit previous-velocity feedforward lag. NOT measured calibration.
        import xml.etree.ElementTree as ET
        urdf = ASSET_ROOT/'test/urdf/fr3v2_1_urdf/robots/fr3v2_1_franka_hand.urdf'
        limits = {j.attrib['name']:float(j.find('limit').attrib['effort'])
                  for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
        for joint in prefab.joints:
            if joint.name in {f'fr3_joint{i}' for i in range(1,8)}:
                joint.friction = physics.ArticulatedJointFrictionParams()
                joint.effort_limit = limits[joint.name.replace('fr3_joint','fr3v2_1_joint')]
    _append_gripper(prefab, name_suffix)
    if world_from_root is not None:
        prefab.world_from_root = world_from_root
    _bake_default_pose(prefab)
    joints = [_copy_joint(prefab.joints[i]) for i in range(len(prefab.joints))]
    links = [_copy_link(prefab.links[i]) for i in range(len(prefab.links))]
    skeleton = physics.ArticulatedActorParams(
        name=prefab.name,
        world_from_root=prefab.world_from_root,
        joints=joints,
        links=links,
    )
    rest_transforms = _root_from_links(joints, links)
    gel_shapes: list[Any] = []
    gel_rest_root: dict[str, np.ndarray] = {}
    gel_rest_sensor_surface: dict[str, np.ndarray] = {}
    gel_contact_nodes: dict[str, np.ndarray] = {}
    gel_surface_grid_indices: dict[str, np.ndarray] = {}
    for side, link_index in (("left", 11), ("right", 12)):
        # Soft shapes are authored in articulation-root coordinates. Mochi
        # applies ``world_from_root`` when the skinned actor starts stepping.
        root_from_sensor = rest_transforms[link_index]
        if material.geometry not in ('legacy_box', 'source_surface', 'matched_box'):
            raise ValueError(f'Unknown gel geometry: {material.geometry}')
        gel_path = LEGACY_GEL_SHAPE if material.geometry == 'legacy_box' else GELSIGHT_ROOT / 'generated' / f'gel_{material.geometry}.mochi.json'
        shape = physics.load_shape_from_file(str(gel_path), bake_transform=root_from_sensor)
        gel_shapes.append(shape)
        mesh = physics.get_shape_mesh(shape)
        rest = np.asarray(mesh.coordinates, dtype=float).reshape(-1, 3)
        gel_rest_root[side] = rest
        # Transform the baked root-frame nodes back into the authored sensor frame.
        rotation = Rotation.from_rotvec(
            np.asarray(root_from_sensor.rotation.to_rotation_vector(), dtype=float)
        )
        sensor_rotation = rotation.inv()
        sensor_translation = np.asarray(
            root_from_sensor.translation, dtype=float
        )
        sensor_rest = sensor_rotation.apply(rest - sensor_translation)
        if material.geometry != 'legacy_box':
            from tactile_grid import SurfaceGrid
            metadata = json.loads(gel_path.with_suffix('.metadata.json').read_text())
            indices = np.asarray(metadata['marker_indices'], dtype=np.int32)
            marker_xy = sensor_rest[indices, :2].reshape(-1, 2)
            exposed = np.asarray(metadata['exposed_nodes'], dtype=np.int32)
            closest = np.argmin(np.linalg.norm(sensor_rest[exposed, None, :2] - marker_xy[None], axis=2), axis=1)
            mapping = {int(n): tuple(np.unravel_index(int(c), (7,9))) for n,c in zip(exposed,closest)}
            gel_surface_grid_indices[side] = SurfaceGrid(indices, mapping)
            gel_contact_nodes[side] = exposed
            gel_rest_sensor_surface[side] = sensor_rest[indices]
            continue
        contact_nodes = np.flatnonzero(
            # FP32 world-space baking at multi-metre tiled offsets introduces
            # sub-micron variation across an otherwise planar gel surface.
            np.isclose(sensor_rest[:, 2], sensor_rest[:, 2].min(), atol=2e-6)
        )
        if len(contact_nodes) != 7 * 9:
            raise RuntimeError(
                "GelSight surface must contain 63 exposed nodes; "
                f"found {len(contact_nodes)}"
            )
        # Recover the regular axes from their extrema. This is robust to the
        # sub-micron FP32 roundoff introduced by baking articulation transforms.
        x_values = np.linspace(
            sensor_rest[contact_nodes, 0].min(),
            sensor_rest[contact_nodes, 0].max(),
            7,
        )
        y_values = np.linspace(
            sensor_rest[contact_nodes, 1].min(),
            sensor_rest[contact_nodes, 1].max(),
            9,
        )
        grid_indices = np.full((7, 9), -1, dtype=np.int32)
        for node_index in contact_nodes:
            x_index = int(np.argmin(np.abs(x_values - sensor_rest[node_index, 0])))
            y_index = int(np.argmin(np.abs(y_values - sensor_rest[node_index, 1])))
            if grid_indices[x_index, y_index] >= 0:
                raise RuntimeError("duplicate node while constructing GelSight surface grid")
            grid_indices[x_index, y_index] = node_index
        if np.any(grid_indices < 0):
            raise RuntimeError("incomplete GelSight surface grid")
        gel_contact_nodes[side] = grid_indices.ravel()
        gel_surface_grid_indices[side] = grid_indices
        gel_rest_sensor_surface[side] = sensor_rest[grid_indices]

    ss_params = physics.SoftSkinnedActorParams(
        skeleton_params=skeleton,
        soft_params=[
            _gel_params("left_gelsight_gel", gel_shapes[0], material),
            _gel_params("right_gelsight_gel", gel_shapes[1], material),
        ],
        soft_attach_links=[
            "franka_left_gelsight_housing",
            "franka_right_gelsight_housing",
        ],
        has_gravity=True,
        has_inertia=False,
        has_stress=False,
        enable_colliding_links=True,
    )
    actor = scene.create_soft_skinned_actor(ss_params)
    nested_links, nested_gels = _resolve_nested(scene, actor)
    # The generated watertight housing colliders are conservative convex
    # proxies. Their inner corners touch each other before the visible
    # HydroShear housings do and otherwise impose an artificial ~27 mm minimum
    # tool width. The Franka joint limits remain the mechanical stop.
    scene.enable_actor_contact_symmetric(
        nested_links["franka_left_gelsight_housing"].get_handle(),
        nested_links["franka_right_gelsight_housing"].get_handle(),
        False,
        physics.IncludeNestedActors.NO,
    )
    # The gels are kinematically attached to the sensor housings. Their regular
    # bounding-box FEM volume intersects conservative robot collision meshes, so
    # explicitly exclude every robot link from tactile contact. Environment and
    # manipulated-object contacts remain enabled.
    for gel in nested_gels.values():
        for link in nested_links.values():
            scene.enable_actor_contact_symmetric(
                gel.get_handle(),
                link.get_handle(),
                False,
                physics.IncludeNestedActors.NO,
            )
    # Preserve the detailed HydroShear housing geometry for visualization while
    # retaining the watertight convex proxy for physics.
    from superdex.physics.utils import render_model_registry
    housing_visual = str(GELSIGHT_ROOT / "generated/housing_visual.glb")
    for side in ("left", "right"):
        render_model_registry.register(
            scene,
            nested_links[f"franka_{side}_gelsight_housing"].get_handle(),
            housing_visual,
            physics.TransformRT(),
            physics.Real3(1.0, 1.0, 1.0),
        )
    if finger_coupling_stiffness_n_m>0:
        # The stock Franka URDF has finger_joint2 mimic finger_joint1. A native
        # bilateral transmission enforces equal opening displacement without
        # attaching the TOOL or freezing the common opening/closing mode.
        info=actor.get_articulated_shape_info()
        indices=[list(info.joint_names).index(f'franka_{side}_finger_joint') for side in ('left','right')]
        transmission=physics.experimental.add_linear_transmission(actor,
            physics.experimental.LinearTransmissionParams(joint_indices=indices,joint_coefficients=[1.,-1.]))
        physics.experimental.attach_displacement_control_actuator(actor,transmission,
            physics.experimental.DisplacementControlActuatorParams(target_displacement=0.,
                stiffness=finger_coupling_stiffness_n_m,damping=20.,allow_compressive_force=True))
    return SoftGripper(
        actor=actor,
        prefab=prefab,
        links=nested_links,
        gels={
            "left": nested_gels["left_gelsight_gel"],
            "right": nested_gels["right_gelsight_gel"],
        },
        gel_rest_root=gel_rest_root,
        gel_rest_sensor_surface=gel_rest_sensor_surface,
        gel_contact_nodes=gel_contact_nodes,
        gel_surface_grid_indices=gel_surface_grid_indices,
        finger_dofs=_finger_dofs(actor),
    )


def create_osc(context: Any, gripper: SoftGripper, name_suffix: str = "") -> Any:
    handle = context.create_controller(
        "BASIC_OSC_PD", gripper.prefab, gripper.actor, f"gelsight_osc{name_suffix}"
    )
    controller = context.get_controller(handle)
    if controller is None:
        raise RuntimeError("failed to create BASIC_OSC_PD controller")
    controller.initialize(
        f"{gripper.prefab.name}/fr3_link0", f"{gripper.prefab.name}/fr3_link8"
    )
    return controller
