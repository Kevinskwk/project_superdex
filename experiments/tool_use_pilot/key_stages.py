"""Independent key insertion and prepared-seated turning verification tasks.

Legacy combined KeyDecision is deliberately unchanged. This is a scripted
mechanics collector, with explicitly recorded privileged tool-pose feedback.
"""

from dataclasses import asdict, dataclass
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from decisions import DecisionSpec, KeyDecision, box_union, interpolate, stop_reason, write_episode
from pilot import metrics, smooth, stack


@dataclass(frozen=True)
class KeyStageSpec(DecisionSpec):
    kind: str = "key"
    stage: str = "insertion"
    case: str = "aligned"
    revision: int = 7
    contact_edge_m: float = 0.008
    turn_command_deg: float = 100.0

    def __post_init__(self):
        duration = self.duration
        super().__post_init__()
        if self.stage not in ("insertion", "turning") or self.kind != "key":
            raise ValueError("key stage must be insertion or turning")
        if self.stage == "turning" and (self.lateral_m or self.yaw_deg):
            raise ValueError("turning uses an aligned prepared seat, not an insertion attempt")
        if self.contact_edge_m <= 0:
            raise ValueError("positive contact mesh edge length required")
        object.__setattr__(self, "duration", duration or (17.0 if self.stage == "insertion" else 26.0))
        object.__setattr__(self, "branch_time", 4.0)


def stage_target(spec, t):
    def v(z=0, yaw=0):
        return np.array([0, 0, z, 0, 0, np.deg2rad(yaw)], dtype=float)
    knots = [(0, v(), "prepared_grasp"), (1.5, v(), "settle"), (4, v(), "ready")]
    if spec.stage == "insertion":
        knots += [(10, v(-.019), "seated_hold"), (12, v(-.019), "withdraw"),
                  (16, v(), "done"), (17, v(), "done")]
        knots[2] = (4, v(), "insert")
    else:
        angle = 0 if spec.case == "hold_control" else spec.turn_command_deg
        knots += [(16, v(yaw=angle), "turned_hold"), (18, v(yaw=angle), "unturn"),
                  (24, v(), "done"), (26, v(), "done")]
        knots[2] = (4, v(), "turn")
    return interpolate(knots, t)


def tapered_gate():
    # A single closed rectangular ring: no overlapping wall seams.
    vertices = []
    for z, hx, hy in ((0, .0076, .0056), (-.003, .0056, .0036)):
        for x, y in ((-.020, -.020), (.020, -.020), (.020, .020), (-.020, .020)):
            vertices.append([x, y, z])
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            vertices.append([x, y, z])
    faces = []
    for j in range(4):
        k = (j + 1) % 4
        for a, b, c, d in ((j, k, k+4, j+4), (j+8, j+12, k+12, k+8),
                            (j, j+8, k+8, k), (j+4, k+4, k+12, j+12)):
            faces += [[a, b, c], [a, c, d]]
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()
    return mesh


def pocket_mesh():
    # Rotor top is 4 mm below the fixed gate mouth; cavity floor is at -16 mm.
    w, h, outer, depth = .0112, .0072, .024, .012
    parts = [([outer, (outer-h)/2, depth], [0, s*(outer+h)/4, -depth/2]) for s in (-1, 1)]
    parts += [([(outer-w)/2, h, depth], [s*(outer+w)/4, 0, -depth/2]) for s in (-1, 1)]
    parts += [([outer, outer, .003], [0, 0, -depth-.0015])]
    return box_union(parts)


def measured_seat(blade_vertices, tool_position, tool_rotation, socket_position,
                  socket_rotation, rotor_position, rotor_rotation):
    """Contain the actual chamfered blade, not its larger imaginary box corners."""
    world = tool_rotation.apply(blade_vertices) + tool_position
    in_rotor = rotor_rotation.inv().apply(world - rotor_position)
    in_socket = socket_rotation.inv().apply(world - socket_position)
    # Same 0.1 mm lateral seating tolerance as the initial implementation;
    # the independent strict 0.3 mm penetration gate remains unchanged.
    seated = (np.all(abs(in_rotor[:, 0]) <= .0057)
              and np.all(abs(in_rotor[:, 1]) <= .0037)
              and in_socket[:, 2].min() >= -.016
              and in_socket[:, 2].max() <= -.003)
    return bool(seated), float(-in_socket[:, 2].min())


class KeyStageWorld(KeyDecision):
    def refine(self, mesh):
        # Uniform subdivision preserves a conforming closed surface.
        # subdivide_to_size can leave T-junctions between different refinement levels.
        while mesh.edges_unique_length.max() > self.spec.contact_edge_m:
            mesh = mesh.subdivide()
        return mesh

    def make_tool_mesh(self):
        # Stitch handle, round shaft and rectangular blade into ONE closed solid.
        # The legacy touching closed primitives have coincident internal caps.
        angles = np.arange(24) * 2 * np.pi / 24
        unit = np.c_[np.cos(angles), np.sin(angles)]
        rings = []
        for z in np.linspace(.025, -.025, 11):
            rings.append((z, .009, .012))
        rings += [(z, .0025, None) for z in np.linspace(-.025, -.061, 10)]
        rings += [(-.061, .005, .003), (-.065, .005, .003), (-.069, .005, .003)]
        vertices, faces = [], []
        for z, hx, hy in rings:
            radius = np.full(24, hx) if hy is None else 1 / np.maximum(abs(unit[:, 0])/hx, abs(unit[:, 1])/hy)
            vertices.extend(np.c_[unit * radius[:, None], np.full(24, z)])
        for layer in range(len(rings)-1):
            for j in range(24):
                a, b = layer*24+j, layer*24+(j+1)%24
                faces += [[a, a+24, b+24], [a, b+24, b]]
        for layer, reverse in ((0, False), (len(rings)-1, True)):
            center = len(vertices)
            vertices.append([0, 0, rings[layer][0]])
            for j in range(24):
                triangle = [center, layer*24+j, layer*24+(j+1)%24]
                faces.append(triangle[::-1] if reverse else triangle)
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.fix_normals()
        return self.refine(mesh)

    def contact_params(self, friction):
        contact = self.p.ContactParams()
        contact.coulomb_friction_coefficient = friction
        contact.penalty_coefficient = 5e10
        contact.penalty_smoothing_half_distance = .00005
        contact.penalty_threshold_default = .00002
        return contact

    def actor(self, name, mesh, pos, rot, mass, friction, static=False, gravity=True):
        p = self.p
        shape = p.create_tri_mesh_shape(np.asarray(mesh.vertices, np.float32).ravel(),
                                       np.asarray(mesh.faces, np.int32).ravel())
        a = self.scene.create_rigid_actor(name=name, shape=shape, mass=mass,
            contact=self.contact_params(friction), collider_type=p.ColliderType.MESH,
            has_gravity=gravity and not static,
            world_from_local=p.TransformRT(translation=pos, rotation=p.Quaternion.from_rotation_vector(rot.as_rotvec())))
        if static:
            com = a.get_center_of_mass_transform()
            a.add_boundary_condition_dofs_world(np.arange(6, dtype=np.int32),
                np.r_[np.asarray(com.translation), rot.as_rotvec()])
        return a

    def build_environment(self):
        p = self.p
        prepared = .019 if self.spec.stage == "turning" else 0
        self.socket_origin = self.origin + self.rot.apply([self.spec.lateral_m, 0, -.073 + prepared])
        self.socket_rot = self.rot * Rotation.from_euler("z", self.spec.yaw_deg, degrees=True)
        gate_mesh, rotor_mesh = self.refine(tapered_gate()), self.refine(pocket_mesh())
        gate = self.actor("keyed_gate", gate_mesh, self.socket_origin, self.socket_rot, 1, self.spec.friction, static=True)
        base_mesh = trimesh.creation.box([.060, .060, .006])
        rotor_shape = p.create_tri_mesh_shape(np.asarray(rotor_mesh.vertices, np.float32).ravel(), np.asarray(rotor_mesh.faces, np.int32).ravel())
        base_shape = p.create_tri_mesh_shape(np.asarray(base_mesh.vertices, np.float32).ravel(), np.asarray(base_mesh.faces, np.int32).ravel())
        params = p.ArticulatedActorParams(name="independent_key_socket")
        params.world_from_root = p.TransformRT(translation=self.socket_origin,
            rotation=p.Quaternion.from_rotation_vector(self.socket_rot.as_rotvec()))
        params.joints = [p.ArticulatedJointParams(name="weld", type=p.ArticulatedJointType.HARD),
            p.ArticulatedJointParams(name="rotor", type=(p.ArticulatedJointType.REVOLUTE if self.spec.stage == "turning" else p.ArticulatedJointType.HARD),
                axis=[0, 0, 1], parent_link_from_joint=p.TransformRT(translation=[0, 0, .021]))]
        params.links = [p.ArticulatedLinkParams(name="socket_base", parent_link=-1,
            parent_joint_from_link=p.TransformRT(translation=[0, 0, -.025]), shape=base_shape,
            collider_type=p.ColliderType.MESH, density=1000),
            p.ArticulatedLinkParams(name="socket_rotor", parent_link=0, shape=rotor_shape,
                collider_type=p.ColliderType.MESH, density=.06/rotor_mesh.volume,
                contact=self.contact_params(self.spec.friction))]
        self.fixture = self.scene.create_articulated_actor(params)
        self.base, self.rotor = [self.scene.get_actor(h) for h in self.fixture.get_nested_link_actors()]
        self.env = [gate, self.rotor, self.base]
        self.environment_meshes = [gate_mesh, rotor_mesh, base_mesh]
        self.slider = None
        self.table_top = self.socket_origin[2]

    def __init__(self, spec):
        super().__init__(spec)
        self.blade_vertices = self.mesh.vertices[self.mesh.vertices[:, 2] <= -.061 + 1e-8].copy()
        self.last_force = 0.0
        self.force_guard = False
        self.guard_z = 0.0
        self.rotation_correction = np.zeros(3)
        self.prepared_seat_verified = False

    def apply_environment_force(self):
        if self.spec.stage == "turning":
            super().apply_environment_force()

    def command_at(self, t, branch="nominal", anchor_command=None):
        value, phase = stage_target(self.spec, t)
        self.nominal_tool_target = value[:3].copy()
        self.trajectory_time = t
        actual = self.rot.inv().apply(np.asarray(self.tool.get_root_transform().translation) - self.origin)
        if self.spec.stage == "insertion" and 4 <= t < 12 and self.last_force > 3 and not self.force_guard:
            self.force_guard = True
            self.guard_z = actual[2] + .0005
        if self.force_guard and t >= 4:
            # Stop advancing rather than driving a misaligned blade through the gate.
            # Withdrawal MUST start from the guarded position, not the original
            # deep target (which would cause a discontinuous plunge at 12 s).
            value[2] = self.guard_z * (1 - smooth((t - 12) / 4))
            phase = "blocked_force_guard" if t < 12 else "guarded_withdraw"
        if t >= 1.5:
            self.tracking_correction = np.clip(self.tracking_correction + self.spec.dt *
                np.clip(4 * (value[:3] - actual), -.003, .003), -.012, .012)
            value[:3] += self.tracking_correction
            # Track the authored TOOL orientation, not the fixture orientation.
            # A small grasp tilt becomes a millimetre-scale blade-tip error.
            actual_rotation = self.rot.inv() * Rotation.from_rotvec(
                np.asarray(self.tool.get_root_transform().rotation.to_rotation_vector()))
            goal_rotation = Rotation.from_rotvec(value[3:])
            error = (goal_rotation * actual_rotation.inv()).as_rotvec()
            self.rotation_correction = np.clip(self.rotation_correction + self.spec.dt *
                np.clip(4 * error, -np.deg2rad(2), np.deg2rad(2)), -np.deg2rad(12), np.deg2rad(12))
            value[3:] = (Rotation.from_rotvec(self.rotation_correction) * goal_rotation).as_rotvec()
            if self.spec.stage == "turning":
                # OSC controls the EE frame ~161 mm above the grasped tool root.
                # Rotate about the TOOL root, including pitch/roll corrections;
                # otherwise a small orientation correction sweeps the seated key
                # sideways and produces unintended bending loads / grasp slip.
                offset = self.rot.inv().apply(self.origin - self.ee0)
                value[:3] += offset - Rotation.from_rotvec(value[3:]).apply(offset)
        return value[:3], phase, value[3:]

    def step(self, t, branch="nominal", anchor_command=None):
        row = super().step(t, branch, anchor_command)
        root = self.tool.get_root_transform()
        tool_rotation = Rotation.from_rotvec(np.asarray(root.rotation.to_rotation_vector()))
        center_world = np.asarray(root.translation) + tool_rotation.apply([0, 0, -.065])
        center = self.socket_rot.inv().apply(center_world - self.socket_origin)
        rotor_rotation = Rotation.from_rotvec(np.asarray(self.rotor.get_root_transform().rotation.to_rotation_vector()))
        seated, depth = measured_seat(self.blade_vertices, np.asarray(root.translation),
            tool_rotation, self.socket_origin, self.socket_rot,
            np.asarray(self.rotor.get_root_transform().translation), rotor_rotation)
        row["blade_center_socket_m"] = center
        row["insertion_depth_m"] = depth
        row["seat_valid"] = bool(seated)
        row["force_guard"] = self.force_guard
        row["task_progress"] = row["insertion_depth_m"] if self.spec.stage == "insertion" else row["rotor_angle_rad"]
        row["rewards"] = row["task_progress"]
        row["scripted_rotation_correction_rad"] = self.rotation_correction.copy()
        if self.spec.stage == "turning" and 3.5 <= t < 4:
            self.prepared_seat_verified = self.prepared_seat_verified or seated
        row["prepared_seat_verified"] = self.prepared_seat_verified
        self.last_force = float(np.linalg.norm(row["extrinsic_contact_wrench"][:3]))
        return row


def stage_audit(data, spec, abort):
    result = metrics(data, spec)
    active = ~data["initialization"]
    evaluation = active & (data["timestamps"] >= 4)
    achieved = data["seat_valid"] & evaluation
    if spec.stage == "insertion":
        achieved &= data["insertion_depth_m"] >= .014
    else:
        achieved &= abs(data["rotor_angle_rad"] - np.pi/2) <= np.deg2rad(5)
    n = max(1, round(.3 / spec.dt))
    result["task_success"] = bool(len(achieved) >= n and np.any(np.convolve(achieved.astype(int), np.ones(n), "valid") == n))
    result["clearance_valid"] = bool(data["rigid_penetration_m"][active].max() < .0003)
    result["prepared_seat_verified"] = bool(data["prepared_seat_verified"].any())
    result["physical_valid"] &= result["clearance_valid"] and abort is None
    if spec.stage == "turning":
        result["physical_valid"] &= result["prepared_seat_verified"]
    result["imitation_eligible"] = result["physical_valid"] and result["task_success"]
    result["tactile_valid"] &= result["physical_valid"]
    result.pop("max_progress_mm", None)
    result.update(stage=spec.stage, abort_reason=abort,
        max_insertion_depth_mm=float(1000 * data["insertion_depth_m"].max()),
        max_rotor_angle_deg=float(np.rad2deg(data["rotor_angle_rad"].max())),
        peak_extrinsic_force_n=float(np.linalg.norm(data["extrinsic_contact_wrench"][active, :3], axis=1).max()),
        peak_extrinsic_torque_nm=float(np.linalg.norm(data["extrinsic_contact_wrench"][active, 3:], axis=1).max()),
        force_guard_triggered=bool(data["force_guard"].any()))
    return result


def worker(spec_dict, output, render=False):
    import superdex.physics as p
    p.initialize(num_worker_threads=0)
    spec = KeyStageSpec(**spec_dict)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f"key_{spec.stage}_{spec.case}_{spec.episode_id:04d}"
    world = None
    try:
        start = time.perf_counter()
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        world = KeyStageWorld(spec)
        world.initial_hash = hashlib.sha256(world.checkpoint()[0]).hexdigest()
        recorder = None
        if render:
            from visuals import Recorder
            recorder = Recorder(world, output / name)
        rows, abort = [], None
        for i in range(round(spec.duration / spec.dt)):
            row = world.step(i * spec.dt)
            rows.append(row)
            if i % 400 == 0:
                print(f"{name}: t={i*spec.dt:.1f}s depth={row['insertion_depth_m']*1000:.2f}mm rotor={np.rad2deg(row['rotor_angle_rad']):.1f}deg seat={row['seat_valid']}", flush=True)
            if recorder and i % max(1, round(.05 / spec.dt)) == 0:
                recorder.frame(row)
            abort = stop_reason(row) if i * spec.dt >= 1.5 else None
            if spec.stage == "turning" and i * spec.dt >= 4 and not world.prepared_seat_verified:
                abort = "prepared_seat_not_verified"
            if abort:
                break
        data = stack(rows)
        result = stage_audit(data, spec, abort)
        write_episode(output / f"{name}.h5", data, spec, world, result)
        with h5py.File(output / f"{name}.h5", "a") as f:
            f.attrs.update(task_kind=f"key_{spec.stage}", task_stage=spec.stage,
                task_progress_units="m" if spec.stage == "insertion" else "rad",
                schema_version="vt_acwm_superdex_key_stages_v1", collector_pose_feedback=True,
                initial_condition="prepared seated grasp" if spec.stage == "turning" else "prepared grasp above socket",
                implementation_hash=source_hash)
            f.attrs["oracle_channels"] += ",blade_center_socket_m,prepared_seat_verified,scripted_rotation_correction_rad"
        if recorder:
            recorder.close(data)
        summary = dict(spec=asdict(spec), metrics=result, seconds=time.perf_counter()-start)
        (output / f"{name}.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        if world:
            world.close()
        p.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    specs = json.loads(args.specs.read_text()) if args.specs else [
        asdict(KeyStageSpec(episode_id=0, stage="insertion", case="aligned")),
        asdict(KeyStageSpec(episode_id=1, stage="insertion", case="lateral_miss", lateral_m=.003)),
        asdict(KeyStageSpec(episode_id=2, stage="turning", case="aligned")),
        asdict(KeyStageSpec(episode_id=3, stage="turning", case="hold_control")),
    ]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(worker, s, str(args.output), args.render) for s in specs]
        for job in jobs:
            print(json.dumps(job.result()), flush=True)


if __name__ == "__main__":
    main()
