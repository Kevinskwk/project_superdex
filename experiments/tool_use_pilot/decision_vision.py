"""Reconstruct ideal/visible PCDs from v2 geometry; never fill hidden surfaces."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def transform(points, pose):
    return Rotation.from_quat(pose[3:]).apply(points) + pose[:3]


def surface(mesh, n, rng):
    triangles = np.asarray(mesh.triangles)
    index = rng.choice(len(triangles), n, p=mesh.area_faces / mesh.area)
    uv = rng.random((n, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    tri = triangles[index]
    return (
        tri[:, 0]
        + uv[:, :1] * (tri[:, 1] - tri[:, 0])
        + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
    )


@dataclass(frozen=True)
class Observation:
    mode: str = "visible"
    camera: int = 0
    noise_m: float = 0.0
    patch_fraction: float = 0.0
    count: int = 1024
    seed: int = 0


class Reconstructor:
    def __init__(self, h5):
        self.meshes = []
        self.roles = []
        for key in sorted(h5["geometry"], key=int):
            g = h5["geometry"][key]
            self.meshes.append(
                trimesh.Trimesh(g["vertices"][:], g["faces"][:], process=False)
            )
            self.roles.append(g.attrs["role"])
        self.eyes = h5["camera/eyes_world"][:]
        self.target = h5["camera/target_world"][:]
        self.width = int(h5["camera"].attrs["width"])
        self.height = int(h5["camera"].attrs["height"])
        self.fov = float(h5["camera"].attrs["vertical_fov_deg"])
        self.face_roles = np.concatenate(
            [
                np.repeat(
                    {"object": 0, "environment": 1, "occluder": 2}[r], len(m.faces)
                )
                for m, r in zip(self.meshes, self.roles)
            ]
        )

    def world_mesh(self, poses):
        return trimesh.util.concatenate(
            [
                trimesh.Trimesh(transform(m.vertices, p), m.faces, process=False)
                for m, p in zip(self.meshes, poses)
            ]
        )

    def visible(self, poses, camera=0, gel_surfaces=None):
        eye = self.eyes[camera]
        forward = self.target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        yy, xx = np.mgrid[: self.height, : self.width]
        scale = np.tan(np.deg2rad(self.fov) / 2)
        directions = (
            forward
            + ((xx.ravel() + 0.5) / self.width * 2 - 1)[:, None]
            * (self.width / self.height)
            * scale
            * right
            + (1 - (yy.ravel() + 0.5) / self.height * 2)[:, None] * scale * up
        )
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        mesh = self.world_mesh(poses)
        labels_by_face = self.face_roles
        if gel_surfaces is not None:
            faces = []
            for i in range(6):
                for j in range(8):
                    a = i * 9 + j
                    b = (i + 1) * 9 + j
                    faces.extend([[a, b, a + 1], [a + 1, b, b + 1]])
            patches = [
                trimesh.Trimesh(
                    np.asarray(surface).reshape(-1, 3), faces, process=False
                )
                for surface in gel_surfaces
            ]
            mesh = trimesh.util.concatenate([mesh, *patches])
            labels_by_face = np.r_[
                labels_by_face, np.full(len(faces) * len(patches), 2)
            ]
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        ray = RayMeshIntersector(mesh)
        locations, rays, triangles = ray.intersects_location(
            np.broadcast_to(eye, directions.shape), directions, multiple_hits=False
        )
        labels = labels_by_face[triangles]
        return [locations[labels == role] for role in (0, 1)]

    def clouds(self, poses, observation, gel_surfaces=None):
        cfg = observation
        rng = np.random.default_rng(cfg.seed)
        if cfg.mode == "ideal":
            clouds = []
            for role in ("object", "environment"):
                parts = [
                    trimesh.Trimesh(transform(m.vertices, p), m.faces, process=False)
                    for m, p, r in zip(self.meshes, poses, self.roles)
                    if r == role
                ]
                clouds.append(surface(trimesh.util.concatenate(parts), 4096, rng))
        else:
            clouds = self.visible(poses, cfg.camera, gel_surfaces)
        result = []
        counts = []
        for stream, points in enumerate(clouds):
            # A spatial patch anchored in the common camera plane. Same seed
            # reuses patch direction/bias across a causal observation window.
            augmentation = np.random.default_rng(
                np.random.SeedSequence([cfg.seed, stream, 1])
            )
            sampling = np.random.default_rng(
                np.random.SeedSequence([cfg.seed, stream, 2])
            )
            direction = augmentation.normal(size=3)
            direction /= np.linalg.norm(direction)
            bias = augmentation.normal(0, cfg.noise_m / 2, (1, 3))
            if len(points) and cfg.patch_fraction:
                score = (points - self.target) @ direction
                points = points[score >= np.quantile(score, cfg.patch_fraction)]
            counts.append(len(points))
            if not len(points):
                result.append(np.zeros((cfg.count, 3), np.float32))
                continue
            indices = sampling.choice(
                len(points), cfg.count, replace=len(points) < cfg.count
            )
            points = points[indices].copy()
            if cfg.noise_m:
                points += bias + augmentation.normal(0, cfg.noise_m, points.shape)
            result.append(points.astype(np.float32))
        return np.stack(result), np.asarray(counts, dtype=np.int32)


def normalize_and_metric(clouds, counts, ee_pose):
    normalized = []
    metric = []
    inv = Rotation.from_quat(ee_pose[3:]).inv()
    for points, count in zip(clouds, counts):
        if count == 0:
            normalized.append(np.zeros_like(points))
            metric.append(np.zeros(17))
            continue
        local = inv.apply(points - ee_pose[:3])
        center = local.mean(0)
        scale = max(float(np.linalg.norm(local - center, axis=1).max()), 1e-6)
        normalized.append((local - center) / scale)
        # Observed metric geometry, never simulator poses: centroid, extent,
        # covariance and coverage. Units remain metres / metres squared.
        covariance = np.cov(local.T)
        metric.append(
            np.r_[center, np.ptp(local, axis=0), covariance.ravel(), scale, 1.0]
        )
    return np.asarray(normalized, dtype=np.float32), np.asarray(
        metric, dtype=np.float32
    ).ravel()


def legacy_clouds(h5, index):
    """Read old stored clouds without requiring v2 geometry."""
    d = h5["observations"]
    key = "objectpointcloud" if "objectpointcloud" in d else "point_cloud"
    return d[key][index], d["env_point_cloud"][index]
