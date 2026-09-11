#!/usr/bin/env python3
"""Prepare a small, reproducible SCFields tool set for SuperDex experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "assets" / "scfields"
LOCK_PATH = HERE / "scfields_assets.lock.json"
HF_REVISION = "923c0b409f65eaefec9ae1ffd5c695aa8aa6d7ca"
HF_BASE = (
    f"https://huggingface.co/datasets/Kevinskwk/scfields-release/resolve/{HF_REVISION}"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(relative_path: str, destination: Path, expected: str) -> None:
    """Fetch immutable input; never accept corrupt caches or publish partial files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(
                f"Source checksum mismatch; preserve/inspect this file: {destination}"
            )
        return
    url = f"{HF_BASE}/{urllib.parse.quote(relative_path, safe='/')}"
    request = urllib.request.Request(url, headers={"User-Agent": "superdex-scfields/1"})
    with tempfile.TemporaryDirectory(
        prefix=".download-", dir=destination.parent
    ) as tmp:
        partial = Path(tmp) / "asset"
        with (
            urllib.request.urlopen(request, timeout=45) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        if sha256(partial) != expected:
            raise ValueError(f"Downloaded source checksum mismatch: {relative_path}")
        # Same-filesystem hard link publishes atomically without replacing a cache.
        os.link(partial, destination)


def _canonical_regular(
    mesh: trimesh.Trimesh, metadata: dict[str, Any]
) -> tuple[trimesh.Trimesh, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    if metadata.get("tool_type") in {"hex_prism", "hex_pen"}:
        # The authored hexagon's wider transverse axis is X. Put it on the
        # gripper-closing Y axis so the metadata width is the grasped width.
        vertices[:, [0, 1]] = vertices[:, [1, 0]]
    grasp_z = float(vertices[:, 2].max()) - float(metadata.get("grasp_offset", 0.05))
    grasp = np.array([0.0, 0.0, grasp_z])
    vertices -= grasp
    canonical = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    return canonical, grasp


def _canonical_peeler(
    mesh: trimesh.Trimesh, grasp_offset: float = 0.05
) -> tuple[trimesh.Trimesh, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    centered = vertices - vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    long_axis = vh[0]
    transverse = vh[1:]
    projection = centered @ long_axis
    low = vertices[projection <= np.quantile(projection, 0.12)]
    high = vertices[projection >= np.quantile(projection, 0.88)]

    def radial_extent(points: np.ndarray) -> float:
        local = (points - points.mean(axis=0)) @ transverse.T
        return float(np.linalg.norm(local, axis=1).mean())

    # The narrower end is the handle end. Canonical +Z points from head to handle,
    # matching the procedural tools whose grasp region is near maximum Z.
    if radial_extent(low) < radial_extent(high):
        long_axis = -long_axis
        projection = -projection
    # Put the thinner transverse PCA direction on local Y, the gripper's
    # closing axis. This maximizes the number of peelers that fit between the
    # two 24 mm GelSight faces without changing scale.
    transverse_extents = np.ptp(centered @ transverse.T, axis=0)
    y_index = int(np.argmin(transverse_extents))
    y_axis = transverse[y_index]
    x_axis = transverse[1 - y_index]
    if np.dot(np.cross(x_axis, y_axis), long_axis) < 0.0:
        x_axis = -x_axis
    rotation = np.column_stack((x_axis, y_axis, long_axis))
    local = centered @ rotation
    grasp_z = float(local[:, 2].max()) - min(grasp_offset, 0.3 * np.ptp(local[:, 2]))
    grasp_local = np.array([0.0, 0.0, grasp_z])
    local -= grasp_local
    canonical = trimesh.Trimesh(vertices=local, faces=mesh.faces, process=False)
    grasp_source = vertices.mean(axis=0) + grasp_local @ rotation.T
    return canonical, grasp_source


def _mesh_record(
    name: str,
    family: str,
    source: Path,
    canonical_path: Path,
    metadata: dict[str, Any],
    source_relative: str,
) -> dict[str, Any]:
    loaded = trimesh.load(source, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise RuntimeError(f"{source} did not contain a triangular mesh")
    if family == "peeler":
        canonical, grasp_source = _canonical_peeler(
            loaded, float(metadata.get("grasp_offset", 0.05))
        )
    else:
        canonical, grasp_source = _canonical_regular(loaded, metadata)
    canonical.remove_unreferenced_vertices()
    surface_path = canonical_path.parent / "surface" / canonical_path.name
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.export(surface_path)
    collision = canonical.copy()
    collision_mode = "original_watertight"
    if not collision.is_watertight:
        trimesh.repair.fix_normals(collision)
        trimesh.repair.fill_holes(collision)
        collision_mode = "repaired"
    if not collision.is_watertight or abs(float(collision.volume)) < 1e-12:
        collision = collision.convex_hull
        collision_mode = "convex_hull"
    if float(collision.volume) < 0.0:
        collision.invert()
        collision_mode += "_winding_fixed"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    collision.export(canonical_path)
    bounds = np.asarray(collision.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    volume = abs(float(collision.volume))
    density = float(metadata.get("density", 1200.0))
    mass = float(np.clip(volume * density, 0.035, 0.30))
    inertia = np.asarray(collision.moment_inertia, dtype=float)
    if not collision.is_watertight or not np.isfinite(inertia).all():
        # Box inertia is a stable fallback for open/downsampled meshes.
        x, y, z = extents
        inertia = np.diag(
            [
                mass * (y * y + z * z) / 12.0,
                mass * (x * x + z * z) / 12.0,
                mass * (x * x + y * y) / 12.0,
            ]
        )
    else:
        inertia *= mass / max(abs(float(collision.mass)), 1e-12)
    return {
        "name": name,
        "family": family,
        "source": source_relative,
        "source_path": str(source.resolve()),
        "canonical_path": str(canonical_path.resolve()),
        "surface_path": str(surface_path.resolve()),
        "source_sha256": sha256(source),
        "canonical_sha256": sha256(canonical_path),
        "surface_sha256": sha256(surface_path),
        "collision_mode": collision_mode,
        "size_quantile": metadata.get("size_quantile", "peeler"),
        "density_kg_m3": density,
        "mass_kg": mass,
        "center_mass_local_m": np.asarray(collision.center_mass, dtype=float).tolist(),
        "inertia_local_kg_m2": inertia.tolist(),
        "friction": float(metadata.get("friction", 1.0)),
        "grasp_offset_m": float(metadata.get("grasp_offset", 0.05)),
        "grasp_point_source": grasp_source.tolist(),
        "bounds_local_m": bounds.tolist(),
        "extents_m": extents.tolist(),
        "watertight": bool(collision.is_watertight),
        "vertices": int(len(collision.vertices)),
        "faces": int(len(collision.faces)),
        "surface_vertices": int(len(canonical.vertices)),
        "surface_faces": int(len(canonical.faces)),
    }


def asset_lock() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text())
    if lock["revision"] != HF_REVISION:
        raise ValueError("Asset lock and downloader revision disagree")
    return lock


def _local_path(root: Path, record: dict, field: str) -> Path:
    """Resolve new relative paths or relocate legacy workstation absolute paths."""
    value = Path(record[field])
    if value.is_absolute():
        # Never reach back into another checkout just because that old path exists.
        parts = value.parts
        if "scfields" not in parts:
            raise ValueError(f"Cannot relocate legacy asset path: {value}")
        value = Path(*parts[parts.index("scfields") + 1 :])
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes manifest root: {value}")
    return resolved


def prepare(root: Path = DEFAULT_ROOT, capsule_root: Path | None = None) -> Path:
    """Rebuild the exact 27 tools used by the recorded experiments.

    Primitive task fixtures are generated by the benchmark, not downloaded here.
    Historical capsule copies are optional and are not used by the current pilot.
    Existing complete caches are checked and reused, never silently overwritten.
    """
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        load_manifest(manifest_path)
        return manifest_path
    lock = asset_lock()
    records = []
    # Only promote generated meshes after ALL locked hashes match.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as tmp:
        staging = Path(tmp)
        for entry in lock["tools"]:
            name = entry["name"]
            source = root / "source" / (name + ".obj")
            download(entry["source"], source, entry["source_sha256"])
            values = dict(
                tool_type=entry["family"],
                density=entry["density_kg_m3"],
                friction=entry["friction"],
                grasp_offset=entry["grasp_offset_m"],
                size_quantile=entry["size_quantile"],
            )
            record = _mesh_record(
                name,
                entry["family"],
                source,
                staging / "canonical" / (name + ".obj"),
                values,
                entry["source"],
            )
            for field in ("canonical", "surface"):
                if record[field + "_sha256"] != entry[field + "_sha256"]:
                    raise ValueError(
                        f"Generated {field} mismatch for {name}; use pinned generation dependencies"
                    )
                path = Path(record[field + "_path"])
                record[field + "_path"] = str(path.relative_to(staging))
            record["source_path"] = str(source.relative_to(root))
            records.append(record)
        for record in records:
            for field in ("canonical_path", "surface_path"):
                relative = record[field]
                target = root / relative
                expected = record[field.replace("_path", "_sha256")]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if sha256(target) != expected:
                        raise ValueError(
                            f"Refusing to overwrite modified mesh: {target}"
                        )
                else:
                    os.link(staging / relative, target)
    capsules = []
    if capsule_root is not None:
        # Explicit legacy option only; absence must not require a TacSL checkout.
        for index in (1, 10, 28):
            source = capsule_root / f"capsule_{index}.obj"
            target = root / "source/capsules" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = sha256(source)
            if target.exists() and sha256(target) != expected:
                raise ValueError(f"Refusing to overwrite capsule: {target}")
            if not target.exists():
                shutil.copy2(source, target)
            capsules.append(
                dict(
                    name=source.stem,
                    path=str(target.relative_to(root)),
                    sha256=expected,
                )
            )
    manifest = dict(
        schema_version="scfields_superdex_assets_v1",
        source_repository="https://github.com/Kevinskwk/SCFields",
        source_dataset=lock["source_dataset"],
        source_revision=lock["revision"],
        asset_lock_sha256=sha256(LOCK_PATH),
        license=lock["license"],
        selection="Frozen recorded 27-tool selection; exact source/collision/surface hashes",
        benchmark_tools=lock["benchmark_tools"],
        tools=records,
        capsules=capsules,
    )
    with manifest_path.open("x") as out:
        json.dump(manifest, out, indent=2)
        out.write("\n")
    load_manifest(manifest_path)
    return manifest_path


def load_manifest(path: Path = DEFAULT_ROOT / "manifest.json") -> dict[str, Any]:
    """Check pinned assets and return resolved paths without rewriting the file."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "scfields_superdex_assets_v1":
        raise ValueError(f"unsupported SCFields asset manifest: {path}")
    lock = asset_lock()
    expected = {r["name"]: r for r in lock["tools"]}
    if [r["name"] for r in manifest["tools"]] != list(expected):
        raise ValueError(
            "SCFields selection/order differs from the recorded asset lock"
        )
    for record in manifest["tools"]:
        entry = expected[record["name"]]
        for field in ("canonical", "surface", "source"):
            filename = _local_path(path.parent, record, field + "_path")
            digest = entry[field + "_sha256"]
            if record[field + "_sha256"] != digest:
                raise ValueError(
                    f"Manifest checksum differs from lock: {record['name']}/{field}"
                )
            if not filename.is_file() or sha256(filename) != digest:
                raise ValueError(
                    f"Missing or modified SCFields {field} asset: {filename}"
                )
            record[field + "_path"] = str(filename)
    for record in manifest.get("capsules", []):
        record["path"] = str(_local_path(path.parent, record, "path"))
    manifest["benchmark_tools"] = lock["benchmark_tools"]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--check", action="store_true", help="Offline checksum check only"
    )
    parser.add_argument(
        "--capsule-root", type=Path, help="Optional legacy capsule copies"
    )
    args = parser.parse_args()
    path = (
        args.output.resolve() / "manifest.json"
        if args.check
        else prepare(args.output.resolve(), args.capsule_root)
    )
    manifest = load_manifest(path)
    print(f"verified {len(manifest['tools'])} pinned tools at {path}")


if __name__ == "__main__":
    main()
