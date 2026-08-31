#!/usr/bin/env python3
"""Prepare a small, reproducible SCFields tool set for SuperDex experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "assets" / "scfields"
HF_BASE = "https://huggingface.co/datasets/Kevinskwk/scfields-release/resolve/main"
TOOL_METADATA = "assets/tools/yaml/tool_asset_info.yaml"
REGULAR_FAMILIES = (
    "cylinder",
    "rectangle",
    "hex_prism",
    "scraper",
    "cylinder_pen",
    "hex_pen",
    "square_pen",
)
RAW_PEELERS = ("peeler_7", "peeler_1", "peeler_10")
COMBINED_PEELERS = (
    "peeler_1_head_rectangular_handle_h0_96_hnd0_99",
    "peeler_6_head_tool_cylinder_6_h0_90_hnd1_03",
    "peeler_11_head_tool_hex_prism_12_h0_90_hnd1_02",
)
DEFAULT_CAPSULE_ROOT = Path(
    os.environ.get(
        "SCFIELDS_CAPSULE_ROOT",
        PROJECT_ROOT.parent / "tacsl" / "IsaacGymEnvs" / "assets" / "shapes" / "mesh",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(relative_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    url = f"{HF_BASE}/{urllib.parse.quote(relative_path, safe='/')}"
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "superdex-scfields/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(destination)


def parse_simple_tool_yaml(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the flat generated tool metadata without adding a YAML dependency."""
    result: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line == "tools:":
            continue
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            name = line.strip()[:-1]
            current = result.setdefault(name, {})
            continue
        if current is None or not line.startswith("    ") or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value in ("", "null", "None"):
            current[key] = None
        else:
            try:
                current[key] = float(value) if any(c in value for c in ".eE") else int(value)
            except ValueError:
                current[key] = value.strip("'\"")
    return result


def select_regular_tools(metadata: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for family in REGULAR_FAMILIES:
        candidates = [
            (name, values)
            for name, values in metadata.items()
            if values.get("tool_type") == family
            and 0.022
            <= float(values["width"] if family in {"hex_prism", "hex_pen"} else values["thickness"])
            <= 0.030
        ]
        if len(candidates) < 3:
            raise RuntimeError(f"SCFields metadata has too few feasible {family} tools")
        candidates.sort(
            key=lambda item: float(item[1]["length"])
            * float(item[1]["thickness"])
            * float(item[1]["width"])
        )
        for quantile, label in ((0.1, "small"), (0.5, "median"), (0.9, "large")):
            index = int(round(quantile * (len(candidates) - 1)))
            name, values = candidates[index]
            selected.append((name, {**values, "size_quantile": label}))
    return selected


def _canonical_regular(mesh: trimesh.Trimesh, metadata: dict[str, Any]) -> tuple[trimesh.Trimesh, np.ndarray]:
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


def _canonical_peeler(mesh: trimesh.Trimesh, grasp_offset: float = 0.05) -> tuple[trimesh.Trimesh, np.ndarray]:
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
        canonical, grasp_source = _canonical_peeler(loaded, float(metadata.get("grasp_offset", 0.05)))
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


def prepare(root: Path = DEFAULT_ROOT, capsule_root: Path = DEFAULT_CAPSULE_ROOT) -> Path:
    source_root = root / "source"
    canonical_root = root / "canonical"
    metadata_path = source_root / "tools" / "tool_asset_info.yaml"
    download(TOOL_METADATA, metadata_path)
    metadata = parse_simple_tool_yaml(metadata_path)
    records: list[dict[str, Any]] = []
    for name, values in select_regular_tools(metadata):
        family = str(values["tool_type"])
        relative = f"assets/tools/mesh/{family}/{name}.obj"
        source = source_root / "tools" / family / f"{name}.obj"
        download(relative, source)
        records.append(
            _mesh_record(name, family, source, canonical_root / f"{name}.obj", values, relative)
        )
    for name in RAW_PEELERS:
        relative = f"assets/peeler_raw/mesh/{name}/downsampled.obj"
        source = source_root / "peelers" / f"{name}.obj"
        download(relative, source)
        records.append(
            _mesh_record(
                name, "peeler", source, canonical_root / f"{name}.obj",
                {"density": 1200.0, "friction": 1.0, "grasp_offset": 0.05}, relative,
            )
        )
    for name in COMBINED_PEELERS:
        relative = f"assets/peeler_combined/mesh/{name}.obj"
        source = source_root / "peelers_combined" / f"{name}.obj"
        download(relative, source)
        records.append(
            _mesh_record(
                name, "peeler", source, canonical_root / f"{name}.obj",
                {"density": 1200.0, "friction": 1.0, "grasp_offset": 0.05}, relative,
            )
        )
    capsules = []
    for index in (1, 10, 28):
        source = capsule_root / f"capsule_{index}.obj"
        if not source.exists():
            raise FileNotFoundError(
                f"required SCFields capsule is missing: {source}; pass --capsule-root "
                "or set SCFIELDS_CAPSULE_ROOT"
            )
        destination = source_root / "capsules" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        capsules.append(
            {
                "name": f"capsule_{index}",
                "source_path": str(source),
                "path": str(destination.resolve()),
                "sha256": sha256(destination),
            }
        )
    manifest = {
        "schema_version": "scfields_superdex_assets_v1",
        "source_repository": "https://github.com/Kevinskwk/SCFields",
        "source_dataset": "https://huggingface.co/datasets/Kevinskwk/scfields-release",
        "selection": "volume quantiles 0.1/0.5/0.9 after 22-30 mm grasp-width filter",
        "tools": records,
        "capsules": capsules,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def load_manifest(path: Path = DEFAULT_ROOT / "manifest.json") -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "scfields_superdex_assets_v1":
        raise ValueError(f"unsupported SCFields asset manifest: {path}")
    for record in manifest["tools"]:
        canonical = Path(record["canonical_path"])
        if not canonical.exists() or sha256(canonical) != record["canonical_sha256"]:
            raise ValueError(f"missing or modified canonical SCFields asset: {canonical}")
        surface = Path(record.get("surface_path", record["canonical_path"]))
        if not surface.exists() or sha256(surface) != record.get(
            "surface_sha256", record["canonical_sha256"]
        ):
            raise ValueError(f"missing or modified SCFields surface asset: {surface}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capsule-root", type=Path, default=DEFAULT_CAPSULE_ROOT)
    args = parser.parse_args()
    path = prepare(args.output.resolve(), args.capsule_root.resolve())
    manifest = load_manifest(path)
    print(f"prepared {len(manifest['tools'])} tools at {path}")


if __name__ == "__main__":
    main()
