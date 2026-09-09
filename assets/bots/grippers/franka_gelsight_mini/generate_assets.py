#!/usr/bin/env python3
"""Fetch pinned HydroShear meshes and generate SuperDex collision/soft assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
GENERATED = HERE / "generated"
COMMIT = "f815b82fdf3451852acd918933020a82cede1f3b"
BASE_URL = (
    "https://raw.githubusercontent.com/MMintLab/hydroshear/"
    f"{COMMIT}/assets/tacsl/mesh/gs_mini_meshes"
)
FILES = {
    "panda_attach_gs_mini.obj": "2dea2f4ba5c9830a71c71c0d852d48b622cdd6d03c06f679ef61738ec304fda6",
    "panda_adapter_gsmini.obj": "eeab1cff056784c459eda40816a632d5ecd0d7bbed3dce724f3e9d85f4b9a5f2",
    "gsmini_shell_hollow_transformed.obj": "22fc9cdab1bcc58d3ee4e9a980bd89803bcac914d6e0817ef567fd6394157e9e",
    "gsmini_elastomer_transformed_both_sides.obj": "6e1d91bc09bd7eb29aab9fc1e1472725fb2b9166fe8833c77562c5a78da04a27",
    "gsmini_elastomer_transformed_enclosed.obj": "20ce7f04752c0d6e92645b6d811cb619a3ba482bbc6cf82d72a3c6d13aba8541",
}


def fetch_sources() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = SOURCE / name
        data = path.read_bytes() if path.exists() else urllib.request.urlopen(
            f"{BASE_URL}/{name}", timeout=60
        ).read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {name}: {actual}")
        if not path.exists():
            path.write_bytes(data)


def as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def transform_mesh(mesh: trimesh.Trimesh, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> trimesh.Trimesh:
    result = mesh.copy()
    matrix = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    matrix[:3, 3] = xyz
    result.apply_transform(matrix)
    return result


def write_mochi(path: Path, vertices: np.ndarray, elements: np.ndarray, constrained: np.ndarray | None = None) -> None:
    payload: dict[str, object] = {
        "mesh": {
            "nodesPerElement": int(elements.shape[1]),
            "coordinates": vertices.astype(float).ravel().tolist(),
            "connectivity": elements.astype(int).ravel().tolist(),
        }
    }
    if constrained is not None:
        payload["constrainedNodes"] = constrained.astype(int).tolist()
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def generate_housing() -> None:
    # Fold the HydroShear attachment, adapter, and shell into one rigid body.
    # Their transformed source meshes already share the GelSight sensor frame.
    attach = as_mesh(SOURCE / "panda_attach_gs_mini.obj")
    adapter = as_mesh(SOURCE / "panda_adapter_gsmini.obj")
    shell = as_mesh(SOURCE / "gsmini_shell_hollow_transformed.obj")
    compound = trimesh.util.concatenate((attach, adapter, shell))
    # The source shell has separate visual groups and is not a closed collision
    # volume. Use its watertight convex envelope for rigid-body collision/inertia.
    collision = compound.convex_hull
    write_mochi(
        GENERATED / "housing_collision.mochi.json",
        collision.vertices,
        collision.faces,
    )
    # HydroShear's reference URDF applies this correction only to visual meshes
    # (the supplied collision meshes stay in the sensor frame).
    visual = transform_mesh(compound, (0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0))
    visual.export(GENERATED / "housing_visual.glb")


def generate_gel_gmsh(target_size: float) -> None:
    try:
        import gmsh
    except ImportError as error:
        raise RuntimeError("Install the regeneration dependency with: uv pip install gmsh") from error

    source = SOURCE / "gsmini_elastomer_transformed_enclosed.obj"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("gelsight_mini_gel")
        gmsh.merge(str(source))
        gmsh.model.mesh.classifySurfaces(math.radians(40), True, True)
        gmsh.model.mesh.createGeometry()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size * 0.7)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coordinates, dtype=float).reshape(-1, 3)
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
        element_types, _, element_nodes = gmsh.model.mesh.getElements(dim=3)
        tetra_tags = None
        for kind, nodes in zip(element_types, element_nodes):
            _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(kind)
            if nodes_per_element == 4:
                tetra_tags = np.asarray(nodes, dtype=np.int64).reshape(-1, 4)
                break
        if tetra_tags is None:
            raise RuntimeError("Gmsh did not generate first-order tetrahedra")
        tets = np.vectorize(tag_to_index.__getitem__, otypes=[np.int64])(tetra_tags)
    finally:
        gmsh.finalize()

    signed = np.einsum(
        "ij,ij->i",
        vertices[tets[:, 1]] - vertices[tets[:, 0]],
        np.cross(vertices[tets[:, 2]] - vertices[tets[:, 0]], vertices[tets[:, 3]] - vertices[tets[:, 0]]),
    ) / 6.0
    valid = np.abs(signed) >= 1e-15
    removed_degenerate = int(np.count_nonzero(~valid))
    tets = tets[valid]
    signed = signed[valid]
    flipped = signed < 0
    tets[flipped, 2], tets[flipped, 3] = tets[flipped, 3].copy(), tets[flipped, 2].copy()
    # Dropping degenerate elements can orphan nodes, which would introduce zero
    # rows in the FEM mass/stiffness matrices. Compact the volume mesh.
    used = np.unique(tets)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vertices = vertices[used]
    tets = remap[tets]

    # The maximum-Z face meets the shell/camera backing; the protruding minimum-Z
    # face is exposed to objects.
    z_max = float(vertices[:, 2].max())
    constrained = np.flatnonzero(vertices[:, 2] >= z_max - 2.5e-4)
    if len(constrained) < 8:
        raise RuntimeError("too few gel attachment nodes selected")
    write_mochi(GENERATED / "gel_tet.mochi.json", vertices, tets, constrained)
    metadata = {
        "source_commit": COMMIT,
        "method": "gmsh",
        "target_edge_size_m": target_size,
        "num_nodes": int(len(vertices)),
        "num_tetrahedra": int(len(tets)),
        "removed_degenerate_tetrahedra": removed_degenerate,
        "num_constrained_nodes": int(len(constrained)),
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "contact_surface_axis": "-Z",
    }
    (GENERATED / "gel_tet.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def generate_gel_structured(resolution: tuple[int, int, int]) -> None:
    """Create an efficient regular FEM mesh from the supplied gel's exact bounds."""
    source = as_mesh(SOURCE / "gsmini_elastomer_transformed_enclosed.obj")
    lower, upper = np.asarray(source.bounds, dtype=float)
    nx, ny, nz = resolution
    axes = [np.linspace(lower[i], upper[i], resolution[i]) for i in range(3)]
    vertices = np.array(
        [[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]],
        dtype=float,
    )

    def node(i: int, j: int, k: int) -> int:
        return k * ny * nx + j * nx + i

    tets: list[list[int]] = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                h = [
                    node(i,j,k), node(i+1,j,k), node(i+1,j+1,k), node(i,j+1,k),
                    node(i,j,k+1), node(i+1,j,k+1), node(i+1,j+1,k+1), node(i,j+1,k+1),
                ]
                tets.extend(
                    ([h[0],h[1],h[3],h[4]], [h[1],h[2],h[3],h[6]],
                     [h[1],h[3],h[4],h[6]], [h[1],h[4],h[5],h[6]],
                     [h[3],h[4],h[6],h[7]])
                )
    tets_array = np.asarray(tets, dtype=np.int64)
    constrained = np.flatnonzero(vertices[:, 2] >= upper[2] - 1e-9)
    write_mochi(GENERATED / "gel_tet.mochi.json", vertices, tets_array, constrained)
    metadata = {
        "source_commit": COMMIT,
        "method": "structured_bounds",
        "resolution": list(resolution),
        "num_nodes": int(len(vertices)),
        "num_tetrahedra": int(len(tets_array)),
        "num_constrained_nodes": int(len(constrained)),
        "bounds_m": [lower.tolist(), upper.tolist()],
        "contact_surface_axis": "-Z",
        "volume_note": "Regular volume uses the exact source bounds; rounded edge detail remains in the visual source mesh.",
    }
    (GENERATED / "gel_tet.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", type=float, default=0.002)
    parser.add_argument("--method", choices=("source_surface", "structured", "gmsh"), default="source_surface")
    parser.add_argument(
        "--resolution",
        nargs=3,
        type=int,
        default=(7, 9, 2),
        metavar=("NX", "NY", "NZ"),
        help="structured method only: FEM node resolution (legacy box)",
    )
    args = parser.parse_args()
    if args.target_size <= 0:
        parser.error("--target-size must be positive")
    GENERATED.mkdir(parents=True, exist_ok=True)
    fetch_sources()
    generate_housing()
    if args.method == "source_surface":
        from generate_surface_gel import generate
        generate()
    elif args.method == "gmsh":
        generate_gel_gmsh(args.target_size)
    else:
        if min(args.resolution) < 2:
            parser.error("all structured resolution values must be at least 2")
        generate_gel_structured(tuple(args.resolution))


if __name__ == "__main__":
    main()
