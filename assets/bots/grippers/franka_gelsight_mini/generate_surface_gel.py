"""Small source-fitted FEM gels with a separate 2 mm marker lattice.

The exposed height is sampled from the pinned enclosed HydroShear mesh. This is
a piecewise-linear approximation, not a claim of mesh-converged mechanics.
Run with the project .venv Python; the legacy assets are never overwritten.
"""

import json
import numpy as np
import trimesh
from generate_assets import SOURCE, GENERATED, COMMIT, write_mochi


def front_height(mesh, xy):
    triangles = np.asarray(mesh.triangles)
    a = triangles[:, 0, :2]
    b = triangles[:, 1, :2] - a
    c = triangles[:, 2, :2] - a
    det = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
    good = np.abs(det) > 1e-15
    a, b, c, det, triangles = a[good], b[good], c[good], det[good], triangles[good]
    d = np.asarray(xy) - a
    u = (d[:, 0] * c[:, 1] - d[:, 1] * c[:, 0]) / det
    v = (b[:, 0] * d[:, 1] - b[:, 1] * d[:, 0]) / det
    inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    if not inside.any():
        return None
    z = (
        triangles[:, 0, 2]
        + u * (triangles[:, 1, 2] - triangles[:, 0, 2])
        + v * (triangles[:, 2, 2] - triangles[:, 0, 2])
    )
    return float(z[inside].min())


def generate():
    mesh = trimesh.load(
        SOURCE / "gsmini_elastomer_transformed_enclosed.obj",
        force="mesh",
        process=False,
    )
    lower, upper = mesh.bounds
    center = (lower[:2] + upper[:2]) / 2
    # Outer shoulder nodes plus exactly 7 x 9 interior marker nodes.
    x = np.r_[lower[0], center[0] + np.arange(-3, 4) * 0.002, upper[0]]
    y = np.r_[lower[1], center[1] + np.arange(-4, 5) * 0.002, upper[1]]
    xy = np.array([[xx, yy] for yy in y for xx in x])
    heights = []
    for i, point in enumerate(xy):
        z = front_height(mesh, point)
        # The source has rounded XY corners. Pull only unsupported outer nodes
        # inward to the source outline; interior marker coordinates never move.
        for scale in np.linspace(0.999, 0.90, 100):
            if z is not None:
                break
            xy[i] = center + scale * (point - center)
            z = front_height(mesh, xy[i])
        if z is None or z >= upper[2] - 1e-6:
            raise RuntimeError("Could not resolve exposed source height")
        heights.append(z)
    heights = np.asarray(heights)
    nx, ny, nz = len(x), len(y), 3
    node = lambda i, j, k: k * nx * ny + j * nx + i
    tets = []
    # Six tetrahedra sharing a consistent body diagonal: conforming faces.
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                h = [
                    node(i, j, k),
                    node(i + 1, j, k),
                    node(i + 1, j + 1, k),
                    node(i, j + 1, k),
                    node(i, j, k + 1),
                    node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1),
                    node(i, j + 1, k + 1),
                ]
                tets.extend(
                    [
                        [h[a] for a in t]
                        for t in [
                            (0, 1, 2, 6),
                            (0, 2, 3, 6),
                            (0, 3, 7, 6),
                            (0, 7, 4, 6),
                            (0, 4, 5, 6),
                            (0, 5, 1, 6),
                        ]
                    ]
                )
    tets = np.asarray(tets)
    markers = np.array([[node(i, j, 0) for j in range(1, 10)] for i in range(1, 8)])
    for kind in ("source_surface", "matched_box"):
        front = heights if kind == "source_surface" else np.full(len(xy), lower[2])
        vertices = np.vstack(
            [
                np.c_[xy, (1 - alpha) * front + alpha * upper[2]]
                for alpha in np.linspace(0, 1, nz)
            ]
        )
        tet = tets.copy()
        det = np.linalg.det(vertices[tet[:, 1:]] - vertices[tet[:, :1]])
        neg = det < 0
        tet[neg, 1], tet[neg, 2] = tet[neg, 2].copy(), tet[neg, 1].copy()
        assert np.min(np.abs(det)) > 1e-15
        path = GENERATED / f"gel_{kind}.mochi.json"
        write_mochi(path, vertices, tet, np.arange((nz - 1) * nx * ny, nz * nx * ny))
        faces = np.concatenate(
            [tet[:, idx] for idx in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]
        )
        unique, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
        exposed = np.setdiff1d(
            np.unique(unique[counts == 1]), np.arange((nz - 1) * nx * ny, nz * nx * ny)
        )
        metadata = dict(
            source_commit=COMMIT,
            method=kind,
            num_nodes=len(vertices),
            num_tetrahedra=len(tet),
            marker_pitch_xy_m=0.002,
            marker_indices=markers.tolist(),
            exposed_nodes=exposed.tolist(),
            contact_surface_axis="-Z",
            marker_span_xy_m=[0.012, 0.016],
            note="Source-height sampled coarse FEM; projected marker pitch, curved-surface 3D distance may be larger. matched_box shares XY topology.",
        )
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(kind, len(vertices), len(tet), "marker XY pitch = 2 mm")


if __name__ == "__main__":
    generate()
