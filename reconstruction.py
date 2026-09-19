"""CPU Poisson baseline. This is NOT SuGaR training or density-level sampling.

Surface-aligned splat centers approximate a surface; arbitrary radiance-field
centers do not. Report the assumption and retain the source splat for comparison.
"""

import io
import time
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def decode(data, extension="ply", min_opacity=0.15):
    if not 0 <= min_opacity < 1:
        raise ValueError("Minimum opacity must be between zero and one.")
    if extension == "splat":
        if len(data) % 32:
            raise ValueError("A .splat record must be 32 bytes.")
        dt = np.dtype(
            [
                ("xyz", "<f4", 3),
                ("scale", "<f4", 3),
                ("rgba", "u1", 4),
                ("rotation", "u1", 4),
            ]
        )
        v = np.frombuffer(data, dtype=dt)
        xyz, scales = v["xyz"].astype(float), v["scale"].astype(float)
        q = (v["rotation"].astype(float) - 128) / 128
        q = q[:, [1, 2, 3, 0]]  # file wxyz -> scipy xyzw
        color, opacity = v["rgba"][:, :3] / 255.0, v["rgba"][:, 3] / 255.0
        normal = None
    else:
        ply = PlyData.read(io.BytesIO(data))
        if "vertex" not in ply:
            raise ValueError(
                "PLY must have vertices; compressed PLY is not supported by the CPU converter."
            )
        v = ply["vertex"].data
        names = v.dtype.names
        if any(k.startswith("binding_") for k in names):
            raise ValueError(
                (
                    'This avatar uses mesh-local Gaussian coordinates. Export it '
                    'to world space with its matching FLAME surface before '
                    'Poisson extraction; raw coordinates are not a face.'
                )
            )
        if not all(k in names for k in ("x", "y", "z")):
            raise ValueError(
                "Export an uncompressed Gaussian PLY with x/y/z properties."
            )
        xyz = np.column_stack([v[k] for k in ("x", "y", "z")]).astype(float)
        scales = (
            np.exp(
                np.clip(np.column_stack([v[f"scale_{k}"] for k in range(3)]), -20, 10)
            )
            if "scale_0" in names
            else None
        )
        q = (
            np.column_stack([v[f"rot_{k}"] for k in (1, 2, 3, 0)]).astype(float)
            if "rot_0" in names
            else None
        )
        opacity = (
            1 / (1 + np.exp(-np.clip(v["opacity"], -30, 30)))
            if "opacity" in names
            else np.ones(len(xyz))
        )
        if "f_dc_0" in names:
            color = np.clip(
                0.5
                + 0.28209479177387814
                * np.column_stack([v[f"f_dc_{k}"] for k in range(3)]),
                0,
                1,
            )
        elif "red" in names:
            color = np.column_stack([v[k] for k in ("red", "green", "blue")]) / 255.0
        else:
            color = np.tile([0.66, 0.43, 0.32], (len(xyz), 1))
        normal = (
            np.column_stack([v[k] for k in ("nx", "ny", "nz")]).astype(float)
            if "nx" in names
            else None
        )
        if normal is not None and np.mean(np.linalg.norm(normal, axis=1)) < 0.1:
            normal = None
    if len(xyz) < 100:
        raise ValueError("At least 100 surface samples are required.")
    if len(xyz) > 2_000_000:
        raise ValueError(
            "Crop the face to fewer than two million splats before conversion."
        )
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(color).all(axis=1)
        & (opacity > min_opacity)
    )
    if q is not None:
        valid &= np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0.01)
    if scales is not None:
        valid &= np.isfinite(scales).all(axis=1) & (scales > 0).all(axis=1)
    if normal is not None:
        valid &= np.isfinite(normal).all(axis=1) & (
            np.linalg.norm(normal, axis=1) > 0.01
        )
    if np.count_nonzero(valid) < 100:
        raise ValueError(
            "Too few valid Gaussian samples remain after opacity filtering."
        )
    xyz, color = xyz[valid], color[valid]
    normal = normal[valid] if normal is not None else None
    alignment = None
    if scales is not None:
        scales = scales[valid]
        ordered = np.sort(scales, axis=1)
        alignment = float(np.median(ordered[:, 0] / np.maximum(ordered[:, 1], 1e-12)))
    if normal is None and scales is not None and q is not None:
        q = q[valid]
        mats = Rotation.from_quat(q).as_matrix()
        normal = mats[np.arange(len(mats)), :, np.argmin(scales, axis=1)]
    return xyz, color, normal, alignment


def reconstruct(data, extension="ply", depth=7, max_triangles=26000, min_opacity=0.15):
    started = time.perf_counter()
    xyz, colors, normals, alignment = decode(data, extension, min_opacity)
    source_count = len(xyz)
    if source_count < 100:
        raise ValueError("Too few valid, opaque surface samples remain.")
    # Isolated face expected. Backgrounds should be cropped before import.
    lo, hi = np.quantile(xyz, [0.005, 0.995], axis=0)
    extent = float(np.max(hi - lo))
    if extent < 1e-9:
        raise ValueError("The point cloud has no measurable extent.")
    center = (lo + hi) / 2
    scale = 0.28 / max(hi[1] - lo[1], extent * 0.3)
    crop = ((xyz >= lo - extent * 0.04) & (xyz <= hi + extent * 0.04)).all(axis=1)
    xyz, colors = xyz[crop], colors[crop]
    normals = normals[crop] if normals is not None else None
    # Deterministic bounded CPU work. Never synthesize absent face features.
    if len(xyz) > 120000:
        ids = np.random.default_rng(42).choice(len(xyz), 120000, replace=False)
        xyz, colors = xyz[ids], colors[ids]
        normals = normals[ids] if normals is not None else None
    xyz = (xyz - center) * scale
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.colors = o3d.utility.Vector3dVector(colors)
    pc, keep = pc.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.5)
    if len(pc.points) < 100:
        raise ValueError("Insufficient connected surface after outlier removal.")
    if normals is None:
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=40)
        )
        normal_source = "estimated local PCA normals"
    else:
        pc.normals = o3d.utility.Vector3dVector(normals[keep])
        pc.normalize_normals()
        normal_source = "input normals or smallest covariance axis"
    try:
        pc.orient_normals_consistent_tangent_plane(30)
    except RuntimeError as exc:
        raise ValueError(
            "Could not orient the surface normals. Use a denser cropped face scan."
        ) from exc
    pts, ns = np.asarray(pc.points), np.asarray(pc.normals)
    if np.mean(np.sum((pts - np.median(pts, axis=0)) * ns, axis=1)) < 0:
        pc.normals = o3d.utility.Vector3dVector(-ns)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth, scale=1.08, linear_fit=False, n_threads=4
    )
    mesh.remove_vertices_by_mask(np.asarray(density) < np.quantile(density, 0.025))
    box = pc.get_axis_aligned_bounding_box()
    box = box.scale(1.035, box.get_center())
    mesh = mesh.crop(box)
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    if len(mesh.triangles) < 30:
        raise ValueError(
            "No usable surface reconstructed. Check crop, normals, and splat alignment."
        )
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    distances, nearest = cKDTree(pts).query(vertices, k=3)
    weights = 1 / np.maximum(distances, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    rgb = (np.asarray(pc.colors)[nearest] * weights[:, :, None]).sum(axis=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    elapsed = time.perf_counter() - started
    return {
        "positions": vertices.astype(np.float32).ravel().tolist(),
        "normals": np.asarray(mesh.vertex_normals).astype(np.float32).ravel().tolist(),
        "colors": rgb.astype(np.float32).ravel().tolist(),
        "indices": np.asarray(mesh.triangles).ravel().tolist(),
        "transform": {"center": center.tolist(), "scale": scale},
        "stats": {
            "sourceSplats": source_count,
            "surfaceSamples": len(pc.points),
            "vertices": len(vertices),
            "triangles": len(mesh.triangles),
            "seconds": round(elapsed, 3),
            "depth": depth,
            "medianThinness": alignment,
            "normalSource": normal_source,
            "minimumOpacity": min_opacity,
            "meanSurfaceDistanceMm": round(float(distances[:, 0].mean() * 1000), 3),
            "method": "Poisson baseline from oriented splat centers; no surface alignment training",
            "limitation": (
                'Requires a cropped, surface-aligned face. Vertex colors retain '
                'DC appearance only. Not a guarantee of likeness or valid '
                'deformation topology.'
            ),
        },
    }
