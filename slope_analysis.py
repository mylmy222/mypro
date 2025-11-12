"""Executable template for slope structural analysis from point clouds.

This module operationalises the step-by-step workflow described in the user
brief for detecting structural planes, clustering them into joint sets, and
identifying potentially unstable wedges directly from a slope point cloud.

The implementation favours clarity and extensibility. Each stage of the
workflow is isolated into a function so that practitioners can tweak or swap
components (e.g. replace DBSCAN with a stereonet-specific clustering method,
inject pore-pressure data, or refine the limit-equilibrium calculations).

The code is intended as a *ready-to-run* starting point: given a point-cloud
file it will produce CSV inventories of planes and wedges together with an
annotated PLY for visual review. Several numerical shortcuts are used to keep
the template compact; comments highlight where more rigorous treatment can be
added if required.

The module implements the workflow outlined in the user specification:

1.  Clean the raw point cloud and estimate per-point normals.
2.  Extract planar structural patches via iterative RANSAC and region growing.
3.  Cluster plane poles into joint sets and derive dip/strike statistics.
4.  Combine plane pairs to flag kinematically admissible wedge candidates,
    optionally computing a simplified factor of safety.
5.  Re-colour the point cloud to highlight risky areas and export tabular
    summaries for further engineering review.

Open3D provides the geometric primitives, while NumPy handles linear algebra.
The template is deliberately verbose with logging statements so practitioners
can monitor progress and tune parameters on their own datasets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Open3D is required to run this script. Install it with `pip install open3d`."
    ) from exc

try:  # Optional dependency for clustering
    from sklearn.cluster import DBSCAN

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency guard
    _HAS_SKLEARN = False


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers


@dataclass
class PlanePatch:
    """Structural plane extracted from the point cloud."""

    id: int
    indices: np.ndarray
    normal: np.ndarray
    offset: float
    area: float
    dip: float
    strike: float
    group_id: Optional[int] = None

    def centroid(self, cloud: o3d.geometry.PointCloud) -> np.ndarray:
        pts = np.asarray(cloud.points)[self.indices]
        return np.mean(pts, axis=0)

    def to_point_cloud(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return cloud.select_by_index(self.indices.tolist())


@dataclass
class WedgeCandidate:
    """Pairwise wedge defined by two structural planes."""

    plane_a: int
    plane_b: int
    line_direction: np.ndarray
    dip: float
    strike: float
    exposed: bool
    volume: float
    factor_of_safety: Optional[float]
    risk_level: str


@dataclass
class AnalysisParams:
    """Tunable parameters controlling the pipeline behaviour."""

    voxel_size: float = 0.01
    outlier_nb_points: int = 30
    outlier_radius: float = 0.05
    normal_k_min: int = 30
    normal_k_max: int = 80
    ransac_threshold: float = 0.02
    min_plane_points: int = 400
    region_angle_threshold_deg: float = 8.0
    region_distance: float = 0.03
    alpha_radius: float = 0.05
    friction_angle_deg: float = 32.0
    do_limit_equilibrium: bool = False
    unit_weight: float = 26.0  # kN/m^3
    cohesion: float = 0.0
    water_unit_weight: float = 9.81
    pore_pressure_head: float = 0.0
    slope_hint: Optional[Tuple[float, float]] = None  # (dip, dip-direction) in degrees
    min_wedge_volume: float = 0.01


# ---------------------------------------------------------------------------
# Helper utilities


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-length vector")
    return vec / norm


def _upper_hemisphere(normal: np.ndarray) -> np.ndarray:
    return normal if normal[2] >= 0 else -normal


def _dip_and_strike(normal: np.ndarray) -> Tuple[float, float]:
    n = _upper_hemisphere(_normalise(normal))
    dip = math.degrees(math.acos(np.clip(n[2], -1.0, 1.0)))
    strike = math.degrees(math.atan2(n[0], n[1])) % 360.0
    return dip, strike


def _plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    demeaned = points - centroid
    _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
    normal = vh[-1]
    d = -float(np.dot(normal, centroid))
    return _normalise(normal), d


def _ensure_directory(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_n = _normalise(v1)
    v2_n = _normalise(v2)
    return math.degrees(math.acos(np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Core processing blocks


def denoise_and_downsample(pc: o3d.geometry.PointCloud, params: AnalysisParams) -> o3d.geometry.PointCloud:
    LOGGER.info("Applying statistical outlier removal")
    filtered, _ = pc.remove_statistical_outlier(nb_neighbors=params.outlier_nb_points, std_ratio=1.0)
    LOGGER.info("Applying radius outlier removal")
    filtered, _ = filtered.remove_radius_outlier(nb_points=max(5, params.outlier_nb_points // 2), radius=params.outlier_radius)
    if params.voxel_size > 0:
        LOGGER.info("Voxel downsampling to %.3f m", params.voxel_size)
        filtered = filtered.voxel_down_sample(params.voxel_size)
    return filtered


def estimate_normals(pc: o3d.geometry.PointCloud, params: AnalysisParams) -> None:
    LOGGER.info("Estimating normals with adaptive neighbourhood sizes")
    tree = o3d.geometry.KDTreeFlann(pc)
    pts = np.asarray(pc.points)
    normals = np.zeros_like(pts)
    for i, point in enumerate(pts):
        for k in range(params.normal_k_min, params.normal_k_max + 1, 5):
            _, idx, _ = tree.search_knn_vector_3d(point, k)
            if len(idx) < 3:
                continue
            neighbourhood = pts[idx, :]
            normal, _ = _plane_from_points(neighbourhood)
            normals[i] = normal
            break
    pc.normals = o3d.utility.Vector3dVector(normals)


def region_grow_plane(
    cloud: o3d.geometry.PointCloud,
    seed_indices: Iterable[int],
    params: AnalysisParams,
) -> np.ndarray:
    normals = np.asarray(cloud.normals)
    points = np.asarray(cloud.points)
    kd_tree = o3d.geometry.KDTreeFlann(cloud)
    threshold_rad = math.radians(params.region_angle_threshold_deg)
    region: set[int] = set(seed_indices)
    queue = list(seed_indices)
    while queue:
        current = queue.pop()
        [_, neighbours, _] = kd_tree.search_radius_vector_3d(points[current], params.region_distance)
        for nb in neighbours:
            if nb in region:
                continue
            angle = math.acos(np.clip(np.dot(normals[current], normals[nb]), -1.0, 1.0))
            if angle > threshold_rad:
                continue
            region.add(nb)
            queue.append(nb)
    return np.array(sorted(region), dtype=int)


def extract_planes(cloud: o3d.geometry.PointCloud, params: AnalysisParams) -> List[PlanePatch]:
    remaining: set[int] = set(range(len(cloud.points)))
    patches: List[PlanePatch] = []
    plane_id = 0
    while len(remaining) >= params.min_plane_points:
        current_indices = np.array(sorted(remaining), dtype=int)
        working_cloud = cloud.select_by_index(current_indices.tolist())
        plane_model, inliers = working_cloud.segment_plane(
            distance_threshold=params.ransac_threshold,
            ransac_n=3,
            num_iterations=500,
        )
        if len(inliers) < params.min_plane_points:
            LOGGER.info("Stopping RANSAC: best plane has only %d inliers", len(inliers))
            break
        inlier_global = current_indices[np.array(inliers, dtype=int)]
        region_indices = region_grow_plane(cloud, inlier_global, params)
        points = np.asarray(cloud.points)[region_indices]
        normal, offset = _plane_from_points(points)
        dip, strike = _dip_and_strike(normal)
        area = estimate_patch_area(points, params)
        patches.append(
            PlanePatch(
                id=plane_id,
                indices=region_indices,
                normal=normal,
                offset=offset,
                area=area,
                dip=dip,
                strike=strike,
            )
        )
        plane_id += 1
        for idx in region_indices:
            remaining.discard(int(idx))
    LOGGER.info("Detected %d structural planes", len(patches))
    return patches


def estimate_patch_area(points: np.ndarray, params: AnalysisParams) -> float:
    patch_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            patch_cloud, params.alpha_radius
        )
        return float(mesh.get_surface_area())
    except RuntimeError:
        return float(len(points)) * (params.voxel_size**2)


def cluster_planes(patches: Sequence[PlanePatch]) -> Dict[int, List[int]]:
    if not patches:
        return {}
    poles = np.array([_upper_hemisphere(p.normal) for p in patches])
    if _HAS_SKLEARN:
        clustering = DBSCAN(eps=0.12, min_samples=2).fit(poles)
        labels = clustering.labels_
    else:  # pragma: no cover - fallback heuristic
        LOGGER.warning("scikit-learn not installed; all planes assigned to group 0")
        labels = np.zeros(len(patches), dtype=int)
    groups: Dict[int, List[int]] = {}
    for patch, label in zip(patches, labels):
        patch.group_id = int(label)
        groups.setdefault(int(label), []).append(patch.id)
    return groups


def detect_slope_plane(patches: Sequence[PlanePatch], params: AnalysisParams) -> Optional[PlanePatch]:
    if not patches:
        return None
    if params.slope_hint is None:
        return max(patches, key=lambda p: p.area)
    dip, dip_dir = params.slope_hint
    dip_rad = math.radians(dip)
    dip_dir_rad = math.radians(dip_dir)
    normal = np.array(
        [
            math.sin(dip_rad) * math.sin(dip_dir_rad),
            math.sin(dip_rad) * math.cos(dip_dir_rad),
            math.cos(dip_rad),
        ]
    )
    return min(patches, key=lambda p: _angle_between(p.normal, normal))


def generate_wedges(
    patches: Sequence[PlanePatch],
    slope_plane: Optional[PlanePatch],
    params: AnalysisParams,
) -> List[WedgeCandidate]:
    if slope_plane is None:
        return []
    wedges: List[WedgeCandidate] = []
    slope_normal = slope_plane.normal
    for i, plane_a in enumerate(patches):
        for plane_b in patches[i + 1 :]:
            line_dir = np.cross(plane_a.normal, plane_b.normal)
            if np.linalg.norm(line_dir) < 1e-6:
                continue
            line_dir = _normalise(line_dir)
            dip = math.degrees(math.asin(abs(line_dir[2])))
            strike = math.degrees(math.atan2(line_dir[0], line_dir[1])) % 360.0
            exposed = is_daylighting(line_dir, slope_normal)
            volume = estimate_wedge_volume(plane_a, plane_b, slope_plane, params)
            if volume < params.min_wedge_volume:
                continue
            fs = None
            risk = "stable"
            if exposed and dip > params.friction_angle_deg:
                risk = "potential"
            if params.do_limit_equilibrium:
                fs = compute_factor_of_safety(
                    volume=volume,
                    unit_weight=params.unit_weight,
                    friction_angle_deg=params.friction_angle_deg,
                    cohesion=params.cohesion,
                    slope_normal=slope_normal,
                    line_dir=line_dir,
                    pore_pressure_head=params.pore_pressure_head,
                    water_unit_weight=params.water_unit_weight,
                )
                risk = classify_risk(fs)
            wedges.append(
                WedgeCandidate(
                    plane_a=plane_a.id,
                    plane_b=plane_b.id,
                    line_direction=line_dir,
                    dip=dip,
                    strike=strike,
                    exposed=exposed,
                    volume=volume,
                    factor_of_safety=fs,
                    risk_level=risk,
                )
            )
    return wedges


def is_daylighting(line_dir: np.ndarray, slope_normal: np.ndarray, tolerance_deg: float = 20.0) -> bool:
    slope_dip_direction = np.cross(np.array([0.0, 0.0, 1.0]), slope_normal)
    if np.linalg.norm(slope_dip_direction) < 1e-8:
        slope_dip_direction = _normalise(np.array([-slope_normal[1], slope_normal[0], 0.0]))
    else:
        slope_dip_direction = _normalise(slope_dip_direction)
    return _angle_between(line_dir, slope_dip_direction) < tolerance_deg


def estimate_wedge_volume(
    plane_a: PlanePatch,
    plane_b: PlanePatch,
    slope_plane: PlanePatch,
    params: AnalysisParams,
) -> float:
    characteristic_length = params.voxel_size if params.voxel_size > 0 else 0.05
    representative_area = min(plane_a.area, plane_b.area)
    return float(representative_area * characteristic_length)


def compute_factor_of_safety(
    volume: float,
    unit_weight: float,
    friction_angle_deg: float,
    cohesion: float,
    slope_normal: np.ndarray,
    line_dir: np.ndarray,
    pore_pressure_head: float,
    water_unit_weight: float,
) -> float:
    weight = volume * unit_weight
    slip_direction = _normalise(np.cross(line_dir, slope_normal))
    normal_force = weight * abs(np.dot(slope_normal, np.array([0.0, 0.0, 1.0])))
    pore = pore_pressure_head * water_unit_weight
    shear = weight * abs(np.dot(line_dir, slip_direction))
    friction = (normal_force - pore) * math.tan(math.radians(friction_angle_deg))
    resistance = cohesion + friction
    return resistance / shear if shear else float("inf")


def classify_risk(fs: Optional[float]) -> str:
    if fs is None:
        return "potential"
    if fs < 1.0:
        return "high"
    if fs < 1.2:
        return "medium"
    return "low"


def annotate_point_cloud(
    cloud: o3d.geometry.PointCloud,
    patches: Sequence[PlanePatch],
    wedges: Sequence[WedgeCandidate],
) -> o3d.geometry.PointCloud:
    annotated = o3d.geometry.PointCloud(cloud)
    colors = np.zeros((len(cloud.points), 3)) + 0.6
    rng = np.random.default_rng(seed=42)
    for patch in patches:
        color = rng.random(3)
        colors[patch.indices] = color
    risk_palette = {
        "high": np.array([1.0, 0.0, 0.0]),
        "medium": np.array([1.0, 0.6, 0.0]),
        "low": np.array([0.2, 0.8, 0.2]),
        "potential": np.array([0.9, 0.9, 0.0]),
        "stable": np.array([0.7, 0.7, 0.7]),
    }
    for wedge in wedges:
        color = risk_palette.get(wedge.risk_level, np.array([0.7, 0.7, 0.7]))
        involved = np.concatenate([
            patches[wedge.plane_a].indices,
            patches[wedge.plane_b].indices,
        ])
        colors[involved] = color
    annotated.colors = o3d.utility.Vector3dVector(colors)
    return annotated


def export_results(
    out_dir: pathlib.Path,
    cloud: o3d.geometry.PointCloud,
    patches: Sequence[PlanePatch],
    wedges: Sequence[WedgeCandidate],
    annotated_cloud: o3d.geometry.PointCloud,
) -> None:
    _ensure_directory(out_dir)
    with open(out_dir / "planes.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "group", "dip", "strike", "area", "cx", "cy", "cz"])
        for patch in patches:
            cx, cy, cz = patch.centroid(cloud)
            writer.writerow([
                patch.id,
                patch.group_id,
                f"{patch.dip:.2f}",
                f"{patch.strike:.2f}",
                f"{patch.area:.4f}",
                f"{cx:.3f}",
                f"{cy:.3f}",
                f"{cz:.3f}",
            ])
    with open(out_dir / "wedges.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "plane_a",
            "plane_b",
            "dip",
            "strike",
            "exposed",
            "volume",
            "fs",
            "risk",
        ])
        for wedge in wedges:
            writer.writerow([
                wedge.plane_a,
                wedge.plane_b,
                f"{wedge.dip:.2f}",
                f"{wedge.strike:.2f}",
                wedge.exposed,
                f"{wedge.volume:.4f}",
                "" if wedge.factor_of_safety is None else f"{wedge.factor_of_safety:.3f}",
                wedge.risk_level,
            ])
    o3d.io.write_point_cloud(str(out_dir / "annotated.ply"), annotated_cloud)


# ---------------------------------------------------------------------------
# Command-line entry point


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze slope point clouds and identify wedges")
    parser.add_argument("--input", required=True, help="Input point-cloud file (PLY/PCD/etc.)")
    parser.add_argument("--output", default="output", help="Directory for generated artefacts")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel down-sampling size (m)")
    parser.add_argument("--friction-angle", type=float, default=32.0, help="Friction angle for kinematic check (deg)")
    parser.add_argument("--limit-equilibrium", action="store_true", help="Enable simplified limit-equilibrium FS")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/…)")
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> Tuple[o3d.geometry.PointCloud, List[PlanePatch], List[WedgeCandidate], o3d.geometry.PointCloud]:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    params = AnalysisParams(voxel_size=args.voxel_size, friction_angle_deg=args.friction_angle, do_limit_equilibrium=args.limit_equilibrium)
    LOGGER.info("Loading point cloud %s", args.input)
    cloud = o3d.io.read_point_cloud(args.input)
    LOGGER.info("Loaded %d points", len(cloud.points))
    cleaned = denoise_and_downsample(cloud, params)
    LOGGER.info("After filtering/downsampling: %d points", len(cleaned.points))
    estimate_normals(cleaned, params)
    patches = extract_planes(cleaned, params)
    groups = cluster_planes(patches)
    LOGGER.info("Detected %d structural groups", len(groups))
    slope_plane = detect_slope_plane(patches, params)
    LOGGER.info("Selected slope plane id: %s", None if slope_plane is None else slope_plane.id)
    wedges = generate_wedges(patches, slope_plane, params)
    LOGGER.info("Generated %d wedge candidates", len(wedges))
    annotated = annotate_point_cloud(cleaned, patches, wedges)
    return cleaned, patches, wedges, annotated


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cleaned, patches, wedges, annotated = run_pipeline(args)
    export_results(pathlib.Path(args.output), cleaned, patches, wedges, annotated)
    LOGGER.info("Analysis finished. Artefacts saved under %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
