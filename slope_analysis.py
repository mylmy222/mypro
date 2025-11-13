"""Comprehensive slope structural analysis pipeline for point clouds.

This module follows the workflow outlined in the user specification: the user
selects a point-cloud file through a dialog, the cloud is cleaned and analysed
for structural planes, those planes are grouped into joint sets with spacing
statistics, and wedge combinations are evaluated after the user chooses which
sets to retain.  The implementation deliberately avoids any learned components;
all detection and grouping relies on geometric heuristics such as RANSAC,
region growing, and spherical DBSCAN.

Key capabilities implemented here:

* Outlier removal + voxel down-sampling tuned from the local point spacing.
* Iterative plane extraction (RANSAC seeds refined by region growing).
* Automatic slope-plane recognition and TIN construction.
* Trace-line computation via plane∩slope-TIN, including start/end vertices.
* Joint-set clustering, spacing estimation (projected along group normals), and
  derivation of linear structures from trace lines.
* Interactive dialog (Tkinter listbox) that allows the analyst to choose which
  joint sets progress to wedge evaluation.
* Kinematic wedge screening (daylighting + plunge checks) and an optional limit
  equilibrium back-of-the-envelope FoS estimate.
* CSV/JSON/PLY exports so the workflow can be audited downstream.

The code is intentionally verbose with logging statements so the operator can
monitor each stage and tune parameters for new datasets.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Geometry backend
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Open3D is required for this script. Install it with `pip install open3d`."
    ) from exc

try:  # Optional clustering backend
    from sklearn.cluster import DBSCAN

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency guard
    _HAS_SKLEARN = False

try:
    from scipy.spatial import Delaunay
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "SciPy is required for triangulation/volume estimates. Install it with `pip install scipy`."
    ) from exc

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses


@dataclass
class TraceLine:
    """Intersection trace between a structural plane and the slope surface."""

    start: np.ndarray
    end: np.ndarray

    @property
    def vector(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.vector))


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
    trace: Optional[TraceLine] = None

    def centroid(self, cloud: o3d.geometry.PointCloud) -> np.ndarray:
        pts = np.asarray(cloud.points)[self.indices]
        return np.mean(pts, axis=0)

    def plane_equation(self) -> Tuple[np.ndarray, float]:
        return self.normal, self.offset


@dataclass
class LinearStructure:
    """Lineation derived from trace lines or explicit linear clusters."""

    id: int
    start: np.ndarray
    end: np.ndarray
    length: float
    azimuth: float
    dip: float
    dip_direction: float
    group_id: Optional[int]


@dataclass
class JointGroup:
    id: int
    plane_ids: List[int]
    normal: np.ndarray
    dip: float
    strike: float
    spacing: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "id": self.id,
            "dip": self.dip,
            "strike": self.strike,
            "spacing": None if self.spacing is None else float(self.spacing),
            "count": len(self.plane_ids),
        }


@dataclass
class WedgeCandidate:
    """Potential wedge formed by two structural planes."""

    plane_a: int
    plane_b: int
    group_a: Optional[int]
    group_b: Optional[int]
    line_direction: np.ndarray
    dip: float
    strike: float
    exposed: bool
    volume: float
    factor_of_safety: Optional[float]
    risk_level: str

    def to_row(self) -> List[str]:
        return [
            str(self.plane_a),
            str(self.plane_b),
            f"{self.dip:.2f}",
            f"{self.strike:.2f}",
            str(self.exposed),
            f"{self.volume:.4f}",
            "" if self.factor_of_safety is None else f"{self.factor_of_safety:.3f}",
            self.risk_level,
        ]


@dataclass
class AnalysisParams:
    """Parameters steering the pipeline behaviour."""

    voxel_size: float = 0.05
    outlier_nb_points: int = 24
    outlier_radius: float = 0.12
    normal_radius: float = 0.25
    normal_max_nn: int = 90
    ransac_threshold: float = 0.08
    min_plane_points: int = 600
    region_angle_threshold_deg: float = 10.0
    region_distance: float = 0.25
    alpha_radius: float = 0.35
    min_trace_length: float = 0.4
    line_dip_deg: float = 70.0
    friction_angle_deg: float = 32.0
    cohesion: float = 0.0
    unit_weight: float = 26.0  # kN/m^3
    water_unit_weight: float = 9.81
    pore_pressure_head: float = 0.0
    slope_hint: Optional[Tuple[float, float]] = None
    min_wedge_volume: float = 0.05
    wedge_nominal_thickness: float = 0.5
    daylight_tolerance_deg: float = 20.0
    plunging_min_deg: float = 10.0
    do_limit_equilibrium: bool = False


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
    strike = (math.degrees(math.atan2(n[0], n[1])) + 360.0) % 360.0
    return dip, strike


def _plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    demeaned = points - centroid
    _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
    normal = _normalise(vh[-1])
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip(np.dot(_normalise(v1), _normalise(v2)), -1.0, 1.0)))


def _ensure_directory(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_point_cloud_file(path: Optional[str]) -> pathlib.Path:
    if path:
        return pathlib.Path(path)
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="选择点云文件 (PLY/PCD)",
            filetypes=[("Point clouds", "*.ply *.pcd *.xyz"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception as exc:  # pragma: no cover - GUI guard
        raise SystemExit("Tkinter file dialog failed; please provide --input path.") from exc
    if not filename:
        raise SystemExit("No point-cloud file selected.")
    return pathlib.Path(filename)


def average_point_spacing(cloud: o3d.geometry.PointCloud) -> float:
    tree = o3d.geometry.KDTreeFlann(cloud)
    pts = np.asarray(cloud.points)
    if len(pts) < 2:
        return 0.1
    samples = np.linspace(0, len(pts) - 1, min(256, len(pts)), dtype=int)
    dists: List[float] = []
    for idx in samples:
        _, idxs, d = tree.search_knn_vector_3d(pts[idx], 2)
        if len(idxs) >= 2:
            dists.append(math.sqrt(d[1]))
    return float(np.median(dists)) if dists else 0.1


# ---------------------------------------------------------------------------
# Plane extraction


def denoise_and_downsample(cloud: o3d.geometry.PointCloud, params: AnalysisParams) -> o3d.geometry.PointCloud:
    filtered = cloud
    if len(cloud.points) == 0:
        return filtered
    LOGGER.info("Statistical outlier removal")
    filtered, _ = filtered.remove_statistical_outlier(nb_neighbors=params.outlier_nb_points, std_ratio=1.0)
    LOGGER.info("Radius outlier removal")
    filtered, _ = filtered.remove_radius_outlier(nb_points=max(6, params.outlier_nb_points // 2), radius=params.outlier_radius)
    if params.voxel_size > 0:
        LOGGER.info("Voxel down-sampling at %.3f m", params.voxel_size)
        filtered = filtered.voxel_down_sample(params.voxel_size)
    return filtered


def estimate_normals(cloud: o3d.geometry.PointCloud, params: AnalysisParams) -> None:
    radius = max(params.normal_radius, params.voxel_size * 4.0)
    LOGGER.info("Estimating normals (radius %.3f m, max %d nn)", radius, params.normal_max_nn)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=params.normal_max_nn))
    cloud.orient_normals_consistent_tangent_plane(50)


def region_grow_plane(
    cloud: o3d.geometry.PointCloud,
    seed_indices: Iterable[int],
    params: AnalysisParams,
) -> np.ndarray:
    normals = np.asarray(cloud.normals)
    points = np.asarray(cloud.points)
    kd_tree = o3d.geometry.KDTreeFlann(cloud)
    region: set[int] = set(seed_indices)
    queue = list(seed_indices)
    threshold = math.radians(params.region_angle_threshold_deg)
    while queue:
        current = queue.pop()
        [_, neighbours, _] = kd_tree.search_radius_vector_3d(points[current], params.region_distance)
        for nb in neighbours:
            if nb in region:
                continue
            angle = math.acos(np.clip(np.dot(normals[current], normals[nb]), -1.0, 1.0))
            if angle > threshold:
                continue
            region.add(nb)
            queue.append(nb)
    return np.array(sorted(region), dtype=int)


def estimate_patch_area(points: np.ndarray, params: AnalysisParams) -> float:
    patch_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(patch_cloud, params.alpha_radius)
        return float(mesh.get_surface_area())
    except RuntimeError:
        return float(len(points)) * (params.voxel_size**2)


def extract_planes(cloud: o3d.geometry.PointCloud, params: AnalysisParams) -> List[PlanePatch]:
    remaining = set(range(len(cloud.points)))
    patches: List[PlanePatch] = []
    plane_id = 0
    while len(remaining) >= params.min_plane_points:
        current_indices = np.array(sorted(remaining), dtype=int)
        working_cloud = cloud.select_by_index(current_indices.tolist())
        plane_model, inliers = working_cloud.segment_plane(
            distance_threshold=params.ransac_threshold,
            ransac_n=3,
            num_iterations=800,
        )
        if len(inliers) < params.min_plane_points:
            LOGGER.info("Stopping RANSAC: best plane has %d inliers", len(inliers))
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
    LOGGER.info("Detected %d planar structures", len(patches))
    return patches


# ---------------------------------------------------------------------------
# Slope plane + traces


def detect_slope_plane(patches: Sequence[PlanePatch], params: AnalysisParams) -> Optional[PlanePatch]:
    if not patches:
        return None
    if params.slope_hint is None:
        slope = max(patches, key=lambda p: p.area)
        LOGGER.info("Slope plane selected by max area (id %d)", slope.id)
        return slope
    dip, dip_dir = params.slope_hint
    dip_rad = math.radians(dip)
    dip_dir_rad = math.radians(dip_dir)
    hint_normal = np.array(
        [
            math.sin(dip_rad) * math.sin(dip_dir_rad),
            math.sin(dip_rad) * math.cos(dip_dir_rad),
            math.cos(dip_rad),
        ]
    )
    slope = min(patches, key=lambda p: _angle_between(p.normal, hint_normal))
    LOGGER.info("Slope plane selected by hint (id %d)", slope.id)
    return slope


def build_slope_tin(slope_patch: PlanePatch, cloud: o3d.geometry.PointCloud) -> Optional[Dict[str, np.ndarray]]:
    if slope_patch is None:
        return None
    pts = np.asarray(cloud.points)[slope_patch.indices]
    if len(pts) < 3:
        LOGGER.warning("Slope plane has insufficient points for TIN")
        return None
    normal = slope_patch.normal
    ref = np.array([0.0, 0.0, 1.0])
    axis1 = np.cross(ref, normal)
    if np.linalg.norm(axis1) < 1e-6:
        axis1 = np.cross(np.array([1.0, 0.0, 0.0]), normal)
    axis1 = _normalise(axis1)
    axis2 = _normalise(np.cross(normal, axis1))
    uv = np.column_stack([pts @ axis1, pts @ axis2])
    try:
        delaunay = Delaunay(uv)
    except Exception:
        LOGGER.warning("Failed to build slope TIN (degenerate projection)")
        return None
    return {
        "vertices": pts,
        "simplices": delaunay.simplices,
        "axis1": axis1,
        "axis2": axis2,
        "normal": normal,
        "offset": slope_patch.offset,
    }


def _intersect_plane_with_triangle(plane_normal: np.ndarray, plane_offset: float, triangle: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    distances = triangle @ plane_normal + plane_offset
    segments: List[np.ndarray] = []
    for start, end, d_start, d_end in zip(
        triangle,
        np.roll(triangle, -1, axis=0),
        distances,
        np.roll(distances, -1),
    ):
        if d_start == 0 and d_end == 0:
            segments.extend([start, end])
        elif d_start == 0:
            segments.append(start)
        elif d_end == 0:
            segments.append(end)
        elif d_start * d_end < 0:
            t = d_start / (d_start - d_end)
            point = start + t * (end - start)
            segments.append(point)
    if len(segments) >= 2:
        return segments[0], segments[1]
    return None


def compute_trace_for_patch(patch: PlanePatch, slope_tin: Optional[Dict[str, np.ndarray]]) -> Optional[TraceLine]:
    if slope_tin is None:
        return None
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for simplex in slope_tin["simplices"]:
        tri = slope_tin["vertices"][simplex]
        result = _intersect_plane_with_triangle(patch.normal, patch.offset, tri)
        if result is None:
            continue
        segments.append(result)
    if not segments:
        return None
    # Assemble by chaining nearest endpoints.
    chain = list(segments[0])
    used = [False] * len(segments)
    used[0] = True
    tolerance = max(1e-3, 0.05)
    extended = True
    while extended:
        extended = False
        for idx, (a, b) in enumerate(segments):
            if used[idx]:
                continue
            if np.linalg.norm(chain[-1] - a) < tolerance:
                chain.append(b)
                used[idx] = True
                extended = True
            elif np.linalg.norm(chain[-1] - b) < tolerance:
                chain.append(a)
                used[idx] = True
                extended = True
    length = float(np.sum(np.linalg.norm(np.diff(np.asarray(chain), axis=0), axis=1)))
    if length < 1e-6:
        return None
    return TraceLine(start=np.asarray(chain[0]), end=np.asarray(chain[-1]))


def attach_traces(patches: Sequence[PlanePatch], slope_tin: Optional[Dict[str, np.ndarray]]) -> None:
    for patch in patches:
        patch.trace = compute_trace_for_patch(patch, slope_tin)


# ---------------------------------------------------------------------------
# Grouping + spacing


def cluster_planes(patches: Sequence[PlanePatch]) -> Dict[int, JointGroup]:
    if not patches:
        return {}
    poles = np.array([_upper_hemisphere(p.normal) for p in patches])
    if _HAS_SKLEARN:
        clustering = DBSCAN(eps=math.radians(12.0), min_samples=3, metric="euclidean").fit(poles)
        labels = clustering.labels_
    else:  # pragma: no cover - fallback
        LOGGER.warning("scikit-learn not installed; assigning all planes to group 0")
        labels = np.zeros(len(patches), dtype=int)
    groups: Dict[int, List[PlanePatch]] = {}
    for patch, label in zip(patches, labels):
        patch.group_id = int(label)
        groups.setdefault(int(label), []).append(patch)
    joint_groups: Dict[int, JointGroup] = {}
    for gid, members in groups.items():
        normals = np.array([m.normal for m in members])
        normal = _normalise(np.mean(normals, axis=0))
        dip, strike = _dip_and_strike(normal)
        joint_groups[gid] = JointGroup(
            id=gid,
            plane_ids=[m.id for m in members],
            normal=normal,
            dip=dip,
            strike=strike,
        )
    return joint_groups


def compute_group_spacing(group: JointGroup, patches: Sequence[PlanePatch], cloud: o3d.geometry.PointCloud) -> None:
    centroids = [patches[pid].centroid(cloud) for pid in group.plane_ids if patches[pid].trace is not None]
    if len(centroids) < 2:
        group.spacing = None
        return
    distances = sorted(float(np.dot(group.normal, c)) for c in centroids)
    diffs = np.diff(distances)
    group.spacing = float(np.median(diffs)) if len(diffs) else None


# ---------------------------------------------------------------------------
# Linear structures


def derive_linear_structures(patches: Sequence[PlanePatch], params: AnalysisParams) -> List[LinearStructure]:
    structures: List[LinearStructure] = []
    line_id = 0
    for patch in patches:
        if patch.trace is None:
            continue
        vector = patch.trace.vector
        length = np.linalg.norm(vector)
        if length < params.min_trace_length:
            continue
        azimuth = (math.degrees(math.atan2(vector[0], vector[1])) + 360.0) % 360.0
        dip_dir = (patch.strike + 90.0) % 360.0
        dip = patch.dip if patch.dip > 0 else params.line_dip_deg
        structures.append(
            LinearStructure(
                id=line_id,
                start=patch.trace.start,
                end=patch.trace.end,
                length=float(length),
                azimuth=azimuth,
                dip=dip,
                dip_direction=dip_dir,
                group_id=patch.group_id,
            )
        )
        line_id += 1
    return structures


# ---------------------------------------------------------------------------
# Interactive selection


def select_groups_dialog(groups: Dict[int, JointGroup]) -> List[int]:
    if not groups:
        return []
    try:
        import tkinter as tk
    except Exception:
        LOGGER.warning("Tkinter unavailable; defaulting to all groups")
        return list(groups.keys())
    try:
        root = tk.Tk()
    except tk.TclError:
        LOGGER.warning("No display for Tkinter; defaulting to all groups")
        return list(groups.keys())
    root.title("选择参与锲体分析的结构面组")
    tk.Label(root, text="按住 Ctrl/Shift 可多选；确定即开始计算").pack(padx=10, pady=6)
    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50)
    ids = sorted(groups.keys())
    for idx, gid in enumerate(ids):
        info = groups[gid]
        text = f"组 {gid}: dip {info.dip:.1f}° / strike {info.strike:.1f}° / {len(info.plane_ids)} 面"
        listbox.insert(idx, text)
        listbox.selection_set(idx)
    listbox.pack(padx=10, pady=10)
    selection: List[int] = []

    def on_ok() -> None:
        selected = listbox.curselection()
        selection.extend(ids[i] for i in selected)
        root.destroy()

    tk.Button(root, text="确定", command=on_ok).pack(pady=10)
    root.mainloop()
    if not selection:
        LOGGER.warning("未选择任何组，默认使用全部")
        return ids
    return selection


# ---------------------------------------------------------------------------
# Wedge analysis


def is_daylighting(line_dir: np.ndarray, slope_normal: np.ndarray, tolerance_deg: float) -> bool:
    slope_dip_direction = np.cross(np.array([0.0, 0.0, 1.0]), slope_normal)
    if np.linalg.norm(slope_dip_direction) < 1e-8:
        slope_dip_direction = _normalise(np.array([-slope_normal[1], slope_normal[0], 0.0]))
    else:
        slope_dip_direction = _normalise(slope_dip_direction)
    return _angle_between(line_dir, slope_dip_direction) < tolerance_deg


def estimate_wedge_volume(plane_a: PlanePatch, plane_b: PlanePatch, params: AnalysisParams) -> float:
    representative_area = min(plane_a.area, plane_b.area)
    return float(representative_area * params.wedge_nominal_thickness)


def compute_factor_of_safety(
    volume: float,
    params: AnalysisParams,
    slope_normal: np.ndarray,
    line_dir: np.ndarray,
) -> float:
    weight = volume * params.unit_weight
    slip_direction = _normalise(np.cross(line_dir, slope_normal))
    normal_force = weight * abs(np.dot(slope_normal, np.array([0.0, 0.0, 1.0])))
    pore = params.pore_pressure_head * params.water_unit_weight
    shear = weight * abs(np.dot(line_dir, slip_direction))
    friction = (normal_force - pore) * math.tan(math.radians(params.friction_angle_deg))
    resistance = params.cohesion + friction
    return resistance / shear if shear else float("inf")


def classify_risk(fs: Optional[float], exposed: bool) -> str:
    if not exposed:
        return "stable"
    if fs is None:
        return "potential"
    if fs < 1.0:
        return "high"
    if fs < 1.2:
        return "medium"
    return "low"


def generate_wedges(
    patches: Sequence[PlanePatch],
    groups: Dict[int, JointGroup],
    slope_plane: Optional[PlanePatch],
    params: AnalysisParams,
    selected_groups: Sequence[int],
) -> List[WedgeCandidate]:
    if slope_plane is None:
        return []
    group_lookup = {gid: groups[gid] for gid in selected_groups if gid in groups}
    slope_normal = slope_plane.normal
    wedges: List[WedgeCandidate] = []
    for gid_a, gid_b in itertools.combinations(group_lookup.keys(), 2):
        group_a, group_b = group_lookup[gid_a], group_lookup[gid_b]
        intersection_dir = np.cross(group_a.normal, group_b.normal)
        if np.linalg.norm(intersection_dir) < 1e-6:
            continue
        intersection_dir = _normalise(intersection_dir)
        plunge = math.degrees(math.asin(abs(intersection_dir[2])))
        if plunge < params.plunging_min_deg:
            continue
        if not is_daylighting(intersection_dir, slope_normal, params.daylight_tolerance_deg):
            continue
        for pid_a in group_a.plane_ids:
            for pid_b in group_b.plane_ids:
                plane_a = patches[pid_a]
                plane_b = patches[pid_b]
                line_dir = np.cross(plane_a.normal, plane_b.normal)
                if np.linalg.norm(line_dir) < 1e-6:
                    continue
                line_dir = _normalise(line_dir)
                dip = math.degrees(math.asin(abs(line_dir[2])))
                strike = (math.degrees(math.atan2(line_dir[0], line_dir[1])) + 360.0) % 360.0
                exposed = is_daylighting(line_dir, slope_normal, params.daylight_tolerance_deg)
                volume = estimate_wedge_volume(plane_a, plane_b, params)
                if volume < params.min_wedge_volume:
                    continue
                fs = None
                if params.do_limit_equilibrium:
                    fs = compute_factor_of_safety(volume, params, slope_normal, line_dir)
                risk = classify_risk(fs, exposed)
                wedges.append(
                    WedgeCandidate(
                        plane_a=plane_a.id,
                        plane_b=plane_b.id,
                        group_a=gid_a,
                        group_b=gid_b,
                        line_direction=line_dir,
                        dip=dip,
                        strike=strike,
                        exposed=exposed,
                        volume=volume,
                        factor_of_safety=fs,
                        risk_level=risk,
                    )
                )
    LOGGER.info("Generated %d wedge candidates", len(wedges))
    return wedges


# ---------------------------------------------------------------------------
# Export & visualisation


def annotate_point_cloud(
    cloud: o3d.geometry.PointCloud,
    patches: Sequence[PlanePatch],
    wedges: Sequence[WedgeCandidate],
) -> o3d.geometry.PointCloud:
    annotated = o3d.geometry.PointCloud(cloud)
    colors = np.zeros((len(cloud.points), 3)) + 0.6
    rng = np.random.default_rng(42)
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
        involved = np.concatenate([patches[wedge.plane_a].indices, patches[wedge.plane_b].indices])
        colors[involved] = color
    annotated.colors = o3d.utility.Vector3dVector(colors)
    return annotated


def export_results(
    out_dir: pathlib.Path,
    cloud: o3d.geometry.PointCloud,
    patches: Sequence[PlanePatch],
    groups: Dict[int, JointGroup],
    linear_structures: Sequence[LinearStructure],
    wedges: Sequence[WedgeCandidate],
    annotated_cloud: o3d.geometry.PointCloud,
) -> None:
    _ensure_directory(out_dir)
    with open(out_dir / "planes.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "id",
                "group",
                "dip",
                "strike",
                "area",
                "trace_length",
                "trace_sx",
                "trace_sy",
                "trace_sz",
                "trace_ex",
                "trace_ey",
                "trace_ez",
            ]
        )
        for patch in patches:
            trace = patch.trace
            sx = sy = sz = ex = ey = ez = ""
            tlen = ""
            if trace is not None:
                sx, sy, sz = (f"{float(v):.3f}" for v in trace.start)
                ex, ey, ez = (f"{float(v):.3f}" for v in trace.end)
                tlen = f"{trace.length:.3f}"
            writer.writerow(
                [
                    patch.id,
                    patch.group_id,
                    f"{patch.dip:.2f}",
                    f"{patch.strike:.2f}",
                    f"{patch.area:.3f}",
                    tlen,
                    sx,
                    sy,
                    sz,
                    ex,
                    ey,
                    ez,
                ]
            )
    with open(out_dir / "groups.json", "w", encoding="utf-8") as fp:
        json.dump({gid: group.to_dict() for gid, group in groups.items()}, fp, indent=2)
    with open(out_dir / "linear_structures.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "group", "length", "azimuth", "dip", "dip_direction"])
        for line in linear_structures:
            writer.writerow(
                [
                    line.id,
                    line.group_id,
                    f"{line.length:.3f}",
                    f"{line.azimuth:.2f}",
                    f"{line.dip:.2f}",
                    f"{line.dip_direction:.2f}",
                ]
            )
    with open(out_dir / "wedges.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["plane_a", "plane_b", "dip", "strike", "exposed", "volume", "fs", "risk"])
        for wedge in wedges:
            writer.writerow(wedge.to_row())
    o3d.io.write_point_cloud(str(out_dir / "annotated.ply"), annotated_cloud)


# ---------------------------------------------------------------------------
# CLI + pipeline


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-cloud slope structural analysis")
    parser.add_argument("--input", help="Input point-cloud file (optional; dialog if omitted)")
    parser.add_argument("--output", default="output", help="Directory for generated artefacts")
    parser.add_argument("--limit-equilibrium", action="store_true", help="Enable FoS calculation")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    parser.add_argument("--friction-angle", type=float, default=32.0, help="Friction angle for kinematic check (deg)")
    parser.add_argument("--line-dip", type=float, default=70.0, help="Fallback dip for lineations without planar parents")
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> Tuple[
    o3d.geometry.PointCloud,
    List[PlanePatch],
    Dict[int, JointGroup],
    List[LinearStructure],
    List[WedgeCandidate],
    o3d.geometry.PointCloud,
]:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    point_cloud_path = choose_point_cloud_file(args.input)
    LOGGER.info("Loading point cloud %s", point_cloud_path)
    cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    if cloud.is_empty():
        raise SystemExit("Point cloud is empty")
    params = AnalysisParams(friction_angle_deg=args.friction_angle, line_dip_deg=args.line_dip)
    spacing = average_point_spacing(cloud)
    params.voxel_size = max(params.voxel_size, spacing * 1.5)
    params.ransac_threshold = max(params.ransac_threshold, spacing * 2.0)
    params.region_distance = max(params.region_distance, spacing * 4.0)
    cleaned = denoise_and_downsample(cloud, params)
    LOGGER.info("After filtering/downsampling: %d points", len(cleaned.points))
    estimate_normals(cleaned, params)
    patches = extract_planes(cleaned, params)
    slope_plane = detect_slope_plane(patches, params)
    slope_tin = build_slope_tin(slope_plane, cleaned)
    attach_traces(patches, slope_tin)
    groups = cluster_planes(patches)
    for group in groups.values():
        compute_group_spacing(group, patches, cleaned)
    LOGGER.info("Detected %d joint sets", len(groups))
    linear_structures = derive_linear_structures(patches, params)
    selected_ids = select_groups_dialog(groups)
    params.do_limit_equilibrium = args.limit_equilibrium
    wedges = generate_wedges(patches, groups, slope_plane, params, selected_ids)
    annotated = annotate_point_cloud(cleaned, patches, wedges)
    return cleaned, patches, groups, linear_structures, wedges, annotated


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    (
        cleaned,
        patches,
        groups,
        linear_structures,
        wedges,
        annotated,
    ) = run_pipeline(args)
    export_results(pathlib.Path(args.output), cleaned, patches, groups, linear_structures, wedges, annotated)
    LOGGER.info("Analysis finished. Artefacts saved under %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
