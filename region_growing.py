#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Region Growing Segmentation for CloudCompare Binary PCD + Joint Sets Clustering
-------------------------------------------------------------------------------

要点：
- 输入：CloudCompare 二进制 PCD（含法向）；曲率在颜色 R 通道 [0,1]
- 区域生长：低曲率作种子；距离+法向角度+曲率三重判定；多核 KDTree；tqdm 进度；AABB 交互框选
- 区域法向：对每个区域点做 PCA/SVD 平面拟合得到法向（更稳健）
- 结构面组分组（方向角版本）：
    先将区域法向统一到 **y>0 半球**，然后用**真实方向角（不取绝对值，0~180°）**构造距离矩阵做 DBSCAN
- 单区大面片规则：若区域点数 ≥ 总点数的 `major_singleton_percent%`（默认 1%），即使为噪声也强制成独立组
- ⭐ 新增：分组步骤**不再将任何已成区域划为噪声**；DBSCAN 产生的 -1 会被分配到最近簇；若无簇则各自成组
- 导出：整体与各区域、labels、summary、config；每个结构面组与未分组点云；可选等面积极投影图
- 可视化：按区域上色、按结构面组上色

依赖：
    pip install open3d scipy numpy tqdm
    pip install scikit-learn matplotlib
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import datetime as _dt
import json
import math
import os
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

# --------- optional file dialog fallback (when pcd arg omitted) ----------
def askopenfilename_pcd() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        fp = filedialog.askopenfilename(
            title="选择 CloudCompare 导出的 .pcd 文件",
            filetypes=[("PCD files", "*.pcd"), ("All files", "*.*")]
        )
        root.destroy()
        return fp if fp else None
    except Exception:
        return None
# ------------------------------------------------------------------------

try:
    import open3d as o3d
except Exception as e:
    raise RuntimeError("Open3D is required. Please `pip install open3d`") from e

try:
    from scipy.spatial import cKDTree, ConvexHull
except Exception as e:
    raise RuntimeError("SciPy is required. Please `pip install scipy`") from e

# 结构面组聚类（可选）
try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None  # 若未安装，将在使用时提示


default_workers = -1


@dataclass
class GrowConfig:
    radius: float = 0.2                  # 距离/邻域半径（米）
    angle_deg: float = 20.0              # 法向阈值（度）
    curvature_max: float = 0.1           # 曲率上限（R 通道单位 0..1）
    curvature_delta_max: Optional[float] = 0.5  # 与种子曲率差的上限（可选）
    seed_percentile: float = 5.0         # 取最低 X% 曲率作为种子
    min_region_size: int = 100           # 最小区域点数
    precompute_neighbors: bool = True    # 是否预计算全部邻域
    neighbor_mem_cap_mb: int = 800       # 邻域表内存上限（超过则改为按需查询）
    workers: int = default_workers       # cKDTree 并行查询核心数（<=0 表示所有核心）
    use_region_mean_normal: bool = True  # 生长时用区域平均法向（效率优先）
    random_seed: int = 42                # 随机种子（可复现）


@dataclass
class PlaneRegion:
    region_id: int
    point_indices: np.ndarray
    normal: np.ndarray
    centroid: np.ndarray
    offset: float
    polygon: np.ndarray
    area: float
    dip: float
    dipdir: float
    strike: float


@dataclass
class SlopePlane:
    normal: np.ndarray
    dip: float
    dipdir: float
    strike: float
    source: str


@dataclass
class WedgeConfig:
    daylight_tol_deg: float = 20.0
    friction_angle_deg: float = 30.0
    line_buffer: float = 0.15
    parallel_tol: float = 1e-3


@dataclass
class WedgeResult:
    region_a: int
    region_b: int
    beta_deg: float
    daylight_angle_deg: float
    risk: str
    highlight_indices: np.ndarray
    line_start: np.ndarray
    line_end: np.ndarray


# --------------------- Utilities ---------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_str() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n

def angle_to_color(n: np.ndarray) -> Tuple[float, float, float]:
    """根据法向方向映射颜色（HSV）"""
    n = n / (np.linalg.norm(n) + 1e-12)
    az = math.atan2(n[1], n[0])                # [-pi, pi]
    el = math.acos(np.clip(n[2], -1.0, 1.0))   # [0, pi]
    hue = (az + math.pi) / (2.0 * math.pi)     # [0,1]
    val = 1.0 - (el / math.pi) * 0.5
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, val)
    return (r, g, b)

def set_id_to_color(k: int, K: int) -> Tuple[float, float, float]:
    """生成组颜色（HSV 均匀分布）；k=-1 使用灰色"""
    if k < 0 or K <= 0:
        return (0.7, 0.7, 0.7)
    hue = (k % K) / float(K)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (r, g, b)


def orthonormal_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """给定法向返回平面内两个单位正交基"""
    n = normalize(n)
    if np.allclose(n, 0.0):
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    # 选择与法向不平行的向量
    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(ref, n))
    v = normalize(np.cross(n, u))
    return u, v


def polygon_area_2d(coords: np.ndarray) -> float:
    if coords.shape[0] < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def dipdir_dip_to_normal(dipdir_deg: float, dip_deg: float) -> np.ndarray:
    dipdir_rad = math.radians(dipdir_deg)
    dip_rad = math.radians(dip_deg)
    # Dip direction azimuth measured clockwise from north (y) -> convert to x,y
    # Assuming coordinate system: x-east, y-north, z-up
    dir_x = math.sin(dipdir_rad)
    dir_y = math.cos(dipdir_rad)
    dir_z = 0.0
    dip_dir_vec = np.array([dir_x, dir_y, dir_z], dtype=np.float64)
    dip_dir_vec = normalize(dip_dir_vec)
    # Normal = rotate dip direction by dip upwards
    normal = np.array([
        -math.sin(dip_rad) * dir_x,
        -math.sin(dip_rad) * dir_y,
        math.cos(dip_rad)
    ], dtype=np.float64)
    return normalize(normal)


def slope_vectors_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = normalize(normal)
    strike_vec = normalize(np.cross(np.array([0.0, 0.0, 1.0]), normal))
    if np.allclose(strike_vec, 0.0):
        strike_vec = np.array([1.0, 0.0, 0.0])
    dip_vec = normalize(np.cross(normal, strike_vec))
    return dip_vec, strike_vec

def load_cc_pcd(pcd_path: Path) -> o3d.geometry.PointCloud:
    """读取 CloudCompare 二进制 PCD，要求包含法向和颜色（R 为曲率）"""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise ValueError(f"Failed to read or empty point cloud: {pcd_path}")
    if not pcd.has_normals():
        raise ValueError("Input PCD must contain normals (nx,ny,nz).")
    if not pcd.has_colors():
        raise ValueError("Input PCD must contain colors; curvature is expected in R channel.")
    return pcd

def interactive_select_aabb(pcd: o3d.geometry.PointCloud, window_name: str = "选择两个点定义 ROI (AABB)") -> Tuple[o3d.geometry.AxisAlignedBoundingBox, np.ndarray]:
    """
    SHIFT + 左键选择 2 个点，生成 AABB。返回 (AABB, mask)
    """
    print("\n[ROI] 操作：在窗口中按住 SHIFT + 左键，依次选择两个点作为对角；按 Q 退出。\n")
    try:
        picked = o3d.visualization.draw_geometries_with_editing([pcd], window_name=window_name)
    except Exception:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        picked = vis.get_picked_points()

    if len(picked) < 2:
        raise RuntimeError(f"需要选择 2 个点作为 AABB，对应当前选择数量：{len(picked)}")

    pts = np.asarray(pcd.points)
    a = pts[picked[0]]
    b = pts[picked[1]]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    aabb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    mask = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
            (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]) &
            (pts[:, 2] >= lo[2]) & (pts[:, 2] <= hi[2]))
    return aabb, mask


# -------------------- Neighbor build --------------------
def build_neighbors(points: np.ndarray, radius: float, workers: int,
                    mem_cap_mb: int, precompute: bool):
    tree = cKDTree(points)
    neighbors: Optional[List[np.ndarray]] = None
    if not precompute:
        return tree, None

    # 采样估计平均邻域规模，做内存预估
    rng = np.random.default_rng(42)
    N = len(points)
    if N == 0:
        return tree, None
    sample_idx = rng.choice(N, size=min(256, N), replace=False)
    sample_counts = [len(tree.query_ball_point(points[i], r=radius)) for i in sample_idx]
    avg_k = float(np.mean(sample_counts)) if sample_counts else 0.0
    estimated_bytes = N * max(avg_k, 1.0) * 4.0
    estimated_mb = estimated_bytes / (1024 ** 2)

    if estimated_mb > mem_cap_mb:
        print(f"[邻域] 预计邻域存储 {estimated_mb:.1f} MB > 上限 {mem_cap_mb} MB，改为按需查询")
        return tree, None

    print(f"[邻域] 全体邻域并行构建，workers={workers}，预计内存 ~{estimated_mb:.1f} MB")
    idx_arrays = tree.query_ball_point(points, r=radius, workers=workers)
    neighbors = [np.asarray(ix, dtype=np.int32) for ix in idx_arrays]
    return tree, neighbors


# -------------------- Plane fitting helper --------------------
def fit_plane_normal(pts_region: np.ndarray) -> np.ndarray:
    """
    用 PCA/SVD 对区域点坐标拟合平面，取最小特征值对应特征向量为法向（单位向量）。
    本版将方向统一到 y>0 半球；若 y≈0，使用 z<0 作为平局规则将其翻到“上方”。
    """
    n = pts_region.shape[0]
    if n < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    c = pts_region.mean(axis=0)
    X = pts_region - c
    C = X.T @ X
    try:
        w, v = np.linalg.eigh(C)
        idx_min = int(np.argmin(w))
        nvec = v[:, idx_min]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)

        # 统一到 y>0 半球；若 y≈0 用 z 作为平局规则
        eps = 1e-12
        if (nvec[1] < 0.0) or (abs(nvec[1]) <= eps and nvec[2] < 0.0):
            nvec = -nvec

        return nvec.astype(np.float64)
    except Exception:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)


# -------------------- Region Growing --------------------
def region_growing(points: np.ndarray,
                   normals: np.ndarray,
                   curvature: np.ndarray,
                   cfg: GrowConfig,
                   tree: cKDTree,
                   neighbors: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    返回：
      labels: [N] int32，区域编号；-1 表示未标记/噪声
      num_regions: 区域数
      region_mean_normals: [num_regions, 3] 区域法向（PCA 平面拟合，统一到 y>0 半球）
    """
    N = len(points)
    labels = np.full(N, -1, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    angle_thr = math.radians(cfg.angle_deg)

    # 种子：最低曲率百分位
    thresh_seed = np.percentile(curvature, cfg.seed_percentile)
    seed_idx = np.where(curvature <= thresh_seed)[0]
    seed_idx = seed_idx[np.argsort(curvature[seed_idx])]  # 低曲率优先

    region_id = 0
    region_normals: List[np.ndarray] = []

    pbar = tqdm(total=len(seed_idx), desc="区域生长（种子）", unit="seed")
    for s in seed_idx:
        pbar.update(1)
        if visited[s]:
            continue

        labels[s] = region_id
        visited[s] = True
        q = [s]
        members = [s]
        norm_sum = normals[s].astype(np.float64).copy()
        seed_curv = curvature[s]
        size = 1

        while q:
            i = q.pop()
            # 候选邻域
            if neighbors is not None:
                neigh = neighbors[i]
            else:
                neigh = np.asarray(tree.query_ball_point(points[i], r=cfg.radius), dtype=np.int32)
            if neigh.size == 0:
                continue

            # 去除已访问
            neigh = neigh[~visited[neigh]]
            if neigh.size == 0:
                continue

            # 曲率判据
            mask_curv = curvature[neigh] <= cfg.curvature_max
            if cfg.curvature_delta_max is not None:
                mask_curv &= np.abs(curvature[neigh] - seed_curv) <= cfg.curvature_delta_max
            if not np.any(mask_curv):
                continue
            cand = neigh[mask_curv]

            # 法向夹角判据（用当前区域平均法向）
            if cfg.use_region_mean_normal and size > 0:
                mean_n = norm_sum / (np.linalg.norm(norm_sum) + 1e-12)
            else:
                mean_n = normals[s]
            dots = np.einsum('ij,j->i', normals[cand], mean_n)
            dots = np.clip(dots, -1.0, 1.0)
            ang = np.arccos(dots)
            cand = cand[ang <= angle_thr]
            if cand.size == 0:
                continue

            # 接受并入队
            labels[cand] = region_id
            visited[cand] = True
            q.extend(cand.tolist())
            members.extend(cand.tolist())
            norm_sum += normals[cand].sum(axis=0)
            size += cand.size

        # 过小区域剔除（作为噪声）
        if size < cfg.min_region_size:
            labels[labels == region_id] = -1
            continue

        # 用 PCA 平面拟合得到区域法向（统一到 y>0 半球）
        pts_region = points[np.asarray(members, dtype=np.int32)]
        nvec = fit_plane_normal(pts_region)
        region_normals.append(nvec)

        region_id += 1
    pbar.close()

    num_regions = region_id
    if num_regions == 0:
        return labels, 0, np.zeros((0, 3), dtype=np.float64)
    region_mean_normals = np.vstack(region_normals)
    return labels, num_regions, region_mean_normals


# -------------------- Region Stats --------------------
def compute_region_stats(points: np.ndarray, labels: np.ndarray, num_regions: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (region_sizes [R], region_centroids [R,3])"""
    valid = labels >= 0
    region_sizes = np.bincount(labels[valid], minlength=num_regions).astype(np.int64)
    sums = np.zeros((num_regions, 3), dtype=np.float64)
    np.add.at(sums, labels[valid], points[valid])
    with np.errstate(invalid='ignore', divide='ignore'):
        centroids = sums / region_sizes[:, None]
    centroids[~np.isfinite(centroids)] = 0.0
    return region_sizes, centroids


def build_plane_regions(points: np.ndarray,
                        labels: np.ndarray,
                        region_mean_normals: np.ndarray) -> List[PlaneRegion]:
    regions: List[PlaneRegion] = []
    num_regions = region_mean_normals.shape[0]
    for rid in range(num_regions):
        idx = np.where(labels == rid)[0]
        if idx.size < 3:
            continue
        pts = points[idx]
        centroid = pts.mean(axis=0)
        normal = normalize(region_mean_normals[rid])
        offset = -float(np.dot(normal, centroid))
        dip, dipdir, strike = normal_to_geology(normal)
        u, v = orthonormal_basis_from_normal(normal)
        local = (pts - centroid) @ np.stack([u, v], axis=1)
        try:
            hull = ConvexHull(local)
            hull_coords = local[hull.vertices]
        except Exception:
            hull_coords = local
        area = abs(polygon_area_2d(hull_coords))
        poly3d = centroid + hull_coords @ np.stack([u, v], axis=0)
        regions.append(PlaneRegion(
            region_id=rid,
            point_indices=idx,
            normal=normal,
            centroid=centroid,
            offset=offset,
            polygon=poly3d,
            area=area,
            dip=dip,
            dipdir=dipdir,
            strike=strike
        ))
    return regions


# -------------------- Joint Sets Clustering (+ promotion & de-noise) --------------------
def normal_to_geology(n: np.ndarray) -> Tuple[float, float, float]:
    """n 已归一；返回 (dip, dipdir, strike) in degrees（使用 z 作为倾角参考）"""
    nz = abs(n[2])
    dip = math.degrees(math.acos(np.clip(nz, -1.0, 1.0)))  # 0..90
    dipdir = (math.degrees(math.atan2(n[1], n[0])) + 360.0) % 360.0
    strike = (dipdir - 90.0) % 360.0
    return dip, dipdir, strike

def estimate_kappa(Rbar: float) -> float:
    """Fisher 模型集中度 κ 的常用近似"""
    Rbar = float(np.clip(Rbar, 0.0, 0.999999))
    if Rbar < 1e-6:
        return 0.0
    return (Rbar * (3.0 - Rbar**2)) / (1.0 - Rbar**2 + 1e-12)


def select_slope_plane(regions: List[PlaneRegion],
                       slope_region: Optional[int],
                       slope_normal: Optional[np.ndarray],
                       slope_dipdir: Optional[float],
                       slope_dip: Optional[float]) -> Optional[SlopePlane]:
    if slope_normal is not None:
        n = normalize(np.asarray(slope_normal, dtype=np.float64))
        dip, dipdir, strike = normal_to_geology(n)
        return SlopePlane(normal=n, dip=dip, dipdir=dipdir, strike=strike, source="manual-normal")

    if slope_dipdir is not None and slope_dip is not None:
        n = dipdir_dip_to_normal(float(slope_dipdir), float(slope_dip))
        dip, dipdir, strike = normal_to_geology(n)
        return SlopePlane(normal=n, dip=dip, dipdir=dipdir, strike=strike, source="manual-dip")

    chosen: Optional[PlaneRegion] = None
    if slope_region is not None:
        for reg in regions:
            if reg.region_id == slope_region:
                chosen = reg
                break
    else:
        if regions:
            chosen = max(regions, key=lambda r: r.area)

    if chosen is None:
        return None

    n = normalize(chosen.normal)
    # 让坡面法向朝向外部（z 分量通常为负）
    if n[2] > 0:
        n = -n
    dip, dipdir, strike = normal_to_geology(n)
    return SlopePlane(normal=n, dip=dip, dipdir=dipdir, strike=strike, source=f"region-{chosen.region_id}")

def cluster_joint_sets(region_mean_normals: np.ndarray,
                       region_weights: np.ndarray,
                       total_points: int,
                       eps_deg: float = 5.0,
                       min_samples: int = 3,
                       major_singleton_percent: float = 1.0):
    """
    按“有向法向”聚类（0~180°）：
      先把法向统一到 y>0 半球（若 y≈0，用 z 作为平局规则），再用方向角 arccos(n·m) 构造距离矩阵。
    分组后：
      - 应用“单区大面片”提升；
      - ⭐ 确保不产生分组噪声：所有区域都会被分到某个 set（无 -1）。
    """
    if DBSCAN is None:
        raise RuntimeError("需要 scikit-learn 才能进行结构面组聚类，请先 `pip install scikit-learn`")

    R = region_mean_normals.shape[0]
    if R == 0:
        return np.zeros((0,), dtype=int), 0, np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,), dtype=int), np.zeros((0,))

    # 统一到 y>0 半球并单位化（与拟合阶段一致）
    n = region_mean_normals.copy()
    n = normalize_rows(n)
    eps = 1e-12
    flip_mask = (n[:, 1] < 0.0) | ((np.abs(n[:, 1]) <= eps) & (n[:, 2] < 0.0))
    n[flip_mask] *= -1.0

    # 方向角距离矩阵（不取绝对值）：D ∈ [0, π]
    dots = np.clip(n @ n.T, -1.0, 1.0)
    D = np.arccos(dots)

    if R > 8000:
        print(f"[警告] 区域数 R={R} 很大，角度距离矩阵为 {R}x{R} 可能占用较多内存。")

    eps_rad = math.radians(eps_deg)
    clu = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric="precomputed")
    raw_labels = clu.fit_predict(D)

    # 初步映射：连续正标签，噪声为 -1
    uniq = [c for c in np.unique(raw_labels) if c >= 0]
    mapping = {c: i for i, c in enumerate(sorted(uniq))}
    set_labels_regions = np.array([mapping.get(c, -1) for c in raw_labels], dtype=int)

    # ------ 单区大面片“强制成组” ------
    if major_singleton_percent > 0.0 and total_points > 0:
        threshold = (major_singleton_percent / 100.0) * float(total_points)
        noise_idx = np.where(set_labels_regions < 0)[0]
        promote_idx = [rid for rid in noise_idx if float(region_weights[rid]) >= threshold]
        if promote_idx:
            next_id = (max(set_labels_regions) + 1) if np.any(set_labels_regions >= 0) else 0
            for rid in promote_idx:
                set_labels_regions[rid] = next_id
                next_id += 1

    # ------ ⭐ 取消分组噪声：把剩余 -1 全部归入某个组 ------
    if np.any(set_labels_regions < 0):
        neg_idx = np.where(set_labels_regions < 0)[0]

        if np.any(set_labels_regions >= 0):
            # 先用当前正标签估计各簇的加权均值方向
            uniq_pos = sorted([c for c in np.unique(set_labels_regions) if c >= 0])
            K_pos = len(uniq_pos)
            means = np.zeros((K_pos, 3), dtype=np.float64)
            for i, k in enumerate(uniq_pos):
                idx = np.where(set_labels_regions == k)[0]
                w = region_weights[idx].astype(np.float64)
                vec = (n[idx] * w[:, None]).sum(axis=0)
                L = np.linalg.norm(vec)
                means[i] = vec / (L + 1e-12)

            # 把每个 -1 指派到“方向角最近”的已有簇（不设阈值，确保无 -1）
            dm = np.clip(n[neg_idx] @ means.T, -1.0, 1.0)   # 余弦相似度
            nearest = np.argmax(dm, axis=1)                 # 最大余弦 => 最小角
            nearest_sets = np.array([uniq_pos[j] for j in nearest], dtype=int)
            set_labels_regions[neg_idx] = nearest_sets
        else:
            # 没有任何正标签：每个区域单独成组（不留 -1）
            set_labels_regions = np.arange(R, dtype=int)

    # ------ 重新连续化并统计 ------
    uniq_final = [c for c in np.unique(set_labels_regions) if c >= 0]
    mapping2 = {c: i for i, c in enumerate(sorted(uniq_final))}
    set_labels_regions = np.array([mapping2.get(c, -1) for c in set_labels_regions], dtype=int)
    K = len(uniq_final)

    set_mean_normals = np.zeros((K, 3), dtype=np.float64)
    set_kappa = np.zeros((K,), dtype=np.float64)
    set_sizes = np.zeros((K,), dtype=int)
    set_weights = np.zeros((K,), dtype=np.float64)

    for k in range(K):
        idx = np.where(set_labels_regions == k)[0]
        set_sizes[k] = idx.size
        w = region_weights[idx].astype(np.float64)
        set_weights[k] = float(w.sum())
        vec = (n[idx] * w[:, None]).sum(axis=0)
        L = np.linalg.norm(vec)
        m = vec / L if L > 0 else np.array([0.0, 0.0, 1.0])
        set_mean_normals[k] = m
        Rbar = L / (w.sum() + 1e-12)
        set_kappa[k] = estimate_kappa(Rbar)

    return set_labels_regions, K, set_mean_normals, set_kappa, set_sizes, set_weights


def solve_line_between_planes(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float) -> Tuple[np.ndarray, np.ndarray]:
    direction = np.cross(n1, n2)
    norm_dir = np.linalg.norm(direction)
    if norm_dir <= 1e-12:
        return np.zeros(3), np.zeros(3)
    direction = direction / norm_dir
    A = np.vstack([n1, n2, direction])
    b = -np.array([d1, d2, 0.0], dtype=np.float64)
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        point = np.zeros(3)
    return point, direction


def distance_to_line(points: np.ndarray, point_on_line: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = points - point_on_line
    proj = v @ direction
    closest = point_on_line + np.outer(proj, direction)
    dist = np.linalg.norm(points - closest, axis=1)
    return dist, proj


def detect_wedges(regions: List[PlaneRegion],
                  slope_plane: Optional[SlopePlane],
                  cfg: WedgeConfig,
                  roi_points: np.ndarray) -> Tuple[List[WedgeResult], np.ndarray]:
    risk_levels = np.zeros(roi_points.shape[0], dtype=np.int32)
    if slope_plane is None or len(regions) < 2:
        return [], risk_levels

    dip_vec, _ = slope_vectors_from_normal(slope_plane.normal)
    slope_out = slope_plane.normal
    results: List[WedgeResult] = []

    for reg_a, reg_b in combinations(regions, 2):
        cross_norm = np.linalg.norm(np.cross(reg_a.normal, reg_b.normal))
        if cross_norm <= cfg.parallel_tol:
            continue
        p, direction = solve_line_between_planes(reg_a.normal, reg_a.offset, reg_b.normal, reg_b.offset)
        if np.allclose(direction, 0.0):
            continue

        if np.dot(direction, slope_out) < 0:
            direction = -direction

        daylight_cos = np.clip(np.dot(direction, dip_vec), -1.0, 1.0)
        daylight_angle = math.degrees(math.acos(abs(daylight_cos)))
        daylight = daylight_angle <= cfg.daylight_tol_deg

        beta = math.degrees(math.asin(np.clip(abs(direction[2]), 0.0, 1.0)))
        risk = "stable"
        if daylight and beta > cfg.friction_angle_deg:
            risk = "kinematic"
        elif daylight:
            risk = "daylight"

        combined_idx = np.concatenate([reg_a.point_indices, reg_b.point_indices])
        pts = roi_points[combined_idx]
        dist, proj = distance_to_line(pts, p, direction)
        mask = dist <= cfg.line_buffer
        if not np.any(mask):
            continue
        highlight_idx = combined_idx[mask]
        proj_vals = proj[mask]
        seg_min = float(np.min(proj_vals))
        seg_max = float(np.max(proj_vals))
        start = p + seg_min * direction
        end = p + seg_max * direction

        priority = {"stable": 1, "daylight": 2, "kinematic": 3}
        for idx in highlight_idx:
            risk_levels[idx] = max(risk_levels[idx], priority[risk])

        results.append(WedgeResult(
            region_a=reg_a.region_id,
            region_b=reg_b.region_id,
            beta_deg=beta,
            daylight_angle_deg=daylight_angle,
            risk=risk,
            highlight_indices=highlight_idx,
            line_start=start,
            line_end=end
        ))

    return results, risk_levels


# -------------------- Coloring --------------------
def colorize_by_region_normal(labels: np.ndarray,
                              region_mean_normals: np.ndarray) -> np.ndarray:
    N = labels.shape[0]
    colors = np.zeros((N, 3), dtype=np.float32)
    for rid in range(region_mean_normals.shape[0]):
        col = np.array(angle_to_color(region_mean_normals[rid]), dtype=np.float32)
        colors[labels == rid] = col
    colors[labels < 0] = np.array([0.7, 0.7, 0.7], dtype=np.float32)
    return colors

def colorize_by_set_id(point_region_labels: np.ndarray,
                       set_labels_regions: np.ndarray,
                       num_sets: int) -> np.ndarray:
    """将每个点映射到 set_id 并上色；未归类为灰色"""
    N = point_region_labels.shape[0]
    colors = np.zeros((N, 3), dtype=np.float32)
    colors[:] = np.array([0.7, 0.7, 0.7], dtype=np.float32)
    if set_labels_regions.size == 0:
        return colors
    valid = point_region_labels >= 0
    rid_for_points = point_region_labels[valid]
    set_for_points = set_labels_regions[rid_for_points]
    for k in range(num_sets):
        col = np.array(set_id_to_color(k, num_sets), dtype=np.float32)
        mask = np.zeros_like(valid)
        mask[valid] = (set_for_points == k)
        colors[mask] = col
    return colors


def colorize_wedge_risk(risk_levels: np.ndarray, base_colors: Optional[np.ndarray] = None) -> np.ndarray:
    palette = {
        0: np.array([0.6, 0.6, 0.6], dtype=np.float32),
        1: np.array([0.8, 0.8, 0.2], dtype=np.float32),
        2: np.array([1.0, 0.65, 0.0], dtype=np.float32),
        3: np.array([0.9, 0.05, 0.1], dtype=np.float32),
    }
    colors = np.zeros((risk_levels.shape[0], 3), dtype=np.float32)
    if base_colors is not None:
        colors[:] = base_colors.astype(np.float32)
    else:
        for level, col in palette.items():
            colors[risk_levels == level] = col
    for level, col in palette.items():
        mask = risk_levels == level
        if np.any(mask):
            colors[mask] = col
    return colors


# -------------------- Saving (regions) --------------------
def save_region_outputs(base_dir: Path, base_name: str,
                        pcd_roi: o3d.geometry.PointCloud,
                        labels: np.ndarray,
                        region_mean_normals: np.ndarray,
                        cfg: GrowConfig) -> Dict[str, str]:
    out_dir = base_dir / f"{base_name}_segments_{now_str()}"
    ensure_dir(out_dir)

    pts = np.asarray(pcd_roi.points)
    nrms = np.asarray(pcd_roi.normals)
    colors = colorize_by_region_normal(labels, region_mean_normals)

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(pts)
    pcd_colored.normals = o3d.utility.Vector3dVector(nrms)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    artifacts: Dict[str, str] = {}
    fused_path = out_dir / f"{base_name}_segmented_colored.ply"
    o3d.io.write_point_cloud(str(fused_path), pcd_colored, write_ascii=False, print_progress=True)
    artifacts["colored_segmentation"] = str(fused_path)

    labels_path = out_dir / f"{base_name}_labels.npy"
    np.save(labels_path, labels)
    artifacts["labels_npy"] = str(labels_path)

    # summary
    num_regions = region_mean_normals.shape[0]
    summary_rows = []
    for rid in range(num_regions):
        k = int(np.sum(labels == rid))
        n = region_mean_normals[rid] if num_regions > 0 else np.array([0, 0, 0])
        summary_rows.append({"region_id": rid, "size": k, "mean_nx": float(n[0]), "mean_ny": float(n[1]), "mean_nz": float(n[2])})
    summary_path = out_dir / f"{base_name}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region_id", "size", "mean_nx", "mean_ny", "mean_nz"])
        w.writeheader()
        w.writerows(summary_rows)
    artifacts["summary_csv"] = str(summary_path)

    cfg_path = out_dir / f"{base_name}_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2, ensure_ascii=False)
    artifacts["config_json"] = str(cfg_path)

    artifacts["out_dir"] = str(out_dir)

    # 各区域并行导出
    region_dir = out_dir / "regions"
    ensure_dir(region_dir)

    def save_one_region(rid: int) -> Tuple[int, str]:
        idx = np.where(labels == rid)[0]
        if idx.size < cfg.min_region_size:
            return rid, ""
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts[idx])
        sub.normals = o3d.utility.Vector3dVector(nrms[idx])
        sub.colors = o3d.utility.Vector3dVector(colors[idx])
        p = region_dir / f"region_{rid:04d}_n{idx.size}.ply"
        o3d.io.write_point_cloud(str(p), sub, write_ascii=False)
        return rid, str(p)

    print("[保存] 按区域导出点云 ...")
    futures = []
    with ThreadPoolExecutor(max_workers=max(os.cpu_count() or 4, 4)) as ex:
        for rid in range(num_regions):
            futures.append(ex.submit(save_one_region, rid))
        for _ in tqdm(as_completed(futures), total=len(futures), unit="region"):
            pass

    return artifacts


# -------------------- Saving (joint sets, per-group & ungrouped) --------------------
def save_joint_sets(out_dir: Path,
                    base_name: str,
                    points: np.ndarray,
                    normals: np.ndarray,
                    region_labels_points: np.ndarray,
                    region_sizes: np.ndarray,
                    set_labels_regions: np.ndarray,
                    set_mean_normals: np.ndarray,
                    set_kappa: np.ndarray) -> Dict[str, str]:
    """
    导出：
      - face_to_set.csv（区域→组映射，含区域大小）
      - joint_sets.csv（每组统计，包括 dip/dipdir/strike/κ）
      - sets_colored.ply（整体按结构面组上色）
      - sets/ 目录：每个结构面组一个 ply（set_XXXX_n*.ply）
      - sets/set_-1_n*.ply：未分组点云（存在才导出）
    """
    artifacts: Dict[str, str] = {}

    # region -> set 映射表
    R = set_labels_regions.shape[0]
    face_map_rows = []
    for rid in range(R):
        face_map_rows.append({"region_id": rid, "set_id": int(set_labels_regions[rid]), "region_size": int(region_sizes[rid])})
    face_map_path = out_dir / f"{base_name}_face_to_set.csv"
    with open(face_map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region_id", "set_id", "region_size"])
        w.writeheader()
        w.writerows(face_map_rows)
    artifacts["face_to_set_csv"] = str(face_map_path)

    # 组统计
    K = set_mean_normals.shape[0]
    set_rows = []

    # 点级别的 set_id（-1 表示未分组或未归属区域）
    point_set = np.full(points.shape[0], -1, dtype=int)
    valid = region_labels_points >= 0
    if R > 0:
        point_set[valid] = set_labels_regions[region_labels_points[valid]]
    set_point_counts = np.bincount(point_set[point_set >= 0], minlength=K).astype(int)

    for k in range(K):
        n = set_mean_normals[k]
        dip, dipdir, strike = normal_to_geology(n)
        set_rows.append({
            "set_id": k,
            "num_regions": int(np.sum(set_labels_regions == k)),
            "num_points": int(set_point_counts[k]),
            "mean_nx": float(n[0]),
            "mean_ny": float(n[1]),
            "mean_nz": float(n[2]),
            "dip": float(dip),
            "dipdir": float(dipdir),
            "strike": float(strike),
            "kappa": float(set_kappa[k]),
        })
    sets_csv = out_dir / f"{base_name}_joint_sets.csv"
    with open(sets_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set_id","num_regions","num_points","mean_nx","mean_ny","mean_nz","dip","dipdir","strike","kappa"])
        w.writeheader()
        w.writerows(set_rows)
    artifacts["joint_sets_csv"] = str(sets_csv)

    # 整体按组着色
    colors = colorize_by_set_id(region_labels_points, set_labels_regions, K)
    pcd_sets = o3d.geometry.PointCloud()
    pcd_sets.points = o3d.utility.Vector3dVector(points)
    pcd_sets.normals = o3d.utility.Vector3dVector(normals)
    pcd_sets.colors = o3d.utility.Vector3dVector(colors)
    fused_path = out_dir / f"{base_name}_sets_colored.ply"
    o3d.io.write_point_cloud(str(fused_path), pcd_sets, write_ascii=False, print_progress=True)
    artifacts["sets_colored_ply"] = str(fused_path)

    # 每组与未分组：全部导出
    sets_dir = out_dir / "sets"
    ensure_dir(sets_dir)
    print("[保存] 按结构面组（含未分组）导出点云 ...")

    def save_one_set_generic(k: int) -> Tuple[int, str]:
        if k >= 0:
            idx = np.where(point_set == k)[0]
        else:
            idx = np.where(point_set < 0)[0]   # 未分组（-1 或无区域）
        if idx.size == 0:
            return k, ""
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(points[idx])
        sub.normals = o3d.utility.Vector3dVector(normals[idx])
        col = np.array(set_id_to_color(k, max(K, 1)), dtype=np.float32)
        sub.colors = o3d.utility.Vector3dVector(np.tile(col, (idx.size, 1)))
        name = f"set_{k:04d}_n{idx.size}.ply" if k >= 0 else f"set_-1_n{idx.size}.ply"
        p = sets_dir / name
        o3d.io.write_point_cloud(str(p), sub, write_ascii=False)
        return k, str(p)

    futures = []
    with ThreadPoolExecutor(max_workers=max(os.cpu_count() or 4, 4)) as ex:
        futures.append(ex.submit(save_one_set_generic, -1))  # 未分组
        for k in range(K):
            futures.append(ex.submit(save_one_set_generic, k))
        for _ in tqdm(as_completed(futures), total=len(futures), unit="set"):
            pass

    artifacts["sets_dir"] = str(sets_dir)
    return artifacts


def save_wedge_outputs(out_dir: Path,
                       base_name: str,
                       points: np.ndarray,
                       normals: np.ndarray,
                       risk_levels: np.ndarray,
                       wedge_results: List[WedgeResult]) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    if not wedge_results:
        return artifacts

    wedge_rows = []
    for res in wedge_results:
        wedge_rows.append({
            "region_a": res.region_a,
            "region_b": res.region_b,
            "beta_deg": float(res.beta_deg),
            "daylight_angle_deg": float(res.daylight_angle_deg),
            "risk": res.risk,
            "highlight_points": int(res.highlight_indices.size),
        })
    wedge_csv = out_dir / f"{base_name}_wedges.csv"
    with open(wedge_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region_a", "region_b", "beta_deg", "daylight_angle_deg", "risk", "highlight_points"])
        w.writeheader()
        w.writerows(wedge_rows)
    artifacts["wedge_csv"] = str(wedge_csv)

    risk_colors = colorize_wedge_risk(risk_levels)
    pcd_risk = o3d.geometry.PointCloud()
    pcd_risk.points = o3d.utility.Vector3dVector(points)
    pcd_risk.normals = o3d.utility.Vector3dVector(normals)
    pcd_risk.colors = o3d.utility.Vector3dVector(risk_colors)
    risk_path = out_dir / f"{base_name}_risk_colored.ply"
    o3d.io.write_point_cloud(str(risk_path), pcd_risk, write_ascii=False, print_progress=True)
    artifacts["risk_colored_ply"] = str(risk_path)

    line_points = []
    lines = []
    line_colors = []
    for idx, res in enumerate(wedge_results):
        line_points.append(res.line_start)
        line_points.append(res.line_end)
        lines.append([2 * idx, 2 * idx + 1])
        if res.risk == "kinematic":
            line_colors.append([0.9, 0.0, 0.1])
        elif res.risk == "daylight":
            line_colors.append([1.0, 0.65, 0.0])
        else:
            line_colors.append([0.6, 0.6, 0.6])
    if line_points:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64)),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        )
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
        line_path = out_dir / f"{base_name}_wedge_lines.ply"
        o3d.io.write_line_set(str(line_path), line_set)
        artifacts["wedge_lines_ply"] = str(line_path)

    return artifacts


# -------------------- Stereonet plot --------------------
def plot_stereonet_equal_area(out_dir: Path,
                              base_name: str,
                              region_mean_normals: np.ndarray,
                              set_labels_regions: np.ndarray) -> Optional[str]:
    """
    绘制等面积极投影（Lambert）：极点=区域平均法向。
    这里仍统一到 z>=0 来做展示（行业习惯为下半/上半球投影），不影响分组逻辑。
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[提示] 未安装 matplotlib，跳过极投影图绘制。请 `pip install matplotlib`")
        return None

    if region_mean_normals.size == 0:
        return None

    n = region_mean_normals.copy()
    n[n[:, 2] < 0] *= -1.0  # 仅用于展示
    n = normalize_rows(n)
    # Lambert 等面积投影：r = sqrt(2) * sin(theta/2)，theta = arccos(nz)
    theta = np.arccos(np.clip(n[:, 2], -1.0, 1.0))
    r = np.sqrt(2.0) * np.sin(theta / 2.0)
    # 在水平面上的方向
    xy_norm = np.linalg.norm(n[:, :2], axis=1) + 1e-12
    x = r * (n[:, 0] / xy_norm)
    y = r * (n[:, 1] / xy_norm)

    labels = set_labels_regions
    K = int(np.max(labels)) + 1 if labels.size and np.max(labels) >= 0 else 0
    colors = np.array([set_id_to_color(k, max(K, 1)) for k in range(max(K, 1))])

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    circle = plt.Circle((0, 0), np.sqrt(2.0), fill=False)
    ax.add_artist(circle)
    if K == 0:
        plt.scatter(x, y, s=8)
    else:
        for k in range(K):
            idx = (labels == k)
            if np.any(idx):
                plt.scatter(x[idx], y[idx], s=10, label=f"Set {k}", c=[colors[k]])
    plt.title("Equal-Area Stereonet (Poles of Regions)")
    if K > 0:
        plt.legend(loc="upper right", fontsize=8)
    plt.xlabel("X"); plt.ylabel("Y")
    out_png = out_dir / f"{base_name}_stereonet.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    return str(out_png)


# -------------------- CLI --------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Region Growing + Joint Sets for CloudCompare Binary PCD")
    # 让 pcd 可选；缺省时弹文件选择对话框
    p.add_argument("pcd", type=str, nargs="?", help="Input CloudCompare binary PCD path")
    p.add_argument("--select", action="store_true", help="Interactive ROI: pick 2 points to form an AABB")
    # 区域生长参数
    p.add_argument("--radius", type=float, default=0.2, help="Neighborhood radius / distance threshold (meters)")
    p.add_argument("--angle-deg", type=float, default=20.0, help="Normal angle threshold (degrees)")
    p.add_argument("--curv-max", type=float, default=0.1, help="Curvature absolute threshold (R channel units 0..1)")
    p.add_argument("--curv-delta", type=float, default=0.5, help="Optional |curvature - curvature_seed| <= delta")
    p.add_argument("--seed-percentile", type=float, default=5.0, help="Seeds from lowest X%% curvature")
    p.add_argument("--min-region-size", type=int, default=100, help="Discard regions smaller than this")
    p.add_argument("--no-precompute", action="store_true", help="Disable neighbor precomputation to save memory")
    p.add_argument("--neighbor-mem-cap", type=int, default=800, help="Neighbor list memory cap in MB")
    p.add_argument("--workers", type=int, default=default_workers, help="cKDTree workers (<=0 means all cores)")
    p.add_argument("--show", action="store_true", help="Show colored segmentation by region normals")

    # 结构面组聚类参数（方向角版本）
    p.add_argument("--no-cluster-sets", action="store_true", help="Disable joint sets clustering")
    p.add_argument("--sets-eps-deg", type=float, default=5.0, help="DBSCAN eps in degrees on directional angle (0~180°)")
    p.add_argument("--sets-min-samples", type=int, default=3, help="DBSCAN min_samples (min regions per set)")
    p.add_argument("--major-singleton-percent", type=float, default=1.0,
                   help="Promote a single region to its own set if its point count ≥ this percent of total ROI points (0 disables).")
    p.add_argument("--show-sets", action="store_true", help="Show colored segmentation by joint sets")
    p.add_argument("--plot-stereonet", action="store_true", help="Plot equal-area stereonet (requires matplotlib)")
    # 楔形体分析与坡面
    p.add_argument("--wedge-analysis", action="store_true", help="Enable wedge detection and risk labelling")
    p.add_argument("--wedge-daylight-deg", type=float, default=20.0,
                   help="Daylight tolerance between wedge line and slope dip direction (degrees)")
    p.add_argument("--friction-angle", type=float, default=30.0,
                   help="Friction angle for kinematic wedge screening (degrees)")
    p.add_argument("--wedge-line-buffer", type=float, default=0.15,
                   help="Distance threshold (meters) to tag points around wedge intersection lines")
    p.add_argument("--slope-region", type=int, help="Use the specified region id as slope reference")
    p.add_argument("--slope-normal", type=float, nargs=3, help="Manual slope normal vector (overrides region selection)")
    p.add_argument("--slope-dipdir", type=float, help="Manual slope dip direction azimuth (degrees)")
    p.add_argument("--slope-dip", type=float, help="Manual slope dip (degrees)")
    # 兼容旧参数（不再需要；保留不报错）
    p.add_argument("--export-sets", action="store_true", help="(Deprecated) Sets are always exported now.")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # 兼容：未提供 pcd 路径则弹出文件选择对话框
    if not args.pcd:
        print("[输入] 未提供 pcd 参数，将弹出文件选择器 ...")
        chosen = askopenfilename_pcd()
        if not chosen:
            parser.error("未选择任何文件；请提供 pcd 路径或在对话框中选择。")
        args.pcd = chosen

    in_path = Path(args.pcd).expanduser().resolve()
    if not in_path.exists():
        parser.error(f"输入文件不存在：{in_path}")

    print(f"[输入] {in_path}")
    pcd = load_cc_pcd(in_path)

    points_all = np.asarray(pcd.points)
    normals_all = np.asarray(pcd.normals)
    colors_all = np.asarray(pcd.colors)  # [0,1]
    curvature_all = colors_all[:, 0].copy()  # R 通道

    # ROI
    if args.select:
        aabb, mask = interactive_select_aabb(pcd)
        print(f"[ROI] AABB min={aabb.get_min_bound()}, max={aabb.get_max_bound()}")
    else:
        print("[ROI] 未启用交互选择（--select），使用全量点")
        mask = np.ones(len(points_all), dtype=bool)

    points = points_all[mask]
    normals = normals_all[mask]
    curvature = curvature_all[mask]

    if points.shape[0] == 0:
        raise RuntimeError("ROI 内无点，无法分割")

    # 配置
    cfg = GrowConfig(
        radius=float(args.radius),
        angle_deg=float(args.angle_deg),
        curvature_max=float(args.curv_max),
        curvature_delta_max=(float(args.curv_delta) if args.curv_delta is not None else None),
        seed_percentile=float(args.seed_percentile),
        min_region_size=int(args.min_region_size),
        precompute_neighbors=not args.no_precompute,
        neighbor_mem_cap_mb=int(args.neighbor_mem_cap),
        workers=int(args.workers),
    )

    # 阶段 1：邻域
    print("[阶段] 1/6 邻域构建")
    tree, neighbors = build_neighbors(points, cfg.radius, cfg.workers, cfg.neighbor_mem_cap_mb, cfg.precompute_neighbors)

    # 阶段 2：区域生长
    print("[阶段] 2/6 区域生长分割")
    region_labels_points, num_regions, region_mean_normals = region_growing(points, normals, curvature, cfg, tree, neighbors)
    print(f"[结果] 区域数量：{num_regions}，有效点数：{int(np.sum(region_labels_points>=0))} / {len(region_labels_points)}")

    # 区域统计（权重=区域点数；也可替换为区域面积）
    region_sizes, _ = compute_region_stats(points, region_labels_points, num_regions)
    plane_regions = build_plane_regions(points, region_labels_points, region_mean_normals)
    if plane_regions:
        largest = max(plane_regions, key=lambda r: r.area)
        print(f"[区域] 面积最大区域 id={largest.region_id} 面积≈{largest.area:.3f} m² dip={largest.dip:.1f}° dipdir={largest.dipdir:.1f}°")

    # 阶段 3：保存区域结果
    print("[阶段] 3/6 保存区域结果")
    pcd_roi = o3d.geometry.PointCloud()
    pcd_roi.points = o3d.utility.Vector3dVector(points)
    pcd_roi.normals = o3d.utility.Vector3dVector(normals)

    base_dir = in_path.parent
    base_name = in_path.stem
    artifacts = save_region_outputs(base_dir, base_name, pcd_roi, region_labels_points, region_mean_normals, cfg)
    out_dir = Path(artifacts["out_dir"])
    for k, v in artifacts.items():
        print(f"  - {k}: {v}")

    # 阶段 4：结构面组聚类 + 提升规则 + 导出 + 可视化
    if not args.no_cluster_sets and num_regions > 0:
        print("[阶段] 4/6 结构面组聚类（方向角，y>0 半球；不产生分组噪声）")
        try:
            set_labels_regions, K, set_mean_normals, set_kappa, set_sizes, set_weights = cluster_joint_sets(
                region_mean_normals,
                region_sizes,
                total_points=points.shape[0],
                eps_deg=args.sets_eps_deg,
                min_samples=args.sets_min_samples,
                major_singleton_percent=args.major_singleton_percent
            )
            print(f"[结构面组] 组数：{K}（全部区域已归组，无 -1）")

            # 保存结构面组相关结果（始终导出每组与未分组）
            set_artifacts = save_joint_sets(out_dir, base_name,
                                            points, normals,
                                            region_labels_points,
                                            region_sizes,
                                            set_labels_regions,
                                            set_mean_normals,
                                            set_kappa)
            for k, v in set_artifacts.items():
                print(f"  - {k}: {v}")

            # 极投影图（可选）
            if args.plot_stereonet:
                png = plot_stereonet_equal_area(out_dir, base_name, region_mean_normals, set_labels_regions)
                if png:
                    print(f"  - stereonet_png: {png}")

            # 3D 显示（按组着色）
            if args.show_sets:
                colors_sets = colorize_by_set_id(region_labels_points, set_labels_regions, K)
                pcd_sets = o3d.geometry.PointCloud()
                pcd_sets.points = o3d.utility.Vector3dVector(points)
                pcd_sets.normals = o3d.utility.Vector3dVector(normals)
                pcd_sets.colors = o3d.utility.Vector3dVector(colors_sets)
                o3d.visualization.draw_geometries([pcd_sets], window_name="Joint Sets Result")

        except RuntimeError as e:
            print(f"[提示] 跳过结构面组聚类：{e}")

    else:
        print("[结构面组] 已禁用（--no-cluster-sets）或无有效区域。")

    # 阶段 5：楔形体分析
    wedge_results: List[WedgeResult] = []
    risk_levels = np.zeros(points.shape[0], dtype=np.int32)
    if args.wedge_analysis and num_regions > 1:
        print("[阶段] 5/6 楔形体判识与风险标注")
        slope_normal = np.asarray(args.slope_normal, dtype=np.float64) if args.slope_normal else None
        slope_plane = select_slope_plane(plane_regions, args.slope_region, slope_normal, args.slope_dipdir, args.slope_dip)
        if slope_plane is None:
            print("[提示] 无法确定坡面参考，跳过楔形体分析。")
        else:
            print(f"[坡面] 来源={slope_plane.source} dip={slope_plane.dip:.2f}° dipdir={slope_plane.dipdir:.2f}° strike={slope_plane.strike:.2f}°")
            wedge_cfg = WedgeConfig(
                daylight_tol_deg=float(args.wedge_daylight_deg),
                friction_angle_deg=float(args.friction_angle),
                line_buffer=float(args.wedge_line_buffer)
            )
            wedge_results, risk_levels = detect_wedges(plane_regions, slope_plane, wedge_cfg, points)
            print(f"[楔体] 检出 {len(wedge_results)} 组候选")
            if wedge_results:
                wedge_artifacts = save_wedge_outputs(out_dir, base_name, points, normals, risk_levels, wedge_results)
                for k, v in wedge_artifacts.items():
                    print(f"  - {k}: {v}")
    else:
        print("[阶段] 5/6 楔形体分析已跳过（条件不足或未启用 --wedge-analysis）。")

    # 阶段 5：可视化（按区域法向）
    if args.show:
        print("[阶段] 6/6 可视化（按区域法向）")
        colors = colorize_by_region_normal(region_labels_points, region_mean_normals)
        pcd_colored = o3d.geometry.PointCloud()
        pcd_colored.points = o3d.utility.Vector3dVector(points)
        pcd_colored.normals = o3d.utility.Vector3dVector(normals)
        pcd_colored.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd_colored], window_name="Region Growing Result")

    print("\n[完成] 处理结束 🎉")


if __name__ == "__main__":
    main()
