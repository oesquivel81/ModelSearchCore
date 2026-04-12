import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Optional, Dict

# --- Utilidades geométricas y de curva ---
def bbox_intersection(b1: BoundingBox, b2: BoundingBox) -> bool:
    return not (b1.x2 < b2.x1 or b1.x1 > b2.x2 or b1.y2 < b2.y1 or b1.y1 > b2.y2)

def point_to_polyline_distance(point: Tuple[float, float], polyline: List[Tuple[float, float]]) -> float:
    px, py = point
    return min(np.hypot(px - x, py - y) for x, y in polyline)

def project_point_to_curve(point: Tuple[float, float], curve: List[Tuple[float, float]]):
    dists = [np.hypot(point[0] - x, point[1] - y) for x, y in curve]
    idx = int(np.argmin(dists))
    return idx / (len(curve)-1), idx

# --- Selección de regiones por experimento ---
def select_regions_by_experiment(regions: List[RegionNode], experiment_mode: str, curve: Optional[List[Tuple[float, float]]] = None, curve_radius: float = 10.0) -> List[RegionNode]:
    selected = []
    if experiment_mode == "all_patches":
        selected = regions.copy()
    elif experiment_mode == "curve_selected_patches":
        if curve is None:
            raise ValueError("Se requiere curva para este experimento")
        for r in regions:
            dist = point_to_polyline_distance(r.centroid, curve)
            if dist <= curve_radius:
                r.lives_near_curve = True
                param, idx = project_point_to_curve(r.centroid, curve)
                r.curve_param = param
                r.order_index = idx
                selected.append(r)
    elif experiment_mode == "curve_all_patches_nerve":
        if curve is None:
            raise ValueError("Se requiere curva para este experimento")
        for r in regions:
            dist = point_to_polyline_distance(r.centroid, curve)
            r.lives_near_curve = dist <= curve_radius
            param, idx = project_point_to_curve(r.centroid, curve)
            r.curve_param = param
            r.order_index = idx
            selected.append(r)
    else:
        raise ValueError(f"Modo de experimento desconocido: {experiment_mode}")
    return selected

# --- Intersecciones y nervio simplicial ---
def compute_region_intersections(regions: List[RegionNode]):
    n = len(regions)
    for i in range(n):
        for j in range(i+1, n):
            if bbox_intersection(regions[i].bbox, regions[j].bbox):
                regions[i].intersections.add(regions[j].region_id)
                regions[j].intersections.add(regions[i].region_id)
    for r in regions:
        r.num_intersections = len(r.intersections)

def build_nerve_simplicial_complex(regions: List[RegionNode], max_dim: int = 2):
    from itertools import combinations
    simplexes = []
    for r in regions:
        simplexes.append((r.region_id,))
    for r1, r2 in combinations(regions, 2):
        if r2.region_id in r1.intersections:
            simplexes.append(tuple(sorted([r1.region_id, r2.region_id])))
    if max_dim >= 2:
        for r1, r2, r3 in combinations(regions, 3):
            ids = [r1.region_id, r2.region_id, r3.region_id]
            if (r2.region_id in r1.intersections and
                r3.region_id in r1.intersections and
                r3.region_id in r2.intersections):
                simplexes.append(tuple(sorted(ids)))
    simplex_count = {r.region_id: 0 for r in regions}
    for s in simplexes:
        for rid in s:
            simplex_count[rid] += 1
    for r in regions:
        r.num_simplices = simplex_count[r.region_id]
    return simplexes

# --- Exportación y tabla ---
def build_region_table(regions: List[RegionNode]) -> pd.DataFrame:
    return pd.DataFrame([{
        "region_id": r.region_id,
        "patient_id": r.patient_id,
        "config_id": r.config.config_id,
        "filter_name": r.config.filter_name,
        "use_variance": r.config.use_variance,
        "variance_mode": r.config.variance_mode,
        "patch_size": r.config.patch_size,
        "stride": r.config.stride,
        "variance_kernel": r.config.variance_kernel,
        "bbox": (r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2),
        "centroid": r.centroid,
        "curve_param": r.curve_param,
        "order_index": r.order_index,
        "intersections": sorted(r.intersections),
        "num_intersections": len(r.intersections),
        "num_simplices": len(r.nerve_simplices) if hasattr(r, 'nerve_simplices') else 0,
        "lives_in_union": r.lives_in_union,
        "lives_in_intersection": r.lives_in_intersection,
        "support_label": r.support_label,
        "metadata": r.metadata,
    } for r in regions])

def export_regions_to_json(regions: List[RegionNode], path: Optional[str] = None) -> str:
    data = build_region_table(regions).to_dict(orient="records")
    js = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(js)
    return js

# --- Visualización ---
def plot_regions_curve_and_nerve(
    image: Optional[np.ndarray],
    regions: List[RegionNode],
    curve: Optional[List[Tuple[float, float]]] = None,
    simplexes: Optional[List[Tuple[str, ...]]] = None,
    show_only_selected: bool = True,
    color_by: str = "cluster"
):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    if image is not None:
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    if curve is not None:
        cx, cy = zip(*curve)
        ax.plot(cx, cy, "-b", lw=2, label="Curva guía")
    color_map = plt.cm.get_cmap("tab10")
    for idx, r in enumerate(regions):
        if show_only_selected and not getattr(r, 'lives_near_curve', True):
            continue
        x1, y1, x2, y2 = r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2
        color = color_map(idx % 10)
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, lw=2))
        ax.plot(r.centroid[0], r.centroid[1], "o", color=color, label=f"{r.region_id}")
    if simplexes is not None:
        id2centroid = {r.region_id: r.centroid for r in regions}
        for s in simplexes:
            if len(s) == 2:
                c1, c2 = id2centroid[s[0]], id2centroid[s[1]]
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "-r", lw=1)
            elif len(s) == 3:
                c = [id2centroid[rid] for rid in s]
                poly = plt.Polygon(c, fill=None, edgecolor="g", lw=1, linestyle=":")
                ax.add_patch(poly)
    ax.legend()
    plt.show()
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Iterable

class RegionSelectionMode(str, Enum):
    ALL = "all"
    CURVE = "curve"
    NERVE = "nerve"

@dataclass
class RegionConfig:
    config_id: str
    filter_name: str
    use_variance: bool
    variance_mode: str
    patch_size: tuple[int, int] | None = None
    stride: int | None = None
    variance_kernel: int | None = None

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def intersects(self, other: "BoundingBox") -> bool:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return ix2 > ix1 and iy2 > iy1

    def intersection_box(self, other: "BoundingBox") -> "BoundingBox | None":
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 > ix1 and iy2 > iy1:
            return BoundingBox(ix1, iy1, ix2, iy2)
        return None

@dataclass(frozen=True, order=True)
class Simplex:
    vertices: tuple[str, ...]

    def __post_init__(self):
        if not self.vertices:
            raise ValueError("A simplex must contain at least one vertex.")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("A simplex cannot contain repeated vertices.")
        object.__setattr__(self, "vertices", tuple(sorted(self.vertices)))

    @property
    def dimension(self) -> int:
        return len(self.vertices) - 1

    def faces(self) -> set["Simplex"]:
        if len(self.vertices) == 1:
            return set()
        return {
            Simplex(self.vertices[:i] + self.vertices[i + 1 :])
            for i in range(len(self.vertices))
        }

@dataclass
class RegionNode:
    region_id: str
    patient_id: str
    config: RegionConfig
    bbox: BoundingBox
    centroid: tuple[float, float] = field(init=False)
    curve_param: float | None = None
    order_index: int | None = None
    intersections: set[str] = field(default_factory=set)
    nerve_simplices: set[Simplex] = field(default_factory=set)
    lives_in_union: bool = False
    lives_in_intersection: bool = False
    support_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.centroid = self.bbox.centroid

    def add_intersection(self, other_region_id: str) -> None:
        self.intersections.add(other_region_id)
        self.lives_in_intersection = True

    def add_simplex(self, simplex: Simplex) -> None:
        self.nerve_simplices.add(simplex)

    @property
    def x(self) -> float:
        return self.centroid[0]

    @property
    def y(self) -> float:
        return self.centroid[1]

@dataclass
class RegionSet:
    patient_id: str

    config: RegionConfig
    mode: RegionSelectionMode
    regions: list[RegionNode] = field(default_factory=list)

    def add_region(self, region: RegionNode) -> None:
        if region.patient_id != self.patient_id:
            raise ValueError("Region patient_id does not match RegionSet patient_id.")
        self.regions.append(region)

    def get_region(self, region_id: str) -> RegionNode | None:
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None

    @staticmethod
    def _all_intersect_together(boxes: Iterable[BoundingBox]) -> bool:
        boxes = list(boxes)
        if not boxes:
            return False

        x1 = max(b.x1 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        x2 = min(b.x2 for b in boxes)
        y2 = min(b.y2 for b in boxes)
        return x2 > x1 and y2 > y1

    def select_regions(self) -> list[RegionNode]:
        if self.mode == RegionSelectionMode.ALL:
            return list(self.regions)

        if self.mode == RegionSelectionMode.CURVE:
            return self.sort_by_curve()

        if self.mode == RegionSelectionMode.NERVE:
            return [r for r in self.regions if len(r.nerve_simplices) > 0]

        raise ValueError(f"Unsupported mode: {self.mode}")

    def to_dicts(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self.regions:
            rows.append(
                {
                    "region_id": r.region_id,
                    "patient_id": r.patient_id,
                    "config_id": r.config.config_id,
                    "filter_name": r.config.filter_name,
                    "use_variance": r.config.use_variance,
                    "variance_mode": r.config.variance_mode,
                    "patch_size": r.config.patch_size,
                    "stride": r.config.stride,
                    "variance_kernel": r.config.variance_kernel,
                    "bbox": (r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2),
                    "centroid": r.centroid,
                    "curve_param": r.curve_param,
                    "order_index": r.order_index,
                    "intersections": sorted(r.intersections),
                    "num_intersections": len(r.intersections),
                    "num_simplices": len(r.nerve_simplices),
                    "lives_in_union": r.lives_in_union,
                    "lives_in_intersection": r.lives_in_intersection,
                    "support_label": r.support_label,
                    "metadata": r.metadata,
                }
            )
        return rows
