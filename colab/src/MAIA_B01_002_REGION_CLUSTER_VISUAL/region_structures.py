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
