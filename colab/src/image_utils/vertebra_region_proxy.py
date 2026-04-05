import numpy as np

from .vertebra_component_extractor import VertebraComponentExtractor


class VertebraRegion:
    """Contenedor inmutable de una región de vértebra."""

    __slots__ = ("idx", "patch_img", "patch_mask", "bbox", "area", "centroid_x", "centroid_y", "quality_label", "quality_score")

    def __init__(self, idx, patch_img, patch_mask, bbox, area, centroid_x, centroid_y, quality_label, quality_score):
        self.idx = idx
        self.patch_img = patch_img
        self.patch_mask = patch_mask
        self.bbox = bbox
        self.area = area
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.quality_label = quality_label
        self.quality_score = quality_score

    def __repr__(self):
        h, w = self.patch_img.shape[:2]
        return (
            f"VertebraRegion(idx={self.idx}, size={h}x{w}, "
            f"area={self.area}, cy={self.centroid_y:.1f}, "
            f"quality={self.quality_label})"
        )


class VertebraRegionProxy:
    """
    Proxy que envuelve VertebraComponentExtractor y expone las regiones
    buenas como arrays de objetos VertebraRegion listos para CNN.

    Uso:
        proxy = VertebraRegionProxy.from_extractor(component_extractor)
        proxy.regions        # lista de VertebraRegion (solo good)
        proxy.patch_images   # np.ndarray (N, H_i, W_i) lista de arrays
        proxy.patch_masks    # np.ndarray lista
        proxy.bboxes         # list[(x1,y1,x2,y2)]
        proxy.areas          # np.ndarray (N,)
        proxy.centroids_y    # np.ndarray (N,)

        # Para CNN con tamaño fijo:
        X, Y = proxy.to_tensors(target_size=(128, 128))
    """

    def __init__(self, regions):
        self.regions = regions

    @classmethod
    def from_extractor(
        cls,
        extractor,
        mode="tight",
        include_labels=None,
        **quality_kwargs
    ):
        """
        Construye el proxy a partir de un VertebraComponentExtractor ya ejecutado.

        Args:
            extractor: VertebraComponentExtractor con .run() y .build_adjusted_bboxes() ya llamados.
            mode: "tight" o "context" para elegir la bbox.
            include_labels: lista de labels a incluir, ej. ["good"], ["good", "doubtful"].
                            Por defecto solo ["good"].
            **quality_kwargs: parámetros extra para classify_all_components.
        """
        if include_labels is None:
            include_labels = ["good"]

        quality_rows = extractor.classify_all_components(mode=mode, **quality_kwargs)
        quality_map = {r["idx"]: r for r in quality_rows}

        regions = []
        for q in quality_rows:
            if q["label"] not in include_labels:
                continue

            idx = q["idx"]
            comp = extractor.components[idx]

            if mode == "tight":
                bbox = comp.get("bbox_tight", comp["bbox"])
            else:
                bbox = comp.get("bbox_context", comp["bbox"])

            regions.append(VertebraRegion(
                idx=idx,
                patch_img=comp["patch_img"],
                patch_mask=comp["patch_mask"],
                bbox=bbox,
                area=comp["area"],
                centroid_x=comp["centroid_x"],
                centroid_y=comp["centroid_y"],
                quality_label=q["label"],
                quality_score=q["score"],
            ))

        return cls(regions)

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        return self.regions[idx]

    def __iter__(self):
        return iter(self.regions)

    def __repr__(self):
        labels = {}
        for r in self.regions:
            labels[r.quality_label] = labels.get(r.quality_label, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"VertebraRegionProxy(n={len(self.regions)}, {parts})"

    @property
    def patch_images(self):
        """Lista de np.ndarray grayscale, uno por región."""
        return [r.patch_img for r in self.regions]

    @property
    def patch_masks(self):
        """Lista de np.ndarray máscara, uno por región."""
        return [r.patch_mask for r in self.regions]

    @property
    def bboxes(self):
        """Lista de tuplas (x1, y1, x2, y2)."""
        return [r.bbox for r in self.regions]

    @property
    def areas(self):
        """np.ndarray con el área de cada componente."""
        return np.array([r.area for r in self.regions], dtype=np.int32)

    @property
    def centroids_y(self):
        """np.ndarray con centroid_y de cada componente."""
        return np.array([r.centroid_y for r in self.regions], dtype=np.float32)

    @property
    def centroids_x(self):
        """np.ndarray con centroid_x de cada componente."""
        return np.array([r.centroid_x for r in self.regions], dtype=np.float32)

    @property
    def indices(self):
        """Lista de índices originales del extractor."""
        return [r.idx for r in self.regions]

    @property
    def quality_labels(self):
        """Lista de etiquetas de calidad."""
        return [r.quality_label for r in self.regions]

    def to_tensors(self, target_size=(128, 128)):
        """
        Redimensiona todos los parches a un tamaño fijo y devuelve
        tensores listos para CNN.

        Returns:
            X: np.ndarray float32 (N, H, W, 1) normalizado [0, 1]
            Y: np.ndarray float32 (N, H, W, 1) máscara binaria [0, 1]
        """
        import cv2

        patches = []
        masks = []

        for r in self.regions:
            p = cv2.resize(r.patch_img, target_size)
            m = cv2.resize(r.patch_mask, target_size, interpolation=cv2.INTER_NEAREST)
            patches.append(p)
            masks.append(m)

        X = np.array(patches, dtype=np.float32)[..., np.newaxis] / 255.0
        Y = (np.array(masks, dtype=np.float32)[..., np.newaxis] > 0).astype(np.float32)

        return X, Y

    def summary(self):
        """Imprime resumen de las regiones."""
        print(f"Total regiones: {len(self.regions)}")
        for r in self.regions:
            h, w = r.patch_img.shape[:2]
            print(
                f"  [{r.idx:>3}] {r.quality_label:>8} "
                f"score={r.quality_score:>5.1f}  "
                f"size={h}x{w}  "
                f"area={r.area}  "
                f"cy={r.centroid_y:.1f}"
            )
