import os
import cv2
import numpy as np

from .vertebra_component_extractor import VertebraComponentExtractor
from .vertebra_region_proxy import VertebraRegionProxy, VertebraRegion


class VertebraRegionBatch:
    """
    Procesa múltiples imágenes y agrupa todas las regiones vertebrales
    con vista por imagen y vista aplanada para CNN.

    Uso:
        batch = VertebraRegionBatch(
            image_paths=["S_100.jpg", "S_101.jpg", ...],
            mask_paths=["mask_S_100.png", "mask_S_101.png", ...],
        )
        batch.process()

        # Vista por imagen
        batch.by_image["S_100"]          # VertebraRegionProxy
        batch.by_image["S_100"][0]       # VertebraRegion

        # Vista aplanada
        batch.all_regions                # [VertebraRegion, ...]
        batch.all_patch_images           # [np.ndarray, ...]

        # Tensor para CNN
        X, Y = batch.to_tensors(target_size=(128, 128))
        origins = batch.region_origins   # ["S_100", "S_100", "S_101", ...]
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        image_names=None,
        # VertebraComponentExtractor params
        min_area=150,
        pad_x=30,
        pad_y=15,
        top_region_ratio=0.35,
        top_pad_x_scale=2,
        top_pad_y_top_scale=3,
        top_pad_y_bottom_scale=0.8,
        # build_adjusted_bboxes params
        pad_x_tight=8,
        pad_y_tight=6,
        top_extra_tight=10,
        # quality filter
        mode="tight",
        include_labels=None,
        save_dir=None,
        **quality_kwargs
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

        if image_names is not None:
            self.image_names = image_names
        else:
            self.image_names = [
                os.path.splitext(os.path.basename(p))[0]
                for p in image_paths
            ]

        self.min_area = min_area
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.top_region_ratio = top_region_ratio
        self.top_pad_x_scale = top_pad_x_scale
        self.top_pad_y_top_scale = top_pad_y_top_scale
        self.top_pad_y_bottom_scale = top_pad_y_bottom_scale

        self.pad_x_tight = pad_x_tight
        self.pad_y_tight = pad_y_tight
        self.top_extra_tight = top_extra_tight

        self.mode = mode
        self.include_labels = include_labels or ["good"]
        self.save_dir = save_dir
        self.quality_kwargs = quality_kwargs

        self.by_image = {}
        self._region_origins = []
        self._failed = []

    def process(self):
        """Ejecuta extracción y clasificación para todas las imágenes."""
        self.by_image = {}
        self._region_origins = []
        self._failed = []

        for img_path, mask_path, name in zip(
            self.image_paths, self.mask_paths, self.image_names
        ):
            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if image is None or mask is None:
                    self._failed.append((name, "imread returned None"))
                    continue

                ext_save_dir = None
                if self.save_dir is not None:
                    ext_save_dir = os.path.join(self.save_dir, name)

                extractor = VertebraComponentExtractor(
                    image=image,
                    local_mask=mask,
                    min_area=self.min_area,
                    pad_x=self.pad_x,
                    pad_y=self.pad_y,
                    top_region_ratio=self.top_region_ratio,
                    top_pad_x_scale=self.top_pad_x_scale,
                    top_pad_y_top_scale=self.top_pad_y_top_scale,
                    top_pad_y_bottom_scale=self.top_pad_y_bottom_scale,
                    save_dir=ext_save_dir,
                ).run()

                extractor.build_adjusted_bboxes(
                    pad_x_tight=self.pad_x_tight,
                    pad_y_tight=self.pad_y_tight,
                    top_extra_tight=self.top_extra_tight,
                )

                proxy = VertebraRegionProxy.from_extractor(
                    extractor,
                    mode=self.mode,
                    include_labels=self.include_labels,
                    **self.quality_kwargs
                )

                self.by_image[name] = proxy

                for _ in proxy.regions:
                    self._region_origins.append(name)

            except Exception as e:
                self._failed.append((name, str(e)))

        return self

    # =========================================================
    # Vista aplanada
    # =========================================================
    @property
    def all_regions(self):
        """Lista aplanada de todas las VertebraRegion de todas las imágenes."""
        regions = []
        for name in self.image_names:
            proxy = self.by_image.get(name)
            if proxy is not None:
                regions.extend(proxy.regions)
        return regions

    @property
    def all_patch_images(self):
        """Lista aplanada de todos los parches (np.ndarray)."""
        return [r.patch_img for r in self.all_regions]

    @property
    def all_patch_masks(self):
        """Lista aplanada de todas las máscaras (np.ndarray)."""
        return [r.patch_mask for r in self.all_regions]

    @property
    def all_bboxes(self):
        """Lista aplanada de todas las bboxes."""
        return [r.bbox for r in self.all_regions]

    @property
    def all_areas(self):
        return np.array([r.area for r in self.all_regions], dtype=np.int32)

    @property
    def all_centroids_y(self):
        return np.array([r.centroid_y for r in self.all_regions], dtype=np.float32)

    @property
    def region_origins(self):
        """Nombre de la imagen de origen para cada región aplanada."""
        return list(self._region_origins)

    # =========================================================
    # Tensor para CNN
    # =========================================================
    def to_tensors(self, target_size=(128, 128)):
        """
        Redimensiona todos los parches y devuelve tensores.

        Returns:
            X: np.ndarray float32 (N_total, H, W, 1) en [0, 1]
            Y: np.ndarray float32 (N_total, H, W, 1) binaria
        """
        regions = self.all_regions
        if len(regions) == 0:
            h, w = target_size
            return (
                np.empty((0, h, w, 1), dtype=np.float32),
                np.empty((0, h, w, 1), dtype=np.float32),
            )

        patches = []
        masks = []
        for r in regions:
            p = cv2.resize(r.patch_img, target_size)
            m = cv2.resize(r.patch_mask, target_size, interpolation=cv2.INTER_NEAREST)
            patches.append(p)
            masks.append(m)

        X = np.array(patches, dtype=np.float32)[..., np.newaxis] / 255.0
        Y = (np.array(masks, dtype=np.float32)[..., np.newaxis] > 0).astype(np.float32)
        return X, Y

    # =========================================================
    # Info
    # =========================================================
    @property
    def total_regions(self):
        return sum(len(p) for p in self.by_image.values())

    @property
    def total_images_processed(self):
        return len(self.by_image)

    @property
    def failed_images(self):
        return list(self._failed)

    def __repr__(self):
        return (
            f"VertebraRegionBatch("
            f"images={self.total_images_processed}, "
            f"regions={self.total_regions}, "
            f"failed={len(self._failed)})"
        )

    def summary(self):
        """Imprime resumen por imagen."""
        print(f"Imágenes procesadas: {self.total_images_processed}")
        print(f"Total regiones: {self.total_regions}")
        print(f"Fallidas: {len(self._failed)}")
        print("-" * 60)

        for name in self.image_names:
            proxy = self.by_image.get(name)
            if proxy is not None:
                labels = {}
                for r in proxy.regions:
                    labels[r.quality_label] = labels.get(r.quality_label, 0) + 1
                parts = ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
                print(f"  {name:>20s}  regions={len(proxy):>3d}  ({parts})")
            else:
                fail_reason = ""
                for fn, reason in self._failed:
                    if fn == name:
                        fail_reason = reason
                        break
                print(f"  {name:>20s}  FAILED: {fail_reason}")
