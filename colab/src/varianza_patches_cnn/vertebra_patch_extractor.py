import os
import cv2
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


@dataclass
class PatchRecord:
    sample_id: str
    split: str
    source_image_path: str
    source_mask_path: str
    component_idx: int
    centroid_x: float
    centroid_y: float
    area: int
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    quality_label: str
    image_patch_path: str
    mask_patch_path: str


class VertebraPatchExtractor:
    """
    Extrae componentes conectadas desde una máscara binaria y guarda:
    - patch_img
    - patch_mask
    - metadata CSV
    """

    def __init__(
        self,
        base_dir: str,
        index_csv: str,
        image_col: str = "radiograph_path",
        mask_col: str = "label_binary_path",
        split_col: str = "split",
        min_area: int = 50,
        pad_x: int = 30,
        pad_y: int = 15,
        include_labels: Optional[List[str]] = None,
        save_root: Optional[str] = None,
    ):
        self.base_dir = base_dir
        self.index_csv = index_csv
        self.image_col = image_col
        self.mask_col = mask_col
        self.split_col = split_col
        self.min_area = min_area
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.include_labels = include_labels or ["good", "doubtful", "bad"]
        self.save_root = save_root
        self.components = []

    def _read_gray(self, rel_path: str) -> np.ndarray:
        full = os.path.join(self.base_dir, rel_path)
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer: {full}")
        return img

    def _extract_components(self, image: np.ndarray, mask: np.ndarray) -> List[Dict]:
        binary = (mask > 0).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        h, w = binary.shape
        comps = []

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])

            cx, cy = centroids[i]

            x1 = max(0, x - self.pad_x)
            y1 = max(0, y - self.pad_y)
            x2 = min(w, x + ww + self.pad_x)
            y2 = min(h, y + hh + self.pad_y)

            patch_img = image[y1:y2, x1:x2]
            patch_mask = mask[y1:y2, x1:x2]

            comps.append({
                "component_idx": i,
                "area": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "bbox": (x1, y1, x2, y2),
                "patch_img": patch_img,
                "patch_mask": patch_mask,
            })

        comps = sorted(comps, key=lambda d: d["centroid_y"])
        for idx, comp in enumerate(comps):
            comp["sorted_idx"] = idx
        return comps

    def _classify_quality(self, comp: Dict) -> str:
        """
        Regla simple baseline. La puedes sustituir por tu clasificador de quality.
        """
        x1, y1, x2, y2 = comp["bbox"]
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        occupancy = comp["area"] / bbox_area
        aspect = (x2 - x1) / max(1, (y2 - y1))

        if comp["area"] < 200 or occupancy < 0.08:
            return "bad"
        if 0.12 <= occupancy and 0.35 <= aspect <= 2.8:
            return "good"
        return "doubtful"

    def extract_and_save(self) -> pd.DataFrame:
        df = pd.read_csv(self.index_csv)
        records: List[PatchRecord] = []

        if self.save_root is None:
            raise ValueError("save_root es obligatorio para guardar patches.")

        images_dir = os.path.join(self.save_root, "patch_images")
        masks_dir = os.path.join(self.save_root, "patch_masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        for row_idx, row in df.iterrows():
            try:
                rel_img = row[self.image_col]
                rel_mask = row[self.mask_col]
                split = row[self.split_col] if self.split_col in row else "train"

                image = self._read_gray(rel_img)
                mask = self._read_gray(rel_mask)

                sample_id = os.path.splitext(os.path.basename(rel_img))[0]

                comps = self._extract_components(image=image, mask=mask)
                self.components.extend(comps)

                for comp in comps:
                    quality = self._classify_quality(comp)
                    if quality not in self.include_labels:
                        continue

                    sorted_idx = comp["sorted_idx"]
                    x1, y1, x2, y2 = comp["bbox"]

                    img_name = f"{sample_id}_v{sorted_idx:02d}.png"
                    mask_name = f"{sample_id}_v{sorted_idx:02d}_mask.png"

                    img_path = os.path.join(images_dir, img_name)
                    mask_path = os.path.join(masks_dir, mask_name)

                    cv2.imwrite(img_path, comp["patch_img"])
                    cv2.imwrite(mask_path, comp["patch_mask"])

                    records.append(PatchRecord(
                        sample_id=sample_id,
                        split=split,
                        source_image_path=rel_img,
                        source_mask_path=rel_mask,
                        component_idx=sorted_idx,
                        centroid_x=comp["centroid_x"],
                        centroid_y=comp["centroid_y"],
                        area=comp["area"],
                        bbox_x1=x1,
                        bbox_y1=y1,
                        bbox_x2=x2,
                        bbox_y2=y2,
                        quality_label=quality,
                        image_patch_path=img_path,
                        mask_patch_path=mask_path,
                    ))

            except Exception as e:
                print(f"[WARN] Error procesando row {row_idx}: {e}")

        out_df = pd.DataFrame([asdict(r) for r in records])
        out_csv = os.path.join(self.save_root, "patch_metadata.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"Patch metadata guardado en: {out_csv}")
        print(f"Total patches guardados: {len(out_df)}")
        return out_df

    def save_patch_grid(self, max_patches=16, figsize=(16, 10), grid_path=None):
        """
        Guarda una grilla visual como la imagen que mostraste.
        """
        import matplotlib.pyplot as plt

        n = min(max_patches, len(self.components))
        if n == 0:
            print("No hay componentes.")
            return

        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for i in range(n):
            comp = self.components[i]
            axes[i].imshow(comp["patch_img"], cmap="gray")
            axes[i].set_title(
                f"id={i}\ny={comp['centroid_y']:.1f} | area={comp['area']}"
            )
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        if grid_path is None:
            if self.save_root is None:
                raise ValueError("Define save_root o grid_path.")
            os.makedirs(self.save_root, exist_ok=True)
            grid_path = os.path.join(self.save_root, "patch_grid.png")

        plt.savefig(grid_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Grilla guardada en: {grid_path}")

    def save_patches_with_metadata(self, sample_id="sample"):
        """
        Guarda patches, máscaras y metadata CSV para luego construir dataset.
        """
        if self.save_root is None:
            raise ValueError("Define save_root para guardar patches.")

        os.makedirs(self.save_root, exist_ok=True)

        img_dir = os.path.join(self.save_root, "patch_images")
        mask_dir = os.path.join(self.save_root, "patch_masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        rows = []

        for i, comp in enumerate(self.components):
            img_name = f"{sample_id}_vertebra_{i:02d}.png"
            mask_name = f"{sample_id}_vertebra_{i:02d}_mask.png"

            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)

            cv2.imwrite(img_path, comp["patch_img"])
            cv2.imwrite(mask_path, comp["patch_mask"])

            x1, y1, x2, y2 = comp["bbox"]

            rows.append({
                "sample_id": sample_id,
                "component_idx": i,
                "centroid_x": comp["centroid_x"],
                "centroid_y": comp["centroid_y"],
                "area": comp["area"],
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "image_patch_path": img_path,
                "mask_patch_path": mask_path,
            })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.save_root, f"{sample_id}_patch_metadata.csv")
        df.to_csv(csv_path, index=False)

        print(f"Metadata guardada en: {csv_path}")
        return df
