import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class VertebraRecord:
    study_id: str
    split: str
    vertebra_idx: int
    vertebra_id: str
    centroid_x: float
    centroid_y: float
    area: int
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    vertebra_img_path: str
    vertebra_mask_path: str


class VertebraRegionExtractor:
    def __init__(self, base_dir, image_col="radiograph_path", mask_col="label_binary_path",
                 min_area=50, pad_x=30, pad_y=15, save_root=None):
        self.base_dir = base_dir
        self.image_col = image_col
        self.mask_col = mask_col
        self.min_area = min_area
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.save_root = save_root

    def _read_gray_rel(self, rel_path):
        full = os.path.join(self.base_dir, rel_path)
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(full)
        return img

    def _extract_components(self, image, mask):
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

            comps.append({
                "area": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "bbox": (x1, y1, x2, y2),
            })

        comps = sorted(comps, key=lambda d: d["centroid_y"])
        return comps

    def extract_all(self, index_csv, split_csv):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio.")

        os.makedirs(self.save_root, exist_ok=True)
        vertebra_img_dir = os.path.join(self.save_root, "vertebra_images")
        vertebra_mask_dir = os.path.join(self.save_root, "vertebra_masks")
        os.makedirs(vertebra_img_dir, exist_ok=True)
        os.makedirs(vertebra_mask_dir, exist_ok=True)

        df = pd.read_csv(index_csv).copy()
        split_df = pd.read_csv(split_csv).copy()

        df["study_id"] = df[self.image_col].apply(lambda p: Path(p).stem)
        df = df.merge(split_df[["study_id", "split"]], on="study_id", how="left")

        rows = []

        for _, row in df.iterrows():
            study_id = row["study_id"]
            split = row["split"]

            image = self._read_gray_rel(row[self.image_col])
            mask = self._read_gray_rel(row[self.mask_col])

            comps = self._extract_components(image, mask)

            for vertebra_idx, comp in enumerate(comps):
                x1, y1, x2, y2 = comp["bbox"]

                vertebra_img = image[y1:y2, x1:x2]
                vertebra_mask = mask[y1:y2, x1:x2]

                vertebra_id = f"{study_id}_v{vertebra_idx:02d}"

                img_path = os.path.join(vertebra_img_dir, f"{vertebra_id}.png")
                mask_path = os.path.join(vertebra_mask_dir, f"{vertebra_id}_mask.png")

                cv2.imwrite(img_path, vertebra_img)
                cv2.imwrite(mask_path, vertebra_mask)

                rows.append(VertebraRecord(
                    study_id=study_id,
                    split=split,
                    vertebra_idx=vertebra_idx,
                    vertebra_id=vertebra_id,
                    centroid_x=comp["centroid_x"],
                    centroid_y=comp["centroid_y"],
                    area=comp["area"],
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    vertebra_img_path=img_path,
                    vertebra_mask_path=mask_path
                ))

        out_df = pd.DataFrame([asdict(r) for r in rows])
        out_csv = os.path.join(self.save_root, "vertebra_regions_metadata.csv")
        out_df.to_csv(out_csv, index=False)

        print(f"Regiones vertebrales guardadas en: {out_csv}")
        print(f"Total vértebras guardadas: {len(out_df)}")
        return out_df


def build_study_split(index_csv, seed=42, image_col="radiograph_path",
                      train_size=0.70, val_size=0.15, test_size=0.15):
    df = pd.read_csv(index_csv)
    study_ids = df[image_col].apply(lambda p: Path(p).stem).unique().tolist()

    rng = np.random.RandomState(seed)
    rng.shuffle(study_ids)

    n = len(study_ids)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    splits = []
    for i in range(n):
        if i < n_train:
            splits.append("train")
        elif i < n_train + n_val:
            splits.append("val")
        else:
            splits.append("test")

    return pd.DataFrame({"study_id": study_ids, "split": splits})


def save_study_split(split_df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    split_df.to_csv(path, index=False)
    return path
