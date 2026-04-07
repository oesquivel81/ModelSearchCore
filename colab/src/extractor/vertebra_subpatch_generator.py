import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class SubpatchRecord:
    study_id: str
    split: str
    vertebra_id: str
    vertebra_idx: int
    subpatch_idx: int
    subpatch_id: str
    grid_row: int
    grid_col: int
    x1: int
    y1: int
    x2: int
    y2: int
    subpatch_img_path: str
    subpatch_mask_path: str


class VertebraSubpatchGenerator:
    def __init__(self, patch_size=(128, 128), subpatch_size=(32, 32),
                 stride=(32, 32), save_root=None):
        self.patch_size = tuple(patch_size)
        self.subpatch_size = tuple(subpatch_size)
        self.stride = tuple(stride)
        self.save_root = save_root

    def _read_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _resize(self, img):
        return cv2.resize(img, self.patch_size, interpolation=cv2.INTER_LINEAR)

    def _extract_windows(self, img):
        H, W = img.shape[:2]
        sh, sw = self.subpatch_size
        th, tw = self.stride

        windows = []
        r = 0
        for y in range(0, H - sh + 1, th):
            c = 0
            for x in range(0, W - sw + 1, tw):
                sub = img[y:y+sh, x:x+sw]
                windows.append((r, c, x, y, x+sw, y+sh, sub))
                c += 1
            r += 1
        return windows

    def generate_all(self, vertebra_csv):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio.")

        os.makedirs(self.save_root, exist_ok=True)
        sub_img_dir = os.path.join(self.save_root, "subpatch_images")
        sub_mask_dir = os.path.join(self.save_root, "subpatch_masks")
        os.makedirs(sub_img_dir, exist_ok=True)
        os.makedirs(sub_mask_dir, exist_ok=True)

        vertebra_df = pd.read_csv(vertebra_csv)
        rows = []

        for _, row in vertebra_df.iterrows():
            study_id = row["study_id"]
            split = row["split"]
            vertebra_id = row["vertebra_id"]
            vertebra_idx = int(row["vertebra_idx"])

            img = self._resize(self._read_gray(row["vertebra_img_path"]))
            mask = self._resize(self._read_gray(row["vertebra_mask_path"]))

            img_windows = self._extract_windows(img)
            mask_windows = self._extract_windows(mask)

            for subpatch_idx, (img_win, mask_win) in enumerate(zip(img_windows, mask_windows)):
                grid_row, grid_col, x1, y1, x2, y2, sub_img = img_win
                _, _, _, _, _, _, sub_mask = mask_win

                subpatch_id = f"{vertebra_id}_sp{subpatch_idx:02d}"

                img_path = os.path.join(sub_img_dir, f"{subpatch_id}.png")
                mask_path = os.path.join(sub_mask_dir, f"{subpatch_id}_mask.png")

                cv2.imwrite(img_path, sub_img)
                cv2.imwrite(mask_path, sub_mask)

                rows.append(SubpatchRecord(
                    study_id=study_id,
                    split=split,
                    vertebra_id=vertebra_id,
                    vertebra_idx=vertebra_idx,
                    subpatch_idx=subpatch_idx,
                    subpatch_id=subpatch_id,
                    grid_row=grid_row,
                    grid_col=grid_col,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    subpatch_img_path=img_path,
                    subpatch_mask_path=mask_path
                ))

        out_df = pd.DataFrame([asdict(r) for r in rows])
        out_csv = os.path.join(self.save_root, "vertebra_subpatches_metadata.csv")
        out_df.to_csv(out_csv, index=False)

        print(f"Subpatches guardados en: {out_csv}")
        print(f"Total subpatches guardados: {len(out_df)}")
        return out_df
