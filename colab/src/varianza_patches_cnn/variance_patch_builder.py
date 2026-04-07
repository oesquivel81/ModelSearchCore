import os
import cv2
import numpy as np
import pandas as pd


class VariancePatchBuilder:
    """
    Construye representaciones:
    - baseline
    - variance_input
    - variance_branch

    También guarda previews:
    [imagen | varianza]
    """

    def __init__(self, patch_size=(128, 128), variance_ksize=5, save_root=None):
        self.patch_size = tuple(patch_size)
        self.variance_ksize = variance_ksize
        self.save_root = save_root

    def _read_gray(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.patch_size, interpolation=cv2.INTER_LINEAR)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return img

    def local_variance_map(self, img: np.ndarray) -> np.ndarray:
        img = self._normalize(img)

        mean = cv2.blur(img, (self.variance_ksize, self.variance_ksize))
        mean_sq = cv2.blur(img * img, (self.variance_ksize, self.variance_ksize))
        var = mean_sq - mean * mean
        var = np.clip(var, 0.0, None)

        if var.max() > 0:
            var = var / (var.max() + 1e-8)

        return var.astype(np.float32)

    def build_baseline(self, patch_img: np.ndarray) -> np.ndarray:
        img = self._normalize(self._resize(patch_img))
        return img[None, ...]

    def build_variance_input(self, patch_img: np.ndarray) -> np.ndarray:
        img = self._normalize(self._resize(patch_img))
        var = self.local_variance_map(img)
        x = np.stack([img, var], axis=0)
        return x.astype(np.float32)

    def build_variance_branch(self, patch_img: np.ndarray):
        img = self._normalize(self._resize(patch_img))
        var = self.local_variance_map(img)
        return img[None, ...].astype(np.float32), var[None, ...].astype(np.float32)

    def save_previews(self, metadata_df: pd.DataFrame, max_samples=100):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio para save_previews.")

        out_dir = os.path.join(self.save_root, "variance_previews")
        os.makedirs(out_dir, exist_ok=True)

        for _, row in metadata_df.head(max_samples).iterrows():
            patch = self._read_gray(row["image_patch_path"])
            patch = self._resize(patch)
            patch = self._normalize(patch)
            var = self.local_variance_map(patch)

            patch_u8 = (patch * 255).astype(np.uint8)
            var_u8 = (var * 255).astype(np.uint8)

            panel = np.concatenate([patch_u8, var_u8], axis=1)

            save_name = f"{row['sample_id']}_vertebra_{int(row['component_idx']):02d}_preview.png"
            cv2.imwrite(os.path.join(out_dir, save_name), panel)
