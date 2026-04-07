import os
import cv2
import numpy as np
import pandas as pd


class VariancePatchProcessor:
    """
    Construye:
    - baseline: imagen sola
    - variance_input: concat(img, var) en 2 canales
    - variance_branch: devuelve imagen y var por separado
    Además puede guardar previews.
    """

    def __init__(
        self,
        patch_size=(128, 128),
        variance_ksize=5,
        save_root=None
    ):
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

    def local_variance_map(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        mean = cv2.blur(img, (self.variance_ksize, self.variance_ksize))
        mean_sq = cv2.blur(img * img, (self.variance_ksize, self.variance_ksize))
        var = mean_sq - mean * mean
        var = np.clip(var, 0.0, None)

        if var.max() > 0:
            var = var / (var.max() + 1e-8)
        return var.astype(np.float32)

    def build_baseline_tensor(self, patch_img: np.ndarray) -> np.ndarray:
        img = self._resize(patch_img).astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return img[None, ...]  # [1,H,W]

    def build_variance_input_tensor(self, patch_img: np.ndarray) -> np.ndarray:
        img = self._resize(patch_img).astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        var = self.local_variance_map(img)
        x = np.stack([img, var], axis=0)  # [2,H,W]
        return x.astype(np.float32)

    def build_variance_branch_tensor(self, patch_img: np.ndarray):
        img = self._resize(patch_img).astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        var = self.local_variance_map(img)
        return img[None, ...].astype(np.float32), var[None, ...].astype(np.float32)

    def save_preview_triplets(self, metadata_df: pd.DataFrame, max_samples=50):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio para previews.")

        out_dir = os.path.join(self.save_root, "variance_previews")
        os.makedirs(out_dir, exist_ok=True)

        for i, row in metadata_df.head(max_samples).iterrows():
            img = self._read_gray(row["image_patch_path"])
            img_r = self._resize(img).astype(np.float32)
            if img_r.max() > 1.0:
                img_r = img_r / 255.0
            var = self.local_variance_map(img_r)

            img_u8 = (img_r * 255).astype(np.uint8)
            var_u8 = (var * 255).astype(np.uint8)

            panel = np.concatenate([img_u8, var_u8], axis=1)
            save_name = f"{row['sample_id']}_v{int(row['component_idx']):02d}_preview.png"
            cv2.imwrite(os.path.join(out_dir, save_name), panel)
