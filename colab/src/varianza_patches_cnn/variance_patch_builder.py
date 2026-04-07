import os
import cv2
import numpy as np
import pandas as pd


class VariancePatchBuilderV2:
    def __init__(
        self,
        patch_size=(128, 128),
        variance_ksize=5,
        save_root=None,
        make_subpatches=False,
        subpatch_size=(64, 64),
        subpatch_stride=(64, 64)
    ):
        self.patch_size = tuple(patch_size)
        self.variance_ksize = variance_ksize
        self.save_root = save_root

        self.make_subpatches = make_subpatches
        self.subpatch_size = tuple(subpatch_size)
        self.subpatch_stride = tuple(subpatch_stride)

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

    def extract_subpatches(self, img_2d: np.ndarray):
        sh, sw = self.subpatch_size
        th, tw = self.subpatch_stride
        H, W = img_2d.shape[:2]

        subpatches = []
        for y in range(0, max(1, H - sh + 1), th):
            for x in range(0, max(1, W - sw + 1), tw):
                sub = img_2d[y:y+sh, x:x+sw]
                if sub.shape[0] == sh and sub.shape[1] == sw:
                    subpatches.append((x, y, sub))
        return subpatches

    def save_outputs(self, metadata_df: pd.DataFrame, max_samples=None):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio.")

        img_dir = os.path.join(self.save_root, "saved_patch_img")
        var_dir = os.path.join(self.save_root, "saved_patch_var")
        npy_dir = os.path.join(self.save_root, "saved_patch_concat_npy")
        sub_dir = os.path.join(self.save_root, "saved_subpatches")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(var_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)
        if self.make_subpatches:
            os.makedirs(sub_dir, exist_ok=True)

        rows = metadata_df if max_samples is None else metadata_df.head(max_samples)

        export_rows = []

        for _, row in rows.iterrows():
            patch = self._read_gray(row["image_patch_path"])
            patch = self._resize(patch)
            patch = self._normalize(patch)

            var = self.local_variance_map(patch)
            concat = np.stack([patch, var], axis=0)  # [2,H,W]

            stem = f"{row['sample_id']}_vertebra_{int(row['component_idx']):02d}"

            img_path = os.path.join(img_dir, f"{stem}_img.png")
            var_path = os.path.join(var_dir, f"{stem}_var.png")
            npy_path = os.path.join(npy_dir, f"{stem}_concat.npy")

            cv2.imwrite(img_path, (patch * 255).astype(np.uint8))
            cv2.imwrite(var_path, (var * 255).astype(np.uint8))
            np.save(npy_path, concat)

            subpatch_count = 0
            if self.make_subpatches:
                subs_img = self.extract_subpatches((patch * 255).astype(np.uint8))
                subs_var = self.extract_subpatches((var * 255).astype(np.uint8))

                for k, ((x, y, s_img), (_, _, s_var)) in enumerate(zip(subs_img, subs_var)):
                    cv2.imwrite(os.path.join(sub_dir, f"{stem}_sub{k:02d}_img.png"), s_img)
                    cv2.imwrite(os.path.join(sub_dir, f"{stem}_sub{k:02d}_var.png"), s_var)
                    subpatch_count += 1

            export_rows.append({
                "sample_id": row["sample_id"],
                "component_idx": int(row["component_idx"]),
                "img_path": img_path,
                "var_path": var_path,
                "concat_npy_path": npy_path,
                "num_subpatches": subpatch_count
            })

        export_df = pd.DataFrame(export_rows)
        export_csv = os.path.join(self.save_root, "saved_outputs_metadata.csv")
        export_df.to_csv(export_csv, index=False)
        print(f"Salidas guardadas en: {export_csv}")
        return export_df
