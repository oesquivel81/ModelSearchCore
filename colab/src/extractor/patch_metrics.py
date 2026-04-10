import cv2
import numpy as np
import pandas as pd

class PatchMetrics:
    def _safe_numeric_array(self, arr):
        """
        Convert input to a clean np.ndarray of float64, removing NaNs.
        Supports pd.DataFrame, pd.Series, np.ndarray, or list.
        """
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            arr = arr.values
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        return arr
    def pad_to_same_shape(self, a, b):
        h = max(a.shape[0], b.shape[0])
        w = max(a.shape[1], b.shape[1])
        aa = np.zeros((h, w), dtype=a.dtype)
        bb = np.zeros((h, w), dtype=b.dtype)
        aa[:a.shape[0], :a.shape[1]] = a
        bb[:b.shape[0], :b.shape[1]] = b
        return aa, bb

    def __init__(self, kernel_size=3, hausdorff_use_edges=True):
        self.kernel_size = kernel_size
        self.hausdorff_use_edges = hausdorff_use_edges

    # =============================
    # LIMPIEZA
    # =============================
    def binarize(self, mask, threshold=0):
        return (mask > threshold).astype(np.uint8)

    def clean_mask(self, mask):
        mask = self.binarize(mask)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = (labels == largest_idx).astype(np.uint8)
        return cleaned

    def extract_edges(self, mask):
        mask_u8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_u8, 50, 150)
        return (edges > 0).astype(np.uint8)

    # =============================
    # MÉTRICAS BÁSICAS
    # =============================
    def dice(self, a, b, eps=1e-8):
        a = (a > 0).astype(np.uint8)
        b = (b > 0).astype(np.uint8)
        inter = np.logical_and(a, b).sum()
        sa = a.sum()
        sb = b.sum()
        return (2.0 * inter) / (sa + sb + eps)

    def iou(self, a, b, eps=1e-8):
        a = (a > 0).astype(np.uint8)
        b = (b > 0).astype(np.uint8)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / (union + eps)

    def mask_points(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def directed_hausdorff(self, A, B):
        if len(A) == 0 or len(B) == 0:
            return np.inf
        max_min = 0.0
        for a in A:
            dists = np.sqrt(((B - a) ** 2).sum(axis=1))
            min_d = dists.min()
            if min_d > max_min:
                max_min = float(min_d)
        return max_min

    def hausdorff(self, a, b):
        A = self.mask_points(a)
        B = self.mask_points(b)
        if len(A) == 0 and len(B) == 0:
            return 0.0
        if len(A) == 0 or len(B) == 0:
            return np.inf
        d_ab = self.directed_hausdorff(A, B)
        d_ba = self.directed_hausdorff(B, A)
        return max(d_ab, d_ba)

    def normalize_hausdorff(self, h, shape):
        hh, ww = shape[:2]
        diag = np.sqrt(hh**2 + ww**2)
        if diag <= 0:
            return np.inf
        return h / diag

    # =============================
    # PREPARACIÓN PARA MÉTRICAS
    # =============================
    def prepare_for_metrics(self, mask):
        cleaned = self.clean_mask(mask)
        if self.hausdorff_use_edges:
            haus_obj = self.extract_edges(cleaned)
        else:
            haus_obj = cleaned
        return cleaned, haus_obj

    # =============================
    # COMPARACIÓN ENTRE PARCHES CONSECUTIVOS
    # =============================
    def compare_consecutive_patches(self, patch_dtos):
        import numpy as np
        import pandas as pd
        rows = []

        if patch_dtos is None or len(patch_dtos) < 2:
            return pd.DataFrame()

        for i in range(len(patch_dtos) - 1):
            a = patch_dtos[i]
            b = patch_dtos[i + 1]

            img_a = np.asarray(a.image)
            img_b = np.asarray(b.image)
            mask_a = np.asarray(a.mask) if a.mask is not None else None
            mask_b = np.asarray(b.mask) if b.mask is not None else None

            # Normalización básica de shapes
            if img_a.ndim == 3:
                img_a_gray = img_a.mean(axis=-1)
            else:
                img_a_gray = img_a

            if img_b.ndim == 3:
                img_b_gray = img_b.mean(axis=-1)
            else:
                img_b_gray = img_b


            row = {
                "idx_a": i,
                "idx_b": i + 1,
                "patch_id_a": getattr(a, "patch_id", f"patch_{i}"),
                "patch_id_b": getattr(b, "patch_id", f"patch_{i+1}"),
            }

            # Estadísticas de intensidad (imagen filtrada)
            row["mean_intensity_a"] = float(np.mean(img_a_gray))
            row["mean_intensity_b"] = float(np.mean(img_b_gray))
            row["std_intensity_a"] = float(np.std(img_a_gray))
            row["std_intensity_b"] = float(np.std(img_b_gray))
            row["mean_intensity_diff"] = abs(row["mean_intensity_a"] - row["mean_intensity_b"])
            row["std_intensity_diff"] = abs(row["std_intensity_a"] - row["std_intensity_b"])

            # Métricas de diferencia de imagen filtrada
            # MSE (Mean Squared Error)
            try:
                min_h = min(img_a_gray.shape[0], img_b_gray.shape[0])
                min_w = min(img_a_gray.shape[1], img_b_gray.shape[1])
                img_a_c = img_a_gray[:min_h, :min_w]
                img_b_c = img_b_gray[:min_h, :min_w]
                mse = float(np.mean((img_a_c - img_b_c) ** 2))
                mae = float(np.mean(np.abs(img_a_c - img_b_c)))
                row["mse_img"] = mse
                row["mae_img"] = mae
            except Exception as e:
                row["mse_img"] = np.nan
                row["mae_img"] = np.nan

            # Métricas geométricas de caja
            box_a = getattr(a, "box", None)
            box_b = getattr(b, "box", None)

            if box_a is not None and box_b is not None:
                xa1, ya1, xa2, ya2 = box_a
                xb1, yb1, xb2, yb2 = box_b

                ca_x = (xa1 + xa2) / 2.0
                ca_y = (ya1 + ya2) / 2.0
                cb_x = (xb1 + xb2) / 2.0
                cb_y = (yb1 + yb2) / 2.0

                row["centroid_distance"] = float(np.sqrt((ca_x - cb_x) ** 2 + (ca_y - cb_y) ** 2))

                area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
                area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
                row["area_a"] = float(area_a)
                row["area_b"] = float(area_b)
                row["area_ratio"] = float(min(area_a, area_b) / max(area_a, area_b)) if max(area_a, area_b) > 0 else np.nan
            else:
                row["centroid_distance"] = np.nan
                row["area_a"] = np.nan
                row["area_b"] = np.nan
                row["area_ratio"] = np.nan

            # Métricas de máscara
            if mask_a is not None and mask_b is not None:
                mask_a_bin = (mask_a > 0).astype(np.uint8)
                mask_b_bin = (mask_b > 0).astype(np.uint8)

                h = min(mask_a_bin.shape[0], mask_b_bin.shape[0])
                w = min(mask_a_bin.shape[1], mask_b_bin.shape[1])
                mask_a_bin = mask_a_bin[:h, :w]
                mask_b_bin = mask_b_bin[:h, :w]

                intersection = np.logical_and(mask_a_bin, mask_b_bin).sum()
                union = np.logical_or(mask_a_bin, mask_b_bin).sum()
                # Explicitly cast to int to avoid overflow/underflow
                sum_a = int(mask_a_bin.sum())
                sum_b = int(mask_b_bin.sum())

                dice = (2.0 * intersection) / (sum_a + sum_b + 1e-8)
                iou = intersection / (union + 1e-8)

                row["dice_mask"] = float(dice)
                row["iou_mask"] = float(iou)
                row["mask_pixels_a"] = float(sum_a)  # Store as float for consistency
                row["mask_pixels_b"] = float(sum_b)
                # Use Python int for subtraction, then abs, then cast to float
                row["mask_pixel_diff"] = float(abs(sum_a - sum_b))
            else:
                row["dice_mask"] = np.nan
                row["iou_mask"] = np.nan
                row["mask_pixels_a"] = np.nan
                row["mask_pixels_b"] = np.nan
                row["mask_pixel_diff"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)

    # =============================
    # MATRIZ DE EMPALMAMIENTO ENTRE TODOS LOS PARCHES
    # =============================
    def compute_overlap_matrix(self, patch_dtos, mode="mask", metric="iou"):
        """
        mode: 'mask' o 'box'
        metric: 'iou', 'dice', 'hausdorff', 'hausdorff_norm'
        """
        n = len(patch_dtos)
        M = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                a = patch_dtos[i]
                b = patch_dtos[j]
                if mode == "mask":
                    if a.mask is None or b.mask is None:
                        M[i, j] = np.nan
                        continue
                    clean_a, haus_a = self.prepare_for_metrics(a.mask)
                    clean_b, haus_b = self.prepare_for_metrics(b.mask)
                    # Igualar tamaño
                    clean_a, clean_b = self.pad_to_same_shape(clean_a, clean_b)
                    haus_a, haus_b = self.pad_to_same_shape(haus_a, haus_b)
                elif mode == "box":
                    shape = a.mask.shape if a.mask is not None else (b.bbox[3]-b.bbox[1], b.bbox[2]-b.bbox[0])
                    clean_a = np.zeros(shape, dtype=np.uint8)
                    clean_b = np.zeros(shape, dtype=np.uint8)
                    x1a, y1a, x2a, y2a = a.bbox
                    x1b, y1b, x2b, y2b = b.bbox
                    clean_a[y1a:y2a, x1a:x2a] = 1
                    clean_b[y1b:y2b, x1b:x2b] = 1
                    haus_a = clean_a
                    haus_b = clean_b
                else:
                    raise ValueError("mode debe ser 'mask' o 'box'")
                if metric == "iou":
                    M[i, j] = self.iou(clean_a, clean_b)
                elif metric == "dice":
                    M[i, j] = self.dice(clean_a, clean_b)
                elif metric == "hausdorff":
                    M[i, j] = self.hausdorff(haus_a, haus_b)
                elif metric == "hausdorff_norm":
                    M[i, j] = self.normalize_hausdorff(self.hausdorff(haus_a, haus_b), clean_a.shape)
                else:
                    raise ValueError("Métrica no soportada")
        return pd.DataFrame(M, index=[p.patch_id for p in patch_dtos], columns=[p.patch_id for p in patch_dtos])

    # =============================
    # RESUMEN DE MÉTRICAS
    # =============================
    def summarize_metrics(self, df_metrics):
        # Usar los nombres de columna correctos según compare_consecutive_patches
        return {
            "mean_dice": df_metrics["dice_mask"].mean(),
            "mean_iou": df_metrics["iou_mask"].mean(),
            "mean_hausdorff": df_metrics["hausdorff"].mean() if "hausdorff" in df_metrics else np.nan,
            "mean_hausdorff_norm": df_metrics["hausdorff_norm"].mean() if "hausdorff_norm" in df_metrics else np.nan,
            "max_dice": df_metrics["dice_mask"].max(),
            "max_iou": df_metrics["iou_mask"].max(),
            "min_hausdorff": df_metrics["hausdorff"].min() if "hausdorff" in df_metrics else np.nan,
            "max_hausdorff": df_metrics["hausdorff"].max() if "hausdorff" in df_metrics else np.nan,
            # Métricas de diferencia de imagen filtrada
            "mean_mse_img": df_metrics["mse_img"].mean() if "mse_img" in df_metrics else np.nan,
            "mean_mae_img": df_metrics["mae_img"].mean() if "mae_img" in df_metrics else np.nan,
            "max_mse_img": df_metrics["mse_img"].max() if "mse_img" in df_metrics else np.nan,
            "max_mae_img": df_metrics["mae_img"].max() if "mae_img" in df_metrics else np.nan,
            "min_mse_img": df_metrics["mse_img"].min() if "mse_img" in df_metrics else np.nan,
            "min_mae_img": df_metrics["mae_img"].min() if "mae_img" in df_metrics else np.nan
        }
