import cv2
import numpy as np
import pandas as pd

class PatchMetrics:
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
    def compare_consecutive_patches(self, patch_dtos, mode="mask"):
        """
        patch_dtos: lista de DTOs que tengan al menos:
          - patch_id
          - patient_id
          - mask (np.ndarray)
          - bbox (tuple)
        mode: 'mask' o 'box'
        """
        rows = []
        patch_dtos = sorted(patch_dtos, key=lambda p: p.patch_id)
        for i in range(len(patch_dtos) - 1):
            a = patch_dtos[i]
            b = patch_dtos[i + 1]
            if mode == "mask":
                if a.mask is None or b.mask is None:
                    continue
                clean_a, haus_a = self.prepare_for_metrics(a.mask)
                clean_b, haus_b = self.prepare_for_metrics(b.mask)
            elif mode == "box":
                # Crea máscaras de caja
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
            d = self.dice(clean_a, clean_b)
            j = self.iou(clean_a, clean_b)
            h = self.hausdorff(haus_a, haus_b)
            h_norm = self.normalize_hausdorff(h, clean_a.shape)
            rows.append({
                "patient_id": a.patient_id,
                "patch_id_a": a.patch_id,
                "patch_id_b": b.patch_id,
                "dice": d,
                "iou": j,
                "hausdorff": h,
                "hausdorff_norm": h_norm
            })
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
        return {
            "mean_dice": df_metrics["dice"].mean(),
            "mean_iou": df_metrics["iou"].mean(),
            "mean_hausdorff": df_metrics["hausdorff"].mean(),
            "mean_hausdorff_norm": df_metrics["hausdorff_norm"].mean(),
            "max_dice": df_metrics["dice"].max(),
            "max_iou": df_metrics["iou"].max(),
            "min_hausdorff": df_metrics["hausdorff"].min(),
            "max_hausdorff": df_metrics["hausdorff"].max()
        }
