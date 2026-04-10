# =============================
# SubregionMetrics
# =============================
import os
import cv2
import numpy as np
import pandas as pd
from extractor.patch_dto import PatchDTO, PatchPathDTO, PatchDTOBuilder

# =============================
# Visualización de parches
# =============================
from extractor.patch_viz import show_patches


class SubregionMetrics:
    @staticmethod
    def curve_distance_mean(centerlines_a, centerlines_b):
        """
        Calcula la distancia promedio entre dos curvas (listas de puntos (x, y)),
        interpolando sobre el eje y para que tengan la misma cantidad de puntos.
        """
        if len(centerlines_a) == 0 or len(centerlines_b) == 0:
            return np.nan
        # Interpolación sobre y
        ys_a = np.array([p[1] for p in centerlines_a])
        xs_a = np.array([p[0] for p in centerlines_a])
        ys_b = np.array([p[1] for p in centerlines_b])
        xs_b = np.array([p[0] for p in centerlines_b])

        y_min = max(ys_a.min(), ys_b.min())
        y_max = min(ys_a.max(), ys_b.max())
        n_points = 50
        target_ys = np.linspace(y_min, y_max, n_points)

        interp_xs_a = np.interp(target_ys, ys_a, xs_a)
        interp_xs_b = np.interp(target_ys, ys_b, xs_b)

        dists = np.abs(interp_xs_a - interp_xs_b)
        return float(np.mean(dists))

    def report_experiment_metrics(self, image, mask, boxes, centerline_a=None, centerline_b=None):
        """
        Calcula y retorna un dict con las métricas estándar para un experimento.
        Si se proveen dos centerlines, calcula curve_distance_mean.
        """
        df_neighbors = self.consecutive_metrics(image, mask, boxes)
        report = {
            "mean_iou_box_neighbors": df_neighbors["iou_box"].mean(),
            "mean_dice_mask_neighbors": df_neighbors["dice_crop"].mean(),
            "mean_hausdorff_mask_norm_neighbors": df_neighbors["hausdorff_crop"].mean(),
            "max_iou_box": df_neighbors["iou_box"].max(),
            "max_dice_mask": df_neighbors["dice_crop"].max(),
        }
        if centerline_a is not None and centerline_b is not None:
            report["curve_distance_mean"] = self.curve_distance_mean(centerline_a, centerline_b)
        else:
            report["curve_distance_mean"] = np.nan
            print(report)  # Existing print statement for report

            # Visualización de parches
            from extractor.patch_viz import show_patches
            show_patches(patch_dtos)
            return report

    def __init__(self):
        pass

    # =========================================================
    # BÁSICOS
    # =========================================================
    @staticmethod
    def binarize(mask):
        return (mask > 0).astype(np.uint8)

    @staticmethod
    def make_box_mask(shape, bbox):
        """
        shape: (h, w)
        bbox: (x1, y1, x2, y2)
        """
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox
        out = np.zeros((h, w), dtype=np.uint8)
        out[y1:y2, x1:x2] = 1
        return out

    @staticmethod
    def crop_mask(mask, bbox):
        x1, y1, x2, y2 = bbox
        return mask[y1:y2, x1:x2]

    @staticmethod
    def pad_to_same_shape(a, b):
        h = max(a.shape[0], b.shape[0])
        w = max(a.shape[1], b.shape[1])

        aa = np.zeros((h, w), dtype=a.dtype)
        bb = np.zeros((h, w), dtype=b.dtype)

        aa[:a.shape[0], :a.shape[1]] = a
        bb[:b.shape[0], :b.shape[1]] = b
        return aa, bb

    # =========================================================
    # DICE / IOU
    # =========================================================
    @staticmethod
    def dice(mask_a, mask_b, eps=1e-8):
        a = (mask_a > 0).astype(np.uint8)
        b = (mask_b > 0).astype(np.uint8)

        inter = np.logical_and(a, b).sum()
        sa = a.sum()
        sb = b.sum()

        return (2.0 * inter) / (sa + sb + eps)

    @staticmethod
    def iou(mask_a, mask_b, eps=1e-8):
        a = (mask_a > 0).astype(np.uint8)
        b = (mask_b > 0).astype(np.uint8)

        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()

        return inter / (union + eps)

    # =========================================================
    # HAUSDORFF
    # =========================================================
    @staticmethod
    def mask_points(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    @staticmethod
    def directed_hausdorff_points(A, B):
        """
        A, B: arrays Nx2 y Mx2
        """
        if len(A) == 0 or len(B) == 0:
            return np.inf

        max_min_dist = 0.0
        for a in A:
            dists = np.sqrt(((B - a) ** 2).sum(axis=1))
            min_dist = dists.min()
            if min_dist > max_min_dist:
                max_min_dist = float(min_dist)
        return max_min_dist

    @classmethod
    def hausdorff(cls, mask_a, mask_b):
        a = (mask_a > 0).astype(np.uint8)
        b = (mask_b > 0).astype(np.uint8)

        A = cls.mask_points(a)
        B = cls.mask_points(b)

        if len(A) == 0 and len(B) == 0:
            return 0.0
        if len(A) == 0 or len(B) == 0:
            return np.inf

        d_ab = cls.directed_hausdorff_points(A, B)
        d_ba = cls.directed_hausdorff_points(B, A)

        return max(d_ab, d_ba)

    # =========================================================
    # MÉTRICAS ENTRE CAJAS
    # =========================================================
    def metrics_box_vs_box(self, image_shape, bbox_a, bbox_b):
        """
        Compara dos cajas rasterizándolas como máscaras en la imagen completa.
        """
        box_a = self.make_box_mask(image_shape[:2], bbox_a)
        box_b = self.make_box_mask(image_shape[:2], bbox_b)

        return {
            "dice_box": self.dice(box_a, box_b),
            "iou_box": self.iou(box_a, box_b),
            "hausdorff_box": self.hausdorff(box_a, box_b)
        }

    # =========================================================
    # MÉTRICAS ENTRE MÁSCARAS RECORTADAS
    # =========================================================
    def metrics_crop_vs_crop(self, full_mask, bbox_a, bbox_b):
        """
        Recorta la máscara completa en ambas cajas y compara sus contenidos.
        """
        crop_a = self.crop_mask(full_mask, bbox_a)
        crop_b = self.crop_mask(full_mask, bbox_b)

        crop_a = self.binarize(crop_a)
        crop_b = self.binarize(crop_b)

        crop_a, crop_b = self.pad_to_same_shape(crop_a, crop_b)

        return {
            "dice_crop": self.dice(crop_a, crop_b),
            "iou_crop": self.iou(crop_a, crop_b),
            "hausdorff_crop": self.hausdorff(crop_a, crop_b)
        }

    # =========================================================
    # MÉTRICAS ENTRE SUBREGIONES CONSECUTIVAS
    # =========================================================
    def consecutive_metrics(self, image, full_mask, boxes):
        """
        boxes: lista de dicts con:
        {
            "vertebra_idx": int,
            "bbox": (x1, y1, x2, y2),
            ...
        }
        """
        rows = []

        boxes = sorted(boxes, key=lambda d: d["vertebra_idx"])

        for i in range(len(boxes) - 1):
            a = boxes[i]
            b = boxes[i + 1]

            bbox_a = a["bbox"]
            bbox_b = b["bbox"]

            box_metrics = self.metrics_box_vs_box(
                image_shape=image.shape,
                bbox_a=bbox_a,
                bbox_b=bbox_b
            )

            crop_metrics = self.metrics_crop_vs_crop(
                full_mask=full_mask,
                bbox_a=bbox_a,
                bbox_b=bbox_b
            )

            rows.append({
                "idx_a": a["vertebra_idx"],
                "idx_b": b["vertebra_idx"],
                **box_metrics,
                **crop_metrics
            })

        return pd.DataFrame(rows)

    # =========================================================
    # MATRIZ COMPLETA ENTRE TODAS LAS SUBREGIONES
    # =========================================================
    def pairwise_matrix(self, image, full_mask, boxes, metric="iou_box"):
        """
        metric:
        - dice_box
        - iou_box
        - hausdorff_box
        - dice_crop
        - iou_crop
        - hausdorff_crop
        """
        boxes = sorted(boxes, key=lambda d: d["vertebra_idx"])
        n = len(boxes)

        M = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                bbox_i = boxes[i]["bbox"]
                bbox_j = boxes[j]["bbox"]

                box_metrics = self.metrics_box_vs_box(
                    image_shape=image.shape,
                    bbox_a=bbox_i,
                    bbox_b=bbox_j
                )
                crop_metrics = self.metrics_crop_vs_crop(
                    full_mask=full_mask,
                    bbox_a=bbox_i,
                    bbox_b=bbox_j
                )

                all_metrics = {**box_metrics, **crop_metrics}
                M[i, j] = all_metrics[metric]

        idxs = [b["vertebra_idx"] for b in boxes]
        return pd.DataFrame(M, index=idxs, columns=idxs)

    # =========================================================
    # EXTRA: OVERLAP PURO DE INTERSECCIÓN DE CAJAS
    # =========================================================
    @staticmethod
    def intersection_area(bbox_a, bbox_b):
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        return iw * ih

    @staticmethod
    def bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def overlap_ratio(self, bbox_a, bbox_b, mode="iou"):
        inter = self.intersection_area(bbox_a, bbox_b)
        area_a = self.bbox_area(bbox_a)
        area_b = self.bbox_area(bbox_b)

        if mode == "iou":
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        if mode == "over_a":
            return inter / area_a if area_a > 0 else 0.0

        if mode == "over_b":
            return inter / area_b if area_b > 0 else 0.0

        raise ValueError("mode debe ser 'iou', 'over_a' o 'over_b'")
import itertools
import matplotlib.pyplot as plt

# =============================
# VertebraAutoGridRunner
# =============================
class VertebraAutoGridRunner:
    def __init__(self, extractor):
        self.extractor = extractor

    def run_grid(self, index_csv, split_csv, config_json, image_rel_path, mask_rel_path):
        """
        config_json: dict con estructura:
        {
            "bands": {
                "n_levels": [9],
                "box_w": [130, 140, 150],
                "box_h": [80, 90, 95],
                "adaptive_width": [True]
            },
            "centerline": {
                "n_levels": [9],
                "smooth_win": [15, 21, 25],
                "box_w": [130, 140, 150],
                "box_h": [80, 90, 95],
                "adaptive_width": [True]
            }
        }
        """
        results = []
        for method in ["bands", "centerline"]:
            if method not in config_json:
                continue
            param_grid = config_json[method]
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                params["method"] = method
                # Preview single para visualización
                _, _, _, _, vis = self.extractor.preview_single(
                    image_rel_path=image_rel_path,
                    mask_rel_path=mask_rel_path,
                    method=method,
                    n_levels=params.get("n_levels", 9),
                    box_w=params.get("box_w"),
                    box_h=params.get("box_h"),
                    adaptive_width=params.get("adaptive_width", True),
                    smooth_win=params.get("smooth_win", 21)
                )
                results.append((method, params, vis))
        return results

    def plot_grid(self, results, max_cols=3):
        n = len(results)
        ncols = min(max_cols, n)
        nrows = (n + ncols - 1) // ncols
        plt.figure(figsize=(5 * ncols, 5 * nrows))
        for i, (method, params, vis) in enumerate(results):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(vis[..., ::-1])
            plt.title(f"{method}\n{params}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

# =============================
# VertebraAutoCentroidExtractor
# =============================
from dataclasses import dataclass, asdict

@dataclass
class VertebraAutoBoxRecord:
    study_id: str
    split: str
    vertebra_idx: int
    vertebra_id: str
    method: str
    centroid_x: float
    centroid_y: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    vertebra_img_path: str
    vertebra_mask_path: str


class VertebraAutoCentroidExtractor:
    def concat_channels(self, image, variance_map):
        """
        Combina la imagen y el mapa de varianza como canales (axis=-1).
        Ambos deben tener la misma forma espacial.
        Devuelve un array shape (H, W, 2) tipo float32.
        """
        import numpy as np
        if image.shape != variance_map.shape:
            raise ValueError(f"Las formas no coinciden: image {image.shape}, variance_map {variance_map.shape}")
        stacked = np.stack([image.astype(np.float32), variance_map.astype(np.float32)], axis=-1)
        return stacked

    def compute_local_variance(self, image, kernel_size=5):
        """
        Calcula el mapa de varianza local usando un filtro de ventana cuadrada de tamaño kernel_size.
        """
        import cv2
        import numpy as np
        if kernel_size is None:
            kernel_size = 5
        # Normaliza a float32
        img = image.astype(np.float32)
        mean = cv2.blur(img, (kernel_size, kernel_size))
        mean_sq = cv2.blur(img * img, (kernel_size, kernel_size))
        var = mean_sq - mean * mean
        var = np.clip(var, 0.0, None)
        if var.max() > 0:
            var = var / (var.max() + 1e-8)
        return var.astype(np.float32)
    def apply_filter(self, image, filter_name: str):
        """
        Aplica una secuencia de filtros separados por '+' sobre la imagen.
        """
        if not filter_name or filter_name == "none":
            return image
        filters = filter_name.split("+")
        out = image.copy()
        for f in filters:
            out = self.apply_single_filter(out, f)
        return out

    def apply_single_filter(self, image, filter_name: str):
        """
        Aplica un filtro individual sobre la imagen.
        """
        import cv2
        import numpy as np
        if filter_name == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif filter_name == "median":
            return cv2.medianBlur(image, 5)
        elif filter_name == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif filter_name == "sobel":
            return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        elif filter_name == "scharr":
            return cv2.Scharr(image, cv2.CV_64F, 1, 0)
        elif filter_name == "prewitt":
            kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            return cv2.filter2D(image, -1, kernelx)
        elif filter_name == "laplacian":
            return cv2.Laplacian(image, cv2.CV_64F)
        elif filter_name == "log":
            blur = cv2.GaussianBlur(image, (3,3), 0)
            return cv2.Laplacian(blur, cv2.CV_64F)
        elif filter_name == "canny":
            return cv2.Canny(image, 100, 200)
        elif filter_name == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image)
        elif filter_name == "unsharp_mask":
            gaussian = cv2.GaussianBlur(image, (9,9), 10.0)
            return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        elif filter_name == "local_variance":
            return cv2.blur(np.square(image), (5,5)) - np.square(cv2.blur(image, (5,5)))
        else:
            print(f"[WARN] Filtro desconocido: {filter_name}, se retorna la imagen original.")
            return image

# ---
# Calcula centroides automáticamente desde máscara binaria con dos métodos:
#
# 1) bands:
#    Divide la columna en bandas verticales y calcula un centroide por banda.
#
# 2) centerline:
#    Obtiene una línea central por filas, la suaviza y toma puntos sobre ella.

    def __init__(
        self,
        base_dir,
        image_col="radiograph_path",
        mask_col="label_binary_path",
        save_root=None,
        min_row_pixels=5,
        default_box_w=140,
        default_box_h=90,
        overlap_y=0.20,
        pad_x=20,
        pad_y=10
    ):
        self.base_dir = base_dir
        self.image_col = image_col
        self.mask_col = mask_col
        self.save_root = save_root
        self.min_row_pixels = min_row_pixels
        self.default_box_w = default_box_w
        self.default_box_h = default_box_h
        self.overlap_y = overlap_y
        self.pad_x = pad_x
        self.pad_y = pad_y

    # =========================================================
    # UTILIDADES
    # =========================================================
    def _read_gray_rel(self, rel_path):
        full = os.path.join(self.base_dir, rel_path)
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(full)
        return img

    def _ensure_dirs(self):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio.")

        os.makedirs(self.save_root, exist_ok=True)

        img_dir = os.path.join(self.save_root, "vertebra_images")
        mask_dir = os.path.join(self.save_root, "vertebra_masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        return img_dir, mask_dir

    def _clip_box(self, x1, y1, x2, y2, w, h):
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2)))
        y2 = min(h, int(round(y2)))
        return x1, y1, x2, y2

    def _build_box_from_center(self, cx, cy, box_w, box_h, w, h):
        x1 = cx - box_w / 2
        y1 = cy - box_h / 2
        x2 = cx + box_w / 2
        y2 = cy + box_h / 2
        return self._clip_box(x1, y1, x2, y2, w, h)

    def _mask_bbox(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    # =========================================================
    # MÉTODO 1: BANDS
    # =========================================================
    def _compute_centroids_by_bands(self, mask, n_levels=9):
        """
        Divide el bbox vertical de la máscara en bandas
        y calcula un centroide por banda.
        """
        binary = (mask > 0).astype(np.uint8)
        bbox = self._mask_bbox(binary)
        if bbox is None:
            return []

        x_min, y_min, x_max, y_max = bbox
        total_h = max(1, y_max - y_min + 1)
        band_h = total_h / max(n_levels, 1)

        centroids = []

        for idx in range(n_levels):
            by1 = int(round(y_min + idx * band_h))
            by2 = int(round(y_min + (idx + 1) * band_h))
            by2 = min(by2, binary.shape[0])

            band = binary[by1:by2, :]
            ys, xs = np.where(band > 0)

            if len(xs) == 0:
                continue

            cx = float(xs.mean())
            cy = float(ys.mean() + by1)

            centroids.append({
                "vertebra_idx": idx,
                "centroid_x": cx,
                "centroid_y": cy
            })

        centroids = sorted(centroids, key=lambda d: d["centroid_y"])
        for i, c in enumerate(centroids):
            c["vertebra_idx"] = i

        return centroids

    # =========================================================
    # MÉTODO 2: CENTERLINE
    # =========================================================
    def _extract_centerline_points(self, mask):
        """
        Para cada fila con suficientes píxeles activos,
        toma el promedio en x.
        """
        binary = (mask > 0).astype(np.uint8)
        h, _ = binary.shape

        pts = []
        for y in range(h):
            xs = np.where(binary[y] > 0)[0]
            if len(xs) >= self.min_row_pixels:
                cx = float(xs.mean())
                pts.append((cx, float(y)))

        return pts

    def _smooth_centerline(self, pts, win=21):
        if len(pts) == 0:
            return []

        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)

        if len(xs) < win:
            return list(zip(xs, ys))

        kernel = np.ones(win, dtype=np.float32) / win
        xs_smooth = np.convolve(xs, kernel, mode="same")
        return list(zip(xs_smooth, ys))

    def _compute_centroids_by_centerline(self, mask, n_levels=9, smooth_win=21):
        """
        Obtiene puntos sobre la línea central suavizada.
        """
        pts = self._extract_centerline_points(mask)
        pts = self._smooth_centerline(pts, win=smooth_win)

        if len(pts) == 0:
            return []

        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)

        y_min = float(np.min(ys))
        y_max = float(np.max(ys))

        target_ys = np.linspace(y_min, y_max, n_levels)

        centroids = []
        for idx, ty in enumerate(target_ys):
            nearest_idx = int(np.argmin(np.abs(ys - ty)))
            cx = float(xs[nearest_idx])
            cy = float(ys[nearest_idx])

            centroids.append({
                "vertebra_idx": idx,
                "centroid_x": cx,
                "centroid_y": cy
            })

        centroids = sorted(centroids, key=lambda d: d["centroid_y"])
        for i, c in enumerate(centroids):
            c["vertebra_idx"] = i

        return centroids

    # =========================================================
    # CAJAS
    # =========================================================
    def _boxes_from_centroids(self, mask, centroids, box_w=None, box_h=None, adaptive_width=True):
        h, w = mask.shape[:2]

        if box_w is None:
            box_w = self.default_box_w
        if box_h is None:
            box_h = self.default_box_h

        binary = (mask > 0).astype(np.uint8)
        boxes = []

        for item in centroids:
            cx = item["centroid_x"]
            cy = item["centroid_y"]
            idx = item["vertebra_idx"]

            local_box_w = box_w
            if adaptive_width:
                yy = int(np.clip(round(cy), 0, h - 1))
                row_xs = np.where(binary[yy] > 0)[0]
                if len(row_xs) > 5:
                    row_w = int(row_xs.max() - row_xs.min() + 1)
                    local_box_w = max(box_w, row_w + 2 * self.pad_x)

            x1, y1, x2, y2 = self._build_box_from_center(
                cx=cx,
                cy=cy,
                box_w=local_box_w,
                box_h=box_h,
                w=w,
                h=h
            )

            boxes.append({
                "vertebra_idx": idx,
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox": (x1, y1, x2, y2)
            })

        return boxes

    # =========================================================
    # VISUALIZACIÓN
    # =========================================================
    def draw_boxes(self, image, boxes, color=(255, 255, 0), thickness=2, draw_idx=True):
        """
        color en BGR para OpenCV.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            cx = int(round(item["centroid_x"]))
            cy = int(round(item["centroid_y"]))
            idx = item["vertebra_idx"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

            if draw_idx:
                cv2.putText(
                    vis,
                    str(idx),
                    (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
        return vis

    # =========================================================
    # GUARDADO
    # =========================================================
    def _save_boxes_for_study(
        self,
        study_id,
        split,
        image,
        mask,
        boxes,
        method,
        img_dir,
        mask_dir
    ):
        rows = []

        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            vertebra_idx = item["vertebra_idx"]

            vertebra_img = image[y1:y2, x1:x2]
            vertebra_mask = mask[y1:y2, x1:x2]

            vertebra_id = f"{study_id}_v{vertebra_idx:02d}"

            img_path = os.path.join(img_dir, f"{vertebra_id}.png")
            mask_path = os.path.join(mask_dir, f"{vertebra_id}_mask.png")

            cv2.imwrite(img_path, vertebra_img)
            cv2.imwrite(mask_path, vertebra_mask)

            rows.append(VertebraAutoBoxRecord(
                study_id=study_id,
                split=split,
                vertebra_idx=vertebra_idx,
                vertebra_id=vertebra_id,
                method=method,
                centroid_x=item["centroid_x"],
                centroid_y=item["centroid_y"],
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                vertebra_img_path=img_path,
                vertebra_mask_path=mask_path
            ))

        return rows

    # =========================================================
    # API PRINCIPAL
    # =========================================================
    def extract_all_auto(
        self,
        index_csv,
        split_csv,
        method="bands",
        n_levels=9,
        box_w=None,
        box_h=None,
        adaptive_width=True,
        smooth_win=21
    ):
        """
        method:
        - "bands"
        - "centerline"
        """
        img_dir, mask_dir = self._ensure_dirs()

        df = pd.read_csv(index_csv).copy()
        split_df = pd.read_csv(split_csv).copy()

        df["study_id"] = df[self.image_col].apply(lambda p: Path(p).stem)
        if "split" in df.columns:
            df = df.drop(columns=["split"])
        df = df.merge(split_df[["study_id", "split"]], on="study_id", how="left")

        rows = []

        for _, row in df.iterrows():
            study_id = row["study_id"]
            split = row["split"]

            image = self._read_gray_rel(row[self.image_col])
            mask = self._read_gray_rel(row[self.mask_col])

            if method == "bands":
                centroids = self._compute_centroids_by_bands(
                    mask=mask,
                    n_levels=n_levels
                )
            elif method == "centerline":
                centroids = self._compute_centroids_by_centerline(
                    mask=mask,
                    n_levels=n_levels,
                    smooth_win=smooth_win
                )
            else:
                raise ValueError("method debe ser 'bands' o 'centerline'")

            boxes = self._boxes_from_centroids(
                mask=mask,
                centroids=centroids,
                box_w=box_w,
                box_h=box_h,
                adaptive_width=adaptive_width
            )

            rows.extend(
                self._save_boxes_for_study(
                    study_id=study_id,
                    split=split,
                    image=image,
                    mask=mask,
                    boxes=boxes,
                    method=method,
                    img_dir=img_dir,
                    mask_dir=mask_dir
                )
            )

        out_df = pd.DataFrame([asdict(r) for r in rows])
        out_csv = os.path.join(self.save_root, f"vertebra_auto_boxes_{method}.csv")
        out_df.to_csv(out_csv, index=False)

        print(f"[{method}] metadata guardada en: {out_csv}")
        print(f"[{method}] total cajas guardadas: {len(out_df)}")
        return out_df

    # =========================================================
    # MÉTODO DE PRUEBA INDIVIDUAL
    # =========================================================
    def preview_single(
        self,
        image_rel_path,
        mask_rel_path,
        method="bands",
        n_levels=9,
        box_w=None,
        box_h=None,
        adaptive_width=True,
        smooth_win=21
    ):
        image = self._read_gray_rel(image_rel_path)
        mask = self._read_gray_rel(mask_rel_path)

        if method == "bands":
            centroids = self._compute_centroids_by_bands(mask, n_levels=n_levels)
        elif method == "centerline":
            centroids = self._compute_centroids_by_centerline(mask, n_levels=n_levels, smooth_win=smooth_win)
        else:
            raise ValueError("method debe ser 'bands' o 'centerline'")

        boxes = self._boxes_from_centroids(
            mask=mask,
            centroids=centroids,
            box_w=box_w,
            box_h=box_h,
            adaptive_width=adaptive_width
        )

        vis = self.draw_boxes(image, boxes)
        return image, mask, centroids, boxes, vis

    def get_vertebra_boxes(self, image, mask, n_levels=9, box_w=130, box_h=80, adaptive_width=True):
        """
        Calcula los centroides usando el método de bandas y retorna las cajas generadas.
        Puedes ajustar n_levels, box_w, box_h y adaptive_width según tu flujo.
        """
        centroids = self._compute_centroids_by_bands(mask, n_levels=n_levels)
        boxes = self._boxes_from_centroids(mask, centroids, box_w=box_w, box_h=box_h, adaptive_width=adaptive_width)
        return boxes
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from dataclasses import dataclass, asdict
@dataclass
class VertebraBoxRecord:
    study_id: str
    split: str
    vertebra_idx: int
    vertebra_id: str
    strategy: str
    centroid_x: float
    centroid_y: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    vertebra_img_path: str
    vertebra_mask_path: str


class VertebraBoxExtractor:
    """
    Dos estrategias:

    1) centerline:
       - Usa una máscara global de columna
       - Calcula una línea central aproximada por filas
       - Divide la columna en niveles verticales
       - Genera cajas alrededor de esa línea central

    2) centroids:
       - Usa centroides/anotaciones externas
       - Construye cajas directamente alrededor de cada centroide
    """

    def __init__(
        self,
        base_dir,
        image_col="radiograph_path",
        mask_col="label_binary_path",
        save_root=None,
        pad_x=30,
        pad_y=15,
        min_row_pixels=5,
        default_box_w=120,
        default_box_h=80,
        overlap_y=0.20
    ):
        self.base_dir = base_dir
        self.image_col = image_col
        self.mask_col = mask_col
        self.save_root = save_root
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.min_row_pixels = min_row_pixels
        self.default_box_w = default_box_w
        self.default_box_h = default_box_h
        self.overlap_y = overlap_y

    def _read_gray_rel(self, rel_path):
        full = os.path.join(self.base_dir, rel_path)
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(full)
        return img

    def _ensure_dirs(self):
        if self.save_root is None:
            raise ValueError("save_root es obligatorio.")

        os.makedirs(self.save_root, exist_ok=True)

        img_dir = os.path.join(self.save_root, "vertebra_images")
        mask_dir = os.path.join(self.save_root, "vertebra_masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        return img_dir, mask_dir

    def _clip_box(self, x1, y1, x2, y2, w, h):
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2)))
        y2 = min(h, int(round(y2)))
        return x1, y1, x2, y2

    def _build_box_from_center(self, cx, cy, box_w, box_h, w, h):
        x1 = cx - box_w / 2
        y1 = cy - box_h / 2
        x2 = cx + box_w / 2
        y2 = cy + box_h / 2
        return self._clip_box(x1, y1, x2, y2, w, h)

    # =========================================================
    # ESTRATEGIA 1: CENTERLINE
    # =========================================================
    def _extract_centerline_points(self, mask):
        """
        Para cada fila y de la máscara, calcula el centro x
        de los píxeles positivos.
        """
        binary = (mask > 0).astype(np.uint8)
        h, w = binary.shape

        pts = []
        for y in range(h):
            xs = np.where(binary[y] > 0)[0]
            if len(xs) >= self.min_row_pixels:
                cx = float(xs.mean())
                pts.append((cx, float(y)))

        return pts

    def _smooth_centerline(self, pts, win=21):
        if len(pts) == 0:
            return []

        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)

        if len(xs) < win:
            return list(zip(xs, ys))

        kernel = np.ones(win, dtype=np.float32) / win
        xs_smooth = np.convolve(xs, kernel, mode="same")
        return list(zip(xs_smooth, ys))

    def _boxes_from_centerline(
        self,
        mask,
        n_levels=9,
        box_w=None,
        box_h=None,
        adaptive_width=True
    ):
        """
        Genera cajas vertebrales sobre una línea central suavizada.
        """
        binary = (mask > 0).astype(np.uint8)
        h, w = binary.shape

        pts = self._extract_centerline_points(binary)
        pts = self._smooth_centerline(pts, win=21)

        if len(pts) == 0:
            return []

        ys = np.array([p[1] for p in pts], dtype=np.float32)
        xs = np.array([p[0] for p in pts], dtype=np.float32)

        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        total_h = max(1, y_max - y_min)

        if box_h is None:
            step = total_h / max(n_levels, 1)
            box_h = max(40, int(step * (1.0 + self.overlap_y)))
        if box_w is None:
            box_w = self.default_box_w

        centers_y = np.linspace(y_min, y_max, n_levels)

        boxes = []
        for idx, cy in enumerate(centers_y):
            nearest_idx = int(np.argmin(np.abs(ys - cy)))
            cx = float(xs[nearest_idx])

            local_box_w = box_w
            if adaptive_width:
                yy = int(np.clip(round(cy), 0, h - 1))
                row_xs = np.where(binary[yy] > 0)[0]
                if len(row_xs) > 5:
                    row_w = int(row_xs.max() - row_xs.min() + 1)
                    local_box_w = max(self.default_box_w, row_w + 2 * self.pad_x)

            x1, y1, x2, y2 = self._build_box_from_center(
                cx=cx,
                cy=cy,
                box_w=local_box_w,
                box_h=box_h,
                w=w,
                h=h
            )

            boxes.append({
                "vertebra_idx": idx,
                "centroid_x": cx,
                "centroid_y": float(cy),
                "bbox": (x1, y1, x2, y2)
            })

        return boxes

    # =========================================================
    # ESTRATEGIA 2: CENTROIDS
    # =========================================================
    def _boxes_from_centroids(
        self,
        image_shape,
        centroids,
        box_w=None,
        box_h=None,
        sort_by_y=True
    ):
        """
        centroids: lista de dicts o tuplas.
        Ejemplos válidos:
        [(x1, y1), (x2, y2), ...]
        o
        [{"x": 100, "y": 200}, {"x": 120, "y": 250}]
        """
        h, w = image_shape[:2]
        if box_w is None:
            box_w = self.default_box_w
        if box_h is None:
            box_h = self.default_box_h

        parsed = []
        for c in centroids:
            if isinstance(c, dict):
                cx = float(c["x"])
                cy = float(c["y"])
            else:
                cx = float(c[0])
                cy = float(c[1])

            x1, y1, x2, y2 = self._build_box_from_center(
                cx=cx,
                cy=cy,
                box_w=box_w,
                box_h=box_h,
                w=w,
                h=h
            )

            parsed.append({
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox": (x1, y1, x2, y2)
            })

        if sort_by_y:
            parsed = sorted(parsed, key=lambda d: d["centroid_y"])

        for idx, item in enumerate(parsed):
            item["vertebra_idx"] = idx

        return parsed

    # =========================================================
    # GUARDADO
    # =========================================================
    def _save_boxes_for_study(
        self,
        study_id,
        split,
        image,
        mask,
        boxes,
        strategy,
        img_dir,
        mask_dir
    ):
        rows = []

        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]

            vertebra_img = image[y1:y2, x1:x2]
            vertebra_mask = mask[y1:y2, x1:x2]

            vertebra_idx = item["vertebra_idx"]
            vertebra_id = f"{study_id}_v{vertebra_idx:02d}"

            img_path = os.path.join(img_dir, f"{vertebra_id}.png")
            mask_path = os.path.join(mask_dir, f"{vertebra_id}_mask.png")

            cv2.imwrite(img_path, vertebra_img)
            cv2.imwrite(mask_path, vertebra_mask)

            rows.append(VertebraBoxRecord(
                study_id=study_id,
                split=split,
                vertebra_idx=vertebra_idx,
                vertebra_id=vertebra_id,
                strategy=strategy,
                centroid_x=item["centroid_x"],
                centroid_y=item["centroid_y"],
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                vertebra_img_path=img_path,
                vertebra_mask_path=mask_path
            ))

        return rows

    # =========================================================
    # API PRINCIPAL
    # =========================================================
    def extract_all_centerline(
        self,
        index_csv,
        split_csv,
        n_levels=9,
        box_w=None,
        box_h=None,
        adaptive_width=True
    ):
        img_dir, mask_dir = self._ensure_dirs()

        df = pd.read_csv(index_csv).copy()
        split_df = pd.read_csv(split_csv).copy()

        df["study_id"] = df[self.image_col].apply(lambda p: Path(p).stem)
        if "split" in df.columns:
            df = df.drop(columns=["split"])
        df = df.merge(split_df[["study_id", "split"]], on="study_id", how="left")

        rows = []

        for _, row in df.iterrows():
            study_id = row["study_id"]
            split = row["split"]

            image = self._read_gray_rel(row[self.image_col])
            mask = self._read_gray_rel(row[self.mask_col])

            boxes = self._boxes_from_centerline(
                mask=mask,
                n_levels=n_levels,
                box_w=box_w,
                box_h=box_h,
                adaptive_width=adaptive_width
            )

            rows.extend(
                self._save_boxes_for_study(
                    study_id=study_id,
                    split=split,
                    image=image,
                    mask=mask,
                    boxes=boxes,
                    strategy="centerline",
                    img_dir=img_dir,
                    mask_dir=mask_dir
                )
            )

        out_df = pd.DataFrame([asdict(r) for r in rows])
        out_csv = os.path.join(self.save_root, "vertebra_boxes_centerline.csv")
        out_df.to_csv(out_csv, index=False)

        print(f"[centerline] metadata guardada en: {out_csv}")
        print(f"[centerline] total cajas guardadas: {len(out_df)}")
        return out_df

    def extract_all_centroids(
        self,
        index_csv,
        split_csv,
        centroids_df,
        centroid_study_col="study_id",
        centroid_x_col="centroid_x",
        centroid_y_col="centroid_y",
        box_w=None,
        box_h=None
    ):
        """
        centroids_df:
        DataFrame con columnas:
        - study_id
        - centroid_x
        - centroid_y
        """
        img_dir, mask_dir = self._ensure_dirs()

        df = pd.read_csv(index_csv).copy()
        split_df = pd.read_csv(split_csv).copy()

        df["study_id"] = df[self.image_col].apply(lambda p: Path(p).stem)
        if "split" in df.columns:
            df = df.drop(columns=["split"])
        df = df.merge(split_df[["study_id", "split"]], on="study_id", how="left")

        rows = []

        for _, row in df.iterrows():
            study_id = row["study_id"]
            split = row["split"]

            image = self._read_gray_rel(row[self.image_col])
            mask = self._read_gray_rel(row[self.mask_col])

            study_centroids = centroids_df[
                centroids_df[centroid_study_col] == study_id
            ][[centroid_x_col, centroid_y_col]].copy()

            centroids = [
                (float(r[centroid_x_col]), float(r[centroid_y_col]))
                for _, r in study_centroids.iterrows()
            ]

            boxes = self._boxes_from_centroids(
                image_shape=image.shape,
                centroids=centroids,
                box_w=box_w,
                box_h=box_h,
                sort_by_y=True
            )

            rows.extend(
                self._save_boxes_for_study(
                    study_id=study_id,
                    split=split,
                    image=image,
                    mask=mask,
                    boxes=boxes,
                    strategy="centroids",
                    img_dir=img_dir,
                    mask_dir=mask_dir
                )
            )

        out_df = pd.DataFrame([asdict(r) for r in rows])
        out_csv = os.path.join(self.save_root, "vertebra_boxes_centroids.csv")
        out_df.to_csv(out_csv, index=False)

        print(f"[centroids] metadata guardada en: {out_csv}")
        print(f"[centroids] total cajas guardadas: {len(out_df)}")
        return out_df

    # =========================================================
    # VISUALIZACIÓN
    # =========================================================
    def draw_boxes(self, image, boxes, color=(255, 255, 0), thickness=2, draw_idx=True):
        """
        color en BGR para OpenCV.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            cx = int(round(item["centroid_x"]))
            cy = int(round(item["centroid_y"]))
            idx = item["vertebra_idx"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

            if draw_idx:
                cv2.putText(
                    vis,
                    str(idx),
                    (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        return vis


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
        if "split" in df.columns:
            df = df.drop(columns=["split"])
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


# ========== OVERLAP SUMMARY BLOCK (robust, numpy-safe) ==========
    def summarize_overlap_matrix(self, overlap_matrix, summary):
        """
        Robustly summarize overlap_matrix (pd.DataFrame, pd.Series, or np.ndarray) into summary dict.
        Ensures no FutureWarnings or conversion errors.
        """
        import numpy as np
        import pandas as pd
        def _safe_numeric_array(arr):
            if isinstance(arr, (pd.DataFrame, pd.Series)):
                arr = arr.values
            arr = np.asarray(arr, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            return arr
        if overlap_matrix is not None:
            try:
                overlap_values = _safe_numeric_array(overlap_matrix)
                if overlap_values.size > 0:
                    summary["overlap_mean"] = float(np.mean(overlap_values))
                    summary["overlap_std"] = float(np.std(overlap_values))
                    summary["overlap_min"] = float(np.min(overlap_values))
                    summary["overlap_max"] = float(np.max(overlap_values))
                else:
                    summary["overlap_mean"] = np.nan
                    summary["overlap_std"] = np.nan
                    summary["overlap_min"] = np.nan
                    summary["overlap_max"] = np.nan
            except Exception as e:
                print(f"[WARNING] No se pudieron resumir métricas de overlap_matrix: {e}")
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
