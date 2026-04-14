import os
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_patch_combinations import (
    generate_patch_combinations,
    evaluate_combination,
)


def normalize_filter_names(filter_names):
    if filter_names is None:
        return []

    if isinstance(filter_names, str):
        s = filter_names.strip()
        return [s] if s else []

    if isinstance(filter_names, pd.Series):
        return (
            filter_names.dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s.ne("")]
            .drop_duplicates()
            .tolist()
        )

    if isinstance(filter_names, (list, tuple, set)):
        out = []
        seen = set()
        for x in filter_names:
            if x is None:
                continue
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    s = str(filter_names).strip()
    return [s] if s else []


def safe_parse_patch_size(value):
    if pd.isna(value):
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return tuple(value)
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return tuple(parsed)
    except Exception:
        pass
    return value


def build_folder_name_from_row(row: pd.Series) -> str:
    return (
        f"{row['filter_name']}"
        f"_var-{row['use_variance']}"
        f"_mode-{row['variance_mode']}"
        f"_pk-{row['patch_size']}"
        f"_st-{row['stride']}"
        f"_vk-{row['variance_kernel']}"
    )


class TDABaselineAndFilterProxy:
    def __init__(self, config, patch_folders=None):
        self.config = config
        self.tda_root = config["tda_root"]
        self.patient_id = config["patient_id"]
        self.restrictions = config["restrictions"]
        self.experiment_modes = config["experiment_modes"]
        self.patch_folders = patch_folders or []

        self.metrics = [
            "mean_dice",
            "max_dice",
            "mean_iou",
            "max_iou",
            "mean_hausdorff",
            "mean_hausdorff_norm",
            "min_hausdorff",
            "max_hausdorff",
            "mean_mse_img",
            "max_mse_img",
            "min_mse_img",
            "mean_mae_img",
            "max_mae_img",
            "min_mae_img",
            "mean_intensity_diff",
            "mean_std_intensity_diff",
            "mean_var_diff",
            "mean_grad_mse",
            "mean_grad_mae",
            "mean_centroid_distance",
            "mean_area_ratio",
            "num_boxes",
            "num_patches",
            "input_mean",
            "input_std",
            "input_min",
            "input_max",
            "overlap_mean",
            "overlap_std",
            "overlap_min",
            "overlap_max",
            "cluster",
            "score_cluster",
            "num_imgs",
        ]

        self.non_numeric_metadata = [
            "image_shape",
            "mask_shape",
            "config_id",
            "filter_name",
            "use_variance",
            "variance_mode",
            "patch_size",
            "stride",
            "variance_kernel",
            "found_dir",
            "patch_images_dir",
            "image_names",
            "image_paths",
            "match_found",
            "config_folder",
            "patch_images_path",
        ]

        self.csv_path = os.path.join(
            self.tda_root,
            f"master_config_metrics_{self.patient_id}.csv",
        )
        self.curve_csv = os.path.join(
            self.tda_root,
            f"patches_processor_{self.patient_id}",
            f"centroid_curve_{self.patient_id}.csv",
        )
        self.output_root = os.path.join(
            self.tda_root,
            f"patches_processor_{self.patient_id}",
            self.patient_id,
            "bands",
        )

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No existe CSV maestro: {self.csv_path}")
        if not os.path.exists(self.curve_csv):
            raise FileNotFoundError(f"No existe CSV de centroides: {self.curve_csv}")

        self.df_metrics = pd.read_csv(self.csv_path)
        self.df_curve = pd.read_csv(self.curve_csv)

        requested_filters = normalize_filter_names(config.get("filter_names"))
        if requested_filters:
            self.df_metrics = self.df_metrics[
                self.df_metrics["filter_name"].astype(str).isin(requested_filters)
            ].copy()

        self.df_metrics["config_folder"] = self.df_metrics.apply(build_folder_name_from_row, axis=1)
        self.df_metrics["patch_images_path"] = self.df_metrics["config_folder"].apply(
            lambda folder: os.path.join(self.tda_root, self.patient_id, folder, "patch_images")
        )

        if self.patch_folders:
            allowed = set(map(str, self.patch_folders))
            self.df_metrics = self.df_metrics[
                self.df_metrics["patch_images_path"].astype(str).isin(allowed)
            ].copy()

        self.df_metrics = self.df_metrics.drop_duplicates(
            subset=[
                "filter_name",
                "use_variance",
                "variance_mode",
                "patch_size",
                "stride",
                "variance_kernel",
            ]
        ).reset_index(drop=True)

        self.config_rows = self.df_metrics.to_dict(orient="records")

    # ============================================================
    # UTILIDADES DE I/O Y CANVAS
    # ============================================================

    def _resolve_image_files(self, patch_dir: str) -> list[str]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if not os.path.isdir(patch_dir):
            return []
        return sorted(
            [
                str(p)
                for p in Path(patch_dir).iterdir()
                if p.is_file() and p.suffix.lower() in exts
            ]
        )

    def _read_patch_image(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert("L")
        return np.array(img, dtype=np.float32)

    def _build_dynamic_canvas(self, image_files, curve_df):
        if not image_files or curve_df.empty:
            return None, None, None, {}

        patch_shapes = []
        for img_path in image_files:
            arr = self._read_patch_image(img_path)
            patch_shapes.append(arr.shape)

        max_h = max(h for h, w in patch_shapes)
        max_w = max(w for h, w in patch_shapes)

        min_x = float(curve_df["centroid_x"].min())
        max_x = float(curve_df["centroid_x"].max())
        min_y = float(curve_df["centroid_y"].min())
        max_y = float(curve_df["centroid_y"].max())

        pad_x = max_w
        pad_y = max_h

        canvas_w = int(np.ceil((max_x - min_x) + 2 * pad_x + max_w))
        canvas_h = int(np.ceil((max_y - min_y) + 2 * pad_y + max_h))

        canvas = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
        valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        count_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        offset_x = pad_x - min_x
        offset_y = pad_y - min_y

        meta = {
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "canvas_h": int(canvas_h),
            "canvas_w": int(canvas_w),
        }

        return canvas, valid_mask, count_map, meta

    def _place_patch_on_canvas(self, canvas, valid_mask, count_map, patch_img, cx, cy, offset_x, offset_y):
        h, w = patch_img.shape[:2]

        canvas_cx = int(round(cx + offset_x))
        canvas_cy = int(round(cy + offset_y))

        x1 = canvas_cx - (w // 2)
        y1 = canvas_cy - (h // 2)
        x2 = x1 + w
        y2 = y1 + h

        H, W = canvas.shape[:2]

        clip_x1 = max(0, x1)
        clip_y1 = max(0, y1)
        clip_x2 = min(W, x2)
        clip_y2 = min(H, y2)

        if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
            return (x1, y1, x2, y2)

        px1 = clip_x1 - x1
        py1 = clip_y1 - y1
        px2 = px1 + (clip_x2 - clip_x1)
        py2 = py1 + (clip_y2 - clip_y1)

        patch_crop = patch_img[py1:py2, px1:px2]

        region_canvas = canvas[clip_y1:clip_y2, clip_x1:clip_x2]
        region_count = count_map[clip_y1:clip_y2, clip_x1:clip_x2]

        existing_valid = ~np.isnan(region_canvas)

        region_canvas[~existing_valid] = patch_crop[~existing_valid]
        region_count[~existing_valid] = 1.0

        overlap_mask = existing_valid
        region_canvas[overlap_mask] = (
            region_canvas[overlap_mask] * region_count[overlap_mask] + patch_crop[overlap_mask]
        ) / (region_count[overlap_mask] + 1.0)
        region_count[overlap_mask] += 1.0

        canvas[clip_y1:clip_y2, clip_x1:clip_x2] = region_canvas
        count_map[clip_y1:clip_y2, clip_x1:clip_x2] = region_count
        valid_mask[clip_y1:clip_y2, clip_x1:clip_x2] = 1

        return (x1, y1, x2, y2)

    # ============================================================
    # UTILIDADES DE ESTADÍSTICA
    # ============================================================

    def _safe_bbox_crop(self, arr, bbox):
        if arr is None or bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        H, W = arr.shape[:2]
        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(W, int(x2))
        cy2 = min(H, int(y2))
        if cx1 >= cx2 or cy1 >= cy2:
            return None
        return arr[cy1:cy2, cx1:cx2]

    def _safe_numeric_summary(self, values, prefix=""):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]

        if values.size == 0:
            return {
                f"{prefix}count": 0,
                f"{prefix}mean": np.nan,
                f"{prefix}std": np.nan,
                f"{prefix}min": np.nan,
                f"{prefix}max": np.nan,
                f"{prefix}median": np.nan,
                f"{prefix}q25": np.nan,
                f"{prefix}q75": np.nan,
                f"{prefix}iqr": np.nan,
                f"{prefix}range": np.nan,
            }

        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        return {
            f"{prefix}count": int(values.size),
            f"{prefix}mean": float(np.mean(values)),
            f"{prefix}std": float(np.std(values)),
            f"{prefix}min": float(np.min(values)),
            f"{prefix}max": float(np.max(values)),
            f"{prefix}median": float(np.median(values)),
            f"{prefix}q25": float(q25),
            f"{prefix}q75": float(q75),
            f"{prefix}iqr": float(q75 - q25),
            f"{prefix}range": float(np.max(values) - np.min(values)),
        }

    def _get_region_binary_mask_in_canvas(self, region, canvas_shape):
        bbox = getattr(region, "bbox", None)
        if bbox is None:
            return np.zeros(canvas_shape, dtype=np.uint8)

        x1, y1, x2, y2 = bbox
        H, W = canvas_shape[:2]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))

        mask = np.zeros((H, W), dtype=np.uint8)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 1
        return mask

    def _compute_region_local_features(self, region, canvas, valid_mask):
        """
        Calcula métricas LOCALES reales sobre la región recortada del canvas.
        """
        bbox = getattr(region, "bbox", None)
        canvas_crop = self._safe_bbox_crop(canvas, bbox)
        mask_crop = self._safe_bbox_crop(valid_mask, bbox)

        if canvas_crop is None or mask_crop is None:
            vals = np.array([], dtype=float)
            valid_ratio = np.nan
            patch_height = np.nan
            patch_width = np.nan
            valid_pixel_count = 0
        else:
            vals = canvas_crop[mask_crop == 1]
            vals = vals[~np.isnan(vals)]
            patch_height, patch_width = canvas_crop.shape[:2]
            valid_pixel_count = int(np.sum(mask_crop == 1))
            valid_ratio = float(valid_pixel_count / mask_crop.size) if mask_crop.size > 0 else np.nan

        out = {
            "region_valid_ratio": valid_ratio,
            "region_patch_height": patch_height,
            "region_patch_width": patch_width,
            "region_valid_pixel_count": valid_pixel_count,
        }
        out.update(self._safe_numeric_summary(vals, prefix="region_patch_"))
        return out

    def _compute_pairwise_consecutive_features(self, regions):
        """
        Calcula diferencias entre regiones consecutivas usando:
        - centroides
        - métricas locales reales region_*
        - métricas globales config_* solo como referencia
        """
        rows = []

        if regions is None or len(regions) < 2:
            return pd.DataFrame(rows)

        local_metric_names = [
            "region_patch_mean",
            "region_patch_std",
            "region_patch_min",
            "region_patch_max",
            "region_patch_median",
            "region_patch_q25",
            "region_patch_q75",
            "region_patch_iqr",
            "region_patch_range",
            "region_valid_ratio",
            "region_valid_pixel_count",
        ]

        config_metric_names = [
            "mean_dice",
            "mean_iou",
            "mean_mse_img",
            "mean_mae_img",
            "mean_grad_mse",
            "mean_grad_mae",
            "mean_var_diff",
            "mean_intensity_diff",
        ]

        for i in range(len(regions) - 1):
            a = regions[i]
            b = regions[i + 1]

            ax = pd.to_numeric(pd.Series([getattr(a, "centroid_x", np.nan)]), errors="coerce").astype(float).iloc[0]
            ay = pd.to_numeric(pd.Series([getattr(a, "centroid_y", np.nan)]), errors="coerce").astype(float).iloc[0]
            bx = pd.to_numeric(pd.Series([getattr(b, "centroid_x", np.nan)]), errors="coerce").astype(float).iloc[0]
            by = pd.to_numeric(pd.Series([getattr(b, "centroid_y", np.nan)]), errors="coerce").astype(float).iloc[0]

            dx = bx - ax if pd.notna(ax) and pd.notna(bx) else np.nan
            dy = by - ay if pd.notna(ay) and pd.notna(by) else np.nan
            dist = np.sqrt(dx**2 + dy**2) if pd.notna(dx) and pd.notna(dy) else np.nan

            row = {
                "region_id_i": getattr(a, "region_id", None),
                "region_id_j": getattr(b, "region_id", None),
                "vertebra_idx_i": getattr(a, "vertebra_idx", None),
                "vertebra_idx_j": getattr(b, "vertebra_idx", None),
                "delta_centroid_x": float(dx) if pd.notna(dx) else np.nan,
                "delta_centroid_y": float(dy) if pd.notna(dy) else np.nan,
                "centroid_distance": float(dist) if pd.notna(dist) else np.nan,
            }

            # Diferencias de métricas LOCALES
            for m in local_metric_names:
                av = pd.to_numeric(pd.Series([getattr(a, m, np.nan)]), errors="coerce").astype(float).iloc[0]
                bv = pd.to_numeric(pd.Series([getattr(b, m, np.nan)]), errors="coerce").astype(float).iloc[0]

                if pd.notna(av) and pd.notna(bv):
                    delta = float(bv - av)
                    row[f"delta_{m}"] = delta
                    row[f"abs_delta_{m}"] = abs(delta)
                else:
                    row[f"delta_{m}"] = np.nan
                    row[f"abs_delta_{m}"] = np.nan

            # Diferencias de métricas globales SOLO como referencia
            for m in config_metric_names:
                av = pd.to_numeric(pd.Series([getattr(a, m, np.nan)]), errors="coerce").astype(float).iloc[0]
                bv = pd.to_numeric(pd.Series([getattr(b, m, np.nan)]), errors="coerce").astype(float).iloc[0]

                if pd.notna(av) and pd.notna(bv):
                    delta = float(bv - av)
                    row[f"delta_config_{m}"] = delta
                    row[f"abs_delta_config_{m}"] = abs(delta)
                else:
                    row[f"delta_config_{m}"] = np.nan
                    row[f"abs_delta_config_{m}"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)

    def _compute_window_intersection_metrics(self, regions, canvas, valid_mask):
        """
        Calcula métricas de ventana usando:
        - unión/intersección geométrica
        - métricas LOCALES reales por región
        - métricas globales SOLO como referencia
        """
        if canvas is None or valid_mask is None or not regions:
            return {}

        H, W = canvas.shape[:2]
        region_masks = [self._get_region_binary_mask_in_canvas(r, (H, W)) for r in regions]

        union_mask = np.zeros((H, W), dtype=np.uint8)
        intersection_mask = np.ones((H, W), dtype=np.uint8)

        for rm in region_masks:
            union_mask = np.logical_or(union_mask, rm).astype(np.uint8)
            intersection_mask = np.logical_and(intersection_mask, rm).astype(np.uint8)

        union_mask = np.logical_and(union_mask == 1, valid_mask == 1).astype(np.uint8)
        intersection_mask = np.logical_and(intersection_mask == 1, valid_mask == 1).astype(np.uint8)

        union_vals = canvas[union_mask == 1]
        union_vals = union_vals[~np.isnan(union_vals)]

        intersection_vals = canvas[intersection_mask == 1]
        intersection_vals = intersection_vals[~np.isnan(intersection_vals)]

        union_area = int(np.sum(union_mask == 1))
        intersection_area = int(np.sum(intersection_mask == 1))
        overlap_ratio = float(intersection_area / union_area) if union_area > 0 else np.nan

        xs = [getattr(r, "centroid_x", np.nan) for r in regions]
        ys = [getattr(r, "centroid_y", np.nan) for r in regions]

        centroid_span_x = float(np.nanmax(xs) - np.nanmin(xs)) if len(xs) > 0 else np.nan
        centroid_span_y = float(np.nanmax(ys) - np.nanmin(ys)) if len(ys) > 0 else np.nan

        out = {
            "simplex_dim": len(regions),
            "window_region_count": len(regions),
            "window_union_area": union_area,
            "window_intersection_area": intersection_area,
            "window_overlap_ratio": overlap_ratio,
            "centroid_span_x": centroid_span_x,
            "centroid_span_y": centroid_span_y,
        }

        out.update(self._safe_numeric_summary(union_vals, prefix="window_union_"))
        out.update(self._safe_numeric_summary(intersection_vals, prefix="window_intersection_"))

        # MÉTRICAS LOCALES REALES que sí deben variar
        local_metric_names = [
            "region_patch_mean",
            "region_patch_std",
            "region_patch_min",
            "region_patch_max",
            "region_patch_median",
            "region_patch_q25",
            "region_patch_q75",
            "region_patch_iqr",
            "region_patch_range",
            "region_valid_ratio",
            "region_valid_pixel_count",
        ]

        for m in local_metric_names:
            vals = pd.to_numeric(
                pd.Series([getattr(r, m, np.nan) for r in regions]),
                errors="coerce"
            ).dropna().astype(float).values

            if vals.size == 0:
                out[f"window_{m}_mean"] = np.nan
                out[f"window_{m}_std"] = np.nan
                out[f"window_{m}_min"] = np.nan
                out[f"window_{m}_max"] = np.nan
                out[f"window_{m}_range"] = np.nan
            else:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                out[f"window_{m}_mean"] = float(np.mean(vals))
                out[f"window_{m}_std"] = float(np.std(vals))
                out[f"window_{m}_min"] = vmin
                out[f"window_{m}_max"] = vmax
                out[f"window_{m}_range"] = float(vmax - vmin)

        # MÉTRICAS GLOBALES SOLO de referencia
        config_metric_names = [
            "mean_dice",
            "mean_iou",
            "mean_mse_img",
            "mean_mae_img",
            "mean_grad_mse",
            "mean_grad_mae",
            "mean_var_diff",
            "mean_intensity_diff",
        ]

        for m in config_metric_names:
            vals = pd.to_numeric(
                pd.Series([getattr(r, m, np.nan) for r in regions]),
                errors="coerce"
            ).dropna().astype(float).values

            if vals.size == 0:
                out[f"window_config_{m}_mean"] = np.nan
            else:
                out[f"window_config_{m}_mean"] = float(np.mean(vals))

        pair_df = self._compute_pairwise_consecutive_features(regions)
        if not pair_df.empty:
            centroid_vals = pd.to_numeric(pair_df["centroid_distance"], errors="coerce").dropna().values
            if len(centroid_vals) > 0:
                out["mean_centroid_distance"] = float(np.mean(centroid_vals))
                out["max_centroid_distance"] = float(np.max(centroid_vals))
            else:
                out["mean_centroid_distance"] = np.nan
                out["max_centroid_distance"] = np.nan

            # Diferencias locales reales
            for m in local_metric_names:
                abs_col = f"abs_delta_{m}"
                if abs_col in pair_df.columns:
                    vals = pd.to_numeric(pair_df[abs_col], errors="coerce").dropna().astype(float).values
                    if len(vals) > 0:
                        out[f"window_mean_abs_diff_{m}"] = float(np.mean(vals))
                        out[f"window_max_abs_diff_{m}"] = float(np.max(vals))
                    else:
                        out[f"window_mean_abs_diff_{m}"] = np.nan
                        out[f"window_max_abs_diff_{m}"] = np.nan

        return out

    def _validate_window_geometry(self, regions, window_metrics):
        """
        Valida una ventana usando continuidad por cadena consecutiva
        en lugar de exigir intersección global total.

        Reglas:
        1. Los vertebra_idx deben ser consecutivos.
        2. Cada par consecutivo debe tener:
          - intersección positiva entre sus bbox
          - o al menos un overlap_ratio mínimo
        3. La distancia entre centroides consecutivos no debe ser absurda.

        Devuelve:
        - is_valid_simplex: bool
        - validity_reason: str
        - rejection_reason: str | None
        """
        if not regions:
            return False, "empty_window", "window_has_no_regions"

        # ------------------------------------------------------------
        # Regla 1: índices consecutivos
        # ------------------------------------------------------------
        vertebra_idxs = [getattr(r, "vertebra_idx", None) for r in regions]
        vertebra_idxs = [int(v) for v in vertebra_idxs if v is not None]

        if len(vertebra_idxs) != len(regions):
            return False, "invalid_missing_vertebra_idx", "missing_vertebra_idx"

        diffs = np.diff(vertebra_idxs)
        if not np.all(diffs == 1):
            return False, "invalid_non_consecutive_indices", "vertebra_idx_not_consecutive"

        # ------------------------------------------------------------
        # Regla 2: continuidad por pares consecutivos
        # ------------------------------------------------------------
        pair_df = self._compute_pairwise_consecutive_features(regions)

        if pair_df.empty:
            return False, "invalid_no_pairs", "pairwise_dataframe_empty"

        # Umbrales ajustables
        min_pair_overlap_ratio = 0.01
        max_pair_centroid_distance = 400.0

        # Necesitamos revisar intersección bbox por cada par
        for i in range(len(regions) - 1):
            a = regions[i]
            b = regions[i + 1]

            bbox_a = getattr(a, "bbox", None)
            bbox_b = getattr(b, "bbox", None)

            if bbox_a is None or bbox_b is None:
                return False, "invalid_missing_bbox", "missing_bbox_in_pair"

            ax1, ay1, ax2, ay2 = bbox_a
            bx1, by1, bx2, by2 = bbox_b

            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union_area = area_a + area_b - inter_area

            pair_overlap_ratio = (inter_area / union_area) if union_area > 0 else 0.0

            # Si no hay nada de intersección entre vecinos, se rompe la cadena
            if inter_area <= 0 and pair_overlap_ratio < min_pair_overlap_ratio:
                return (
                    False,
                    "invalid_pairwise_disconnected_window",
                    f"pair_{i}_{i+1}_has_zero_or_tiny_overlap"
                )

            # Revisar distancia entre centroides del par
            row = pair_df.iloc[i]
            centroid_distance = row.get("centroid_distance", np.nan)

            if pd.notna(centroid_distance) and float(centroid_distance) > max_pair_centroid_distance:
                return (
                    False,
                    "invalid_large_centroid_jump",
                    f"pair_{i}_{i+1}_centroid_distance_too_large"
                )

        # ------------------------------------------------------------
        # Regla 3: diagnóstico global suave
        # ------------------------------------------------------------
        mean_centroid_distance = window_metrics.get("mean_centroid_distance", np.nan)
        if pd.notna(mean_centroid_distance) and float(mean_centroid_distance) > max_pair_centroid_distance:
            return False, "invalid_large_mean_centroid_jump", "mean_centroid_distance_too_large"

        # Ya no exigimos intersección global de TODA la ventana.
        # Basta con que la cadena local sea coherente.
        return True, "valid_pairwise_chain_window", None

    # ============================================================
    # CONSTRUCCIÓN DE REGIONES
    # ============================================================

    def _build_regions_for_config(self, row: dict):
        from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_patch_combinations import RegionRecord

        patch_dir = row["patch_images_path"]
        if not os.path.isdir(patch_dir):
            print(f"[WARN] Carpeta patch_images no encontrada: {patch_dir}")
            return [], None, None

        required_curve_cols = {"vertebra_idx", "centroid_x", "centroid_y"}
        missing_curve = required_curve_cols - set(self.df_curve.columns)
        if missing_curve:
            raise ValueError(f"Faltan columnas en centroid_curve: {missing_curve}")

        curve_df = self.df_curve.sort_values("vertebra_idx").reset_index(drop=True).copy()
        image_files = self._resolve_image_files(patch_dir)

        if not image_files:
            print(f"[WARN] No hay imágenes en: {patch_dir}")
            return [], None, None

        n_regions = min(len(image_files), len(curve_df))
        if n_regions == 0:
            return [], None, None

        image_files = image_files[:n_regions]
        curve_df = curve_df.iloc[:n_regions].copy()

        canvas, valid_mask, count_map, meta = self._build_dynamic_canvas(image_files, curve_df)

        regions = []

        for i in range(n_regions):
            img_path = image_files[i]
            c_row = curve_df.iloc[i]
            patch_img = self._read_patch_image(img_path)

            cx = float(c_row["centroid_x"])
            cy = float(c_row["centroid_y"])
            vertebra_idx = int(c_row["vertebra_idx"])

            bbox = self._place_patch_on_canvas(
                canvas=canvas,
                valid_mask=valid_mask,
                count_map=count_map,
                patch_img=patch_img,
                cx=cx,
                cy=cy,
                offset_x=meta["offset_x"],
                offset_y=meta["offset_y"],
            )

            region = RegionRecord(
                region_id=Path(img_path).stem,
                patient_id=self.patient_id,
                config_id=row.get("config_id", None),
                filter_name=row.get("filter_name", None),
                image_path=img_path,
                vertebra_idx=vertebra_idx,
                centroid_x=cx,
                centroid_y=cy,
                use_variance=row.get("use_variance", None),
                variance_mode=row.get("variance_mode", None),
                patch_size=safe_parse_patch_size(row.get("patch_size", None)),
                stride=row.get("stride", None),
                variance_kernel=row.get("variance_kernel", None),
                bbox=bbox,
                centroid=(cx, cy),
                curve_param=None,
                order_index=vertebra_idx,
                lives_near_curve=True,
                split=None,
                metadata={
                    "optional_metadata": {
                        "config_folder": row.get("config_folder"),
                        "patch_images_path": row.get("patch_images_path"),
                        "canvas_offset_x": meta["offset_x"],
                        "canvas_offset_y": meta["offset_y"],
                        "canvas_h": meta["canvas_h"],
                        "canvas_w": meta["canvas_w"],
                    }
                },
            )

            # Propagar todo
            for col in self.df_metrics.columns:
                try:
                    setattr(region, col, row.get(col, np.nan))
                except Exception:
                    pass

            # Asegurar métricas numéricas explícitas
            for m in self.metrics:
                try:
                    setattr(
                        region,
                        m,
                        pd.to_numeric(pd.Series([row.get(m, np.nan)]), errors="coerce").astype(float).iloc[0]
                    )
                except Exception:
                    setattr(region, m, np.nan)

            # Locales reales sobre patch colocado
            local_features = self._compute_region_local_features(region, canvas, valid_mask)
            for k, v in local_features.items():
                setattr(region, k, v)

            regions.append(region)

        print(
            f"[DEBUG] Config '{row.get('config_folder')}' -> "
            f"{len(regions)} regiones, canvas={canvas.shape}, valid_pixels={int(valid_mask.sum())}"
        )

        return regions, canvas, valid_mask

    # ============================================================
    # EXPORTACIÓN SIMPLE SIN DEPENDER DEL FLUJO VIEJO
    # ============================================================

    def _regions_to_dataframe(self, region_records):
        rows = []
        for r in region_records:
            row = {}
            for k, v in r.__dict__.items():
                if isinstance(v, (list, dict, tuple)):
                    row[k] = str(v)
                else:
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def _windows_to_dataframe(self, windows, validity_records):
        rows = []
        for win, rec in zip(windows, validity_records):
            row = dict(rec)
            row["member_region_ids"] = str([r.region_id for r in win])
            row["member_image_paths"] = str([r.image_path for r in win])
            row["member_vertebra_idx"] = str([r.vertebra_idx for r in win])
            rows.append(row)
        return pd.DataFrame(rows)

    def _summary_dataframe(self, regions_df, windows_df):
        summary = {
            "n_regions": len(regions_df),
            "n_windows": len(windows_df),
        }
        if "is_valid_simplex" in windows_df.columns:
            summary["valid_windows"] = int((windows_df["is_valid_simplex"] == True).sum())
            summary["invalid_windows"] = int((windows_df["is_valid_simplex"] == False).sum())
        return pd.DataFrame([summary])

    def _rename_config_metric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renombra métricas heredadas del CSV maestro con prefijo config_.
        """
        if df is None or df.empty:
            return df

        config_metric_cols = [
            "mean_dice",
            "max_dice",
            "mean_iou",
            "max_iou",
            "mean_hausdorff",
            "mean_hausdorff_norm",
            "min_hausdorff",
            "max_hausdorff",
            "mean_mse_img",
            "max_mse_img",
            "min_mse_img",
            "mean_mae_img",
            "max_mae_img",
            "min_mae_img",
            "mean_intensity_diff",
            "mean_std_intensity_diff",
            "mean_var_diff",
            "mean_grad_mse",
            "mean_grad_mae",
            "mean_centroid_distance",
            "mean_area_ratio",
            "num_boxes",
            "num_patches",
            "input_mean",
            "input_std",
            "input_min",
            "input_max",
            "overlap_mean",
            "overlap_std",
            "overlap_min",
            "overlap_max",
            "cluster",
            "score_cluster",
            "num_imgs",
        ]

        rename_map = {}
        for col in config_metric_cols:
            if col in df.columns and f"config_{col}" not in df.columns:
                rename_map[col] = f"config_{col}"

        return df.rename(columns=rename_map)        

    # ============================================================
    # PIPELINE PRINCIPAL
    # ============================================================

    def run(self):
        print("[PIPELINE] SOLO se usarán configuraciones del CSV / array de rutas")
        print(f"[PIPELINE] Total configuraciones: {len(self.config_rows)}")

        for i, row in enumerate(self.config_rows, 1):
            config_folder = row["config_folder"]
            patch_dir = row["patch_images_path"]
            filter_name = row["filter_name"]
            config_id = row.get("config_id", None)

            print(f"\n[PIPELINE] ({i}/{len(self.config_rows)}) Procesando:")
            print(f"  filter_name   : {filter_name}")
            print(f"  config_folder : {config_folder}")
            print(f"  patch_dir     : {patch_dir}")

            if not os.path.isdir(patch_dir):
                print(f"[WARN] No existe carpeta: {patch_dir}")
                continue

            region_records, canvas, valid_mask = self._build_regions_for_config(row)

            print(f"[DEBUG] Regiones encontradas: {len(region_records)}")
            if canvas is not None and valid_mask is not None:
                print(f"[DEBUG] Canvas shape: {canvas.shape}, valid pixels: {int(valid_mask.sum())}")

            if not region_records:
                continue

            self._run_tda_for_regions(
                region_records=region_records,
                filter_name=filter_name,
                config_id=config_id,
                config_folder=config_folder,
                canvas=canvas,
                valid_mask=valid_mask,
            )

    def _run_tda_for_regions(self, region_records, filter_name, config_id, config_folder, canvas=None, valid_mask=None):
        region_records = sorted(region_records, key=lambda r: (r.vertebra_idx, r.order_index))

        print(f"[VALIDACIÓN] Muestra de regiones para '{config_folder}':")
        for r in region_records[:min(5, len(region_records))]:
            print({
                "region_id": r.region_id,
                "image_path": r.image_path,
                "centroid": (r.centroid_x, r.centroid_y),
                "vertebra_idx": r.vertebra_idx,
                "bbox": r.bbox,
                "config_mean_dice": getattr(r, "mean_dice", np.nan),
                "config_mean_iou": getattr(r, "mean_iou", np.nan),
                "region_patch_mean": getattr(r, "region_patch_mean", np.nan),
                "region_patch_std": getattr(r, "region_patch_std", np.nan),
                "region_valid_ratio": getattr(r, "region_valid_ratio", np.nan),
            })

        combos_raw = generate_patch_combinations(
            region_records,
            min_k=self.restrictions["min_k"],
            max_k=self.restrictions["max_k"],
            max_combination_count=self.restrictions["max_combination_count"],
        )

        windows = []
        validity_records = []

        for _, c in combos_raw:
            win_regions = list(c)

            eval_result = evaluate_combination(
                c,
                filter_params=None,
                selection_mode="consecutive_windows",
                experiment_mode="all_patches",
                restrictions=self.restrictions,
            )

            window_metrics = self._compute_window_intersection_metrics(
                regions=win_regions,
                canvas=canvas,
                valid_mask=valid_mask,
            )

            is_valid, validity_reason, rejection_reason = self._validate_window_geometry(
                regions=win_regions,
                window_metrics=window_metrics,
            )

            record = {
                "combination_id": eval_result.combination_id,
                "is_valid_simplex": is_valid,
                "validity_reason": validity_reason,
                "rejection_reason": rejection_reason,
                "k": len(win_regions),
                "filter_name": filter_name,
                "config_id": config_id,
                "config_folder": config_folder,
            }
            record.update(window_metrics)

            windows.append(win_regions)
            validity_records.append(record)

        print(f"[VALIDACIÓN] Muestra de ventanas para '{config_folder}':")
        for rec in validity_records[:min(5, len(validity_records))]:
            print({
                "combination_id": rec["combination_id"],
                "k": rec["k"],
                "is_valid_simplex": rec["is_valid_simplex"],
                "validity_reason": rec["validity_reason"],
                "rejection_reason": rec["rejection_reason"],
                "window_overlap_ratio": rec.get("window_overlap_ratio", np.nan),
                "mean_centroid_distance": rec.get("mean_centroid_distance", np.nan),
                "centroid_span_y": rec.get("centroid_span_y", np.nan),
            })

        outdir = os.path.join(self.output_root, f"pre_tda_{config_folder}")
        os.makedirs(outdir, exist_ok=True)

        prefix = f"{self.patient_id}_{config_folder}"

        if canvas is not None:
            np.save(os.path.join(outdir, f"{prefix}_canvas.npy"), canvas)
        if valid_mask is not None:
            np.save(os.path.join(outdir, f"{prefix}_valid_mask.npy"), valid_mask)

        regions_df = self._regions_to_dataframe(region_records)
        windows_df = self._windows_to_dataframe(windows, validity_records)

        # renombrar métricas globales heredadas del CSV
        regions_df = self._rename_config_metric_columns(regions_df)
        windows_df = self._rename_config_metric_columns(windows_df)

        summary_df = self._summary_dataframe(regions_df, windows_df)

        regions_path = os.path.join(outdir, f"{prefix}_regions_report.csv")
        windows_path = os.path.join(outdir, f"{prefix}_windows_report.csv")
        summary_path = os.path.join(outdir, f"{prefix}_summary_report.csv")
        master_path = os.path.join(outdir, f"{prefix}_master_table.csv")

        regions_df.to_csv(regions_path, index=False)
        windows_df.to_csv(windows_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        regions_df.to_csv(master_path, index=False)

        print(f"[EXPORT] Reportes exportados en {outdir}")
        print(f"[EXPORT] Archivos generados: {{'regions_report': regions_path, 'windows_report': windows_path, 'summary_report': summary_path, 'master_report': master_path}}")