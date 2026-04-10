from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class AblationConfig:
    config_id: str
    filter_name: str
    use_variance: bool
    variance_mode: str
    patch_size: tuple[int, int]
    stride: int
    variance_kernel: int = 7

class PatchAblationRunner:

    def run_one(self, image, mask, patient_id, config: AblationConfig):
        return self._patched_run_one(image, mask, patient_id, config)

    def __init__(self, extractor, patch_builder, metrics):
        self.extractor = extractor
        self.patch_builder = patch_builder
        self.metrics = metrics

    def _patched_run_one(self, image, mask, patient_id, config: AblationConfig):
        import numpy as np
        import pandas as pd
        print(f"\n[TRACE] _patched_run_one patient_id={patient_id}")
        print(f"[TRACE] config={config}")

        # =========================================================
        # 1) FILTRADO BASE
        # =========================================================
        filtered_image = self.extractor.apply_filter(image, config.filter_name)

        if filtered_image is None:
            raise ValueError(f"[ERROR] apply_filter devolvió None para {config.filter_name}")

        # Asegurar ndarray
        filtered_image = np.asarray(filtered_image)

        print(
            f"[TRACE] filtered_image shape={filtered_image.shape}, "
            f"dtype={filtered_image.dtype}, "
            f"mean={float(filtered_image.mean()):.6f}, "
            f"std={float(filtered_image.std()):.6f}"
        )

        # =========================================================
        # 2) MAPA DE VARIANZA SI APLICA
        # =========================================================
        variance_map = None
        model_input = filtered_image

        if getattr(config, "use_variance", False):
            kernel_size = getattr(config, "variance_kernel", 3)

            if kernel_size is None:
                kernel_size = 3

            variance_map = self.extractor.compute_local_variance(
                filtered_image,
                kernel_size=kernel_size
            )

            variance_map = np.asarray(variance_map)

            print(
                f"[TRACE] variance_map shape={variance_map.shape}, "
                f"dtype={variance_map.dtype}, "
                f"mean={float(variance_map.mean()):.6f}, "
                f"std={float(variance_map.std()):.6f}, "
                f"kernel={kernel_size}"
            )

            variance_mode = getattr(config, "variance_mode", "none")

            if variance_mode == "concat_channel":
                model_input = self.extractor.concat_channels(filtered_image, variance_map)

            elif variance_mode == "variance_only":
                model_input = variance_map

            elif variance_mode == "none":
                model_input = filtered_image

            else:
                raise ValueError(f"[ERROR] variance_mode no soportado: {variance_mode}")

        model_input = np.asarray(model_input)

        print(
            f"[TRACE] model_input shape={model_input.shape}, "
            f"dtype={model_input.dtype}, "
            f"mean={float(model_input.mean()):.6f}, "
            f"std={float(model_input.std()):.6f}"
        )

        # =========================================================
        # 3) BOXES
        # =========================================================
        boxes = self.extractor.get_vertebra_boxes(image=filtered_image, mask=mask)

        if boxes is None:
            boxes = []

        print(f"[TRACE] get_vertebra_boxes devolvió {len(boxes)} boxes")
        if len(boxes) > 0:
            print(f"[TRACE] first_box={boxes[0]}")

        # =========================================================
        # 4) PARCHES EN DISCO
        # =========================================================
        method_name = (
            f"{config.filter_name}"
            f"_var-{getattr(config, 'use_variance', False)}"
            f"_mode-{getattr(config, 'variance_mode', 'none')}"
            f"_pk-{getattr(config, 'patch_size', None)}"
            f"_st-{getattr(config, 'stride', None)}"
            f"_vk-{getattr(config, 'variance_kernel', None)}"
        )

        self.patch_builder.build_patch_dtos_on_disk(
            patient_id=patient_id,
            image=model_input,
            mask=mask,
            boxes=boxes,
            method=method_name,
            patch_size=getattr(config, "patch_size", None),
            stride=getattr(config, "stride", None)
        )

        # =========================================================
        # 5) PARCHES EN MEMORIA
        # =========================================================
        patch_dtos_mem = self.patch_builder.build_patch_dtos_in_memory(
            patient_id=patient_id,
            image=model_input,
            mask=mask,
            boxes=boxes,
            method=method_name,
            patch_size=getattr(config, "patch_size", None),
            stride=getattr(config, "stride", None)
        )

        if patch_dtos_mem is None:
            patch_dtos_mem = []

        print(f"[TRACE] patch_dtos_mem={len(patch_dtos_mem)}")

        # =========================================================
        # 6) MÉTRICAS
        # =========================================================
        if len(patch_dtos_mem) < 2:
            print("[WARNING] Menos de 2 patches; no se pueden comparar consecutivos.")
            summary = {
                "mean_dice": np.nan,
                "mean_iou": np.nan,
                "mean_hausdorff": np.nan,
                "mean_hausdorff_norm": np.nan,
                "max_dice": np.nan,
                "max_iou": np.nan,
                "min_hausdorff": np.nan,
                "max_hausdorff": np.nan,
            }
            overlap_matrix = None
            df_consecutive = pd.DataFrame()
        else:
            df_consecutive = self.metrics.compare_consecutive_patches(patch_dtos_mem)
            overlap_matrix = self.metrics.compute_overlap_matrix(patch_dtos_mem)
            summary = self.metrics.summarize_metrics(df_consecutive)

        # =========================================================
        # 7) FEATURES DE AUDITORÍA
        # =========================================================
        summary["num_boxes"] = len(boxes)
        summary["num_patches"] = len(patch_dtos_mem)
        summary["input_mean"] = float(model_input.mean())
        summary["input_std"] = float(model_input.std())
        summary["input_min"] = float(model_input.min())
        summary["input_max"] = float(model_input.max())
        summary["image_shape"] = str(tuple(model_input.shape))
        summary["mask_shape"] = str(tuple(mask.shape)) if mask is not None else None

        if overlap_matrix is not None:
            try:
                summary["overlap_mean"] = float(np.mean(overlap_matrix))
                summary["overlap_std"] = float(np.std(overlap_matrix))
                summary["overlap_min"] = float(np.min(overlap_matrix))
                summary["overlap_max"] = float(np.max(overlap_matrix))
            except Exception as e:
                print(f"[WARNING] No se pudieron resumir métricas de overlap_matrix: {e}")

        # =========================================================
        # 8) RESULTADO FINAL
        # =========================================================
        metrics_df = pd.DataFrame([summary])

        for field in AblationConfig.__annotations__:
            value = getattr(config, field, None)
            if isinstance(value, (tuple, list, dict)):
                value = str(value)
            metrics_df[field] = value

        return metrics_df

    def run_all(self, dataset: List[Dict[str, Any]], configs: List[AblationConfig]) -> pd.DataFrame:
        rows = []
        for sample in dataset:
            for config in configs:
                result = self.run_one(
                    image=sample["image"],
                    mask=sample["mask"],
                    patient_id=sample["patient_id"],
                    config=config
                )
                if isinstance(result, pd.DataFrame):
                    rows.extend(result.to_dict(orient="records"))
                else:
                    rows.append(result)
        return pd.DataFrame(rows)

    def apply_filter(self, image, filter_name: str):
        # stub
        return image

    def compute_variance_map(self, image, kernel_size: int = 7):
        # stub
        return image

    def combine_inputs(self, original, filtered, variance, use_variance: bool, variance_mode: str):
        if not use_variance or variance_mode == "none":
            return filtered

        if variance_mode == "variance_only":
            return variance

        if variance_mode == "concat_channel":
            return np.stack([original, variance], axis=-1)

        if variance_mode == "concat_after_filter":
            return np.stack([filtered, variance], axis=-1)

        if variance_mode == "weighted_sum":
            return 0.7 * filtered + 0.3 * variance

        raise ValueError(f"variance_mode no soportado: {variance_mode}")
