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

    def __init__(self, extractor, patch_builder, metrics):
        self.extractor = extractor
        self.patch_builder = patch_builder
        self.metrics = metrics

    def run_one(self, image, mask, patient_id: str, config: AblationConfig) -> Dict[str, Any]:
        filtered = self.apply_filter(image, config.filter_name)
        variance_map = self.compute_variance_map(filtered if config.filter_name != "none" else image,
                                                 kernel_size=config.variance_kernel)

        model_input = self.combine_inputs(
            original=image,
            filtered=filtered,
            variance=variance_map,
            use_variance=config.use_variance,
            variance_mode=config.variance_mode
        )

        # Calcular centroides y cajas usando métodos existentes
        centroids = self.extractor._compute_centroids_by_bands(mask, n_levels=9)  # Ajusta n_levels si es necesario
        boxes = self.extractor._boxes_from_centroids(mask, centroids, box_w=130, box_h=80, adaptive_width=True)

        patch_dtos = self.patch_builder.build_patch_dtos_in_memory(
            patient_id=patient_id,
            image=model_input,
            mask=mask,
            boxes=boxes,
            method=f"{config.filter_name}_{config.variance_mode}"
        )

        df_consecutive = self.metrics.compare_consecutive_patches(patch_dtos)
        overlap_matrix = self.metrics.compute_overlap_matrix(patch_dtos)
        summary = self.metrics.summarize_metrics(df_consecutive)

        result = {
            "config_id": config.config_id,
            "patient_id": patient_id,
            "filter_name": config.filter_name,
            "use_variance": config.use_variance,
            "variance_mode": config.variance_mode,
            "patch_size": config.patch_size,
            "stride": config.stride,
            **summary
        }

        result["score_final"] = (
            0.4 * result.get("mean_iou", 0.0)
            + 0.4 * result.get("mean_dice", 0.0)
            - 0.2 * result.get("mean_hausdorff_norm", 0.0)
        )

        return result

    def run_all(self, dataset: List[Dict[str, Any]], configs: List[AblationConfig]) -> pd.DataFrame:
        rows = []
        for sample in dataset:
            for config in configs:
                rows.append(
                    self.run_one(
                        image=sample["image"],
                        mask=sample["mask"],
                        patient_id=sample["patient_id"],
                        config=config
                    )
                )
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
