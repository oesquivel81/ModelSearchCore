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

    def _patched_run_one(self, image, mask, patient_id, config: AblationConfig):
        boxes = self.extractor.get_vertebra_boxes(image=image, mask=mask)

        patch_dtos = self.patch_builder.build_patch_dtos_in_memory(
            patient_id=patient_id,
            image=image,
            mask=mask,
            boxes=boxes,
            method=f"{config.filter_name}_{config.variance_mode}"
        )

        df_consecutive = self.metrics.compare_consecutive_patches(patch_dtos)
        overlap_matrix = self.metrics.compute_overlap_matrix(patch_dtos)
        summary = self.metrics.summarize_metrics(df_consecutive)

        import pandas as pd
        metrics_df = pd.DataFrame([summary])
        for field in AblationConfig.__annotations__:
            metrics_df[field] = getattr(config, field)
        return metrics_df

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
