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
        print(f"[TRACE] _patched_run_one llamado para patient_id={patient_id}, config={config}")
        # Aplica el filtro configurado antes de extraer boxes y parches
        filtered_image = self.extractor.apply_filter(image, config.filter_name)
        boxes = self.extractor.get_vertebra_boxes(image=filtered_image, mask=mask)
        print(f"[TRACE] get_vertebra_boxes devolvió {len(boxes)} boxes")

        # Guardar parches y máscaras en disco usando la imagen filtrada
        patch_dtos = self.patch_builder.build_patch_dtos_on_disk(
            patient_id=patient_id,
            image=filtered_image,
            mask=mask,
            boxes=boxes,
            method=f"{config.filter_name}_{config.variance_mode}"
        )

        # Para métricas, puedes usar la imagen filtrada o la original según tu flujo
        from extractor.patch_dto import PatchDTO
        patch_dtos_mem = self.patch_builder.build_patch_dtos_in_memory(
            patient_id=patient_id,
            image=filtered_image,
            mask=mask,
            boxes=boxes,
            method=f"{config.filter_name}_{config.variance_mode}"
        )

        df_consecutive = self.metrics.compare_consecutive_patches(patch_dtos_mem)
        overlap_matrix = self.metrics.compute_overlap_matrix(patch_dtos_mem)
        summary = self.metrics.summarize_metrics(df_consecutive)

        import pandas as pd
        metrics_df = pd.DataFrame([summary])
        for field in AblationConfig.__annotations__:
            value = getattr(config, field)
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
