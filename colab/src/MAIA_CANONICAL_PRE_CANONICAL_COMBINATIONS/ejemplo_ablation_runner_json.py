import json
import numpy as np
from dataclasses import dataclass

@dataclass
class AblationConfig:
    config_id: str
    filter_name: str
    use_variance: bool
    variance_mode: str
    patch_size: tuple
    stride: int
    variance_kernel: int = 7

class DummyExtractor:
    def apply_filter(self, image, filter_name):
        # Simula un filtro (devuelve la imagen igual)
        return image
    def compute_local_variance(self, image, kernel_size):
        # Simula un mapa de varianza (devuelve la imagen igual)
        return image
    def concat_channels(self, img1, img2):
        return np.stack([img1, img2], axis=-1)
    def get_vertebra_boxes(self, image, mask):
        # Simula una caja
        return [(0,0,10,10)]

class DummyPatchBuilder:
    def build_patch_dtos_on_disk(self, **kwargs):
        pass
    def build_patch_dtos_in_memory(self, **kwargs):
        # Simula dos parches
        return [object(), object()]

class DummyMetrics:
    def compare_consecutive_patches(self, patches):
        return {"dice": 1.0}
    def compute_overlap_matrix(self, patches):
        return np.array([[1.0, 0.5],[0.5,1.0]])
    def summarize_metrics(self, df):
        return {"mean_dice": 1.0, "mean_iou": 1.0}

class PatchAblationRunner:
    def __init__(self, extractor, patch_builder, metrics):
        self.extractor = extractor
        self.patch_builder = patch_builder
        self.metrics = metrics
    def run_one(self, image, mask, patient_id, config: AblationConfig):
        filtered_image = self.extractor.apply_filter(image, config.filter_name)
        boxes = self.extractor.get_vertebra_boxes(filtered_image, mask)
        self.patch_builder.build_patch_dtos_on_disk(patient_id=patient_id, image=filtered_image, mask=mask, boxes=boxes, method=config.filter_name, patch_size=config.patch_size, stride=config.stride)
        patches = self.patch_builder.build_patch_dtos_in_memory(patient_id=patient_id, image=filtered_image, mask=mask, boxes=boxes, method=config.filter_name, patch_size=config.patch_size, stride=config.stride)
        summary = self.metrics.summarize_metrics(None)
        summary["num_boxes"] = len(boxes)
        summary["num_patches"] = len(patches)
        return summary

if __name__ == "__main__":
    # Lee parámetros desde un archivo JSON externo
    with open("ablation_config.json", "r") as f:
        config_data = json.load(f)
    config = AblationConfig(
        config_id=config_data["config_id"],
        filter_name=config_data["filter_name"],
        use_variance=config_data["use_variance"],
        variance_mode=config_data["variance_mode"],
        patch_size=tuple(config_data["patch_size"]),
        stride=config_data["stride"],
        variance_kernel=config_data.get("variance_kernel", 7)
    )
    # Simula una imagen y máscara
    image = np.ones((32,32), dtype=np.uint8)
    mask = np.ones((32,32), dtype=np.uint8)
    runner = PatchAblationRunner(DummyExtractor(), DummyPatchBuilder(), DummyMetrics())
    result = runner.run_one(image, mask, patient_id="P001", config=config)
    print("Resultado:", result)
