import os
import pandas as pd
from typing import List
from extractor.patch_dto import PatchDTOBuilder
from .patch_ablation_runner import PatchAblationRunner, AblationConfig
from colab.src.MAIA_B01_001_BL_ch32_lr3_px30.vertebra_visualization_proxy import VertebraVisualizationProxy

class AblationOrchestrator:
    def __init__(self, dataset_csv: str, save_root: str, configs: List[AblationConfig], extractor, patch_metrics):
        self.dataset_csv = dataset_csv
        self.save_root = save_root
        self.configs = configs
        self.extractor = extractor
        self.patch_metrics = patch_metrics
        self.patch_builder = PatchDTOBuilder(save_root=save_root)
        self.runner = PatchAblationRunner(extractor, self.patch_builder, patch_metrics)

    def run(self):
        # Leer dataset CSV con columnas: patient_id, image_path, mask_path
        df = pd.read_csv(self.dataset_csv)
        all_results = []
        log = []
        for idx, row in df.iterrows():
            patient_id = row['patient_id']
            image_path = row['image_path']
            mask_path = row['mask_path']
            # Cargar imagen y máscara
            vis_proxy = VertebraVisualizationProxy(image_path, mask_path)
            image = vis_proxy.img
            mask = vis_proxy.mask
            sample = {"image": image, "mask": mask, "patient_id": patient_id}
            # Ejecutar ablation para todas las configs
            results = self.runner.run_all([sample], self.configs)
            all_results.append(results)
            log.append({"patient_id": patient_id, "image_path": image_path, "mask_path": mask_path, "n_configs": len(self.configs)})
        # Concatenar y guardar métricas
        df_results = pd.concat(all_results, ignore_index=True)
        metrics_path = os.path.join(self.save_root, "metrics_ablation.csv")
        df_results.to_csv(metrics_path, index=False)
        # Guardar log/summary
        log_path = os.path.join(self.save_root, "ablation_log.csv")
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"Métricas guardadas en: {metrics_path}")
        print(f"Log guardado en: {log_path}")
        print(f"Parches y máscaras guardados en: {self.save_root}/patch_images y {self.save_root}/patch_masks")
