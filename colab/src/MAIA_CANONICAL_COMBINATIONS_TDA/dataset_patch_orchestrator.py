import os
import json
import pandas as pd
import cv2
from .apply_filter_chain import apply_filter_chain

class DatasetPatchOrchestrator:
    def __init__(self, config_json_path):
        with open(config_json_path, 'r') as f:
            self.config = json.load(f)
        self.dataset_csv = self.config['dataset_csv']
        self.save_root = self.config['save_root']
        self.filters = self.config['filters']
        self.patch_size = tuple(self.config.get('patch_size', (128, 128)))
        self.stride = self.config.get('stride', 32)
        self.centroid_curve_dir = self.config.get('centroid_curve_dir', self.save_root)
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.centroid_curve_dir, exist_ok=True)

    def run(self):
        df = pd.read_csv(self.dataset_csv)
        for idx, row in df.iterrows():
            patient_id = str(row['patient_id'])
            img_path = row['image_path']
            mask_path = row['mask_path']
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"[WARN] No se pudo cargar imagen o máscara para {patient_id}")
                continue
            # 1. Generar parches y curva de centroides sobre la imagen original
            try:
                from colab.src.image_utils.vertebra_component_extractor import VertebraComponentExtractor
                patch_dir = os.path.join(self.save_root, f"{patient_id}_original")
                os.makedirs(patch_dir, exist_ok=True)
                extractor = VertebraComponentExtractor(
                    image=img,
                    local_mask=mask,
                    min_area=150,
                    pad_x=20,
                    pad_y=15,
                    save_dir=patch_dir
                )
                extractor.run()
                df_meta = extractor.save_patches_with_metadata(sample_id=patient_id)
                print(f"Parches y centroides originales generados para {patient_id}")
            except Exception as e:
                print(f"[ERROR] Falló la generación de parches/centroides originales para {patient_id}: {e}")
                continue

            # 2. Aplicar filtros a cada parche generado
            if df_meta is not None and not df_meta.empty:
                for filt in self.filters:
                    filt_dir = os.path.join(self.save_root, f"{patient_id}_{filt.replace('+','_')}")
                    os.makedirs(filt_dir, exist_ok=True)
                    img_dir = os.path.join(patch_dir, "patch_images")
                    for i, row_patch in df_meta.iterrows():
                        patch_img_path = row_patch["image_patch_path"]
                        patch_img = cv2.imread(patch_img_path, cv2.IMREAD_GRAYSCALE)
                        if patch_img is None:
                            print(f"[WARN] No se pudo cargar el parche {patch_img_path}")
                            continue
                        filtered_patch = apply_filter_chain(patch_img, filt)
                        out_patch_path = os.path.join(filt_dir, os.path.basename(patch_img_path))
                        cv2.imwrite(out_patch_path, filtered_patch)
                    print(f"Parches filtrados guardados para {patient_id} con filtro {filt}")

# Ejemplo de uso:
# orchestrator = DatasetPatchOrchestrator('config_orchestrator.json')
# orchestrator.run()
