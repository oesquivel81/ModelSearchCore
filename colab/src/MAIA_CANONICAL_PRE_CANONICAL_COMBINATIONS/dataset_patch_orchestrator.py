import os
import json
import pandas as pd
import cv2
from .apply_filter_chain import apply_filter_chain


class DatasetPatchOrchestrator:
    def __init__(self, config):
        # Permite pasar un dict o un path a JSON
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            raise TypeError("config debe ser un dict o un path a archivo JSON")
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

        failed_cases = []
        dir_root = self.config.get('dir_root', None)
        for idx, row in df.iterrows():
            patient_id = str(row['patient_id'])
            img_path = row.get('radiograph_path', None)
            mask_path = row.get('mask_path', None)
            # Si dir_root está definido y la ruta no es absoluta, concatena
            if dir_root:
                if img_path and not os.path.isabs(img_path):
                    img_path = os.path.join(dir_root, img_path)
                if mask_path and not os.path.isabs(mask_path):
                    mask_path = os.path.join(dir_root, mask_path)
            print(f"[INFO] Paciente {patient_id} - radiograph_path: {img_path}")
            print(f"[INFO] Paciente {patient_id} - mask_path: {mask_path}")
            if not img_path or not os.path.exists(img_path):
                print(f"[WARN] No existe la radiografía para {patient_id}: {img_path}")
                failed_cases.append({'patient_id': patient_id, 'reason': 'radiograph_not_found', 'radiograph_path': img_path})
                continue
            if not mask_path or not os.path.exists(mask_path):
                print(f"[WARN] No existe la máscara para {patient_id}: {mask_path}")
                failed_cases.append({'patient_id': patient_id, 'reason': 'mask_not_found', 'mask_path': mask_path})
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"[WARN] No se pudo cargar imagen o máscara para {patient_id}")
                failed_cases.append({'patient_id': patient_id, 'reason': 'no_image_or_mask', 'radiograph_path': img_path, 'mask_path': mask_path})
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
                # Validar que la máscara permita combinaciones válidas
                if not extractor.has_valid_combinations():
                    print(f"[WARN] Máscara no permite combinaciones válidas para {patient_id}")
                    failed_cases.append({'patient_id': patient_id, 'reason': 'invalid_mask_combinations'})
                    continue
                extractor.run()
                df_meta = extractor.save_patches_with_metadata(sample_id=patient_id)
                # Seleccionar los mejores parches según la máscara
                if hasattr(extractor, 'select_best_patches'):
                    df_meta = extractor.select_best_patches(df_meta)
                # Validar overlay corresponde al borde
                if hasattr(extractor, 'validate_overlay_borders'):
                    if not extractor.validate_overlay_borders(df_meta):
                        print(f"[WARN] Overlay no corresponde al borde para {patient_id}")
                        failed_cases.append({'patient_id': patient_id, 'reason': 'overlay_mismatch'})
                        continue
                print(f"Parches y centroides originales generados para {patient_id}")
            except Exception as e:
                print(f"[ERROR] Falló la generación de parches/centroides originales para {patient_id}: {e}")
                failed_cases.append({'patient_id': patient_id, 'reason': str(e)})
                continue

            # 2. Aplicar filtros a cada parche generado
            if df_meta is not None and not df_meta.empty:
                for filt in self.filters:
                    filt_dir = os.path.join(self.save_root, f"{patient_id}_{filt.replace('+','_')}")
                    os.makedirs(filt_dir, exist_ok=True)
                    img_dir = os.path.join(patch_dir, "patch_images")
                    print(f"[INFO] Procesando filtro '{filt}' para paciente {patient_id} en {filt_dir}")
                    for i, row_patch in df_meta.iterrows():
                        patch_img_path = row_patch["image_patch_path"]
                        patch_img = cv2.imread(patch_img_path, cv2.IMREAD_GRAYSCALE)
                        if patch_img is None:
                            print(f"[WARN] No se pudo cargar el parche {patch_img_path}")
                            continue
                        filtered_patch = apply_filter_chain(patch_img, filt)
                        out_patch_path = os.path.join(filt_dir, os.path.basename(patch_img_path))
                        cv2.imwrite(out_patch_path, filtered_patch)
                        print(f"[INFO] Guardado parche filtrado: {out_patch_path}")
                    print(f"[INFO] Parches filtrados guardados para {patient_id} con filtro {filt} en {filt_dir}")

        # Guardar CSV con los casos fallidos
        if failed_cases:
            failed_df = pd.DataFrame(failed_cases)
            failed_csv_path = os.path.join(self.save_root, "failed_centroid_cases.csv")
            failed_df.to_csv(failed_csv_path, index=False)
            print(f"Casos fallidos guardados en {failed_csv_path}")

# Ejemplo de uso:
# orchestrator = DatasetPatchOrchestrator('config_orchestrator.json')
# orchestrator.run()
