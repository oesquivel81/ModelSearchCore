from extractor.patch_viz import show_patches
import os
import pandas as pd
from extractor.vertebra_region_extractor import VertebraAutoCentroidExtractor, PatchDTOBuilder
from extractor.patch_metrics import PatchMetrics

class CentroidCurveProxy:
    def __init__(self, config: dict):
        self.config = config
        self.base_path = os.path.abspath(config["base_path"])
        self.img_rel_path = config["img_rel_path"]
        self.mask_rel_path = config["mask_rel_path"]
        self.n_levels = config.get("n_levels", 9)
        self.box_w = config.get("box_w", 130)
        self.box_h = config.get("box_h", 80)
        self.adaptive_width = config.get("adaptive_width", True)
        self.split = config.get("split", "unspecified")
        # Nuevo: directorio base de salida y subdirectorio por paciente
        self.base_output_dir = config.get("base_output_dir", ".")
        # Siempre inferir patient_id del nombre de la imagen
        self.patient_id = os.path.splitext(os.path.basename(self.img_rel_path))[0]
        self.patches_processor_dir = os.path.join(self.base_output_dir, f"patches_processor_{self.patient_id}")
        os.makedirs(self.patches_processor_dir, exist_ok=True)


    def run_all(self):
        # 1. Cargar imagen y máscara
        extractor = VertebraAutoCentroidExtractor(
            base_dir=self.base_path,
            image_col="radiograph_path",
            mask_col="label_binary_path"
        )
        image = extractor._read_gray_rel(self.img_rel_path)
        mask = extractor._read_gray_rel(self.mask_rel_path)

        # 2. Extraer centroides por bandas
        centroids = extractor._compute_centroids_by_bands(mask, n_levels=self.n_levels)
        boxes = extractor._boxes_from_centroids(mask, centroids, box_w=self.box_w, box_h=self.box_h, adaptive_width=self.adaptive_width)

        # 3. Guardar curva de centroides en CSV en el subdirectorio
        patient_id = self.patient_id
        df_centroids = pd.DataFrame(centroids)
        df_centroids["split"] = self.split
        df_centroids["patient_id"] = patient_id
        csv_curve = os.path.join(self.patches_processor_dir, f"centroid_curve_{patient_id}.csv")
        df_centroids.to_csv(csv_curve, index=False)
        print(f"Curva de centroides guardada en {csv_curve}")

        # 4. Construir DTOs y guardar imágenes si se solicita
        patch_builder = PatchDTOBuilder(save_root=self.patches_processor_dir)
        save_patches = self.config.get("save_patches", False)
        # Siempre construye los parches en memoria para métricas y visualización
        patch_dtos = patch_builder.build_patch_dtos_in_memory(
            patient_id=patient_id,
            image=image,
            mask=mask,
            boxes=boxes,
            method="bands",
            add_overlay=True
        )
        self.patches = patch_dtos  # Para visualización externa
        # Si se solicita, también guarda los parches en disco
        if save_patches:
            patch_builder.build_patch_dtos_on_disk(
                patient_id=patient_id,
                image=image,
                mask=mask,
                boxes=boxes,
                method="bands"
            )
            print(f"Parches guardados en {self.patches_processor_dir}/patch_images y patch_masks")

        # 5. Calcular métricas consecutivas (máscara)
        metrics = PatchMetrics(kernel_size=3, hausdorff_use_edges=True)
        df_metrics_mask = metrics.compare_consecutive_patches(patch_dtos)
        print("Métricas consecutivas (máscara):")
        print(df_metrics_mask)

        # 6. Calcular matriz de empalmamiento (IoU, máscara)
        overlap_matrix = metrics.compute_overlap_matrix(patch_dtos, mode="mask", metric="iou")
        print("\nMatriz de empalmamiento (IoU, máscara):")
        print(overlap_matrix)

        # 7. Resumen de métricas
        df_summary = metrics.summarize_metrics(df_metrics_mask)
        print("\nResumen de métricas:")
        for k, v in df_summary.items():
            print(f"{k}: {v}")

    def show_patches(self, show_mask=False, show_overlay=True):
        """
        Visualiza los parches generados usando show_patches de patch_viz.
        """
        if not hasattr(self, 'patches') or self.patches is None:
            print("Primero ejecuta run_all() para generar los parches.")
            return
        show_patches(self.patches, show_mask=show_mask, show_overlay=show_overlay)
