import cv2
import numpy as np
import pandas as pd
from extractor.vertebra_region_extractor import VertebraAutoCentroidExtractor, PatchDTOBuilder
from extractor.patch_metrics import PatchMetrics

# Configuración de paths y parámetros
base_path = "/content/drive/MyDrive/MaIA_Scoliosis_Dataset (1)"
img_rel_path = "Scoliosis/S_100.jpg"
mask_rel_path = "LabelMultiClass_Gray_JPG/LabelMulti_S_100.jpg"

# Extraer imagen y máscara
extractor = VertebraAutoCentroidExtractor(
    base_dir=base_path,
    image_col="radiograph_path",
    mask_col="label_binary_path"
)
image = extractor._read_gray_rel(img_rel_path)
mask = extractor._read_gray_rel(mask_rel_path)

# Extraer parches con método bands
centroids = extractor._compute_centroids_by_bands(mask, n_levels=9)
boxes = extractor._boxes_from_centroids(mask, centroids, box_w=130, box_h=80, adaptive_width=True)

# Construir DTOs
patch_builder = PatchDTOBuilder()
patch_dtos = patch_builder.build_patch_dtos_in_memory(
    patient_id="S_100",
    image=image,
    mask=mask,
    boxes=boxes,
    method="bands",
    add_overlay=True
)

# Calcular métricas consecutivas (máscara)
metrics = PatchMetrics(kernel_size=3, hausdorff_use_edges=True)
df_metrics_mask = metrics.compare_consecutive_patches(patch_dtos, mode="mask")
print("Métricas consecutivas (máscara):")
print(df_metrics_mask)

# Calcular matriz de empalmamiento (IoU, máscara)
overlap_matrix = metrics.compute_overlap_matrix(patch_dtos, mode="mask", metric="iou")
print("\nMatriz de empalmamiento (IoU, máscara):")
print(overlap_matrix)

# Resumen de métricas
df_summary = metrics.summarize_metrics(df_metrics_mask)
print("\nResumen de métricas:")
for k, v in df_summary.items():
    print(f"{k}: {v}")
