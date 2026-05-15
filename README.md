
# SCOLIOSIS_VERTEBRA_PEAK_ANALYSIS_20260515_200103

Backup reproducible del análisis de vértebras/body peaks, curva central,
centroides, geometría radiográfica, Cobb angle, clustering y visualizaciones.

## Contenido principal

- global_outputs/
  - RADIOGRAPH_GEOMETRY_METRICS
  - CENTROIDS_GEOMETRY_RELATION
  - COMPOSITE_BINARY_BOUNDARY_INTER_PEAKS
  - BODY_VERTEBRA_PEAK_SHAPE_CLUSTERING
  - BODY_VERTEBRA_VISIBLE_ORDER_CLEAN
  - PCA / clustering outputs

- patients_minimal/
  - normalized_full_image_used.png
  - pred_binary_confidence.png
  - pred_boundary.png
  - pred_intervertebral.png
  - pred_ordinal.png
  - ordered_vertebra_mask_curve_peaks.png
  - ordered_vertebra_overlay_curve_peaks.png
  - curve_peaks_detected.csv
  - composite_curve_profiles.csv
  - curve_peak_wave_metrics.csv
  - pasted_patches.csv
  - radiograph_geometry/

## Archivos clave

- BACKUP_MANIFEST_SHA256.csv
- global_copy_log.csv
- patient_minimal_copy_log.csv
- restore_config.json

## Uso en otro cuaderno

1. Copiar o descomprimir este backup en /content.
2. Ejecutar el bloque RESTORE incluido.
3. Cargar tablas desde:
   /content/PATIENT_RECONSTRUCTED_PREDICTIONS_FROM_PATCHES/...

Fecha de creación: 2026-05-15T20:01:50.339672
