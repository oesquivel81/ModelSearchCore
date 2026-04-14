# Experimento: Análisis Pre-TDA por Filtro con Métricas y Trazabilidad Espacial

## Descripción General
Este experimento implementa un pipeline robusto para el análisis previo a TDA (Topological Data Analysis) sobre parches de imágenes médicas, asegurando:
- Procesamiento independiente por filtro (sin mezclar regiones de diferentes filtros en ninguna ventana o combinación).
- Trazabilidad espacial completa: cada región/patch mantiene su centroid, métricas y ruta de imagen.
- Combinatorias solo de ventanas consecutivas.
- Exportación de reportes y CSVs con métricas, centroides, rutas de imagen y selección de mejores parches por cluster.

## Estructura del Stack
- **tda_patch_combinations.py**: Lógica de combinaciones, dataclasses (RegionRecord, CombinationRecord), carga de regiones desde centroides y métricas.
- **tda_baseline_and_filter_proxy.py**: Orquestador del pipeline, itera por filtro, integra PreTDAMetricsBuilder, exporta reportes y CSVs.
- **pre_tda_metrics_builder.py**: Agregador y exportador robusto de métricas.
- **master_config_metrics_S_37.csv**: Archivo de métricas por región (debe tener columna vertebra_idx y columnas de métricas).
- **patch_images_{filtro}/**: Carpeta con imágenes de parches por filtro.
- **centroid_curve_{paciente}.csv**: Centroides de cada región.
- **top_por_cluster_{paciente}.csv**: Selección de mejores parches por cluster.

## Flujo de Ejecución
1. **Preparar archivos de entrada:**
   - master_config_metrics_S_37.csv con vertebra_idx y métricas.
   - centroid_curve_{paciente}.csv con centroides.
   - patch_images_{filtro}/ con imágenes de parches.
2. **Ejecutar pipeline:**
   - El orquestador itera por cada filtro, carga regiones, asocia centroides, métricas e imágenes.
   - Se generan combinaciones solo de ventanas consecutivas.
   - Se validan combinaciones para evitar mezcla de filtros.
   - Se exportan reportes y CSVs con toda la información.
   - Se seleccionan y exportan los mejores parches por cluster.
3. **Revisar salidas:**
   - pre_tda_windows_report.csv: Reporte general con métricas por región y combinación.
   - pre_tda_master_table.csv: Tabla maestra consolidada.
   - top_por_cluster_{paciente}.csv: Mejores parches por cluster.

## Ejemplo de Ejecución
```bash
# (Asegúrate de tener pandas y numpy instalados)
python tda_baseline_and_filter_proxy.py --config config.yaml
```

## Notas
- Si alguna métrica falta para una región, se llenará con np.nan.
- Los nombres de filtro deben coincidir exactamente con los nombres de las carpetas patch_images_{filtro}.
- El pipeline es robusto a errores de importación y mezcla de filtros.

## Contacto
Para dudas o mejoras, contactar a Tavo o al responsable del pipeline.
