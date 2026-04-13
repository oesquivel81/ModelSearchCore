# MAIA_B01_002_REGION_CLUSTER_VISUAL

## Descripción General
Módulo para experimentos de análisis topológico y combinatorio sobre parches de imágenes médicas. Permite extracción de centroides, generación y evaluación de combinaciones de parches, análisis respecto a curvas de referencia y exportación de resultados.

## Componentes y Métodos Principales

### tda_baseline_and_filter_proxy.py
**Clase:** `TDABaselineAndFilterProxy`
- `__init__(self, config)`: Inicializa el proxy con la configuración.
- `_find_filters(self)`: Busca y lista los filtros disponibles.
- `_load_patches(self, patch_dir)`: Carga parches de un directorio como `PatchPathDTO`.
- `_patch_to_region(self, patch)`: Convierte un `PatchPathDTO` en `RegionRecord`.
- `run(self)`: Orquesta la ejecución: carga centroides, procesa baseline y filtros, ejecuta TDA.
- `_run_tda_for_patches(self, patches, label, curve)`: Convierte parches a regiones, genera combinaciones, evalúa y exporta resultados.

### tda_experiment_proxy.py
**Clase:** `TDAExperimentProxy`
- `__init__(self, config)`: Inicializa el proxy con la configuración.
- `run(self)`: Lee centroides, selecciona mejores parches por cluster, ejecuta experimentos TDA y exporta resultados.

### tda_patch_combinations.py
- `RegionRecord`, `CombinationRecord`, `SimplexRecord`, `ExperimentBundle`: Estructuras de datos para regiones, combinaciones y resultados.
- `select_regions(df, mode, curve, curve_radius)`: Convierte un DataFrame en lista de `RegionRecord`.
- `generate_patch_combinations(regions, min_k, max_k, max_combination_count)`: Genera combinaciones de regiones.
- `evaluate_combination(combo, ...)`: Evalúa una combinación y retorna un `CombinationRecord`.
- `build_region_dataframe(regions)`, `build_combination_dataframe(combos)`, `build_simplex_dataframe(combos)`: Construyen DataFrames de resultados.

### region_structures.py
- `select_regions_by_experiment(regions, experiment_mode, curve, curve_radius)`: Selecciona subconjuntos de regiones según el modo.
- `compute_region_intersections(regions)`: Calcula intersecciones entre regiones.
- `build_nerve_simplicial_complex(regions, max_dim)`: Construye el nervio simplicial.
- `build_region_table(regions)`: Convierte regiones a tabla de métricas.
- `plot_regions_curve_and_nerve(image, regions, curve, simplexes, show_only_selected)`: Visualiza regiones, curva y nervio.

## Stack de Ejecución
1. Prepara un archivo de configuración JSON (ver ejemplos en la carpeta).
2. Ejecuta el proxy deseado desde un script o notebook:
   - `TDABaselineAndFilterProxy(config).run()`
   - `TDAExperimentProxy(config).run()`
3. Revisa los resultados exportados en las carpetas de salida.

## Requisitos
- Python 3.x
- pandas, numpy, matplotlib, etc.

---
