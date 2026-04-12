# Descripción de Clases y Funciones para Extracción y Visualización de Parches

## Clases principales

### PatchDTO (patch_dto.py)
Representa un parche en memoria, incluyendo la imagen, máscara, overlay, bounding box y metadatos.
- **Atributos:**
  - patch_id, patient_id, image (np.ndarray), mask (opcional), overlay (opcional), bbox, centroid_x, centroid_y, method.

### PatchPathDTO (patch_dto.py)
Representa un parche referenciado por rutas en disco en vez de arrays en memoria.
- **Atributos:**
  - patch_id, patient_id, image_path, mask_path, bbox, centroid_x, centroid_y, method.

### PatchDTOBuilder (patch_dto.py)
Construye listas de PatchDTO o PatchPathDTO a partir de imágenes, máscaras y cajas.
- **Método clave:**
  - build_patch_dtos_in_memory: genera PatchDTOs en memoria a partir de los datos de entrada.
  - build_patch_dtos_on_disk: genera PatchPathDTOs en disco a partir de los datos de entrada.

### PatchMetrics (patch_metrics.py)
Calcula métricas de segmentación y superposición entre parches (IoU, Dice, Hausdorff, etc).
- **Métodos clave:**
  - compare_consecutive_patches, compute_overlap_matrix, summarize_metrics, dice, iou, hausdorff, etc.

### SubregionMetrics (vertebra_region_extractor.py)
Calcula métricas de experimentos sobre regiones vertebrales, incluyendo comparación de curvas y métricas entre parches.
- **Métodos clave:**
  - report_experiment_metrics (retorna un dict con métricas), curve_distance_mean.

### VertebraAutoCentroidExtractor (vertebra_region_extractor.py)
Extrae centroides y cajas de regiones vertebrales a partir de máscaras, y lee imágenes/máscaras desde disco.
- **Métodos clave:**
  - _read_gray_rel, _compute_centroids_by_bands, _boxes_from_centroids.

### show_patches (patch_viz.py)
Visualiza una lista de PatchDTO o PatchPathDTO en una cuadrícula usando matplotlib.
- **Uso:**
  - show_patches(patch_dtos) muestra los parches generados.

---

## Ejecución de Varias Configuraciones y Selección de Mejores Scores

Cuando se corren varias configuraciones (por ejemplo, diferentes parámetros para la extracción de parches o segmentación), normalmente se sigue este flujo:

1. **Definir configuraciones:**
   - Se crean diferentes sets de parámetros (por ejemplo, distintos n_levels, tamaños de caja, métodos de extracción, etc).
2. **Ejecutar el pipeline para cada configuración:**
   - Para cada configuración, se extraen los parches, se calculan las métricas y se guardan los resultados (por ejemplo, en un DataFrame o archivo CSV/JSON).
3. **Comparar resultados:**
   - Se recopilan los scores de cada configuración (por ejemplo, mean IoU, mean Dice, Hausdorff, etc).
4. **Seleccionar la mejor configuración:**
   - Se escoge la configuración con el mejor score según la métrica de interés (por ejemplo, la mayor media de IoU o Dice).
   - Esto puede hacerse ordenando el DataFrame de resultados y seleccionando la fila con el valor máximo.

### Ejemplo de selección automática:

```python
import pandas as pd
# Supón que tienes un DataFrame df_results con una columna 'mean_iou_box_neighbors'
mejor = df_results.loc[df_results['mean_iou_box_neighbors'].idxmax()]
print('Mejor configuración:', mejor)
```

En resumen, el proceso consiste en automatizar la ejecución de todas las configuraciones, guardar los resultados y luego seleccionar la mejor según la métrica que más te interese.

---

## Guardado automático de parches y máscaras en Google Drive

La forma más conveniente de guardar todos los parches y máscaras en disco (incluyendo Google Drive) es usando el método `build_patch_dtos_on_disk` de la clase `PatchDTOBuilder`.

### Ejemplo de uso:

```python
from extractor.patch_dto import PatchDTOBuilder

# Cambia esta ruta a tu carpeta de Google Drive
save_root = "/content/drive/MyDrive/patches_guardados"
patch_builder = PatchDTOBuilder(save_root=save_root)

patch_path_dtos = patch_builder.build_patch_dtos_on_disk(
    patient_id="S_100",
    image=image,
    mask=mask,
    boxes=boxes,
    method="bands"
)
```

Esto guardará automáticamente las imágenes y máscaras de los parches en carpetas separadas dentro de `save_root` y te devolverá una lista de `PatchPathDTO` con las rutas de cada archivo.

---

## Ejemplo de uso de la clase orquestadora para ablation y guardado en Google Drive

Supón que tienes un archivo CSV en tu Google Drive con las columnas: `patient_id`, `image_path`, `mask_path` (rutas absolutas o relativas a Drive).

```python
from colab.src.MAIA_B01_001_BL_FILTERS.orquestador_ablation import AblationOrchestrator
from colab.src.MAIA_B01_001_BL_FILTERS.patch_ablation_runner import AblationConfig
from extractor.vertebra_region_extractor import VertebraAutoCentroidExtractor
from extractor.patch_metrics import PatchMetrics

# Define tus configuraciones de ablation
configs = [
    AblationConfig(
        config_id="cfg1",
        filter_name="none",
        use_variance=False,
        variance_mode="none",
        patch_size=(128, 128),
        stride=32
    ),
    AblationConfig(
        config_id="cfg2",
        filter_name="gaussian",
        use_variance=True,
        variance_mode="concat_channel",
        patch_size=(128, 128),
        stride=32
    ),
    # Agrega más configs según tus experimentos
]

# Inicializa extractor y métricas según tu pipeline
extractor = VertebraAutoCentroidExtractor(
    base_dir="/content/drive/MyDrive/MaIA_Scoliosis_Dataset (1)",
    image_col="radiograph_path",
    mask_col="label_binary_path"
)
metrics = PatchMetrics(kernel_size=3, hausdorff_use_edges=True)

# Ruta al CSV con tus pacientes
csv_path = "/content/drive/MyDrive/MaIA_Scoliosis_Dataset (1)/dataset_index.csv"
# Carpeta donde se guardarán parches y métricas
save_root = "/content/drive/MyDrive/patches_guardados"

# Crea y ejecuta el orquestador
orq = AblationOrchestrator(
    dataset_csv=csv_path,
    save_root=save_root,
    configs=configs,
    extractor=extractor,
    patch_metrics=metrics
)
orq.run()
```

Esto ejecutará todas las configuraciones de ablation para todos los pacientes del CSV, guardará los parches, las métricas y un log en Google Drive.

---

## Ejemplo: Generar todas las combinaciones de configuraciones para ablation

Puedes usar itertools.product para crear todas las combinaciones posibles de filtros, modos de varianza, tamaños de parche, etc. Ejemplo:

```python
from itertools import product
from colab.src.MAIA_B01_001_BL_FILTERS.patch_ablation_runner import AblationConfig

# Define tus opciones
filtros = ["none", "gaussian", "median"]
use_variance_opts = [False, True]
variance_modes = ["none", "concat_channel", "variance_only"]
patch_sizes = [(128, 128), (256, 256)]
strides = [32, 64]

configs = []
for i, (f, u, v, p, s) in enumerate(product(filtros, use_variance_opts, variance_modes, patch_sizes, strides)):
    configs.append(
        AblationConfig(
            config_id=f"cfg_{i:03d}",
            filter_name=f,
            use_variance=u,
            variance_mode=v,
            patch_size=p,
            stride=s
        )
    )

print(f"Total de configuraciones: {len(configs)}")
# Ahora puedes pasar 'configs' al orquestador
```

Esto te permite lanzar experimentos de ablation masivos y sistemáticos, cubriendo todas las combinaciones relevantes para tu análisis.

---

## Descripción de los filtros disponibles para ablation

- **none**: No se aplica ningún filtro. Se usa la imagen original.
- **gaussian**: Aplica un filtro gaussiano para suavizar la imagen y reducir el ruido. Útil para eliminar detalles finos y resaltar estructuras grandes.
- **median**: Aplica un filtro de mediana, que es efectivo para eliminar ruido tipo "sal y pimienta" sin difuminar los bordes.
- **sobel**: Aplica un filtro de detección de bordes (gradiente). Útil para resaltar contornos y transiciones bruscas en la imagen.
- **laplacian**: Aplica un filtro Laplaciano para resaltar regiones de cambio rápido (bordes y detalles finos).
- **bilateral**: Aplica un filtro bilateral, que suaviza la imagen preservando los bordes. Útil para reducir ruido manteniendo detalles importantes.

Puedes agregar más filtros según tus necesidades, solo asegúrate de implementarlos en el método `apply_filter` de tu runner.

---

## Ejemplo: Ablation sobre una sola imagen/máscara cargada manualmente

Si quieres correr la ablation solo sobre un caso específico (sin usar el CSV), puedes hacerlo así:

```python
from itertools import product
from colab.src.MAIA_B01_001_BL_FILTERS.patch_ablation_runner import PatchAblationRunner, AblationConfig
from extractor.vertebra_region_extractor import VertebraAutoCentroidExtractor
from extractor.patch_metrics import PatchMetrics
from extractor.patch_dto import PatchDTOBuilder

# 1. Cargar imagen y máscara manualmente
base_path = "/content/drive/MyDrive/MaIA_Scoliosis_Dataset (1)"
img_rel_path = "Scoliosis/S_37.jpg"
mask_rel_path = "LabelMultiClass_Gray_JPG/LabelMulti_S_37.jpg"

extractor = VertebraAutoCentroidExtractor(
    base_dir=base_path,
    image_col="radiograph_path",
    mask_col="label_binary_path"
)
image = extractor._read_gray_rel(img_rel_path)
mask = extractor._read_gray_rel(mask_rel_path)

# 2. Define las combinaciones de configuraciones
filtros = ["none", "gaussian", "median"]
use_variance_opts = [False, True]
variance_modes = ["none", "concat_channel", "variance_only"]
patch_sizes = [(128, 128), (256, 256)]
strides = [32, 64]

configs = []
for i, (f, u, v, p, s) in enumerate(product(filtros, use_variance_opts, variance_modes, patch_sizes, strides)):
    configs.append(
        AblationConfig(
            config_id=f"cfg_{i:03d}",
            filter_name=f,
            use_variance=u,
            variance_mode=v,
            patch_size=p,
            stride=s
        )
    )

# 3. Inicializa patch_builder y métricas
patch_builder = PatchDTOBuilder(save_root="/content/drive/MyDrive/patches_guardados")
metrics = PatchMetrics(kernel_size=3, hausdorff_use_edges=True)

# 4. Prepara el dataset con un solo caso
dataset = [{
    "image": image,
    "mask": mask,
    "patient_id": "S_37"
}]

# 5. Ejecuta la ablation solo para este caso
runner = PatchAblationRunner(extractor, patch_builder, metrics)
df_results = runner.run_all(dataset, configs)
df_results.to_csv("/content/drive/MyDrive/patches_guardados/metrics_ablation_S_37.csv", index=False)
print("¡Ablation terminada y métricas guardadas para S_37!")
```

Esto te permite probar todas las configuraciones sobre una sola imagen/máscara, guardando los resultados en Google Drive.



## Análisis de filtros en el experimento de ablación

En esta etapa del análisis de ablación se observó que los filtros sí modifican la señal de la imagen, pero no todas las métricas reflejan ese cambio de la misma manera. En versiones iniciales del pipeline, métricas como **Dice** e **IoU** permanecían casi constantes entre configuraciones, lo que indicaba que estaban más influenciadas por la geometría de las máscaras y de las cajas que por el efecto real del preprocesamiento sobre la imagen. Por esta razón, fue necesario incorporar métricas sensibles al contenido visual de los parches.

A partir de ello, se consideraron métricas de imagen como:

- `mean_mae_img`
- `mean_intensity_diff`
- `mean_var_diff`
- `mean_grad_mae`
- `input_mean` / `input_std`

Estas métricas permiten evaluar cambios en:

- **brillo**
- **contraste**
- **textura**
- **bordes**
- **dispersión global de intensidades**

Por ejemplo:

- filtros como **Laplaciano**, **Sobel**, **Scharr** o **Prewitt** afectan principalmente la estructura de bordes;
- **CLAHE** modifica la distribución de intensidades y el contraste local;
- combinaciones con mapas de varianza pueden resaltar regiones con mayor textura o heterogeneidad.

## Criterio de selección de métricas

Para seleccionar métricas útiles no bastó con observar sus valores de manera aislada, sino que se realizó un análisis de:

1. **variación estadística**, para verificar que realmente cambiaran entre configuraciones;
2. **correlación**, para identificar métricas redundantes.

Cuando dos métricas presentaban una correlación muy alta, se interpretó que describían casi el mismo fenómeno, por lo que convenía conservar solo una. El objetivo no fue quedarse con muchas métricas, sino con un conjunto pequeño pero representativo de fenómenos distintos.

En términos prácticos, se buscó conservar métricas que representaran:

- **cambio global de imagen**
- **cambio de intensidad**
- **cambio de textura**
- **cambio de bordes**
- **dispersión estadística del input**

## Conclusión

Con este criterio, el ranking de configuraciones dejó de depender únicamente de métricas geométricas poco sensibles al filtro y pasó a apoyarse en métricas que sí capturan el efecto real del preprocesamiento. Esto permitió identificar configuraciones visualmente más estables y construir una base más confiable para:

- inspección cualitativa mediante grids de imágenes;
- comparación entre filtros;
- análisis posteriores como **TDA**, **clustering** o **embeddings**.