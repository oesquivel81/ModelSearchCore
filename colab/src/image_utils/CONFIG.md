# Configuración de `VarianceUNetRegionExperiment`

## JSON completo con todas las propiedades

```json
{
    "experiment_name": "unet_variance_regions_v1",
    "drive_root": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset/cnn_varianza_a",
    "seed": 42,
    "execution_mode": "variance_input",

    "extractor": {
        "base_dir": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset (1)",
        "index_csv": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset (1)/dataset_index.csv",
        "image_col": "radiograph_path",
        "mask_col": "label_binary_path",
        "min_area": 150,
        "pad_x": 30,
        "pad_y": 15,
        "top_region_ratio": 0.35,
        "top_pad_x_scale": 2,
        "top_pad_y_top_scale": 3,
        "top_pad_y_bottom_scale": 0.8,
        "pad_x_tight": 8,
        "pad_y_tight": 6,
        "top_extra_tight": 10,
        "mode": "tight",
        "include_labels": ["good"]
    },

    "data": {
        "patch_size": [128, 128],
        "binarize_mask": true
    },

    "model": {
        "type": "variance_input",
        "base_channels": 32,
        "model_name": "unet"
    },

    "training": {
        "batch_size": 8,
        "epochs": 20,
        "lr": 1e-3,
        "num_workers": 2,
        "save_visuals_each_epoch": true,
        "num_visual_samples": 3,
        "best_metric_name": "dice",
        "resume_checkpoint_path": null
    },

    "discord": {
        "webhook_url": null,
        "notify_every_n_epochs": 1
    }
}
```

---

## Propiedades raíz

| Propiedad | Tipo | Default | Descripción |
|---|---|---|---|
| `experiment_name` | str | `"unet_region_experiment"` | Nombre del experimento. Se usa como subcarpeta dentro de `drive_root`. |
| `drive_root` | str | `"./results"` | Ruta raíz donde se guardan todos los artefactos (plots, checkpoints, reportes). |
| `seed` | int | `42` | Semilla global para reproducibilidad (random, numpy, torch). |
| `execution_mode` | str | `"variance_input"` | Modo de ejecución. Se usa como fallback para `model.type` si no se especifica. |

---

## Sección `extractor`

Controla la extracción de regiones vertebrales con `VertebraComponentExtractor` y `VertebraRegionBatch`.

| Propiedad | Tipo | Default | Descripción |
|---|---|---|---|
| `base_dir` | str | **requerido** | Carpeta base del dataset. Las rutas relativas del CSV se resuelven desde aquí. |
| `index_csv` | str | **requerido** | Ruta al archivo CSV con las columnas de imagen y máscara. |
| `image_col` | str | `"radiograph_path"` | Nombre de la columna en el CSV que contiene la ruta a la radiografía. |
| `mask_col` | str | `"label_binary_path"` | Nombre de la columna en el CSV que contiene la ruta a la máscara binaria. |
| `min_area` | int | `150` | Área mínima (en píxeles) para considerar un componente conectado como vértebra candidata. |
| `pad_x` | int | `30` | Padding horizontal (px) alrededor de cada componente al recortar. |
| `pad_y` | int | `15` | Padding vertical (px) alrededor de cada componente al recortar. |
| `top_region_ratio` | float | `0.35` | Fracción superior de la imagen donde se aplica padding aumentado (vértebras cervicales). |
| `top_pad_x_scale` | float | `2` | Multiplicador de `pad_x` para componentes en la zona superior. |
| `top_pad_y_top_scale` | float | `3` | Multiplicador de `pad_y` (arriba) para componentes en la zona superior. |
| `top_pad_y_bottom_scale` | float | `0.8` | Multiplicador de `pad_y` (abajo) para componentes en la zona superior. |
| `pad_x_tight` | int | `8` | Padding tight en X para `build_adjusted_bboxes`. |
| `pad_y_tight` | int | `6` | Padding tight en Y para `build_adjusted_bboxes`. |
| `top_extra_tight` | int | `10` | Padding extra tight para zona superior. |
| `mode` | str | `"tight"` | Modo de bbox: `"tight"` (ajustada) o `"context"` (con contexto amplio). |
| `include_labels` | list[str] | `["good"]` | Labels de calidad a incluir. Opciones: `"good"`, `"doubtful"`, `"bad"`. |

---

## Sección `data`

Configuración del dataset de parches para la CNN.

| Propiedad | Tipo | Default | Descripción |
|---|---|---|---|
| `patch_size` | [H, W] | `[128, 128]` | Tamaño al que se redimensiona cada parche vertebral antes de entrar a la red. |
| `binarize_mask` | bool | `true` | Si `true`, binariza la máscara con umbral 127. Si `false`, normaliza a [0, 1]. |

---

## Sección `model`

Configuración de la arquitectura UNet.

| Propiedad | Tipo | Default | Descripción |
|---|---|---|---|
| `type` | str | `"variance_input"` | Arquitectura de la UNet. Valores: `"baseline"`, `"variance_input"`, `"variance_branch"`. |
| `base_channels` | int | `32` | Número de canales base. Se duplica en cada nivel del encoder (32 → 64 → 128 → 256). |
| `model_name` | str | `"unet"` | Nombre del modelo (se guarda en metadata, no afecta la arquitectura). |

### Arquitecturas disponibles

- **`baseline`**: UNet estándar con 1 canal de entrada.
- **`variance_input`**: Agrega 3 mapas de varianza local (k=3, 5, 9) como canales de entrada → 4 canales totales.
- **`variance_branch`**: Dos encoders paralelos (imagen + varianza) que se fusionan en el nivel 3.

---

## Sección `training`

Configuración del entrenamiento.

| Propiedad | Tipo | Default | Descripción |
|---|---|---|---|
| `batch_size` | int | `8` | Tamaño de batch para DataLoader. |
| `epochs` | int | `20` | Número total de épocas. |
| `lr` | float | `1e-3` | Learning rate para Adam optimizer. |
| `num_workers` | int | `2` | Workers del DataLoader para carga paralela de datos. |
| `save_visuals_each_epoch` | bool | `true` | Si `true`, guarda grids de predicciones y filtros de varianza cada época. |
| `num_visual_samples` | int | `3` | Número de muestras en los grids visuales (predicciones y filtros). |
| `best_metric_name` | str | `"dice"` | Métrica para seleccionar el mejor modelo. Opciones: `"dice"`, `"iou"`, `"f1"`, `"precision"`, `"recall"`, `"hausdorff"`. |
| `resume_checkpoint_path` | str / null | `null` | Ruta a un checkpoint `.pt` para reanudar entrenamiento. Si es `null`, entrena desde cero. |

---

## Sección `discord`

Notificaciones al canal de Discord vía webhook. **Totalmente opcional** — si se omite o `webhook_url` es `null`, no se envían notificaciones.

| Propiedad | Tipo | Default | Descripción |
|---|---|---|---|
| `webhook_url` | str / null | `null` | URL completa del webhook de Discord (`https://discord.com/api/webhooks/...`). |
| `notify_every_n_epochs` | int | `1` | Enviar notificación cada N épocas. Ej: `5` notifica en épocas 5, 10, 15, 20. |

### ¿Qué se envía a Discord?

**Cada N épocas:**
- Reporte de métricas (train_loss, val_loss, dice, iou, precision, recall, f1, hausdorff)
- Marca ⭐ BEST si es la mejor época
- Imagen de curvas de entrenamiento
- Imagen de predicciones de validación

**Al finalizar:**
- Resumen completo con métricas de test
- Curvas finales de entrenamiento
- Predicciones de test
- Métricas por grupo de vértebras

---

## Estructura de salida en Drive

```
drive_root/experiment_name/
├── config.json
├── history.csv
├── summary.json
├── epoch_metrics.jsonl
├── plots/
│   ├── history_curves.png
│   └── epoch_XXX_curves.png
├── epoch_samples/
│   ├── epoch_XXX_train_predictions.png
│   └── epoch_XXX_val_predictions.png
├── filters/
│   └── epoch_XXX_val_sample_XXX.png
├── best_predictions/
│   ├── val_predictions.png
│   └── test_predictions.png
├── checkpoints/
│   ├── best_model.pt
│   └── last_model.pt
└── groups/
    ├── group_distribution.png
    ├── group_patch_samples.png
    ├── group_test_metrics.png
    └── test_group_metrics.json
```
