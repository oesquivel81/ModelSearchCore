# Clases Paramétricas Reutilizables

## utils/

| Clase | Parámetros | Descripción |
|---|---|---|
| `DiscordWebhookNotifier` | `webhook_url`, `experiment_name="experiment"` | Envía métricas e imágenes a Discord vía webhook |
| `VariancePatchBuilderV2` | `patch_size=(128,128)`, `variance_ksize=5`, `save_root=None`, `make_subpatches=False`, `subpatch_size=(64,64)`, `subpatch_stride=(64,64)` | Construye tensores baseline/variance_input/variance_branch con varianza local |

### utils/metrics.py (funciones)

| Función | Parámetros | Descripción |
|---|---|---|
| `dice_from_probs` | `preds, targets, eps=1e-8` | Dice coefficient sobre tensores |
| `iou_from_probs` | `preds, targets, eps=1e-8` | IoU sobre tensores |
| `precision_from_probs` | `preds, targets, eps=1e-8` | Precision sobre tensores |
| `recall_from_probs` | `preds, targets, eps=1e-8` | Recall sobre tensores |
| `f1_from_precision_recall` | `precision, recall, eps=1e-8` | F1 a partir de precision y recall |
| `hausdorff_distance_binary` | `pred, target` | Distancia de Hausdorff entre máscaras binarias |

### utils/helpers.py (funciones)

| Función | Parámetros | Descripción |
|---|---|---|
| `utc_now_iso` | — | Timestamp UTC en formato ISO |
| `set_seed` | `seed=42` | Fija semillas de random, numpy, torch |
| `ensure_dir` | `path` | Crea directorio si no existe |
| `save_json` | `path, data` | Guarda dict como JSON |
| `append_jsonl` | `path, data` | Agrega línea JSON a archivo |
| `normalize_split_value` | `x` | Normaliza "training"→"train", "validation"→"val", etc. |
| `get_disk_free_gb` | `path="."` | GB libres en disco |
| `get_system_metrics` | — | CPU, RAM, GPU, disco |

---

## extractor/

| Clase | Parámetros | Descripción |
|---|---|---|
| `VertebraRegionExtractor` | `base_dir`, `image_col="radiograph_path"`, `mask_col="label_binary_path"`, `min_area=50`, `pad_x=30`, `pad_y=15`, `save_root=None` | Extrae regiones vertebrales de imágenes/máscaras y guarda pares con CSV de metadata |
| `VertebraSubpatchGenerator` | `patch_size=(128,128)`, `subpatch_size=(32,32)`, `stride=(32,32)`, `save_root=None` | Genera subpatches con ventana deslizante sobre regiones vertebrales |

### Dataclasses

| Dataclass | Campos |
|---|---|
| `VertebraRecord` | `study_id, split, vertebra_idx, vertebra_id, centroid_x, centroid_y, area, bbox_x1/y1/x2/y2, vertebra_img_path, vertebra_mask_path` |
| `SubpatchRecord` | `study_id, split, vertebra_id, vertebra_idx, subpatch_idx, subpatch_id, grid_row, grid_col, x1/y1/x2/y2, subpatch_img_path, subpatch_mask_path` |

---

## varianza_patches_cnn/

### Experimento

| Clase | Parámetros | Descripción |
|---|---|---|
| `VertebraVarianceExperimentV2` | `cfg: dict` | Orquestador completo: extracción → varianza → CNN → métricas → Discord |

### Extractores y Datasets

| Clase | Parámetros | Descripción |
|---|---|---|
| `VertebraPatchExtractor` | `base_dir`, `index_csv`, `image_col`, `mask_col`, `split_col`, `min_area=50`, `pad_x=30`, `pad_y=15`, `include_labels=None`, `save_root=None` | Extrae componentes conectados y guarda patches con metadata |
| `VariancePatchProcessor` | `patch_size=(128,128)`, `variance_ksize=5`, `save_root=None` | Procesador legacy de varianza (V1) |
| `VertebraPatchDataset` | `metadata_df`, `processor`, `model_type="baseline"`, `label_mode="ordinal_13"` | Dataset PyTorch para clasificación (V1) |
| `VertebraPatchDatasetV2` | `metadata_df`, `builder`, `model_type="baseline"` | Dataset PyTorch para clasificación (V2, usa builder) |
| `PatchRecord` | dataclass | Metadata de un patch extraído |

### Modelos CNN (clasificación)

| Clase | Parámetros | Descripción |
|---|---|---|
| `ConvEncoder` | `in_channels=1`, `base_channels=32` | Encoder 3 bloques conv-bn-relu-pool + adaptive avg pool |
| `BaselinePatchCNN` | `base_channels=32`, `num_classes=13` | CNN 1 canal → clasificación ordinal |
| `VarianceInputPatchCNN` | `base_channels=32`, `num_classes=13` | CNN 2 canales (imagen + varianza) |
| `VarianceBranchPatchCNN` | `base_channels=32`, `num_classes=13` | CNN dual-encoder (rama imagen + rama varianza) |

### Grid de Experimentos

| Función | Parámetros | Descripción |
|---|---|---|
| `build_experiment_batch` | `model_types, patch_sizes, pad_xs, pad_ys, lrs, base_channels_list, batch_sizes, epochs_list, variance_ksizes, seeds, min_areas` | Genera producto cartesiano de hiperparámetros |
| `run_experiment_grid` | `base_config, experiment_batch` | Ejecuta batch secuencialmente con notificaciones Discord |

---

## image_utils/

### Experimento UNet

| Clase | Parámetros | Descripción |
|---|---|---|
| `VarianceUNetRegionExperiment` | `config: dict` | Entrena UNet sobre regiones vertebrales agrupadas por nº vértebras |

### Extracción y Datos

| Clase | Parámetros | Descripción |
|---|---|---|
| `VertebraComponentExtractor` | `image`, `local_mask`, `min_area=150`, `pad_x=20`, `pad_y=15`, `top_region_ratio=0.30`, `top_pad_x_scale=1.25`, `top_pad_y_top_scale=2.2`, `top_pad_y_bottom_scale=0.8`, `save_dir=None` | Extrae componentes vertebrales con clasificación de calidad y bbox adaptativo |
| `VertebraRegionBatch` | `image_paths`, `mask_paths`, `image_names=None`, `min_area=150`, `pad_x=30`, `pad_y=15`, ... , `mode="tight"`, `include_labels=None`, `save_dir=None` | Procesa múltiples imágenes y agrupa regiones vertebrales |
| `VertebraRegionDataset` | `regions`, `patch_size=(128,128)`, `binarize_mask=True` | Dataset PyTorch para segmentación UNet |
| `VertebraRegion` | `idx, patch_img, patch_mask, bbox, area, centroid_x, centroid_y, quality_label, quality_score` | Contenedor inmutable de una región vertebral |
| `VertebraRegionProxy` | `regions` | Wrapper sobre VertebraComponentExtractor con filtrado por calidad |

### Modelos UNet (segmentación)

| Clase | Parámetros | Descripción |
|---|---|---|
| `LocalVarianceLayer` | `kernel_size=5` | Varianza local como capa nn.Module |
| `VarianceInputLayer` | `kernel_sizes=(3,5,9)` | Multi-escala varianza + concatenación |
| `UNetBaseline` | `in_channels=1`, `out_channels=1`, `base=32` | UNet estándar 4 niveles |
| `UNetVarianceInput` | `in_channels=1`, `out_channels=1`, `base=32` | UNet con entrada de varianza multi-escala |
| `UNetVarianceBranch` | `in_channels=1`, `out_channels=1`, `base=32` | UNet dual-branch (imagen + varianza) |

---

## Resumen

| Paquete | Clases | Dataclasses | nn.Module | Datasets |
|---|---|---|---|---|
| `utils` | 2 | — | — | — |
| `extractor` | 2 | 2 | — | — |
| `varianza_patches_cnn` | 7 | 1 | 4 | 2 |
| `image_utils` | 8 | — | 8 | 1 |
| **Total** | **19** | **3** | **12** | **3** |
