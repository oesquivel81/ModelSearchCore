# Cuadernos

## Unet_final.ipynb — Clases utilizadas

| # | Clase | Herencia | Funcionalidad |
|---|-------|----------|---------------|
| 1 | `SpineEnhancementAndAdaptiveROIViewer` | — | Pipeline completo de preprocesamiento de radiografías: carga imagen, normalización, denoising bilateral, CLAHE, unsharp masking, refinamiento de máscara morfológico, detección de ROI rectangular y ROI anatómica adaptativa (centerline + banda), recorte y visualización de 8 pasos. |
| 2 | `CaseIdentity` | `@dataclass` | DTO que identifica un caso de paciente: `case_key` (ej. `S_21`), `prefix` (S/N), `patient_number`, `diagnosis` (scoliosis/normal). |
| 3 | `ImageCaseRecord` | `@dataclass` | Registro de un caso con rutas a máscara gris y radiografía, más identidad del paciente. |
| 4 | `GrayBlobInfo` | `@dataclass` | Información de un componente conexo extraído de una máscara gris: `class_id`, `class_name`, `area`, `bbox`, `centroid`. |
| 5 | `GrayClassSummary` | `@dataclass` | Resumen de auditoría por clase: conteo de píxeles, presencia, número de componentes, distancia media, flags de calidad (`weak`, `ambiguous`). |
| 6 | `GrayDuplicateReport` | `@dataclass` | Reporte de deduplicación entre componentes: áreas descartada/conservada, IoU, overlap, razón de descarte. |
| 7 | `ExtractedRegionInfo` | `@dataclass` | Metadata de una región extraída: caso, clase, índice de blob, área, bbox, ruta de crop guardado. |
| 8 | `VertebraGrayImageOnlyProcessor` | — | Procesador de máscaras multiclase en escala de grises: clasifica píxeles por gray nearest, extrae componentes conexos, deduplica regiones solapadas, audita clases, visualiza segmentación sobre radiografía. Soporta convención `S_num` / `N_num`. |
| 9 | `UNetExperimentConfig` | `@dataclass` | DTO de configuración del experimento: agrupa `experiment_name`, `drive_root`, `seed`, `execution_mode`, y dicts para `processor`, `data`, `model`, `training`, `checkpointing`, `reporting`, `runtime`. |
| 10 | `DiceLoss` | `nn.Module` | Función de pérdida Dice: 1 - (2·inter + ε)/(union + ε) sobre logits con sigmoid. |
| 11 | `BCEDiceLoss` | `nn.Module` | Pérdida combinada: `BCEWithLogitsLoss` + `DiceLoss`. |
| 12 | `SpineProcessorAdapter` | — | Adaptador que conecta `SpineEnhancementAndAdaptiveROIViewer.process()` al dataset. Selecciona la representación de entrada (`enhanced`, `roi_rect_crop`, `roi_adaptive_crop`, etc.) y su máscara target correspondiente. |
| 13 | `VertebraProcessorAdapter` | — | Adaptador que conecta `VertebraGrayImageOnlyProcessor.process_single()` al dataset. Retorna `segmented_gray` como imagen y `binary_mask` como target. |
| 14 | `ScoliosisSegmentationDataset` | `Dataset` | Dataset PyTorch: lee CSV, invoca el processor para cada sample, extrae imagen/máscara del resultado, redimensiona a `image_size`, normaliza, retorna tensores `[1, H, W]` + metadata (`patient_id`, `sample_id`, paths, arrays numpy). |
| 15 | `UNetExperimentService` | — | Servicio orquestador del experimento completo: setup de seed/directorios, construcción de processor/datasets/modelo/optimizer/scheduler/loss, entrenamiento con métricas por epoch, validación, test, checkpoints (best/last/interrupción/periódico), monitoreo de recursos (CPU/RAM/GPU/disco), tracking de muestras fijas, feature maps, filtros, report grids, resumen final JSON. API pública: `run()` y `resume(checkpoint_path)`. |

---

## JSON de configuración

El notebook usa dos configuraciones JSON. Aquí la estructura completa:

### `unet_spine.json` (modo `spine_enhancement`)

```json
{
  "experiment_name": "unet_spine_roi_v1",
  "drive_root": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset/resultados_experimentos",
  "seed": 42,
  "execution_mode": "spine_enhancement",

  "processor": {
    "type": "SpineEnhancementAndAdaptiveROIViewer",
    "params": {
      "bbox_pad_ratio": 0.08,
      "band_width_ratio": 0.18
    }
  },

  "data": {
    "base_dir": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset",
    "index_csv": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset/dataset_index.csv",
    "split_column": "split",
    "patient_id_column": "patient_id",
    "image_column": "radiograph_path",
    "mask_column": "label_binary_path",
    "image_size": [512, 256],
    "normalize": true,
    "input_representation": "roi_rect_crop"
  },

  "model": {
    "model_name": "unet",
    "in_channels": 1,
    "classes": 1,
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "activation": null
  },

  "training": {
    "epochs": 20,
    "batch_size": 8,
    "learning_rate": 0.0003,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "loss": "bce_dice",
    "num_workers": 2,
    "mixed_precision": false,
    "pin_memory": true,
    "best_metric_name": "dice"
  },

  "checkpointing": {
    "save_best": true,
    "save_last": true,
    "save_interrupt_checkpoint": true,
    "save_epoch_checkpoints": true,
    "epoch_checkpoint_frequency": 5
  },

  "reporting": {
    "save_visual_every_n_epochs": 5,
    "tracked_sample_count": 4,
    "save_feature_maps": true,
    "save_filters": true,
    "max_feature_maps_per_layer": 8,
    "max_filters_to_plot": 16
  },

  "runtime": {
    "device": "cuda",
    "deterministic": true,
    "benchmark": false
  }
}
```

### `unet_vertebra.json` (modo `vertebra_gray`)

```json
{
  "experiment_name": "unet_vertebra_gray_v1",
  "drive_root": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset/resultados_experimentos",
  "seed": 42,
  "execution_mode": "vertebra_gray",

  "processor": {
    "type": "VertebraGrayImageOnlyProcessor",
    "params": {
      "use_multiclass_gray": true,
      "use_multiclass_id": false,
      "keep_largest_component": true,
      "min_component_area": 20,
      "pad": 8
    }
  },

  "data": {
    "base_dir": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset",
    "index_csv": "/content/drive/MyDrive/MaIA_Scoliosis_Dataset/dataset_index.csv",
    "split_column": "split",
    "patient_id_column": "patient_id",
    "image_column": "radiograph_path",
    "mask_column": "multiclass_gray_jpg",
    "image_size": [256, 256],
    "normalize": true
  },

  "model": {
    "model_name": "unet",
    "in_channels": 1,
    "classes": 1,
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "activation": null
  },

  "training": {
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.0003,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "loss": "bce_dice",
    "num_workers": 2,
    "mixed_precision": false,
    "pin_memory": true,
    "best_metric_name": "dice"
  },

  "checkpointing": {
    "save_best": true,
    "save_last": true,
    "save_interrupt_checkpoint": true,
    "save_epoch_checkpoints": true,
    "epoch_checkpoint_frequency": 5
  },

  "reporting": {
    "save_visual_every_n_epochs": 5,
    "tracked_sample_count": 4,
    "save_feature_maps": true,
    "save_filters": true,
    "max_feature_maps_per_layer": 8,
    "max_filters_to_plot": 16
  },

  "runtime": {
    "device": "cuda",
    "deterministic": true,
    "benchmark": false
  }
}
```
