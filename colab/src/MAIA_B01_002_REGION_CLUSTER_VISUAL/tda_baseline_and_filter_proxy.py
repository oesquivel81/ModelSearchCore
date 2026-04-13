def normalize_filter_names(filter_names):
    """
    Normaliza filter_names a una lista de strings únicos.
    Soporta Series, lista, string único, None.
    """
    if filter_names is None:
        return []
    if isinstance(filter_names, str):
        return [filter_names]
    try:
        if isinstance(filter_names, pd.Series):
            return list(filter_names.dropna().unique())
    except Exception:
        pass
    if isinstance(filter_names, (list, tuple, set)):
        return list({str(f) for f in filter_names if f is not None})
    return [str(filter_names)]

import os
import pandas as pd
from pathlib import Path
from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_patch_combinations import generate_patch_combinations, evaluate_combination, ExperimentBundle, export_experiment_bundle

class TDABaselineAndFilterProxy:
    def _patch_to_region(self, patch, filter_name, config_id):
        from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_patch_combinations import RegionRecord
        # patch puede ser PatchPathDTO o PatchDTO
        # Propaga image_path, vertebra_idx, centroid_x/y, split, filter_name y config_id
        return RegionRecord(
            region_id=getattr(patch, 'patch_id', ''),
            patient_id=getattr(patch, 'patient_id', ''),
            config_id=config_id,
            filter_name=filter_name,
            image_path=getattr(patch, 'image_path', ''),
            vertebra_idx=getattr(patch, 'vertebra_idx', None) if hasattr(patch, 'vertebra_idx') else None,
            centroid_x=getattr(patch, 'centroid_x', None),
            centroid_y=getattr(patch, 'centroid_y', None),
            use_variance=None,
            variance_mode=None,
            patch_size=None,
            stride=None,
            variance_kernel=None,
            bbox=getattr(patch, 'bbox', None),
            centroid=(getattr(patch, 'centroid_x', None), getattr(patch, 'centroid_y', None)),
            curve_param=None,
            order_index=getattr(patch, 'vertebra_idx', None) if hasattr(patch, 'vertebra_idx') else None,
            lives_near_curve=None,
            split=getattr(patch, 'split', None) if hasattr(patch, 'split') else None,
            metadata={"optional_metadata": {}}
        )
    def __init__(self, config):
        self.config = config
        self.tda_root = config["tda_root"]
        self.patient_id = config["patient_id"]
        self.restrictions = config["restrictions"]
        self.experiment_modes = config["experiment_modes"]
        self.metrics = config["metrics"]
        self.patient_dir = os.path.join(self.tda_root, f"patches_processor_{self.patient_id}", self.patient_id, "bands")
        self.curve_csv = os.path.join(self.tda_root, f"patches_processor_{self.patient_id}", f"centroid_curve_{self.patient_id}.csv")
        # Si el config tiene 'filter_names', solo usa esos filtros
        if "filter_names" in config:
            clean_filters = normalize_filter_names(config["filter_names"])
            self.filters = [f for f in self._find_filters() if f in clean_filters]
        else:
            self.filters = self._find_filters()
        Path(self.patient_dir).mkdir(parents=True, exist_ok=True)

    def _find_filters(self):
        # Busca subcarpetas patch_images_{filtro} en bands
        bands_dir = os.path.join(self.tda_root, f"patches_processor_{self.patient_id}", self.patient_id, "bands")
        filters = []
        for d in os.listdir(bands_dir):
            if d.startswith("patch_images_"):
                filters.append(d.replace("patch_images_", ""))
        return filters

    def _load_patches(self, patch_dir):
        # Carga todos los parches como PatchPathDTO usando PatchDTOBuilder
        from extractor.patch_dto import PatchPathDTO
        # Asume que los nombres de los archivos siguen el patrón {patient_id}_patch_{idx:02d}.png
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir) if Path(f).suffix.lower() in exts]
        # Crear PatchPathDTO para cada archivo
        patches = []
        for f in files:
            patch_id = os.path.splitext(os.path.basename(f))[0]
            patches.append(
                PatchPathDTO(
                    patch_id=patch_id,
                    patient_id=self.patient_id,
                    image_path=f,
                    mask_path=None,  # Si tienes máscaras, puedes inferir el path aquí
                    bbox=None,
                    centroid_x=None,
                    centroid_y=None,
                    method="bands"
                )
            )
        return patches

    def run(self):
        # 1. Leer curva de centroides
        df_centroids = pd.read_csv(self.curve_csv)
        curve = df_centroids[["centroid_x", "centroid_y"]].values.tolist()
        # 2. Procesar baseline y cada filtro de manera independiente
        all_region_rows = []
        all_window_rows = []
        all_summaries = []
        for filtro in ["baseline"] + self.filters:
            if filtro == "baseline":
                patch_dir = os.path.join(self.patient_dir, "patch_images")
                config_id = "baseline"
            else:
                patch_dir = os.path.join(self.patient_dir, f"patch_images_{filtro}")
                config_id = filtro
            if not os.path.exists(patch_dir):
                continue
            patches = self._load_patches(patch_dir)
            print(f"Procesando filtro {filtro}: {len(patches)} parches")
            region_rows, window_rows, summary = self._run_tda_for_patches(patches, filtro, curve, config_id)
            all_region_rows.extend(region_rows)
            all_window_rows.extend(window_rows)
            all_summaries.extend(summary)
        # Exportar consolidado global
        outdir = os.path.join(self.patient_dir, "pre_tda_consolidado")
        os.makedirs(outdir, exist_ok=True)
        for row in all_region_rows:
            row['record_level'] = 'region'
        for row in all_window_rows:
            row['record_level'] = 'window'
        pd.DataFrame(all_region_rows).to_csv(os.path.join(outdir, 'pre_tda_regions_report.csv'), index=False)
        pd.DataFrame(all_window_rows).to_csv(os.path.join(outdir, 'pre_tda_windows_report.csv'), index=False)
        pd.DataFrame(all_summaries).to_csv(os.path.join(outdir, 'pre_tda_summary_report.csv'), index=False)
        pd.DataFrame(all_region_rows + all_window_rows).to_csv(os.path.join(outdir, 'master_pre_tda_table.csv'), index=False)

    def _run_tda_for_patches(self, patches, filter_name, curve, config_id):
        from MAIA_B01_002_REGION_CLUSTER_VISUAL import tda_patch_combinations as tda_utils
        from MAIA_B01_002_REGION_CLUSTER_VISUAL.pre_tda_metrics_builder import PreTDAMetricsBuilder
        import numpy as np
        import pandas as pd
        # 1. Poblar RegionRecord directamente desde el archivo de centroides (con métricas)
        centroid_path = tda_utils.find_centroid_curve_file(self.tda_root, self.patient_id)
        region_records = tda_utils.load_regions_from_centroid_csv(centroid_path, self.patient_id, config_id, filter_name)
        # 2. Ordenar espacialmente
        region_records = tda_utils.sort_regions_for_spatial_windows(region_records)
        # 3. Validar y mostrar muestra de regiones enriquecidas
        print(f"[VALIDACIÓN] Muestra de regiones para filtro '{filter_name}':")
        for r in region_records[:min(5, len(region_records))]:
            print({
                'region_id': r.region_id,
                'image_path': r.image_path,
                'patient_id': r.patient_id,
                'config_id': r.config_id,
                'filter_name': r.filter_name,
                'centroid': (r.centroid_x, r.centroid_y),
                'order_index': r.order_index,
                'mean_dice': getattr(r, 'mean_dice', None),
                'mean_iou': getattr(r, 'mean_iou', None)
            })
        # 4. Generar ventanas consecutivas
        combos_raw = tda_utils.generate_patch_combinations(
            region_records,
            min_k=self.restrictions['min_k'],
            max_k=self.restrictions['max_k'],
            max_combination_count=self.restrictions['max_combination_count']
        )
        # 5. Evaluar combinaciones y construir validity_records
        combo_records = []
        windows = []
        validity_records = []
        for _, c in combos_raw:
            eval_result = tda_utils.evaluate_combination(
                c,
                filter_params=None,
                selection_mode="consecutive_windows",
                experiment_mode="all_patches",
                restrictions=self.restrictions
            )
            combo_records.append(eval_result)
            windows.append(list(c))
            validity_records.append({
                'combination_id': eval_result.combination_id,
                'is_valid_simplex': eval_result.is_valid_simplex,
                'validity_reason': eval_result.validity_reason,
                'rejection_reason': eval_result.rejection_reason
            })
        # 6. Validar y mostrar muestra de ventanas
        print(f"[VALIDACIÓN] Muestra de ventanas para filtro '{filter_name}':")
        for i, (w, v) in enumerate(zip(windows, validity_records)):
            if i >= min(3, len(windows)):
                break
            print({
                'combination_id': v['combination_id'],
                'member_region_ids': [r.region_id for r in w],
                'member_image_paths': [r.image_path for r in w],
                'k': len(w),
                'filter_name': getattr(w[0], 'filter_name', None),
                'mean_dice': np.nanmean([getattr(r, 'mean_dice', np.nan) for r in w]),
                'mean_iou': np.nanmean([getattr(r, 'mean_iou', np.nan) for r in w])
            })
        # 7. Usar PreTDAMetricsBuilder para construir y exportar reportes
        prefix = f"{self.patient_id}_{filter_name}"
        outdir = os.path.join(self.patient_dir, f"pre_tda_{filter_name}")
        os.makedirs(outdir, exist_ok=True)
        metrics_builder = PreTDAMetricsBuilder(
            metric_columns=[
                "mean_dice", "mean_iou", "mean_mse_img", "mean_mae_img",
                "mean_grad_mse", "mean_grad_mae", "mean_var_diff", "mean_intensity_diff"
            ],
            selection_mode="spatial_consecutive_windows",
            ordering_source="vertebra_idx",
            spatial_file_used=centroid_path
        )
        regions_df = metrics_builder.build_regions_dataframe(region_records)
        windows_df = metrics_builder.build_windows_dataframe(windows, validity_records)
        summary_df = metrics_builder.build_summary_dataframe(regions_df, windows_df)
        paths = metrics_builder.export_all(
            output_dir=outdir,
            regions_df=regions_df,
            windows_df=windows_df,
            summary_df=summary_df,
            prefix=prefix
        )
        print(f"[EXPORT] Reportes previos a TDA y CSV maestro exportados en {outdir}")
        print(f"[EXPORT] Archivos generados: {paths}")
        return regions_df.to_dict(orient='records'), windows_df.to_dict(orient='records'), summary_df.to_dict(orient='records')
