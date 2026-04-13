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
        import pandas as pd
        if isinstance(filter_names, pd.Series):
            return list(filter_names.dropna().unique())
    except ImportError:
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
        import pandas as pd
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
        # 1. Convertir PatchPathDTO a RegionRecord y enriquecer con centroides y métricas
        centroid_path = tda_utils.find_centroid_curve_file(self.tda_root, self.patient_id)
        centroid_df = tda_utils.load_centroid_curve_data(centroid_path, self.patient_id)
        region_records = []
        for p in patches:
            region = self._patch_to_region(p, filter_name, config_id)
            centroid_row, matched = tda_utils.match_region_with_centroid_row(region, centroid_df)
            # Enriquecer con centroid_x, centroid_y, vertebra_idx, split, etc.
            if matched:
                region.centroid_x = centroid_row['centroid_x']
                region.centroid_y = centroid_row['centroid_y']
                region.vertebra_idx = centroid_row['vertebra_idx']
                region.order_index = centroid_row['vertebra_idx']
                region.split = centroid_row['split'] if 'split' in centroid_row else None
                region.metadata = {
                    "split": centroid_row['split'] if 'split' in centroid_row else None,
                    "spatial_file_used": centroid_path,
                    "optional_metadata": {}
                }
            else:
                region.metadata = {
                    "split": None,
                    "spatial_file_used": centroid_path,
                    "optional_metadata": {"centroid_match": False}
                }
            # Propagar métricas si existen en patch o centroid_row
            for m in [
                'mean_dice', 'mean_iou', 'mean_mse_img', 'mean_mae_img',
                'mean_grad_mse', 'mean_grad_mae', 'mean_var_diff', 'mean_intensity_diff']:
                val = getattr(p, m, None)
                if val is None and matched and m in centroid_row:
                    val = centroid_row[m]
                setattr(region, m, val)
            region_records.append(region)
        # 2. Ordenar espacialmente
        region_records = tda_utils.sort_regions_for_spatial_windows(region_records)
        # 3. Generar ventanas consecutivas
        combos_raw = tda_utils.generate_patch_combinations(
            region_records,
            min_k=self.restrictions['min_k'],
            max_k=self.restrictions['max_k'],
            max_combination_count=self.restrictions['max_combination_count']
        )
        # 4. Calcular métricas agregadas y relacionales por ventana
        combo_records = [
            tda_utils.evaluate_combination(
                c,
                filter_params=None,
                selection_mode="consecutive_windows",
                experiment_mode="all_patches",
                restrictions=self.restrictions
            )
            for _, c in combos_raw
        ]
        # 5. Exportar reportes previos y CSV maestro único
        outdir = os.path.join(self.patient_dir, f"pre_tda_{filter_name}")
        os.makedirs(outdir, exist_ok=True)
        # a) Reporte de regiones
        region_rows = [r.__dict__.copy() for r in region_records]
        # b) Reporte de ventanas
        window_rows = []
        for w in combo_records:
            row = w.__dict__.copy()
            # Aplanar window_metrics y window_relational_metrics
            if 'window_metrics' in row and isinstance(row['window_metrics'], dict):
                row.update(row.pop('window_metrics'))
            if 'window_relational_metrics' in row and isinstance(row['window_relational_metrics'], dict):
                row.update(row.pop('window_relational_metrics'))
            window_rows.append(row)
        # c) Reporte resumen
        summary = [{
            "patient_id": self.patient_id,
            "config_id": config_id,
            "filter_name": filter_name,
            "num_regions": len(region_records),
            "num_windows": len(window_rows),
            "num_valid_simplices": sum(1 for w in combo_records if w.is_valid_simplex),
            "k_min": self.restrictions['min_k'],
            "k_max": self.restrictions['max_k'],
            "ordering_source": "vertebra_idx",
            "spatial_file_used": centroid_path
        }]
        # d) Reporte de trazabilidad espacial
        spatial = [
            {
                "region_id": r.region_id,
                "image_path": r.image_path,
                "patient_id": r.patient_id,
                "vertebra_idx": r.vertebra_idx,
                "centroid_x": r.centroid_x,
                "centroid_y": r.centroid_y,
                "split": r.split,
                "config_id": r.config_id,
                "filter_name": r.filter_name
            } for r in region_records
        ]
        # e) Exportar reportes previos
        tda_utils.export_pre_tda_reports(region_rows, window_rows, summary, spatial, outdir)
        # f) Exportar CSV maestro único
        for row in region_rows:
            row['record_level'] = 'region'
        for row in window_rows:
            row['record_level'] = 'window'
        tda_utils.export_master_pre_tda_table(region_rows, window_rows, outdir)
        print(f"Reportes previos a TDA y CSV maestro exportados en {outdir}")
        return region_rows, window_rows, summary
