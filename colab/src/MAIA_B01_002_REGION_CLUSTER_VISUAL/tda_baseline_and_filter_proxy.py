import os
import pandas as pd
from pathlib import Path
from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_patch_combinations import generate_patch_combinations, evaluate_combination, ExperimentBundle, export_experiment_bundle

class TDABaselineAndFilterProxy:
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
            self.filters = [f for f in self._find_filters() if f in config["filter_names"]]
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
        # 2. Procesar baseline
        baseline_dir = os.path.join(self.patient_dir, "patch_images")
        baseline_patches = self._load_patches(baseline_dir)
        print(f"Procesando baseline: {len(baseline_patches)} parches")
        self._run_tda_for_patches(baseline_patches, "baseline", curve)
        # 3. Procesar por filtro
        for filtro in self.filters:
            patch_dir = os.path.join(self.patient_dir, f"patch_images_{filtro}")
            if not os.path.exists(patch_dir):
                continue
            patches = self._load_patches(patch_dir)
            print(f"Procesando filtro {filtro}: {len(patches)} parches")
            self._run_tda_for_patches(patches, filtro, curve)

    def _run_tda_for_patches(self, patches, label, curve):
        # Aquí deberías implementar la lógica de combinaciones y nervio usando las restricciones
        # Por cada combinación válida de parches (según restricciones), calcula el nervio y métricas
        # Este es un esqueleto, debes adaptar a tu lógica de TDA
        combos_raw = generate_patch_combinations(
            patches,
            min_k=self.restrictions['min_k'],
            max_k=self.restrictions['max_k'],
            max_combination_count=self.restrictions['max_combination_count']
        )
        combo_records = [
            evaluate_combination(
                c,
                filter_params=None,
                selection_mode="default",
                experiment_mode="all_patches",
                restrictions=self.restrictions
            )
            for _, c in combos_raw
        ]
        bundle = ExperimentBundle(
            patient_id=self.patient_id,
            config_id=label,
            experiment_mode="all_patches",
            regions=patches,
            combinations=combo_records,
            simplexes=[s for s in combo_records if getattr(s, 'is_valid_simplex', False)],
            filter_params=None
        )
        outdir = os.path.join(self.patient_dir, f"tda_{label}")
        os.makedirs(outdir, exist_ok=True)
        export_experiment_bundle(bundle, outdir=outdir)
        print(f"Resultados TDA guardados en {outdir}")
