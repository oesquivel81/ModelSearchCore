import os
import pandas as pd
import json
from pathlib import Path
from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_patch_combinations import select_regions
from MAIA_B01_002_REGION_CLUSTER_VISUAL.region_structures import (
    select_regions_by_experiment,
    compute_region_intersections,
    build_nerve_simplicial_complex,
    build_region_table,
    plot_regions_curve_and_nerve
)

class TDAExperimentProxy:
    def __init__(self, config):
        self.config = config
        self.centroid_csv = config["centroid_csv"]
        self.experiment_modes = config["experiment_modes"]
        self.curve_radius = config.get("curve_radius", 25)
        self.metrics = config.get("metrics", [])
        self.output_dir = config["output_dir"]
        self.filters = config.get("filters", [])
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        # 1. Leer centroides
        df_centroids = pd.read_csv(self.centroid_csv)
        curve = df_centroids[["centroid_x", "centroid_y"]].values.tolist()
        patient_id = df_centroids["patient_id"].iloc[0] if "patient_id" in df_centroids.columns else "unknown"
        split = df_centroids["split"].iloc[0] if "split" in df_centroids.columns else "unspecified"

        # 2. Inferir base de datos y rutas
        base_dir = str(Path(self.centroid_csv).parents[2])
        filters_dir = base_dir.replace("MAIA_B01_001_BL_ch32_lr3_px30", "MAIA_B01_001_BL_FILTERS")

        # 3. Buscar el archivo de métricas y top configs
        metrics_csv = os.path.join(filters_dir, f"metrics_ablation_{patient_id}.csv")
        if not os.path.exists(metrics_csv):
            raise FileNotFoundError(f"No se encontró el archivo de métricas: {metrics_csv}")
        df = pd.read_csv(metrics_csv)
        # Top por cluster (puedes ajustar el criterio)
        top_por_cluster = df.sort_values("score_cluster", ascending=False).groupby("cluster", as_index=False).head(1).copy()
        mejores_ids = top_por_cluster['config_id'].tolist()
        df_mejores = df[df['config_id'].isin(mejores_ids)].copy()
        regions_mejores = select_regions(df_mejores, mode="all_patches")

        # 4. Ejecutar experimentos
        for modo in self.experiment_modes:
            print(f"\n=== Experimento SOLO MEJORES: {modo} ===")
            selected = select_regions_by_experiment(regions_mejores, modo, curve=curve, curve_radius=self.curve_radius)
            selected = [r for r in selected if getattr(r, 'bbox', None) is not None]
            compute_region_intersections(selected)
            simplexes = build_nerve_simplicial_complex(selected, max_dim=2)
            df_sel = build_region_table(selected)
            # Guardar métricas
            out_csv = os.path.join(self.output_dir, f"tda_{modo}_{patient_id}.csv")
            df_sel.to_csv(out_csv, index=False)
            print(f"Guardado: {out_csv}")
            # Guardar plot
            plot_path = os.path.join(self.output_dir, f"tda_{modo}_{patient_id}.png")
            plot_regions_curve_and_nerve(
                image=None,
                regions=selected,
                curve=curve,
                simplexes=simplexes,
                show_only_selected=False
            )
            # plt.savefig(plot_path)  # Descomenta si quieres guardar la figura

        # 5. (Opcional) Regenerar filtros si se pasan
        if self.filters:
            print("\nRegenerando filtros especificados:")
            for filtro in self.filters:
                print(f"Filtro: {filtro}")
                # Aquí puedes llamar a tu lógica de regeneración de filtros si lo necesitas
                # Por ejemplo: regenerate_filter(filtro, ...)
