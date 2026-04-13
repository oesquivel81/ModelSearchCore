def normalize_filter_names(filter_names):
    """
    Normaliza filter_names a una lista de strings únicos.
    Soporta Series, lista, string único, None.
    """
    if filter_names is None:
        return []
    if isinstance(filter_names, str):
        import pandas as pd
    try:
        import pandas as pd
        if isinstance(filter_names, pd.Series):
            return list(filter_names.dropna().unique())
    except ImportError:
        pass
    if isinstance(filter_names, (list, tuple, set)):
        return list({str(f) for f in filter_names if f is not None})
    return [str(filter_names)]

# === Métricas agregadas y relacionales por ventana ===
def compute_window_metrics(window, metric_names):
    """Calcula media, mediana, std, min, max, rango para cada métrica en la ventana."""
    import numpy as np
    results = {}
    for m in metric_names:
        vals = [getattr(r, m, np.nan) for r in window]
        arr = np.array([v for v in vals if v is not None])
        if arr.size == 0:
            results[f'window_{m}_mean'] = np.nan
            results[f'window_{m}_median'] = np.nan
            results[f'window_{m}_std'] = np.nan
            results[f'window_{m}_min'] = np.nan
            results[f'window_{m}_max'] = np.nan
            results[f'window_{m}_range'] = np.nan
        else:
            results[f'window_{m}_mean'] = float(np.nanmean(arr))
            results[f'window_{m}_median'] = float(np.nanmedian(arr))
            results[f'window_{m}_std'] = float(np.nanstd(arr))
            results[f'window_{m}_min'] = float(np.nanmin(arr))
            results[f'window_{m}_max'] = float(np.nanmax(arr))
            results[f'window_{m}_range'] = float(np.nanmax(arr) - np.nanmin(arr))
    return results

def compute_window_relational_metrics(window, metric_names):
    """Calcula diferencias absolutas y distancias relacionales entre miembros consecutivos de la ventana."""
    import numpy as np
    results = {}
    for m in metric_names:
        vals = [getattr(r, m, np.nan) for r in window]
        arr = np.array([v for v in vals if v is not None])
        if arr.size < 2:
            results[f'window_mean_abs_diff_{m}'] = np.nan
            results[f'window_max_abs_diff_{m}'] = np.nan
        else:
            diffs = np.abs(np.diff(arr))
            results[f'window_mean_abs_diff_{m}'] = float(np.nanmean(diffs))
            results[f'window_max_abs_diff_{m}'] = float(np.nanmax(diffs))
    # Distancias entre centroides consecutivos (robusto a tipos)
    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan
    centroids = [(safe_float(getattr(r, 'centroid_x', np.nan)), safe_float(getattr(r, 'centroid_y', np.nan))) for r in window]
    steps = [np.linalg.norm(np.subtract(centroids[i+1], centroids[i]))
             for i in range(len(centroids)-1)
             if not np.isnan(centroids[i][0]) and not np.isnan(centroids[i][1])
             and not np.isnan(centroids[i+1][0]) and not np.isnan(centroids[i+1][1])]
    if steps:
        results['window_mean_centroid_step'] = float(np.nanmean(steps))
        results['window_max_centroid_step'] = float(np.nanmax(steps))
    else:
        results['window_mean_centroid_step'] = np.nan
        results['window_max_centroid_step'] = np.nan
    return results

# === Exportación de reportes previos y CSV maestro único ===
def export_pre_tda_reports(regions, windows, summary, spatial, outdir):
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(regions).to_csv(os.path.join(outdir, 'pre_tda_regions_report.csv'), index=False)
    pd.DataFrame(windows).to_csv(os.path.join(outdir, 'pre_tda_windows_report.csv'), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(outdir, 'pre_tda_summary_report.csv'), index=False)
    pd.DataFrame(spatial).to_csv(os.path.join(outdir, 'pre_tda_spatial_traceability.csv'), index=False)

def export_master_pre_tda_table(region_rows, window_rows, outdir):
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    all_rows = region_rows + window_rows
    pd.DataFrame(all_rows).to_csv(os.path.join(outdir, 'master_pre_tda_table.csv'), index=False)
import os
import numpy as np
# === Utilidades para centroides y trazabilidad espacial ===
def find_centroid_curve_file(tda_root, patient_id):
    """Busca el archivo centroid_curve_<patient_id>.csv en tda_root."""
    pattern = os.path.join(tda_root, f"patches_processor_{patient_id}", f"centroid_curve_{patient_id}.csv")
    if os.path.exists(pattern):
        return pattern
    raise FileNotFoundError(f"No se encontró archivo de centroides para {patient_id} en {pattern}")

def load_centroid_curve_data(path, patient_id=None):
    """Carga el CSV de centroides y filtra por patient_id si aplica."""
    df = pd.read_csv(path)
    if 'patient_id' in df.columns and patient_id is not None:
        df = df[df['patient_id'] == patient_id]
    return df

def match_region_with_centroid_row(region, centroid_df):
    """Enlaza una región con su fila de centroides por vertebra_idx."""
    idx = getattr(region, 'vertebra_idx', None)
    if idx is None:
        return None, False
    row = centroid_df[centroid_df['vertebra_idx'] == idx]
    if row.empty:
        return None, False
    return row.iloc[0], True

def sort_regions_for_spatial_windows(regions):
    """Ordena regiones por: order_index, curve_param, vertebra_idx, centroid_y, centroid_x, region_id."""
    def key(r):
        return (
            getattr(r, 'order_index', None) if getattr(r, 'order_index', None) is not None else 1e9,
            getattr(r, 'curve_param', None) if getattr(r, 'curve_param', None) is not None else 1e9,
            getattr(r, 'vertebra_idx', None) if getattr(r, 'vertebra_idx', None) is not None else 1e9,
            getattr(r, 'centroid_y', None) if getattr(r, 'centroid_y', None) is not None else 1e9,
            getattr(r, 'centroid_x', None) if getattr(r, 'centroid_x', None) is not None else 1e9,
            str(getattr(r, 'region_id', ''))
        )
    return sorted(regions, key=key)
def load_regions_from_centroid_csv(csv_path: str, patient_id: str, config_id: str, filter_name: str = "") -> list:
    """
    Lee el archivo centroid_curve_<patient_id>.csv y construye RegionRecord para cada fila.
    Espera columnas: vertebra_idx, centroid_x, centroid_y, split, image_path
    """
    df = pd.read_csv(csv_path)
    regions = []
    for _, row in df.iterrows():
        regions.append(RegionRecord(
            region_id = str(row.get("vertebra_idx", "")),
            patient_id = patient_id,
            config_id = config_id,
            filter_name = filter_name,
            image_path = row.get("image_path", ""),
            vertebra_idx = row.get("vertebra_idx", None),
            centroid_x = row.get("centroid_x", None),
            centroid_y = row.get("centroid_y", None),
            use_variance = None,
            variance_mode = None,
            patch_size = None,
            stride = None,
            variance_kernel = None,
            bbox = None,
            centroid = (row.get("centroid_x", None), row.get("centroid_y", None)),
            curve_param = None,
            order_index = row.get("vertebra_idx", None),
            lives_near_curve = None,
            split = row.get("split", None),
            metadata = {"optional_metadata": {}}
        ))
    return regions
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import itertools
import json
import numpy as np

@dataclass
class RegionRecord:
    region_id: str
    patient_id: str
    config_id: str
    filter_name: str
    image_path: str
    vertebra_idx: Any
    centroid_x: Any
    centroid_y: Any
    order_index: Any = None
    curve_param: Any = None
    bbox: Any = None
    split: Any = None
    # Métricas por región
    mean_dice: Any = None
    mean_iou: Any = None
    mean_mse_img: Any = None
    mean_mae_img: Any = None
    mean_grad_mse: Any = None
    mean_grad_mae: Any = None
    mean_var_diff: Any = None
    mean_intensity_diff: Any = None
    # Otros campos
    use_variance: Any = None
    variance_mode: Any = None
    patch_size: Any = None
    stride: Any = None
    variance_kernel: Any = None
    centroid: Any = None
    lives_near_curve: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinationRecord:
    combination_id: str
    patient_id: str
    config_id: str
    filter_name: str
    k: int
    simplex_dim: int
    members: List[RegionRecord]
    member_indices: List[int]
    member_region_ids: List[str]
    member_image_paths: List[str]
    member_centroids: list = field(default_factory=list)
    centroid_span_y: Any = None
    centroid_span_x: Any = None
    mean_centroid_distance: Any = None
    max_centroid_distance: Any = None
    # Métricas agregadas y relacionales (dict)
    window_metrics: dict = field(default_factory=dict)
    window_relational_metrics: dict = field(default_factory=dict)
    is_contiguous_in_order: bool = None
    is_curve_consistent: bool = None
    is_valid_simplex: bool = None
    validity_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    selection_mode: str = "consecutive_windows"
    experiment_mode: str = "all_patches"
    ordering_source: str = "vertebra_idx"
    spatial_file_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimplexRecord:
    combination_id: str
    patient_id: str
    config_id: str
    k: int
    member_region_ids: List[str]
    simplex_dim: int
    experiment_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentBundle:
    patient_id: str
    config_id: str
    experiment_mode: str
    regions: List[RegionRecord]
    combinations: List[CombinationRecord]
    simplexes: List[SimplexRecord]
    filter_params: Dict[str, Any]
    # Optionally add matrices/tensors for TDA
    combination_tensor: Optional[Any] = None
    incidence_matrix: Optional[Any] = None
    adjacency_matrix: Optional[Any] = None
    simplex_membership_map: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

def select_regions(df: pd.DataFrame, mode: str, curve=None, curve_radius=25) -> List[RegionRecord]:
    # Dummy implementation, replace with your logic
    # Should return a list of RegionRecord
    regions = []
    for _, row in df.iterrows():
        regions.append(RegionRecord(
            region_id=row.get('region_id', ''),
            patient_id=row.get('patient_id', ''),
            config_id=row.get('config_id', ''),
            filter_name=row.get('filter_name', ''),
            use_variance=row.get('use_variance', None),
            variance_mode=row.get('variance_mode', None),
            patch_size=row.get('patch_size', None),
            stride=row.get('stride', None),
            variance_kernel=row.get('variance_kernel', None),
            bbox=row.get('bbox', None),
            centroid=row.get('centroid', None),
            curve_param=row.get('curve_param', None),
            order_index=row.get('order_index', None),
            lives_near_curve=row.get('lives_near_curve', None),
            metadata=row.get('metadata', {})
        ))
    return regions

def sort_regions_for_consecutive_windows(regions: List[RegionRecord]) -> List[RegionRecord]:
    """
    Ordena regiones por prioridad: order_index > curve_param > region_id (estable).
    """
    def region_sort_key(r: RegionRecord):
        if r.order_index is not None:
            return (0, r.order_index)
        elif r.curve_param is not None:
            return (1, r.curve_param)
        else:
            return (2, str(r.region_id))
    return sorted(regions, key=region_sort_key)

def generate_patch_combinations(
    regions: List[RegionRecord],
    min_k: int = 2,
    max_k: Optional[int] = None,
    max_combination_count: Optional[int] = None
) -> List[Tuple[int, Tuple[RegionRecord, ...]]]:
    """
    Genera solo ventanas consecutivas de regiones ordenadas.
    Para cada k en [min_k, max_k], produce ventanas deslizantes de paso 1.
    El total es sum(n - k + 1 for k in ...), nunca combinaciones arbitrarias.
    Si max_combination_count está definido, corta el total global.
    """
    n = len(regions)
    if max_k is None:
        max_k = n
    combos = []
    count = 0
    for k in range(min_k, max_k + 1):
        for i in range(n - k + 1):
            window = tuple(regions[i:i + k])
            combos.append((k, window))
            count += 1
            if max_combination_count is not None and count >= max_combination_count:
                return combos
    return combos

def generate_patch_combinations_lazy(
    regions: List[RegionRecord],
    min_k: int = 2,
    max_k: Optional[int] = None,
    max_combination_count: Optional[int] = None
):
    """
    Versión generadora (lazy) de ventanas consecutivas.
    """
    n = len(regions)
    if max_k is None:
        max_k = n
    count = 0
    for k in range(min_k, max_k + 1):
        for i in range(n - k + 1):
            window = tuple(regions[i:i + k])
            yield (k, window)
            count += 1
            if max_combination_count is not None and count >= max_combination_count:
                return

def evaluate_combination(combo: Tuple[RegionRecord, ...], filter_params: Dict[str, Any], selection_mode: str, experiment_mode: str, restrictions: Dict[str, Any]) -> CombinationRecord:
    # Dummy logic for geometric/topological properties
    k = len(combo)
    member_region_ids = [r.region_id for r in combo]
    member_indices = list(range(k))
    # Validar unicidad de filter_name y config_id
    filter_names = set(getattr(r, 'filter_name', None) for r in combo)
    config_ids = set(getattr(r, 'config_id', None) for r in combo)
    patient_ids = set(getattr(r, 'patient_id', None) for r in combo)
    if len(filter_names) != 1 or len(config_ids) != 1 or len(patient_ids) != 1:
        # Rechazar combinación y dejar trazabilidad
        return CombinationRecord(
            combination_id='|'.join(member_region_ids),
            patient_id=patient_ids.pop() if len(patient_ids) == 1 else '',
            config_id=config_ids.pop() if len(config_ids) == 1 else '',
            filter_name=filter_names.pop() if len(filter_names) == 1 else '',
            k=k,
            simplex_dim=k-1,
            members=list(combo),
            member_indices=member_indices,
            member_region_ids=member_region_ids,
            member_image_paths=[getattr(r, 'image_path', '') for r in combo],
            member_centroids=[],
            centroid_span_y=None,
            centroid_span_x=None,
            mean_centroid_distance=None,
            max_centroid_distance=None,
            window_metrics={},
            window_relational_metrics={},
            is_contiguous_in_order=False,
            is_curve_consistent=False,
            is_valid_simplex=False,
            validity_reason="rejected_mixed_filter_or_config",
            rejection_reason="Regiones de filtros/configs distintos en la misma combinación",
            selection_mode=selection_mode,
            experiment_mode=experiment_mode,
            ordering_source="vertebra_idx",
            spatial_file_used="",
            metadata={"rejected_mixed_filter_or_config": True}
        )
    filter_name = filter_names.pop()
    config_id = config_ids.pop()
    patient_id = patient_ids.pop()
    member_image_paths = [getattr(r, 'image_path', '') for r in combo]
    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan
    member_centroids = [(safe_float(getattr(r, 'centroid_x', np.nan)), safe_float(getattr(r, 'centroid_y', np.nan))) for r in combo]
    centroid_y_vals = [c[1] for c in member_centroids if not np.isnan(c[1])]
    centroid_x_vals = [c[0] for c in member_centroids if not np.isnan(c[0])]
    centroid_span_y = float(np.nanmax(centroid_y_vals) - np.nanmin(centroid_y_vals)) if centroid_y_vals else np.nan
    centroid_span_x = float(np.nanmax(centroid_x_vals) - np.nanmin(centroid_x_vals)) if centroid_x_vals else np.nan
    # Distancias entre centroides
    centroid_distances = [float(np.linalg.norm(np.subtract(member_centroids[i+1], member_centroids[i])))
                         for i in range(k-1)
                         if not np.isnan(member_centroids[i][0]) and not np.isnan(member_centroids[i][1])
                         and not np.isnan(member_centroids[i+1][0]) and not np.isnan(member_centroids[i+1][1])]
    mean_centroid_distance = float(np.nanmean(centroid_distances)) if centroid_distances else np.nan
    max_centroid_distance = float(np.nanmax(centroid_distances)) if centroid_distances else np.nan
    # Métricas agregadas y relacionales
    metric_names = [
        'mean_dice', 'mean_iou', 'mean_mse_img', 'mean_mae_img',
        'mean_grad_mse', 'mean_grad_mae', 'mean_var_diff', 'mean_intensity_diff'
    ]
    window_metrics = compute_window_metrics(combo, metric_names)
    window_relational_metrics = compute_window_relational_metrics(combo, metric_names)
    # Determinar si son realmente consecutivos en el orden
    order_indices = [r.order_index for r in combo]
    is_contiguous_in_order = all(
        (order_indices[i] is not None and order_indices[i+1] is not None and order_indices[i+1] - order_indices[i] == 1)
        for i in range(k-1)
    ) if all(r.order_index is not None for r in combo) else True
    simplex_dim = k-1
    # Validación de simplex: por ahora, solo por orden, dejar trazabilidad
    if is_contiguous_in_order:
        is_valid_simplex = False
        validity_reason = "pending_geometric_validation"
        rejection_reason = None
    else:
        is_valid_simplex = False
        validity_reason = "rejected_nonconsecutive"
        rejection_reason = "regiones no consecutivas"

    return CombinationRecord(
        combination_id='|'.join(member_region_ids),
        patient_id=patient_id,
        config_id=config_id,
        filter_name=filter_name,
        k=k,
        simplex_dim=simplex_dim,
        members=list(combo),
        member_indices=member_indices,
        member_region_ids=member_region_ids,
        member_image_paths=member_image_paths,
        member_centroids=member_centroids,
        centroid_span_y=centroid_span_y,
        centroid_span_x=centroid_span_x,
        mean_centroid_distance=mean_centroid_distance,
        max_centroid_distance=max_centroid_distance,
        window_metrics=window_metrics,
        window_relational_metrics=window_relational_metrics,
        is_contiguous_in_order=is_contiguous_in_order,
        is_curve_consistent=True,
        is_valid_simplex=is_valid_simplex,
        validity_reason=validity_reason,
        rejection_reason=rejection_reason,
        selection_mode="consecutive_windows",
        experiment_mode=experiment_mode,
        ordering_source="vertebra_idx",
        spatial_file_used="centroid_curve_<patient_id>.csv",
        metadata={}
    )

def build_region_dataframe(regions: List[RegionRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in regions])

def build_combination_dataframe(combos: List[CombinationRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in combos])

def build_simplex_dataframe(combos: List[CombinationRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in combos if c.is_valid_simplex])

def export_experiment_bundle(bundle: ExperimentBundle, outdir: str):
    # Export DataFrames
    df_regions = build_region_dataframe(bundle.regions)
    df_combos = build_combination_dataframe(bundle.combinations)
    df_simplexes = build_simplex_dataframe(bundle.combinations)
    df_regions.to_csv(f'{outdir}/regions.csv', index=False)
    df_combos.to_csv(f'{outdir}/combinations.csv', index=False)
    df_simplexes.to_csv(f'{outdir}/valid_simplices.csv', index=False)
    # Export JSON
    with open(f'{outdir}/experiment_bundle.json', 'w') as f:
        json.dump(asdict(bundle), f, default=str)

# Example usage (to be adapted in notebook):
# regions = select_regions(df, mode='all_patches')
# combos = generate_patch_combinations(regions, min_k=2, max_k=4)
# filter_params = {...}  # Fill with actual filter info
# restrictions = {...}   # Fill with actual restrictions
# combo_records = [evaluate_combination(c, filter_params, 'mode', 'experiment', restrictions) for _, c in combos]
# bundle = ExperimentBundle(
#     patient_id='P001',
#     config_id='C001',
#     experiment_mode='all_patches',
#     regions=regions,
#     combinations=combo_records,
#     simplexes=[SimplexRecord(
#         combination_id=cr.combination_id,
#         patient_id=cr.patient_id,
#         config_id=cr.config_id,
#         k=cr.k,
#         member_region_ids=cr.member_region_ids,
#         simplex_dim=cr.simplex_dim,
#         experiment_mode=cr.experiment_mode
#     ) for cr in combo_records if cr.is_valid_simplex],
#     filter_params=filter_params
# )
# export_experiment_bundle(bundle, outdir='output_dir')
