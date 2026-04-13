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
    use_variance: Any
    variance_mode: Any
    patch_size: Any
    stride: Any
    variance_kernel: Any
    bbox: Any
    centroid: Any
    curve_param: Any
    order_index: Any
    lives_near_curve: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Add more fields as needed

@dataclass
class CombinationRecord:
    combination_id: str
    patient_id: str
    config_id: str
    k: int
    members: List[RegionRecord]
    member_indices: List[int]
    member_region_ids: List[str]
    filter_name: str
    filter_params: Dict[str, Any]
    curve_params: List[Any]
    order_indices: List[Any]
    bbox_union: Any
    bbox_intersection: Any
    intersection_area: float
    overlap_ratio: float
    centroid_distances: List[float]
    curve_span: float
    is_contiguous_in_order: bool
    is_curve_consistent: bool
    is_valid_simplex: bool
    selection_mode: str
    experiment_mode: str
    validity_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    simplex_dim: Optional[int] = None
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
    # Determinar si son realmente consecutivos en el orden
    order_indices = [r.order_index for r in combo]
    is_contiguous_in_order = all(
        (order_indices[i] is not None and order_indices[i+1] is not None and order_indices[i+1] - order_indices[i] == 1)
        for i in range(k-1)
    ) if all(r.order_index is not None for r in combo) else True  # Si no hay order_index, asumimos True

    member_indices = list(range(k))
    filter_name = combo[0].filter_name if combo else ''
    patient_id = combo[0].patient_id if combo else ''
    config_id = combo[0].config_id if combo else ''
    bbox_union = None
    bbox_intersection = None
    intersection_area = 0.0
    overlap_ratio = 0.0
    centroid_distances = [0.0] * k
    curve_params = [r.curve_param for r in combo]
    curve_span = 0.0
    simplex_dim = k-1
    # Validación de simplex: por ahora, solo por orden, dejar trazabilidad
    if is_contiguous_in_order:
        # Aquí se podría validar intersección real si hay datos geométricos
        # Por ahora, solo dejamos trazabilidad
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
        k=k,
        members=list(combo),
        member_indices=member_indices,
        member_region_ids=member_region_ids,
        filter_name=filter_name,
        filter_params=filter_params,
        curve_params=curve_params,
        order_indices=order_indices,
        bbox_union=bbox_union,
        bbox_intersection=bbox_intersection,
        intersection_area=intersection_area,
        overlap_ratio=overlap_ratio,
        centroid_distances=centroid_distances,
        curve_span=curve_span,
        is_contiguous_in_order=is_contiguous_in_order,
        is_curve_consistent=True,
        is_valid_simplex=is_valid_simplex,
        selection_mode="consecutive_windows",
        experiment_mode=experiment_mode,
        validity_reason=validity_reason,
        rejection_reason=rejection_reason,
        simplex_dim=simplex_dim,
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
