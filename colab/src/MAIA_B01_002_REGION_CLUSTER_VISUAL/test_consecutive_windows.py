# Test mínimo: ventanas consecutivas

from MAIA_B01_002_REGION_CLUSTER_VISUAL.tda_patch_combinations import RegionRecord, generate_patch_combinations, sort_regions_for_consecutive_windows

# Crear 9 regiones ordenadas
regions = [RegionRecord(
    region_id=f"R{i+1}",
    patient_id="P1",
    config_id="cfg",
    filter_name="f",
    use_variance=None,
    variance_mode=None,
    patch_size=None,
    stride=None,
    variance_kernel=None,
    bbox=None,
    centroid=None,
    curve_param=None,
    order_index=i,
    lives_near_curve=None,
    metadata={}
) for i in range(9)]

regions = sort_regions_for_consecutive_windows(regions)
combos = generate_patch_combinations(regions, min_k=2, max_k=4)

print(f"Total ventanas: {len(combos)}")
print("Ejemplo de ventanas para k=2:")
for k, c in combos:
    if k == 2:
        print([r.region_id for r in c])
# Debe imprimir 8 ventanas para k=2, 7 para k=3, 6 para k=4 (total 21)
