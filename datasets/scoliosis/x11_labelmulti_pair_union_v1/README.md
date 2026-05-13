# X11 LabelMulti Pair-Union Shards

Dataset congelado para entrenamiento rápido de CNN multihead de escoliosis.

## Entrada X

X shape:

`X: [B, 11, 224, 224]`

Canales:

0. baseline_raw
1. robust_mad_image
2. band_mask
3. balanced_edge
4. oriented_centered_edge
5. distance_score_band
6. t_map_band
7. normal_x_band
8. normal_y_band
9. tangent_x_band
10. tangent_y_band

## Targets

`y_multiclass: [B, 224, 224] int64, clases 0..24`

`y_binary: [B, 1, 224, 224] float32`

`y_boundary: [B, 1, 224, 224] float32`

`y_intervertebral: [B, 1, 224, 224] float32`

`y_ordinal: [B, 1, 224, 224] float32`

## Estructura

- `shards_npz/`
- `tables/`
- `config/`

## Uso rápido

```python
import numpy as np

data = np.load('shards_npz/shard_00000.npz', allow_pickle=True)

X = data['X']
y_multiclass = data['y_multiclass']
y_binary = data['y_binary']
y_boundary = data['y_boundary']
y_intervertebral = data['y_intervertebral']
y_ordinal = data['y_ordinal']

print(X.shape)
print(data['channel_names'])
```
