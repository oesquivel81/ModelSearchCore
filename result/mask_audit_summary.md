# Resumen de Auditoría de Máscaras — `mask_audit_light.csv`

**Total de imágenes analizadas:** 250

---

## Tipo de imagen

| Tipo | Cantidad |
|------|----------|
| Scoliosis | 179 |
| Normal | 71 |

---

## Estado (status)

| Estado | Cantidad |
|--------|----------|
| repair | 249 |
| discard | 1 |

La única imagen descartada es **LabelMulti_S_107** (Scoliosis), con razón: `id_empty`.

---

## Razones individuales (reasons)

Cada imagen puede tener múltiples razones. Desglose individual:

| Razón | Total | Normal | Scoliosis |
|-------|-------|--------|-----------|
| `id_empty` | 250 | 71 | 179 |
| `gray_noise` | 249 | 71 | 178 |
| `color_id_mismatch` | 249 | 71 | 178 |
| `too_many_quantized_colors` | 174 | 70 | 104 |

---

## Combinaciones de razones

| Combinación de razones | Cantidad |
|------------------------|----------|
| `too_many_quantized_colors` + `gray_noise` + `id_empty` + `color_id_mismatch` | 174 |
| `gray_noise` + `id_empty` + `color_id_mismatch` | 75 |
| `id_empty` (sola) | 1 |

---

## Observaciones

- **Todas** las imágenes (250/250) presentan la razón `id_empty`, lo que indica que la máscara ID está vacía en todo el dataset.
- El 99.6% (249/250) tienen `gray_noise` y `color_id_mismatch`, lo cual sugiere ruido generalizado en las máscaras de grises y discrepancia entre color e ID.
- El 69.6% (174/250) además tienen `too_many_quantized_colors`, indicando exceso de colores cuantizados en la máscara color — más frecuente en imágenes Scoliosis (104/179 = 58%) que en Normal (70/71 = 98.6%).
- Solo 1 imagen fue descartada; las 249 restantes se marcaron para reparación.
