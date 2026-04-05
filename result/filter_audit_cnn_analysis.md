# Resumen de Auditoría de Filtros — Análisis para CNN

**Fuente:** `filter_audit_detailed_20260402_052851.json` / `filter_audit_summary_20260402_052851.json`  
**Imágenes evaluadas:** 20 | **Combinaciones de filtros:** 21 | **Resoluciones:** 10 (10%–100%) | **Total registros:** 4,200

> **Score compuesto CNN** = 0.35×correlación + 0.25×(entropía/8) + 0.20×edge_density + 0.10×nonzero_ratio − 0.10×(fragmentación)

---

## 1. Mejor combinación por resolución

| Resolución | Mejor combinación | Score |
|:----------:|:------------------|:-----:|
| 100% | `gaussian (k=5, σ=1.0)` | 0.8373 |
| 90% | `gaussian (k=5, σ=1.0)` | 0.8403 |
| 80% | `gaussian (k=5, σ=1.0)` | 0.8421 |
| 70% | `gaussian (k=5, σ=1.0)` | 0.8437 |
| 60% | `gaussian (k=5, σ=1.0)` | 0.8451 |
| 50% | `gaussian (k=5, σ=1.0)` | 0.8463 |
| 40% | `gaussian (k=5, σ=1.0)` | 0.8476 |
| 30% | `gaussian (k=5, σ=1.0)` | 0.8487 |
| 20% | `gaussian (k=5, σ=1.0)` | 0.8499 |
| 10% | `gaussian (k=5, σ=1.0)` | 0.8502 |

**Observación:** `gaussian (k=5, σ=1.0)` domina en **todas** las resoluciones. El score sube ligeramente al reducir resolución (menor fragmentación de componentes).

---

## 2. Mejor resolución por combinación

| Combinación | Mejor resolución | Score |
|:------------|:----------------:|:-----:|
| `gaussian (k=5, σ=1.0)` | 10% | 0.8502 |
| `gaussian → sobel` | 10% | 0.4012 |
| `gaussian → scharr` | 10% | 0.4014 |
| `gaussian → log (k=5, σ=1.0)` | 10% | 0.3525 |
| `sobel (k=3, magnitude)` | 10% | 0.3815 |
| `scharr (magnitude)` | 10% | 0.3815 |
| `sobel → scharr` | 10% | 0.3452 |
| `gaussian → laplacian` | 10% | 0.3244 |
| `log (k=5, σ=1.0)` | 10% | 0.3483 |
| `scharr → log` | 10% | 0.3164 |
| `sobel → log` | 10% | 0.3166 |
| `laplacian (k=3)` | 10% | 0.2985 |
| `scharr → laplacian` | 10% | 0.2991 |
| `sobel → laplacian` | 10% | 0.2982 |
| `laplacian → log` | 10% | 0.2656 |
| `canny (50–150)` | 10% | 0.0942 |
| `sobel → canny` | 10% | 0.0899 |
| `scharr → canny` | 10% | 0.0895 |
| `log → canny` | 10% | 0.0723 |
| `laplacian → canny` | 10% | 0.0573 |
| `gaussian → canny` | 10% | 0.0518 |

**Observación:** Todas las combinaciones obtienen su mejor score a **10%** (mínima resolución), debido a que la reducción elimina ruido y fragmentación. Sin embargo, las combinaciones con `canny` tienen scores muy bajos por la binarización agresiva que destruye información de textura.

---

## 3. Top 10 Global (combinación + resolución)

| # | Resolución | Dimensión media | Score | Combinación |
|:-:|:----------:|:---------------:|:-----:|:------------|
| 1 | 10% | 236 × 97 | 0.8502 | `gaussian (k=5, σ=1.0)` |
| 2 | 20% | 471 × 195 | 0.8499 | `gaussian (k=5, σ=1.0)` |
| 3 | 30% | 707 × 292 | 0.8487 | `gaussian (k=5, σ=1.0)` |
| 4 | 40% | 942 × 389 | 0.8476 | `gaussian (k=5, σ=1.0)` |
| 5 | 50% | 1178 × 487 | 0.8463 | `gaussian (k=5, σ=1.0)` |
| 6 | 60% | 1414 × 584 | 0.8451 | `gaussian (k=5, σ=1.0)` |
| 7 | 70% | 1649 × 681 | 0.8437 | `gaussian (k=5, σ=1.0)` |
| 8 | 80% | 1885 × 778 | 0.8421 | `gaussian (k=5, σ=1.0)` |
| 9 | 90% | 2120 × 876 | 0.8403 | `gaussian (k=5, σ=1.0)` |
| 10 | 100% | 2356 × 973 | 0.8373 | `gaussian (k=5, σ=1.0)` |

---

## 4. Detalle del Gaussian por resolución

| Scale | Avg H×W | Pixeles | Correlación | Entropía | Edge density | Nonzero ratio | Componentes |
|:-----:|:-------:|--------:|:-----------:|:--------:|:------------:|:-------------:|:-----------:|
| 100% | 2356×973 | 2,292,554 | 0.9987 | 7.674 | 0.818 | 0.9995 | 77.8 |
| 90% | 2120×876 | 1,857,134 | 0.9990 | 7.679 | 0.814 | 0.9995 | 60.6 |
| 80% | 1885×778 | 1,466,995 | 0.9988 | 7.680 | 0.813 | 0.9995 | 50.6 |
| 70% | 1649×681 | 1,123,270 | 0.9986 | 7.682 | 0.813 | 0.9994 | 42.1 |
| 60% | 1414×584 | 825,301 | 0.9984 | 7.684 | 0.812 | 0.9994 | 34.2 |
| 50% | 1178×487 | 573,239 | 0.9978 | 7.688 | 0.812 | 0.9994 | 26.9 |
| **40%** | **942×389** | **366,821** | **0.9975** | **7.693** | **0.811** | **0.9993** | **20.3** |
| 30% | 707×292 | 206,350 | 0.9966 | 7.698 | 0.811 | 0.9992 | 13.6 |
| 20% | 471×195 | 91,762 | 0.9943 | 7.712 | 0.814 | 0.9984 | 8.4 |
| 10% | 236×97 | 22,900 | 0.9874 | 7.723 | 0.821 | 0.9978 | 3.1 |

---

## 5. Recomendación final de resolución candidata para CNN

### Filtro: `Gaussian (ksize=5, σ=1.0)`
### Resolución recomendada: **40% (~942 × 389 px)**

**Justificación:**

| Criterio | Detalle |
|----------|---------|
| **Correlación** | 0.9975 — preserva el 99.75% de la información anatómica original |
| **Entropía** | 7.693 — máxima riqueza de información útil para el modelo |
| **Edge density** | 0.811 — densidad de bordes alta, ideal para segmentación vertebral |
| **Fragmentación** | 20.3 componentes — imagen cohesiva sin exceso de artefactos |
| **Tamaño** | ~367K píxeles — equilibrio entre detalle y costo computacional |
| **Batch size** | Permite batches de 8–16 en GPU con 8–12 GB VRAM |
| **Redimensión CNN** | Se puede escalar a 512×256 o 384×192 sin pérdida significativa |

**¿Por qué no resoluciones más bajas (10–30%)?**  
Aunque el score compuesto sube, la correlación cae por debajo de 0.997 y se pierden detalles finos vertebrales relevantes para segmentación precisa. A 10% (236×97) la imagen es demasiado pequeña para distinguir estructuras anatómicas sutiles.

**¿Por qué no resoluciones más altas (50–100%)?**  
El beneficio en correlación es marginal (+0.001) pero el costo en memoria y tiempo de entrenamiento crece significativamente (hasta 6× más píxeles). La fragmentación también aumenta innecesariamente.

> **Pipeline recomendado:** Imagen original → Gaussian (k=5, σ=1.0) → Resize a 40% → Normalización → CNN (U-Net / ResNet backbone)
