# Resultados — ejemplo_agentes.py

## Clasificación

### Breast Cancer
- **Shape:** (569, 31) · Nulos: 0 · Duplicados: 0
- **Tipo sugerido:** clasificacion · Target: `target`

| Modelo | Exactitud | F1-Score | Error Global |
|---|---|---|---|
| Random Forest | 96.50% | 0.9650 | 0.0350 |
| KNN | 95.80% | 0.9580 | 0.0420 |
| Gradient Boosting | 95.80% | 0.9580 | 0.0420 |
| AdaBoost | 95.80% | 0.9580 | 0.0420 |
| Decision Tree | 95.10% | 0.9511 | 0.0490 |

**Mejor modelo:** Random Forest — `Modelo aceptado — exactitud 96.50%`

**Matriz de confusión (Random Forest):**
```
[[51  3]
 [ 2 87]]
```

---

### Digits
- **Shape:** (1797, 65) · Nulos: 0 · Duplicados: 0
- **Tipo sugerido:** clasificacion · Target: `target`

| Modelo | Exactitud | F1-Score | Error Global |
|---|---|---|---|
| KNN | 97.78% | 0.9776 | 0.0222 |
| Random Forest | 97.56% | 0.9756 | 0.0244 |
| Gradient Boosting | 97.11% | 0.9712 | 0.0289 |
| Decision Tree | 85.33% | 0.8538 | 0.1467 |
| AdaBoost | 70.44% | 0.7169 | 0.2956 |

**Mejor modelo:** KNN — `Modelo aceptado — exactitud 97.78%`

**Matriz de confusión (Random Forest):**
```
[[42  0  0  0  1  0  0  0  0  0]
 [ 0 36  1  0  0  0  0  0  0  0]
 [ 0  0 38  0  0  0  0  0  0  0]
 [ 0  0  0 42  0  1  0  0  3  0]
 [ 0  0  0  0 55  0  0  0  0  0]
 [ 0  0  0  0  0 57  1  0  0  1]
 [ 0  0  0  0  0  1 44  0  0  0]
 [ 0  0  0  0  0  0  0 40  0  1]
 [ 0  2  0  0  0  0  0  0 36  0]
 [ 0  0  0  0  0  1  0  2  0 45]]
```

---

### Iris
- **Shape:** (150, 5) · Nulos: 0 · Duplicados: 1
- **Tipo sugerido:** clasificacion · Target: `target`

| Modelo | Exactitud | F1-Score | Error Global |
|---|---|---|---|
| KNN | 100.00% | 1.0000 | 0.0000 |
| Decision Tree | 100.00% | 1.0000 | 0.0000 |
| Random Forest | 100.00% | 1.0000 | 0.0000 |
| Gradient Boosting | 100.00% | 1.0000 | 0.0000 |
| AdaBoost | 94.74% | 0.9474 | 0.0526 |

**Mejor modelo:** KNN/DT/RF/GB — `Modelo aceptado — exactitud 100.00%`

**Matriz de confusión (Random Forest):**
```
[[15  0  0]
 [ 0 11  0]
 [ 0  0 12]]
```

---

## Regresión

### Diabetes
- **Shape:** (442, 11) · Nulos: 0 · Duplicados: 0
- **Tipo sugerido:** regresion · Target: `target`

| Modelo | R² | RMSE | MAE |
|---|---|---|---|
| SVM | 0.5005 | 52.5559 | 40.1235 |
| Lasso | 0.4866 | 53.2813 | 41.4842 |
| Ridge | 0.4862 | 53.3029 | 41.4918 |
| Regresión Lineal Múltiple | 0.4849 | 53.3696 | 41.5485 |
| Random Forest | 0.4717 | 54.0511 | 43.0916 |
| Gradient Boosting | 0.3725 | 58.9050 | 46.9585 |
| Árbol de Decisión | 0.3489 | 60.0016 | 47.0390 |

**Mejor modelo:** SVM — `Modelo requiere mejora — R² 0.5005 (umbral 0.7)`

> Dataset conocido por ser difícil de predecir con modelos simples.

---

### California Housing
- **Shape:** (500, 9) · Nulos: 0 · Duplicados: 0 · (muestra de 500 filas)
- **Tipo sugerido:** regresion · Target: `Price`

| Modelo | R² | RMSE | MAE |
|---|---|---|---|
| Gradient Boosting | 0.7568 | 0.6011 | 0.4066 |
| Ridge | 0.7560 | 0.6020 | 0.4589 |
| Regresión Lineal Múltiple | 0.7559 | 0.6022 | 0.4589 |
| Lasso | 0.7556 | 0.6025 | 0.4595 |
| Random Forest | 0.7427 | 0.6182 | 0.4572 |
| SVM | 0.6992 | 0.6685 | 0.4825 |
| Árbol de Decisión | 0.5494 | 0.8182 | 0.5557 |

**Mejor modelo:** Gradient Boosting — `Modelo aceptado — R² 0.7568`

---

## Clustering

### Iris (k=3)
- **Shape:** (150, 4) · Nulos: 0 · Duplicados: 1

| Algoritmo | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| KMeans | 0.4599 | 241.90 | 0.8336 |
| HAC | 0.4467 | 222.72 | 0.8035 |

**Mejor:** KMeans (Silhouette más alto) · HAC (Davies-Bouldin más bajo)

---

### Wine (k=3)
- **Shape:** (178, 13) · Nulos: 0 · Duplicados: 0

| Algoritmo | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| KMeans | 0.2849 | 70.94 | 1.3892 |
| HAC | 0.2774 | 67.65 | 1.4186 |

**Mejor:** KMeans en todas las métricas

---

### Blobs (k=4)
- **Shape:** (300, 2) · Nulos: 0 · Duplicados: 0 · (sintético)

| Algoritmo | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| KMeans | 0.7582 | 2311.47 | 0.3362 |
| HAC | 0.7582 | 2311.47 | 0.3362 |

**Mejor:** KMeans = HAC (idénticos — clusters perfectamente separados)

---

## Resumen general

### Clasificación
| Dataset | Mejor modelo | Exactitud | Decisión |
|---|---|---|---|
| Breast Cancer | Random Forest | 96.50% | ✅ Aceptado |
| Digits | KNN | 97.78% | ✅ Aceptado |
| Iris | KNN / DT / RF / GB | 100.00% | ✅ Aceptado |

### Regresión
| Dataset | Mejor modelo | R² | Decisión |
|---|---|---|---|
| Diabetes | SVM | 0.5005 | ❌ Requiere mejora |
| California Housing | Gradient Boosting | 0.7568 | ✅ Aceptado |

### Clustering (Silhouette Score)
| Dataset | KMeans | HAC | Mejor |
|---|---|---|---|
| Iris | 0.4599 | 0.4467 | KMeans |
| Wine | 0.2849 | 0.2774 | KMeans |
| Blobs | 0.7582 | 0.7582 | Empate |
