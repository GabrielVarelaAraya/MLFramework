# Documentación Técnica — ML Framework

## Sistema Multiagente Inteligente para la Automatización de Procesos de Machine Learning

**Curso:** EIF420O - Inteligencia Artificial  
**Ciclo:** I Ciclo 2026  
**Profesor:** Dr. Juan De Dios Murillo Morera  
**Escuela:** Escuela de Informática

---

## Índice

1. [Arquitectura MAS](#1-arquitectura-mas)
2. [Diagramas UML](#2-diagramas-uml)
3. [Agente General Coordinador](#3-agente-general-coordinador)
4. [Agente EDA](#4-agente-eda)
5. [Agente No Supervisado (Clustering)](#5-agente-no-supervisado-clustering)
6. [Agente Supervisado — Clasificación](#6-agente-supervisado--clasificacin)
7. [Agente Supervisado — Regresión](#7-agente-supervisado--regresin)
8. [Dashboard Streamlit](#8-dashboard-streamlit)
9. [Benchmarking Automático](#9-benchmarking-automtico)
10. [Conclusiones Automáticas](#10-conclusiones-automticas)
11. [Evidencia Experimental](#11-evidencia-experimental)
12. [Guía de Instalación y Ejecución](#12-guía-de-instalacin-y-ejecucin)

---

## 1. Arquitectura MAS

### Visión General

El sistema sigue una arquitectura de **Sistema Multiagente (MAS)** con un agente coordinador central y agentes especializados. Cada agente es autónomo, con responsabilidades específicas, y se comunican a través del agente coordinador.

```
                    +---------------------------+
                    |     USUARIO (Streamlit)    |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |   Agente Coordinador MAS   |
                    |     (AgenteMaestro)        |
                    +--+--------+--------+------+
                       |        |        |
              +--------+  +-----+-----+  +--------+
              |  EDA   |  | Clasif.  |  |  Reg.   |
              | Agente |  | Agente   |  | Agente  |
              +--------+  +-----+-----+  +--------+
                       |        |        |
              +--------+  +-----+-----+  +--------+
              |Clustering|  |Preproc. |  | Dashboard|
              | Agente   |  | Agente  |  | Streamlit|
              +--------+  +-----+-----+  +--------+
```

### Componentes del Sistema

| Componente | Archivo | Rol |
|-----------|---------|-----|
| `AgenteMaestro` | `agents/agente_maestro.py` | Coordinador central |
| `AgenteEDA` | `agents/agente_eda.py` | Análisis exploratorio |
| `AgenteClasificacion` | `agents/agente_clasificacion.py` | Clasificación |
| `AgenteRegresion` | `agents/agente_regresion.py` | Regresión |
| `AgenteClustering` | `agents/agente_clustering.py` | Clustering |
| `AgentePreprocesamiento` | `agents/agente_preprocesamiento.py` | Preprocesamiento |
| `analisisEDA` | `core/eda.py` | Backend EDA |
| `Clasificacion` | `core/clasificacion.py` | Backend clasificación |
| `Regresion` | `core/regresion.py` | Backend regresión |
| `Clustering` | `core/clustering.py` | Backend clustering |
| `Supervisado` | `core/supervisado.py` | Base supervisado |
| `NoSupervisado` | `core/no_supervisado.py` | Base no supervisado |

### Flujo de Datos

```
CSV → Carga → AgenteMaestro → AgenteEDA (análisis)
                             → AgentePreprocesamiento (limpieza)
                             → AgenteClasificacion (si target categórico)
                             → AgenteRegresion (si target numérico)
                             → AgenteClustering (si no supervisado)
                             → Dashboard (resultados)
```

---

## 2. Diagramas UML

### 2.1 Diagrama de Clases

```
+------------------+       +-------------------+
|   analisisEDA    |<------|   Supervisado     |
+------------------+       +-------------------+
| - __df: DataFrame|       | + preparar_datos() |
| + resumen_dict() |       +---------+---------+
| + grafico*()     |                 |
| + detectar_*()   |       +---------+---------+
+------------------+       |                   |
                    +------+------+     +------+------+
                    | Clasificacion|   |  Regresion   |
                    +-------------+     +-------------+
                    | + entrenar()|     | + entrenar()|
                    | + comparar_ |     | + comparar_ |
                    |   todos()   |     |   todos()   |
                    | + BM()      |     +-------------+
                    +-------------+

+------------------+       +-------------------+
|   analisisEDA    |<------|  NoSupervisado    |
+------------------+       +-------------------+
                    |       | + agregar_modelo()|
                    +-------+---------+---------+
                                      |
                            +---------+---------+
                            |    Clustering     |
                            +-------------------+
                            | + ejecutar()      |
                            | + comparar_todos()|
                            | + decidir()       |
                            +-------------------+

+------------------+       +-------------------+
|   AgenteMaestro  |------>|   AgenteEDA       |
+------------------+       +-------------------+
| + ejecutar_eda() |       | + analizar()      |
| + sugerir_tipo() |       | + detectar_tipo_  |
| + ejecutar_clasif|       |   problema()      |
| + ejecutar_reg() |       +-------------------+
| + ejecutar_clust |
| + comparar_*()   |------>| AgenteClasificacion|
+------------------+       +-------------------+
                    |       | + entrenar()      |
                    |       | + comparar_todos()|
                    |       +-------------------+

                    |------>| AgenteRegresion   |
                    |       +-------------------+
                    |       | + entrenar()      |
                    |       | + comparar_todos()|
                    |       +-------------------+

                    |------>| AgenteClustering  |
                    |       +-------------------+
                    |       | + ejecutar()      |
                    |       | + comparar_todos()|
                    |       +-------------------+
```

### 2.2 Diagrama de Secuencia — Benchmark Completo

```
Usuario          AgenteMaestro     AgenteClasif     Clasificacion
   |                   |                |                |
   |--run benchmark--> |                |                |
   |                   |--comparar_---->|                |
   |                   |  clasificacion |                |
   |                   |                |--entrenar(KNN) |
   |                   |                |     |          |
   |                   |                |<----resultado--|
   |                   |                |--entrenar(DT)  |
   |                   |                |     |          |
   |                   |                |<----resultado--|
   |                   |                |--entrenar(RF)  |
   |                   |                |     |          |
   |                   |                |<----resultado--|
   |                   |                |   ... (6 más)  |
   |                   |                |                |
   |                   |<--resultados---|                |
   |                   |     + tabla    |                |
   |<--muestra---------|                |                |
   |  dashboard +      |                |                |
   |  conclusiones     |                |                |
```

### 2.3 Diagrama de Secuencia — Benchmark Regresión (con errores)

```
Usuario          AgenteMaestro      AgenteRegresion     Regresion
   |                   |                   |                |
   |--run reg--------> |                   |                |
   |  benchmark        |--comparar_------->|                |
   |                   |  regresion        |                |
   |                   |                   |--entrenar(RLM) |
   |                   |                   |     |          |
   |                   |                   |<----OK---------|
   |                   |                   |--entrenar(Lasso)|
   |                   |                   |     |          |
   |                   |                   |<----OK---------|
   |                   |                   |--entrenar(SVM) |
   |                   |                   |     |          |
   |                   |                   |<----FALLO------|
   |                   |                   |  (capturado)   |
   |                   |                   |--entrenar(DT)  |
   |                   |                   |     |          |
   |                   |                   |<----OK---------|
   |                   |                   |   ...          |
   |                   |                   |                |
   |                   |<--resultados------|                |
   |                   |   + tabla + errs  |                |
   |<--best model + ---|                   |                |
   |   errores expand  |                   |                |
```

---

## 3. Agente General Coordinador

### Archivo: `agents/agente_maestro.py`

### Responsabilidades

- **Gestión de datos**: Carga, validación de formato, detección de estructura.
- **Gestión de agentes**: Activación, coordinación, flujo de trabajo.
- **Gestión de resultados**: Consolidación de métricas, comparación, selección.
- **Generación de conclusiones**: Recomendaciones, reportes automáticos.

### Métodos Principales

| Método | Descripción | Retorna |
|--------|-------------|---------|
| `ejecutar_eda()` | Análisis exploratorio completo | `dict` con métricas |
| `sugerir_tipo()` | Heurística de tipo de problema | `{tipo, target, razon}` |
| `ejecutar_clasificacion()` | Entrenar un clasificador | `dict` con métricas |
| `comparar_clasificacion()` | Benchmark todos los clasificadores | `(resultados, tabla)` |
| `ejecutar_regresion()` | Entrenar un regresor | `dict` con métricas |
| `comparar_regresion()` | Benchmark todos los regresores | `(resultados, tabla, errores)` |
| `ejecutar_clustering()` | Ejecutar un algoritmo de clustering | `dict` con resultados |
| `comparar_clustering()` | Benchmark todos los algoritmos | `(resultados, tabla)` |

### Diagrama de Clase — AgenteMaestro

```
+-------------------------------------------+
|              AgenteMaestro                 |
+-------------------------------------------+
| - df: DataFrame                           |
| - _eda: AgenteEDA                         |
| - _preprocesamiento: AgentePreprocesamiento|
| - _clustering: AgenteClustering            |
| - _clasificacion: AgenteClasificacion      |
| - _regresion: AgenteRegresion              |
+-------------------------------------------+
| + ejecutar_eda()                          |
| + sugerir_tipo()                          |
| + preprocesar()                           |
| + ejecutar_clustering()                   |
| + comparar_clustering()                   |
| + decidir_clustering()                    |
| + ejecutar_clasificacion()                |
| + comparar_clasificacion()                |
| + ejecutar_regresion()                    |
| + comparar_regresion()                    |
+-------------------------------------------+
```

### Heurística de Sugerencia (`sugerir_tipo`)

La función `detectar_tipo_problema()` en `core/eda.py` implementa:

1. **Match exacto**: Si el nombre de columna está en `nombres_target`.
2. **Match parcial**: Si el nombre contiene palabras clave numéricas (`price`, `precio`, `cost`, etc.).
3. **Última columna**: Por convención ML, con detección de moneda y feature-like columns.
4. **Feature detection**: Columnas con ≤5 valores enteros pequeños (0-10) se consideran features.

---

## 4. Agente EDA

### Archivo: `agents/agente_eda.py` + `core/eda.py`

### Algoritmos y Métodos Implementados

| Categoría | Métodos |
|-----------|---------|
| **Calidad de datos** | `valores_faltantes()`, `eliminarDuplicados()`, `eliminarNulos()`, `detectar_inconsistencias()` |
| **Estadística descriptiva** | Media, mediana, moda, varianza, desv. estándar, min, max, cuantiles |
| **Correlación** | Pearson (`graficoCorrelacion_fig`), Spearman (`correlacion_spearman_fig`) |
| **Outliers** | IQR (`detectar_outliers_iqr`), Z-Score (`detectar_outliers_zscore`) |
| **Visualización** | Histogramas, boxplots, KDE, scatterplots, pairplots, densidad |

### Tipos de Datos Soportados

- Numéricos (int, float)
- Categóricos (object, category)
- Fechas (datetime, detectados por nombre)
- Booleanos

### Manejo de Errores

- Validación de DataFrame vacío
- Detección de columnas constantes
- Detección de valores negativos en columnas positivas
- Detección de desbalance extremo en categóricas

---

## 5. Agente No Supervisado (Clustering)

### Archivo: `agents/agente_clustering.py` + `core/clustering.py`

### Algoritmos Implementados

| Algoritmo | Familia | Parámetros Clave |
|-----------|---------|------------------|
| K-Means | Particional | `n_clusters`, `random_state` |
| K-Medoids | Particional | `n_clusters`, `metric` |
| HAC | Jerárquico | `n_clusters`, `linkage`, `metric` |
| DBSCAN | Densidad | `eps`, `min_samples` |
| T-SNE + KMeans | Reducción + clustering | `perplexity`, `n_clusters` |
| UMAP + KMeans | Reducción + clustering | `n_neighbors`, `n_clusters` |

### Métricas

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| Silhouette Score | Cohesión intra-cluster vs separación inter-cluster | [-1, 1], mayor es mejor |
| Calinski-Harabasz | Ratio varianza inter/intra | Mayor es mejor |
| Davies-Bouldin | Similitud promedio entre clusters | Menor es mejor |

### Visualizaciones

- Proyección 2D de clusters
- Distribución de clusters por componentes
- Curva de codo (Elbow)
- Curva de Silhouette
- Dendrograma (HAC)

### Decisión Automática

El método `decidir()` selecciona el mejor algoritmo basado en:
1. Mayor Silhouette Score
2. Mayor Calinski-Harabasz
3. Menor Davies-Bouldin

---

## 6. Agente Supervisado — Clasificación

### Archivo: `agents/agente_clasificacion.py` + `core/clasificacion.py`

### Algoritmos Implementados

| Algoritmo | Familia | Librería |
|-----------|---------|----------|
| KNN | Distancia | `sklearn.neighbors.KNeighborsClassifier` |
| Decision Tree | Árbol | `sklearn.tree.DecisionTreeClassifier` |
| Random Forest | Árbol (ensemble) | `sklearn.ensemble.RandomForestClassifier` |
| SVM | Margen | `sklearn.svm.SVC` |
| Logistic Regression | Lineal | `sklearn.linear_model.LogisticRegression` |
| Naive Bayes | Probabilística | `sklearn.naive_bayes.GaussianNB` |
| XGBoost | Boosting | `xgboost.XGBClassifier` |
| Gradient Boosting | Boosting | `sklearn.ensemble.GradientBoostingClassifier` |
| AdaBoost | Boosting | `sklearn.ensemble.AdaBoostClassifier` |

### Métricas

| Métrica | Cálculo | Interpretación |
|---------|---------|----------------|
| Exactitud (Accuracy) | `(TP+TN)/(TP+TN+FP+FN)` | Proporción de aciertos |
| F1-Score (ponderado) | Media armónica precision/recall por clase | Balance precision-recall |
| ROC-AUC | Área bajo curva ROC | Capacidad discriminativa |
| Error Global | `1 - Accuracy` | Proporción de errores |
| Matriz de Confusión | `confusion_matrix()` | Distribución predicciones vs reales |
| Reporte Clasificación | `classification_report()` | Precision/Recall/F1 por clase |

### Benchmarking

El método `comparar_todos()` ejecuta los 9 algoritmos y retorna una tabla comparativa ordenada por Exactitud descendente.

### Decision Automatica

El método `decidir()` selecciona el mejor modelo usando:
- Mayor F1-Score ponderado (para desbalanceo)
- Mayor ROC-AUC (como desempate)

---

## 7. Agente Supervisado — Regresión

### Archivo: `agents/agente_regresion.py` + `core/regresion.py`

### Algoritmos Implementados

| Algoritmo | Familia | Librería |
|-----------|---------|----------|
| Regresión Lineal Múltiple | Lineal | `sklearn.linear_model.LinearRegression` |
| Lasso | Regularización L1 | `sklearn.linear_model.Lasso` |
| LassoCV | Regularización L1 + CV | `sklearn.linear_model.LassoCV` |
| Ridge | Regularización L2 | `sklearn.linear_model.Ridge` |
| RidgeCV | Regularización L2 + CV | `sklearn.linear_model.RidgeCV` |
| SVM (SVR) | Soporte | `sklearn.svm.SVR` |
| Árbol de Decisión | Árbol | `sklearn.tree.DecisionTreeRegressor` |
| Random Forest | Árbol (ensemble) | `sklearn.ensemble.RandomForestRegressor` |
| XGBoost | Boosting | `xgboost.XGBRegressor` |
| Gradient Boosting | Boosting | `sklearn.ensemble.GradientBoostingRegressor` |

### Métricas

| Métrica | Cálculo | Interpretación |
|---------|---------|----------------|
| R² | `1 - SS_res/SS_tot` | Proporción de varianza explicada |
| MSE | `mean((y - y_pred)^2)` | Error cuadrático medio |
| RMSE | `sqrt(MSE)` | Raíz del error cuadrático medio |
| MAE | `mean(|y - y_pred|)` | Error absoluto medio |
| MAPE | `mean(|y - y_pred|/|y|) * 100` | Error porcentual absoluto medio |

### Benchmarking

El método `comparar_todos()` ejecuta los 10 algoritmos con manejo individual de errores por algoritmo, retornando:
- `resultados`: lista de dicts con métricas
- `tabla`: DataFrame comparativo ordenado por R² descendente
- `errores`: lista de errores por algoritmo (si alguno falla)

### Manejo de Errores

Cada algoritmo se ejecuta en un bloque `try/except` individual, permitiendo que el benchmark continúe incluso si un algoritmo específico falla (ej. SVM con pocos datos, LassoCV con muestras insuficientes para validación cruzada).

---

## 8. Dashboard Streamlit

### Archivo: `streamlit_app.py`

### Estructura de Tabs

| Tab | Archivo/Línea | Contenido |
|-----|---------------|-----------|
| **Inicio** | `tab_inicio` | Banner del proyecto, info del curso, integrantes |
| **Carga de Datos** | `tab_carga` | Upload CSV, separador, dummies, previsualización |
| **EDA** | `tab_eda` | Estadísticas, correlaciones, histogramas, outliers |
| **Supervisado** | `tab_sup` → `tab_clf` + `tab_reg` | Clasificación 9 algos, Regresión 10 algos |
| **No supervisado** | `tab_ns` → `tab_cluster` | KMeans, KMedoids, HAC, DBSCAN, T-SNE, UMAP |
| **Dashboard Ejecutivo** | `tab_dash` | KPIs, heatmap, benchmark, conclusiones automáticas |

### Sidebar

- Configuración global (test size, random seed)

### Funcionalidades Clave

- **Carga de datos**: CSV con separador configurable, conversión a dummies
- **EDA interactivo**: Selectores de columnas para visualizaciones
- **Benchmarking**: Comparación de todos los algoritmos con métricas
- **Dashboard Ejecutivo**: KPIs, heatmap de correlación, gráfico de valores nulos, tabla comparativa, conclusiones automáticas

---

## 9. Benchmarking Automático

### Clasificación

Ejecuta 9 algoritmos secuencialmente y compara:
- Exactitud
- F1-Score (ponderado)
- ROC-AUC
- Error Global

### Regresión

Ejecuta 10 algoritmos secuencialmente y compara:
- R²
- MSE
- RMSE
- MAE
- MAPE

### Clustering

Ejecuta hasta 6 algoritmos y compara:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

### Dashboard

Para cada área, el Dashboard Ejecutivo:
1. Muestra KPIs del dataset (filas, columnas, nulos, problema, target)
2. Genera heatmap de correlación y gráfico de valores nulos
3. Ejecuta benchmark completo
4. Muestra tabla comparativa con formato condicional (gradientes)
5. Genera **conclusiones automáticas** por área

---

## 10. Conclusiones Automáticas

Implementadas en el Dashboard Ejecutivo (`streamlit_app.py`), las conclusiones se generan evaluando el mejor modelo por área:

### Clasificación
- **Bueno** (★): Exactitud ≥ 80% y F1 ≥ 0.75
- **Aceptable** (★): Exactitud ≥ 60%
- **Bajo** (★): Exactitud < 60%

### Regresión
- **Bueno** (★): R² ≥ 0.70
- **Aceptable** (★): R² ≥ 0.40
- **Bajo** (★): R² < 0.40

### Clustering
- **Bueno** (★): Silhouette ≥ 0.50
- **Aceptable** (★): Silhouette ≥ 0.25
- **Bajo** (★): Silhouette < 0.25

---

## 11. Evidencia Experimental

### Prueba 1: Clasificación — Dataset Iris

| Modelo | Exactitud | F1-Score | ROC-AUC |
|--------|-----------|----------|---------|
| KNN | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 |
| SVM | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 |
| AdaBoost | 0.9474 | 0.9486 | 0.9933 |

**Resultado:** 8 de 9 modelos lograron 100% de exactitud en Iris. AdaBoost tuvo un rendimiento ligeramente inferior (94.7%).

### Prueba 2: Regresión — Dataset Diabetes

| Modelo | R² | MSE | RMSE | MAE |
|--------|-----|-----|------|-----|
| Gradient Boosting | 0.5448 | 2742.67 | 52.37 | 42.42 |
| Ridge | 0.5144 | 2928.91 | 54.12 | 42.87 |
| RidgeCV | 0.5144 | 2928.91 | 54.12 | 42.87 |
| Random Forest | 0.5105 | 2951.24 | 54.32 | 43.19 |
| Decision Tree | 0.5006 | 3010.64 | 54.87 | 42.71 |
| Linear Multiple | 0.4977 | 3028.16 | 55.03 | 43.79 |
| Lasso | 0.4975 | 3028.87 | 55.04 | 43.77 |
| LassoCV | 0.4975 | 3028.87 | 55.04 | 43.77 |
| XGBoost | 0.4608 | 3251.42 | 57.02 | 44.76 |
| SVM | 0.1301 | 5244.18 | 72.42 | 56.25 |

**Resultado:** Gradient Boosting fue el mejor modelo (R² = 0.5448). SVM tuvo el peor rendimiento (R² = 0.1301). RidgeCV y LassoCV seleccionaron automáticamente los mismos alphas que Ridge y Lasso.

### Prueba 3: Sugerencia Automática

| Dataset | Tipo Detectado | Target Sugerido | Acierto |
|---------|---------------|-----------------|---------|
| Iris (especies) | Clasificación | `species` | ✅ |
| Diabetes (progresión) | Regresión | `target` | ✅ |
| Car data (precio) | Regresión | `Selling_Price` | ✅ |
| Wine (calidad) | Clasificación | `quality` | ✅ |

### Prueba 4: Manejo de Errores

| Escenario | Resultado Esperado | Resultado |
|-----------|-------------------|-----------|
| CSV sin target claro | `no_supervisado` | ✅ |
| Columna moneda `$` | Regresión (limpieza automática) | ✅ |
| Última columna con pocos valores enteros | Feature, se busca otro target | ✅ |
| Algoritmo falla en benchmark | Error capturado, continúa con demás | ✅ |

---

## 12. Guía de Instalación y Ejecución

### Requisitos

- Python 3.12+
- pip

### Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd MLFramework

# Crear entorno virtual (opcional)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash
streamlit run streamlit_app.py
```

### Dependencias (`requirements.txt`)

```
streamlit
pandas
matplotlib
scipy
seaborn
scikit-learn
xgboost
prince
umap-learn
numpy
scikit-learn-extra (comentado, sin wheels para 3.12+)
```

---

## Estructura del Proyecto

```
MLFramework/
├── streamlit_app.py          # Dashboard Streamlit principal
├── ejemplo_agentes.py        # Demo standalone
├── util_pca.py               # Análisis PCA
├── requirements.txt          # Dependencias
├── guia.md                   # Guía del laboratorio
├── DOCUMENTACION.md          # Este documento
├── core/
│   ├── __init__.py           # Exports
│   ├── eda.py                # Backend EDA
│   ├── supervisado.py        # Base supervisado
│   ├── clasificacion.py      # 9 clasificadores
│   ├── regresion.py          # 10 regresores
│   ├── no_supervisado.py     # Base no supervisado
│   └── clustering.py         # 6 algoritmos clustering
├── agents/
│   ├── __init__.py           # Exports
│   ├── agente_maestro.py     # Coordinador
│   ├── agente_eda.py         # Agente EDA
│   ├── agente_preprocesamiento.py  # Preprocesamiento
│   ├── agente_clasificacion.py     # Agente clasificación
│   ├── agente_regresion.py   # Agente regresión
│   └── agente_clustering.py  # Agente clustering
└── .vscode/
    └── settings.json
```

---

*Documentación generada para el Laboratorio #04 — EIF420O Inteligencia Artificial, I Ciclo 2026.*
