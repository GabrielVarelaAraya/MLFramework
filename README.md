# MLFramework

**Sistema Multiagente Inteligente para la Automatización de Procesos de Machine Learning**

Dashboard interactivo en Streamlit con agentes especializados para EDA, clasificación, regresión y clustering.

---

## Arquitectura MAS

```
                    +---------------------------+
                    |        Usuario UI         |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |   Agente Coordinador MAS   |
                    |     (AgenteMaestro)        |
                    +--+--------+--------+------+
                       |        |        |
              +--------+  +-----+-----+  +--------+
              |  EDA   |  | Clasif.  |  |  Reg.  |
              +--------+  +-----+-----+  +--------+
              +--------+  +-----+-----+
              |Cluster.|  | Preproc. |
              +--------+  +---------+
```

## Componentes

| Agente | Archivo | Responsabilidad |
|--------|---------|----------------|
| **Coordinador** | `agents/agente_maestro.py` | Orquestar agentes, gestionar flujo, consolidar resultados |
| **EDA** | `agents/agente_eda.py` + `core/eda.py` | Análisis exploratorio, estadísticas, correlaciones, outliers |
| **Clasificación** | `agents/agente_clasificacion.py` + `core/clasificacion.py` | 9 algoritmos de clasificación |
| **Regresión** | `agents/agente_regresion.py` + `core/regresion.py` | 10 algoritmos de regresión |
| **Clustering** | `agents/agente_clustering.py` + `core/clustering.py` | 6 algoritmos de clustering |
| **Preprocesamiento** | `agents/agente_preprocesamiento.py` | Imputación, escalado, encoding |

## Tablero (Streamlit)

| Tab | Contenido |
|-----|-----------|
| **Inicio** | Banner del proyecto, info del curso, integrantes |
| **Carga de Datos** | Upload CSV, separador, dummies, previsualización |
| **EDA** | Estadísticas, histogramas, boxplots, correlaciones, outliers |
| **Supervisado** | Clasificación (9 algos) y Regresión (10 algos) con benchmark |
| **No Supervisado** | Clustering (KMeans, KMedoids, HAC, DBSCAN, T-SNE, UMAP) |
| **Dashboard Ejecutivo** | KPIs, heatmap, benchmark global, conclusiones automáticas |

## Algoritmos

### Clasificación (9)
KNN, Decision Tree, Random Forest, SVM, Logistic Regression, Naive Bayes, XGBoost, Gradient Boosting, AdaBoost

### Regresión (10)
Regresión Lineal Múltiple, Lasso, LassoCV, Ridge, RidgeCV, SVR, Árbol Decisión, Random Forest, XGBoost, Gradient Boosting

### Clustering (6)
K-Means, K-Medoids (PAM), HAC, DBSCAN, T-SNE + KMeans, UMAP + KMeans

## Métricas

- **Clasificación**: Accuracy, F1-Score (ponderado), ROC-AUC, Matriz de Confusión
- **Regresión**: R², MSE, RMSE, MAE, MAPE
- **Clustering**: Silhouette, Calinski-Harabasz, Davies-Bouldin

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dependencias

```
streamlit, pandas, numpy, matplotlib, seaborn,
scikit-learn, xgboost, prince, umap-learn, scipy
```

## Autores

Santiago Azofeifa Benavides · Rubén Ramos Jiménez · Gabriel Varela Araya
