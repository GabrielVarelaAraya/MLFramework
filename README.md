# MLFramework

Un framework integral de **Machine Learning en Python** que proporciona herramientas para análisis exploratorio de datos, aprendizaje supervisado, no supervisado y tareas de clustering.

## Visión General

MLFramework es un framework modular, basado en clases, construido sobre **scikit-learn** y otras librerías populares de ciencia de datos. Proporciona un pipeline organizado para construir y evaluar modelos de machine learning en distintos tipos de problemas.

## Arquitectura

El framework sigue un diseño orientado a objetos con jerarquías de herencia que organizan la funcionalidad por tipo de problema:

```
analisisEDA (Base Class)
├── Supervisado
│   ├── Clasificacion
│   └── Regresion
└── NoSupervisado
    └── Clustering
```


## Componentes Principales

### 1. **analisisEDA** (`eda.py`)
Clase base que provee capacidades de análisis exploratorio de datos (EDA).

**Características:**
- Carga y gestión de datasets desde CSV
- Análisis de tipos de datos
- Conversión de variables categóricas a dummies
- Manejo de valores faltantes y duplicados
- Estadísticas descriptivas (media, mediana, desviación estándar, cuantiles)
- Métodos de visualización:
  - Boxplots para detección de outliers
  - Histogramas y distribuciones
  - Mapas de calor de correlación
  - Gráficos de dispersión (pair plots)
  - Gráficos de densidad
  - Dendrogramas para análisis jerárquico
- Utilidades de visualización personalizadas (barras, radar)

**Métodos Clave:**
```python
tipoDatos()
analisisNumerico()
analisisCompleto()
eliminarColumnas()
renombrarColumnas()
valores_unicos()
valores_faltantes()
eliminarDuplicados()
eliminarNulos()
analisis()
graficoCorrelacion()
graficosDispersion()
```

## 2. Supervisado (`supervisado.py`)
Clase base para tareas supervisadas (clasificación y regresión).

**Funcionalidad:**
- Preparación de datos y división *train-test* (75/25)
- Estandarización de variables con **StandardScaler**
- Gestión de la variable objetivo

---

## 3. Clasificación (`clasificacion.py`)
Modelos de clasificación y herramientas de evaluación.

**Algoritmos soportados:**
- KNN
- Árboles de decisión
- Random Forest
- Gradient Boosting
- AdaBoost

**Métricas:**
- Matriz de confusión
- Exactitud global
- Tasa de error global
- Precisión por clase
- Benchmark de modelos

---

## 4. Regresión (`regresion.py`)
Modelos para variables continuas.

**Algoritmos soportados:**
- Regresión lineal (simple y múltiple)
- Lasso / LassoCV
- Ridge / RidgeCV
- SVM (rbf, linear, poly)
- Árboles de regresión
- Random Forest
- Gradient Boosting

**Métricas de error:**
- RMSE
- MAE
- Error Rate

---

## 5. No Supervisado (`no_supervisado.py`)
Clase base para aprendizaje no supervisado.

**Funcionalidad:**
- Benchmark de modelos
- Evaluación con **Silhouette Score**
- Comparación de algoritmos de clustering

---

## 6. Clustering (`clustering.py`)
Técnicas avanzadas de clustering y reducción de dimensionalidad.

**Algoritmos:**
- K-Means
- K-Medoids
- HAC (Ward, Average, Single, Complete)

**Reducción de dimensionalidad:**
- PCA
- t-SNE
- UMAP

---

## 7. util_pca.py
Utilidades para análisis PCA y visualizaciones especializadas.


## Dependencias

```
streamlit              # Web interface
pandas                 # Data manipulation
matplotlib             # Plotting
seaborn                # Statistical visualizations
scikit-learn           # ML algorithms
prince                 # Categorical encoding
umap-learn             # UMAP algorithm
scikit-learn-extra     # K-Medoids and other extras
numpy                  # Numerical computing
```

## Instalación

```bash
pip install -r requirements.txt
```

## Frontend

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Notas

- Variables numéricas se escalan automáticamente
- Variables categóricas se convierten a dummies
- Split 75/25 con random_state=42
- Clustering usa Silhouette Score
- Visualizaciones interactivas

## Autores

Santiago Azofeifa Benavides
Rubén Ramos Jiménez
Gabriel Varela Araya
