# MLFramework

A comprehensive Python Machine Learning framework that provides tools for exploratory data analysis, supervised learning, unsupervised learning, and clustering tasks.

## Overview

MLFramework is a modular, class-based machine learning framework built on top of scikit-learn and other popular data science libraries. It provides an organized pipeline for building and evaluating machine learning models across different problem types.

## Architecture

The framework is built following an object-oriented design with inheritance hierarchies that organize functionality by problem type:

```
analisisEDA (Base Class)
├── Supervisado
│   ├── Clasificacion
│   └── Regresion
└── NoSupervisado
    └── Clustering
```

## Core Components

### 1. **analisisEDA** (`eda.py`)
The foundational class providing exploratory data analysis (EDA) capabilities.

**Key Features:**
- Dataset loading and management from CSV files
- Data type analysis
- Categorical to dummy variable conversion
- Handling missing values and duplicates
- Descriptive statistics (mean, median, std, quantiles)
- Visualization methods:
  - Boxplots for outlier detection
  - Histograms and distribution plots
  - Correlation heatmaps
  - Scatter plots (pair plots)
  - Density plots
  - Dendrograms for hierarchical analysis
- Custom visualization utilities (bar plots, radar plots)

**Key Methods:**
```python
tipoDatos()                    # Display data types
analisisNumerico()            # Filter numeric columns
analisisCompleto()            # Convert categorical variables to dummies
eliminarColumnas()            # Remove specified columns
renombrarColumnas()           # Rename columns
valores_unicos()              # Count unique values
valores_faltantes()           # Display missing values
eliminarDuplicados()          # Remove duplicate rows
eliminarNulos()               # Remove rows with null values
analisis()                    # Comprehensive statistical summary
graficoCorrelacion()          # Plot correlation heatmap
graficosDispersion()          # Generate pairwise scatter plots
```

### 2. **Supervisado** (`supervisado.py`)
Base class for supervised learning tasks (classification and regression).

**Functionality:**
- Data preparation and train-test splitting (75/25)
- Feature standardization using StandardScaler
- Target variable management

**Key Methods:**
```python
preparar_datos(target='target')  # Prepare X, y splits with scaling
```

### 3. **Clasificacion** (`clasificacion.py`)
Classification models and evaluation tools.

**Supported Algorithms:**
- **KNN** (K-Nearest Neighbors) - Multiple search algorithms (auto, ball_tree, kd_tree, brute)
- **Decision Tree** - Configurable depth and split criteria
- **Random Forest** - Ensemble-based classification
- **Gradient Boosting** (XGBoost style) - Sequential ensemble method
- **AdaBoost** - Adaptive boosting with multiple base estimators

**Evaluation Metrics:**
- Confusion Matrix
- Global Accuracy
- Global Error Rate
- Per-class Precision
- Benchmark Comparison (BM) - Comprehensive model comparison

**Key Methods:**
```python
modeloKNN(X_train, y_train, n_neighbors, algorithm)
DT(min_samples_split, max_depth)
RF(n_estimators, min_samples_split, max_depth)
XG(n_estimators, min_samples_split, max_depth)
ADA(n_estimators)
BM()  # Benchmark all classification models
```

### 4. **Regresion** (`regresion.py`)
Regression models for continuous target variables.

**Supported Algorithms:**
- **Linear Regression** - Simple and multiple
- **Lasso Regression** - L1 regularization
- **LassoCV** - Cross-validated Lasso
- **Ridge Regression** - L2 regularization
- **RidgeCV** - Cross-validated Ridge
- **SVM Regression** - Multiple kernels (rbf, linear, poly)
- **Decision Tree Regressor** - Tree-based regression
- **Random Forest Regressor** - Ensemble regression
- **Gradient Boosting Regressor** - Sequential boosting

**Error Metrics:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- ER (Error Rate)

**Key Methods:**
```python
RegLinSimp(X_train, X_test, y_train, y_test)
RegLinMult(X_train, X_test, y_train, y_test)
RegLasso/RegLassoCV()
RegRidge/RegRidgeCV()
SVM(X_train, X_test, y_train, y_test)
DecisionTreeReg()
RandomForestReg()
XGBoostingReg()
ComparacionALL()  # Compare all regression models
```

### 5. **NoSupervisado** (`no_supervisado.py`)
Base class for unsupervised learning tasks.

**Functionality:**
- Model benchmarking and tracking
- Silhouette score evaluation
- Cluster algorithm comparison

**Key Methods:**
```python
agregar_modelo(algoritmo, n_clusters, silhouette_score)
benchmark()  # Return DataFrame with model comparison
```

### 6. **Clustering** (`clustering.py`)
Advanced clustering and dimensionality reduction techniques.

**Clustering Algorithms:**
- **K-Means** - Partitional clustering
- **K-Medoids** - Robust alternative to K-means
- **Hierarchical Agglomerative Clustering (HAC)** - Ward, Average, Single, Complete linkages

**Dimensionality Reduction:**
- **PCA** (Principal Component Analysis) - Linear dimensionality reduction
- **t-SNE** - Non-linear dimensionality reduction
- **UMAP** - Uniform Manifold Approximation and Projection

**Evaluation:**
- Silhouette Score computation
- Optimal cluster number detection
- Visualization using PCA 2D projections

**Key Methods:**
```python
ACP(n_componentes)        # Principal Component Analysis
HAC()                     # Hierarchical clustering with dendrograms
KMedia()                  # K-Means and K-Medoids clustering
TSNE()                    # t-SNE dimensionality reduction
UMAP(n_componentes, n_neighbors)  # UMAP projection
```

### 7. **util_pca.py**
Utility class for PCA analysis providing specialized visualizations.

## Dependencies

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

## Usage Example

```python
from clasificacion import Clasificacion
from regresion import Regresion
from clustering import Clustering
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Classification Task
clf = Clasificacion(df)
X_train, X_test, y_train, y_test = clf.preparar_datos(target='target')
clf.RF(n_estimators=100, min_samples_split=2, max_depth=10)
clf.BM()  # Compare all classifiers

# Regression Task
reg = Regresion(df)
X_train, X_test, y_train, y_test = reg.preparar_datos(target='target')
reg.ComparacionALL(X_train, X_test, y_train, y_test)

# Clustering Task
clust = Clustering(df)
clust.KMedia()
clust.HAC()
clust.UMAP()
clust.benchmark()
```

## Framework Workflow

### Supervised Learning Pipeline
1. Initialize model class (Clasificacion/Regresion)
2. Prepare data with `preparar_datos()` (automatic scaling)
3. Train model with specific algorithm
4. Make predictions
5. Evaluate using provided metrics
6. Compare multiple models using benchmark tools

### Unsupervised Learning Pipeline
1. Initialize Clustering class
2. Perform dimensionality reduction (PCA, UMAP, t-SNE)
3. Apply clustering algorithms (KMeans, HAC, KMedoids)
4. Evaluate with silhouette scores
5. Benchmark models and track results

## Key Design Patterns

- **Inheritance**: Specialized classes inherit from general base classes
- **Encapsulation**: Private methods (prefixed with `__`) for internal operations
- **Property Decorators**: Safe access to internal data through properties
- **Method Chaining**: Many methods modify data in-place for workflow efficiency
- **Modular Design**: Each algorithm family is separated into dedicated methods

## Visualization Capabilities

- **EDA Plots**: Distributions, correlations, boxplots, scatter plots
- **Classification**: Confusion matrices, performance comparisons
- **Clustering**: Dendrograms, PCA scatter plots, radar plots, TSNE visualizations
- **Dimensionality Reduction**: 2D/3D projections, variance explained plots

## File Structure

```
MLFramework/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── eda.py                      # Exploratory Data Analysis
├── supervisado.py              # Supervised learning base class
├── clasificacion.py            # Classification models
├── regresion.py                # Regression models
├── no_supervisado.py           # Unsupervised learning base class
├── clustering.py               # Clustering algorithms
├── util_pca.py                 # PCA utilities
└── streamlit_app.py            # Web application interface
```

## Installation

```bash
pip install -r requirements.txt
```

## Web Interface

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Notes

- All numeric features are automatically scaled during supervised learning
- Categorical variables are converted to dummy variables in EDA
- Train-test split uses 25% test size with random_state=42 for reproducibility
- Clustering algorithms use silhouette score for optimal cluster number detection
- All visualization methods display plots interactively

## Author

Gabriel Varela Araya

## License

Unlicensed (Private)
