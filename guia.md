**

**ESCUELA DE INFORMÁTICA**

**EIF420O - INTELIGENCIA ARTIFICIAL**

**I CICLO 2026**

**LABORATORIO #04**

**Sistema Multiagente Inteligente para la Automatización de Procesos de Machine Learning mediante Streamlit**

**Profesor:** Dr. Juan De Dios Murillo Morera

**OBJETIVO GENERAL**

Diseñar e implementar un Sistema Multiagente (MAS - Multiagent System) capaz de automatizar el ciclo completo de análisis de datos, aprendizaje automático y generación de reportes inteligentes mediante un dashboard interactivo desarrollado en Streamlit.

**OBJETIVOS ESPECÍFICOS**

- Comprender los fundamentos de los Sistemas Multiagente.
- Diseñar una arquitectura basada en agentes especializados.
- Implementar un Agente General Coordinador.
- Implementar un Agente EDA para análisis exploratorio.
- Implementar un Agente No Supervisado para clustering.
- Implementar un Agente Supervisado para clasificación.
- Implementar un Agente Supervisado para regresión.
- Desarrollar un dashboard interactivo mediante Streamlit.
- Comparar algoritmos mediante benchmarking automático.
- Generar reportes inteligentes y conclusiones automáticas.

**INTRODUCCIÓN**

Los Sistemas Multiagente (MAS) constituyen una de las áreas más importantes dentro de la Inteligencia Artificial Distribuida. Un Sistema Multiagente está compuesto por múltiples entidades autónomas denominadas agentes, capaces de cooperar, coordinarse y comunicarse para resolver problemas complejos.

Los avances recientes en Agentic AI han demostrado que la coordinación inteligente entre agentes especializados permite automatizar tareas tradicionalmente realizadas por analistas de datos, científicos de datos e ingenieros de Machine Learning.

En este laboratorio se desarrollará un sistema compuesto por agentes especializados capaces de analizar datasets, identificar patrones, construir modelos predictivos y presentar resultados mediante dashboards interactivos.

La solución deberá desarrollarse utilizando Python, Scikit-Learn, Pandas, NumPy y Streamlit.

**FUNDAMENTO TEÓRICO**

**Sistemas Multiagente (MAS)**

Un Sistema Multiagente es un conjunto de agentes inteligentes que interactúan entre sí para alcanzar objetivos comunes.

**Características principales**

- Autonomía
- Cooperación
- Coordinación
- Comunicación
- Adaptabilidad
- Toma de decisiones

**Ventajas**

- Distribución de tareas
- Escalabilidad
- Modularidad
- Reutilización de componentes
- Mayor robustez

**Agentic AI**

Agentic AI corresponde a sistemas inteligentes capaces de:

- Planificar tareas.
- Tomar decisiones.
- Utilizar herramientas.
- Coordinar agentes.
- Aprender del entorno.
- Ejecutar procesos complejos de manera autónoma.

**Aprendizaje Supervisado**

Corresponde a técnicas donde existe una variable objetivo.

**Clasificación**

La variable objetivo es categórica.

Ejemplos:

- Potable / No potable.
- Diabético / No diabético.
- Aprobado / Reprobado.

**Regresión**

La variable objetivo es numérica.

Ejemplos:

- Precio.
- Edad.
- Salario.
- Temperatura.
- Densidad de defectos.

**Aprendizaje No Supervisado**

No existe una variable objetivo.

El objetivo es descubrir patrones ocultos.

**Técnicas comunes**

- K-Means
- Clustering Jerárquico
- DBSCAN

**ARQUITECTURA GENERAL DEL SISTEMA**

Usuario  
|  
v  
+-------------------------+  
| Agente Coordinador MAS |  
+-------------------------+  
| | | |  
v v v v  
EDA Cluster Clasif Regres  
\\ | | /  
\\ | | /  
+----+------+----+  
|  
v  
Dashboard Streamlit

**AGENTE GENERAL COORDINADOR**

**Descripción**

Este agente representa el cerebro del sistema.

Será responsable de coordinar el trabajo realizado por los demás agentes.

**Responsabilidades**

**Gestión de datos**

- Cargar datasets.
- Validar formato.
- Detectar estructura.

**Gestión de agentes**

- Activar agentes.
- Coordinar comunicación.
- Gestionar flujo de trabajo.

**Gestión de resultados**

- Consolidar métricas.
- Comparar modelos.
- Seleccionar mejores algoritmos.

**Generación de conclusiones**

- Generar recomendaciones.
- Elaborar reportes automáticos.

**Entradas**

- Dataset CSV.
- Parámetros de configuración.

**Salidas**

- Resultados consolidados.
- Reporte final.
- Dashboard actualizado.

**AGENTE EDA**

**Objetivo**

Analizar y comprender la estructura del dataset.

**Responsabilidades**

**Calidad de datos**

Detectar:

- Valores nulos.
- Valores duplicados.
- Registros inconsistentes.

**Estadística descriptiva**

Calcular:

- Media.
- Mediana.
- Moda.
- Mínimo.
- Máximo.
- Varianza.
- Desviación estándar.

**Análisis de correlación**

Generar:

- Pearson.
- Spearman.
- Heatmaps.

**Detección de outliers**

Aplicar:

- IQR.
- Z-Score.

**Visualización**

Generar:

- Histogramas.
- Boxplots.
- Scatterplots.
- Pairplots.

**Productos**

- Reporte exploratorio.
- Gráficos.
- Recomendaciones de limpieza.

**AGENTE NO SUPERVISADO - CLUSTERING**

**Objetivo**

Identificar grupos o segmentos ocultos dentro de los datos.

**Algoritmos mínimos**

**K-Means**

Determinar:

- Número óptimo de clusters.
- Segmentación principal.

**Clustering Jerárquico**

Generar:

- Dendrogramas.
- Agrupamientos jerárquicos.

**DBSCAN**

Detectar:

- Regiones densas.
- Ruido.

**Métricas**

**Silhouette Score**

Evaluar cohesión de clusters.

**Davies-Bouldin Index**

Evaluar separación.

**Calinski-Harabasz Index**

Evaluar calidad global.

**Productos**

- Clusters identificados.
- Interpretación de segmentos.
- Comparación de algoritmos.

**AGENTE SUPERVISADO - CLASIFICACIÓN**

**Objetivo**

Resolver problemas donde la variable objetivo sea categórica.

**Algoritmos mínimos**

**Familia basada en distancia**

- KNN

**Familia basada en árboles**

- Decision Tree
- Random Forest

**Familia Boosting**

- AdaBoost
- XGBoost

**Familia probabilística**

- Naive Bayes

**Familia lineal**

- Logistic Regression

**Familia basada en márgenes**

- SVM

**Métricas**

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Matriz de Confusión

**Benchmarking**

Comparar:

**Misma familia**

Ejemplo:

- Árboles vs Random Forest

**Diferente familia**

Ejemplo:

- Random Forest vs SVM

**AGENTE SUPERVISADO - REGRESIÓN**

**Objetivo**

Resolver problemas donde la variable objetivo sea continua.

**Algoritmos mínimos**

**Lineales**

- Linear Regression

**Regularización**

- Ridge
- RidgeCV
- Lasso
- LassoCV

**Basados en soporte**

- SVR

**Basados en árboles**

- Decision Tree Regressor
- Random Forest Regressor

**Boosting**

- XGBoost Regressor

**Métricas**

- MAE
- MSE
- RMSE
- R²
- MAPE

**Benchmarking**

Comparar desempeño de:

- Modelos lineales.
- Modelos basados en árboles.
- Modelos de boosting.

**DASHBOARD STREAMLIT**

**Página 1 - Inicio**

Mostrar:

- Nombre del proyecto.
- Integrantes.
- Objetivos.

**Página 2 - Carga de Datos**

Permitir:

- Cargar CSV.
- Visualizar dataset.

**Página 3 - EDA**

Mostrar:

- Estadísticas.
- Correlaciones.
- Histogramas.
- Valores nulos.

**Página 4 - Clustering**

Mostrar:

- Método del Codo.
- Silhouette Score.
- Clusters.

**Página 5 - Clasificación**

Mostrar:

- Accuracy.
- Precision.
- Recall.
- F1.
- AUC.

**Página 6 - Regresión**

Mostrar:

- MAE.
- RMSE.
- R².

**Página 7 - Dashboard Ejecutivo**

Mostrar:

- Mejor algoritmo.
- Comparación general.
- Recomendaciones.

**ACTIVIDADES DEL LABORATORIO**

**Actividad 1**

Diseñar la arquitectura MAS.

**Actividad 2**

Implementar el Agente Coordinador.

**Actividad 3**

Implementar el Agente EDA.

**Actividad 4**

Implementar el Agente Clustering.

**Actividad 5**

Implementar el Agente Clasificación.

**Actividad 6**

Implementar el Agente Regresión.

**Actividad 7**

Integrar todos los agentes.

**Actividad 8**

Construir el Dashboard Streamlit.

**Actividad 9**

Realizar benchmarking automático.

**Actividad 10**

Generar conclusiones automáticas.

**ENTREGABLES**

- Código fuente completo.
- Dashboard Streamlit funcional.
- Documento técnico.
- Video demostrativo.
- Repositorio GitHub.
- Diagramas UML.
- Arquitectura MAS.
- Evidencia experimental.

**EVALUACIÓN**

| **Rubro**            | **Porcentaje** |
| -------------------- | -------------- |
| Arquitectura MAS     | 15%            |
| Agente EDA           | 15%            |
| Agente Clustering    | 15%            |
| Agente Clasificación | 15%            |
| Agente Regresión     | 15%            |
| Dashboard Streamlit  | 10%            |
| Documentación        | 10%            |
| Presentación         | 5%             |
| **TOTAL**            | **100%**       |

**RESULTADO ESPERADO**

Al finalizar el laboratorio, el estudiante habrá desarrollado una plataforma inteligente basada en Sistemas Multiagente capaz de automatizar el análisis de datos, ejecutar procesos de clustering, clasificación y regresión, comparar algoritmos mediante benchmarking y presentar resultados mediante un dashboard interactivo desarrollado en Streamlit. Esto constituye una aproximación moderna a los paradigmas de Agentic AI y AutoML aplicados a la Ciencia de Datos.