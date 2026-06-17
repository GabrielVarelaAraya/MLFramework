import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from agents import AgenteMaestro, AgenteEDA, AgentePreprocesamiento

SEP = "=" * 55


# ════════════════════════════════════════════════════════════════════════════
# PRUEBA 1 — AgenteEDA
# ════════════════════════════════════════════════════════════════════════════
print(f"{SEP}")
print(f"PRUEBA 1 — AgenteEDA")
print(SEP)

data = load_iris()
df_iris = pd.DataFrame(data.data, columns=data.feature_names)
df_iris["target"] = data.target

agente_eda = AgenteEDA(df_iris)

resumen = agente_eda.analizar()
print(f"  Shape        : {resumen['shape']}")
print(f"  Nulos totales: {resumen['nulos_totales']}")
print(f"  Duplicados   : {resumen['duplicados']}")

sugerencia = agente_eda.detectar_tipo_problema()
print(f"\n  Tipo sugerido: {sugerencia['tipo']}")
print(f"  Target       : {sugerencia['target_sugerido']}")
print(f"  Razon        : {sugerencia['razon']}")

print(f"\n  Visualizaciones disponibles:")
print(f"    - grafico_boxplot()")
print(f"    - histogramas()")
print(f"    - distribucion_variables()")
print(f"    - grafico_correlacion()")
print(f"    - graficos_dispersion()")


# ════════════════════════════════════════════════════════════════════════════
# PRUEBA 2 — AgentePreprocesamiento
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print(f"PRUEBA 2 — AgentePreprocesamiento")
print(SEP)

df_sucio = pd.DataFrame({
    'edad': [25, 30, None, 25, 30, 40, None],
    'salario': [50000, 60000, 70000, 50000, 60000, 80000, 90000],
    'ciudad': ['A', 'B', 'A', 'A', 'B', 'C', 'C'],
    'compra': ['Si', 'No', 'Si', 'Si', 'No', 'Si', 'No']
})

print(f"  Dataset original:")
print(f"    Shape: {df_sucio.shape} | Nulos: {df_sucio.isnull().sum().sum()} | Duplicados: {df_sucio.duplicated().sum()}")

agente_prep = AgentePreprocesamiento(df_sucio)

print(f"\n  Aplicando preprocesamiento encadenado...")
agente_prep.eliminar_duplicados()
agente_prep.imputar_nulos(estrategia='mean', columnas=['edad'])
agente_prep.codificar_categoricas(columnas=['ciudad', 'compra'], metodo='dummies')
agente_prep.escalar(columnas=['edad', 'salario'])

df_limpio = agente_prep.get_dataframe()
print(f"\n  Dataset procesado:")
print(f"    Shape: {df_limpio.shape} | Nulos: {df_limpio.isnull().sum().sum()}")
print(f"    Columnas: {list(df_limpio.columns)}")
print(f"\n  Primeras 3 filas:")
print(df_limpio.head(3))


# ════════════════════════════════════════════════════════════════════════════
# PRUEBA 3 — AgenteMaestro coordinando todos los agentes
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print(f"PRUEBA 3 — AgenteMaestro (flujo completo)")
print(SEP)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

maestro = AgenteMaestro(df)

print(f"\n  [1/5] EDA")
resumen = maestro.ejecutar_eda()
sugerencia = maestro.sugerir_tipo()
print(f"        Shape: {resumen['shape']} | Tipo: {sugerencia['tipo']} | Target: {sugerencia['target_sugerido']}")

print(f"\n  [2/5] Preprocesamiento")
prep = maestro.preprocesar()
prep.resumen()

print(f"\n  [3/5] Clasificación")
res_clf = maestro.ejecutar_clasificacion(
    target='target',
    algoritmo='Random Forest',
    params={'n_estimators': 100, 'max_depth': 8}
)
print(f"        Modelo   : {res_clf['modelo_nombre']}")
print(f"        Exactitud: {res_clf['exactitud']:.2%}")
print(f"        Decision : {res_clf['decision']}")

print(f"\n  [4/5] Benchmark clasificación")
_, tabla = maestro.comparar_clasificacion(target='target')
print(tabla.head(3).to_string(index=False))

print(f"\n  [5/5] Clustering")
df_cluster = df.drop(columns=['target']).sample(200, random_state=42)
maestro_cluster = AgenteMaestro(df_cluster)
res_cluster = maestro_cluster.ejecutar_clustering(
    algoritmo='KMeans',
    n_clusters=2,
    normalizar=True
)
m = res_cluster['metricas']
print(f"        Algoritmo   : {res_cluster['algoritmo']}")
print(f"        Silhouette  : {m['silhouette']:.4f}")
print(f"        Calinski-H  : {m['calinski_harabasz']:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# PRUEBA 4 — Regresión con criterios R² + RMSE
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print(f"PRUEBA 4 — Regresión + criterio dual (R² y RMSE)")
print(SEP)

data_reg = load_diabetes()
df_reg = pd.DataFrame(data_reg.data, columns=data_reg.feature_names)
df_reg['target'] = data_reg.target

maestro_reg = AgenteMaestro(df_reg)

print(f"\n  EDA:")
resumen_reg = maestro_reg.ejecutar_eda()
print(f"    Shape: {resumen_reg['shape']} | Nulos: {resumen_reg['nulos_totales']}")

print(f"\n  Regresión con umbral R² y RMSE:")
res_reg = maestro_reg.ejecutar_regresion(
    target='target',
    algoritmo='Ridge',
    umbral_r2=0.7,
    umbral_rmse=55.0
)
print(f"    Modelo  : {res_reg['modelo_nombre']}")
print(f"    R²      : {res_reg['r2']:.4f}")
print(f"    RMSE    : {res_reg['rmse']:.4f}")
print(f"    Decision: {res_reg['decision']}")

print(f"\n  Comparación top 3:")
_, tabla_reg, _ = maestro_reg.comparar_regresion(target='target')
print(tabla_reg.head(3).to_string(index=False))


print(f"\n{SEP}")
print("Todas las pruebas completadas exitosamente.")
print(f"Agentes probados: AgenteEDA, AgentePreprocesamiento,")
print(f"                  AgenteClasificacion, AgenteRegresion,")
print(f"                  AgenteClustering, AgenteMaestro")
print(SEP)
