import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from sklearn.datasets import (load_breast_cancer, load_digits,
                               load_diabetes, load_iris, make_blobs)

from agente_maestro import AgenteMaestro

SEP  = "=" * 55
SEP2 = "-" * 45


def print_eda(maestro, nombre):
    resumen   = maestro.ejecutar_eda()
    sugerencia = maestro.sugerir_tipo()
    print(f"  Shape        : {resumen['shape']}")
    print(f"  Nulos totales: {resumen['nulos_totales']}")
    print(f"  Duplicados   : {resumen['duplicados']}")
    print(f"  Tipo sugerido: {sugerencia['tipo']}  |  target: {sugerencia['target_sugerido']}")
    print(f"  Razon        : {sugerencia['razon']}")


# ════════════════════════════════════════════════════════════════════════════
# CLASIFICACION — Breast Cancer / Digits / Iris
# ════════════════════════════════════════════════════════════════════════════
datasets_clf = {
    "Breast Cancer": (load_breast_cancer, "target"),
    "Digits"       : (load_digits,        "target"),
    "Iris"         : (load_iris,          "target"),
}

for nombre, (loader, target) in datasets_clf.items():
    print(f"\n{SEP}")
    print(f"CLASIFICACION — {nombre}")
    print(SEP)

    data = loader()
    df = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, 'feature_names') else
                      [f"f{i}" for i in range(data.data.shape[1])])
    df[target] = data.target

    maestro = AgenteMaestro(df)
    print_eda(maestro, nombre)

    print(f"\n  {SEP2}")
    res = maestro.ejecutar_clasificacion(
        target=target,
        algoritmo="Random Forest",
        params={"n_estimators": 100, "max_depth": 8},
    )
    print(f"  Modelo    : {res['modelo_nombre']}")
    print(f"  Exactitud : {res['exactitud']:.2%}")
    print(f"  F1-Score  : {res['f1_ponderado']:.4f}")
    print(f"  Decision  : {res['decision']}")
    print(f"  Matriz de confusion:")
    print(res["matriz_confusion"])

    print(f"\n  Comparacion todos los algoritmos:")
    _, tabla = maestro.comparar_clasificacion(target=target)
    print(tabla.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# REGRESION — Diabetes / California Housing (muestra) / Iris numerica
# ════════════════════════════════════════════════════════════════════════════
from sklearn.datasets import fetch_california_housing

datasets_reg = {
    "Diabetes"          : (load_diabetes,           "target"),
    "California Housing": (fetch_california_housing, "Price"),
}

for nombre, (loader, target) in datasets_reg.items():
    print(f"\n{SEP}")
    print(f"REGRESION — {nombre}")
    print(SEP)

    data = loader()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[target] = data.target
    if nombre == "California Housing":
        df = df.sample(500, random_state=42).reset_index(drop=True)

    maestro = AgenteMaestro(df)
    print_eda(maestro, nombre)

    print(f"\n  {SEP2}")
    res = maestro.ejecutar_regresion(
        target=target,
        algoritmo="Gradient Boosting",
        params={"n_estimators": 100, "max_depth": 4},
    )
    print(f"  Modelo  : {res['modelo_nombre']}")
    print(f"  R2      : {res['r2']:.4f}")
    print(f"  RMSE    : {res['rmse']:.4f}")
    print(f"  MAE     : {res['mae']:.4f}")
    print(f"  Decision: {res['decision']}")

    print(f"\n  Comparacion todos los algoritmos:")
    _, tabla, errores = maestro.comparar_regresion(target=target)
    print(tabla.to_string(index=False))
    if errores:
        print(f"  Fallaron: {[e['algoritmo'] for e in errores]}")


# ════════════════════════════════════════════════════════════════════════════
# CLUSTERING — Iris / Digits (reducido) / make_blobs
# ════════════════════════════════════════════════════════════════════════════
from sklearn.datasets import load_wine
import numpy as np

datasets_cl = {
    "Iris"    : (load_iris,  3),
    "Wine"    : (load_wine,  3),
    "Blobs"   : (None,       4),
}

for nombre, (loader, k) in datasets_cl.items():
    print(f"\n{SEP}")
    print(f"CLUSTERING — {nombre}  (k={k})")
    print(SEP)

    if loader is None:
        X, _ = make_blobs(n_samples=300, centers=k, cluster_std=1.2, random_state=42)
        df = pd.DataFrame(X, columns=["x1", "x2"])
    else:
        data = loader()
        df = pd.DataFrame(data.data, columns=data.feature_names
                          if hasattr(data, 'feature_names')
                          else [f"f{i}" for i in range(data.data.shape[1])])

    maestro = AgenteMaestro(df)
    print_eda(maestro, nombre)

    print(f"\n  {SEP2}")
    for algo in ["KMeans", "HAC"]:
        kwargs = {"linkage": "ward"} if algo == "HAC" else {}
        res_c = maestro.ejecutar_clustering(
            algoritmo=algo, n_clusters=k,
            normalizar=True, random_state=42, **kwargs,
        )
        m = res_c["metricas"]
        print(f"  {algo:8s} | Silhouette: {m['silhouette']:.4f} "
              f"| Calinski: {m['calinski_harabasz']:8.2f} "
              f"| Davies-Bouldin: {m['davies_bouldin']:.4f}")

    print(f"\n  Benchmark acumulado:")
    print(maestro._clustering.benchmark().to_string(index=False))

print(f"\n{SEP}")
print("Ejemplo completado.")
print(SEP)
