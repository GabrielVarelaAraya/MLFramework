from core.clustering import Clustering


class AgenteClustering:

    def __init__(self, df):
        self._backend = Clustering(df)

    def ejecutar(self, algoritmo, n_clusters=3, columnas=None,
                 normalizar=True, random_state=42, **kwargs):
        return self._backend.ejecutar(
            algoritmo=algoritmo,
            n_clusters=n_clusters,
            columnas=columnas,
            normalizar=normalizar,
            random_state=random_state,
            **kwargs,
        )

    def perfil_clusters(self, resultado):
        return self._backend.perfil_clusters(resultado)

    def benchmark(self):
        return self._backend.benchmark()

    def comparar_todos(self, n_clusters=3, columnas=None, normalizar=True, random_state=42):
        return self._backend.comparar_todos(
            n_clusters=n_clusters,
            columnas=columnas,
            normalizar=normalizar,
            random_state=random_state,
        )

    # ── Visualizaciones ──────────────────────────────────────────────────────
    def plot_proyeccion_2d(self, resultado):
        return Clustering.plot_proyeccion_2d_fig(
            resultado["X_proj"], resultado["labels"], resultado["n_clusters"],
            titulo=f"{resultado['algoritmo']} — {resultado['n_clusters']} clusters",
        )

    def plot_distribucion_clusters(self, resultado):
        return Clustering.plot_distribucion_clusters_fig(
            resultado["labels"], resultado["n_clusters"]
        )

    def plot_silhouette_curve(self, resultado, k_min=2, k_max=10, random_state=42):
        X = resultado["X_proj"] if resultado["algoritmo"].lower() in ("t-sne", "umap") \
            else resultado["X"]
        return Clustering.plot_silhouette_curve_fig(X, k_min, k_max, random_state)

    def plot_dendrograma(self, resultado, metodo="ward"):
        return Clustering.plot_dendrograma_fig(resultado["X"], metodo=metodo)

    def decidir(self, resultados):
        return self._backend.decidir(resultados)
