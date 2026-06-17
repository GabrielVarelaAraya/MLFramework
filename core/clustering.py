from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, pairwise_distances)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

try:
    from sklearn_extra.cluster import KMedoids
    _HAS_KMEDOIDS = True
except ImportError:
    _HAS_KMEDOIDS = False

try:
    import umap.umap_ as umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

from core.no_supervisado import NoSupervisado


class _KMedoidsFallback:
    """K-Medoids (PAM, iteración de Voronoi) sin dependencias externas.

    Se usa como respaldo cuando scikit-learn-extra no está instalado
    (por ejemplo en Python 3.12+, donde esa librería no ofrece wheels y
    suele fallar al compilar). Expone una API mínima compatible con
    sklearn_extra.cluster.KMedoids: ``fit_predict``, ``labels_`` y
    ``cluster_centers_``.
    """

    def __init__(self, n_clusters=3, metric="euclidean", random_state=42,
                 max_iter=300, n_init=10):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init

    def _una_corrida(self, D, n, rng):
        medoides = rng.choice(n, self.n_clusters, replace=False)
        labels = np.argmin(D[:, medoides], axis=1)
        for _ in range(self.max_iter):
            nuevos = medoides.copy()
            for k in range(self.n_clusters):
                miembros = np.where(labels == k)[0]
                if len(miembros) == 0:
                    continue
                costos = D[np.ix_(miembros, miembros)].sum(axis=1)
                nuevos[k] = miembros[int(np.argmin(costos))]
            nuevas_labels = np.argmin(D[:, nuevos], axis=1)
            if np.array_equal(nuevos, medoides):
                break
            medoides, labels = nuevos, nuevas_labels
        # Costo total = suma de distancias de cada punto a su medoide
        costo = D[np.arange(n), medoides[labels]].sum()
        return medoides, labels, costo

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        D = pairwise_distances(X, metric=self.metric)
        rng = np.random.RandomState(self.random_state)

        mejor = None
        for _ in range(self.n_init):
            medoides, labels, costo = self._una_corrida(D, n, rng)
            if mejor is None or costo < mejor[2]:
                mejor = (medoides, labels, costo)

        medoides, labels, _ = mejor
        self.medoid_indices_ = medoides
        self.cluster_centers_ = X[medoides]
        self.labels_ = labels
        return labels


def _crear_kmedoids(n_clusters, metric, random_state, max_iter=300):
    """Devuelve KMedoids de scikit-learn-extra si está disponible; si no, el respaldo nativo."""
    if _HAS_KMEDOIDS:
        return KMedoids(n_clusters=n_clusters, metric=metric,
                        random_state=random_state, max_iter=max_iter)
    return _KMedoidsFallback(n_clusters=n_clusters, metric=metric,
                             random_state=random_state, max_iter=max_iter)


class Clustering(NoSupervisado):

    def __init__(self, df):
        super().__init__(df)

    # ── API principal para Streamlit ─────────────────────────────────────────
    def ejecutar(self, algoritmo, n_clusters=3, columnas=None,
                 normalizar=True, random_state=42, **kwargs):
        """Ejecuta un algoritmo de clustering y retorna resultados estructurados."""
        df_num = self.df.select_dtypes(include='number').dropna()
        if columnas is not None:
            df_num = df_num[columnas]
        if df_num.shape[1] < 2:
            raise ValueError("Se necesitan al menos 2 columnas numéricas.")

        X = StandardScaler().fit_transform(df_num) if normalizar else df_num.values
        X_proj = X  # espacio donde se calcularán las métricas

        algoritmo_lower = algoritmo.lower()
        if algoritmo_lower == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=random_state,
                           n_init=kwargs.get("n_init", 10),
                           max_iter=kwargs.get("max_iter", 500))
            labels = model.fit_predict(X)
            centros = model.cluster_centers_

        elif algoritmo_lower == "kmedoids":
            model = _crear_kmedoids(n_clusters, kwargs.get("metric", "euclidean"),
                                    random_state, max_iter=kwargs.get("max_iter", 500))
            labels = model.fit_predict(X)
            centros = model.cluster_centers_

        elif algoritmo_lower == "hac":
            linkage_method = kwargs.get("linkage", "ward")
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method,
                                            metric='euclidean' if linkage_method == 'ward' else kwargs.get("metric", "euclidean"))
            labels = model.fit_predict(X)
            centros = self._compute_centroids(X, labels, n_clusters)

        elif algoritmo_lower == "t-sne":
            X_proj = TSNE(n_components=2, random_state=random_state).fit_transform(X)
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = model.fit_predict(X_proj)
            centros = self._compute_centroids(X, labels, n_clusters)

        elif algoritmo_lower == "umap":
            if not _HAS_UMAP:
                raise ImportError("umap-learn no está instalado.")
            reducer = umap.UMAP(n_components=2, n_neighbors=kwargs.get("n_neighbors", 15),
                                random_state=random_state)
            X_proj = reducer.fit_transform(X)
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = model.fit_predict(X_proj)
            centros = self._compute_centroids(X, labels, n_clusters)

        elif algoritmo_lower == "dbscan":
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            # DBSCAN puede generar ruido (etiqueta -1)
            unique_labels = np.unique(labels)
            n_clusters_found = len(unique_labels[unique_labels != -1])
            # Actualizar n_clusters con el número real encontrado
            n_clusters = n_clusters_found
            centros = self._compute_centroids(X, labels, n_clusters) if n_clusters > 0 else np.array([])

        else:
            raise ValueError(f"Algoritmo desconocido: {algoritmo}")

        metricas = self._calc_metricas(X_proj, labels)
        self.agregar_modelo(algoritmo, n_clusters, metricas["silhouette"])

        return {
            "algoritmo": algoritmo,
            "labels": labels,
            "centros": centros,
            "n_clusters": n_clusters,
            "X": X,
            "X_proj": X_proj,
            "columnas": list(df_num.columns),
            "metricas": metricas,
            "df_original": df_num,
        }

    @staticmethod
    def _compute_centroids(X, labels, n_clusters):
        return np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

    @staticmethod
    def _calc_metricas(X, labels):
        unique = np.unique(labels)
        if len(unique) < 2:
            return {"silhouette": float('nan'), "calinski_harabasz": float('nan'),
                    "davies_bouldin": float('nan')}
        return {
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        }

    # ── Visualizaciones que retornan figuras ─────────────────────────────────
    @staticmethod
    def plot_proyeccion_2d_fig(X, labels, n_clusters, titulo="Proyección 2D",
                                xlabel="PC1", ylabel="PC2"):
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            var_exp = pca.explained_variance_ratio_
            xlabel = f"PC1 ({var_exp[0]*100:.1f}% var)"
            ylabel = f"PC2 ({var_exp[1]*100:.1f}% var)"
        else:
            coords = X

        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        palette = sns.color_palette("Set2", n_clusters)
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[k]],
                       label=f"Cluster {k}", alpha=0.75,
                       edgecolors="white", linewidths=0.3, s=60)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)
        ax.legend(framealpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_distribucion_clusters_fig(labels, n_clusters):
        counts = pd.Series(labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        bars = ax.bar([f"Cluster {i}" for i in counts.index], counts.values,
                      color=sns.color_palette("Set2", n_clusters),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Cantidad de puntos")
        ax.set_title("Tamaño de cada cluster")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_silhouette_curve_fig(X, k_min=2, k_max=10, random_state=42):
        ks = list(range(k_min, min(k_max + 1, len(X))))
        scores = []
        for k in ks:
            lbl = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
            scores.append(silhouette_score(X, lbl))
        best_k = ks[int(np.argmax(scores))]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        ax.plot(ks, scores, marker="o", color="#4f8bf9", linewidth=2.5, markersize=8)
        ax.fill_between(ks, scores, alpha=0.15, color="#4f8bf9")
        ax.axvline(best_k, color="#00c49a", linestyle="--", alpha=0.7,
                   label=f"Mejor k = {best_k}")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score vs k")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return fig, best_k, max(scores)

    @staticmethod
    def plot_dendrograma_fig(X, metodo="ward", etiquetas=None):
        Z = linkage(X, method=metodo, metric='euclidean')
        fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
        dendrogram(Z, labels=etiquetas, ax=ax, leaf_font_size=8)
        ax.set_title(f"Dendrograma — método: {metodo}")
        ax.grid(False)
        plt.tight_layout()
        return fig

    def perfil_clusters(self, resultado):
        """Retorna DataFrame con medias por cluster."""
        df = resultado["df_original"].copy()
        df["Cluster"] = resultado["labels"]
        return df.groupby("Cluster").mean().round(3)

    def comparar_todos(self, n_clusters=3, columnas=None, normalizar=True, random_state=42):
        """Compara todos los algoritmos disponibles."""
        algoritmos = ["KMeans", "KMedoids", "HAC"]
        if _HAS_UMAP:
            algoritmos.append("UMAP")

        resultados = []
        for algo in algoritmos:
            try:
                res = self.ejecutar(
                    algoritmo=algo,
                    n_clusters=n_clusters,
                    columnas=columnas,
                    normalizar=normalizar,
                    random_state=random_state,
                )
                resultados.append(res)
            except Exception:
                pass

        if not resultados:
            raise ValueError("Ningún algoritmo pudo ejecutarse.")

        tabla = pd.DataFrame([{
            "Algoritmo": r["algoritmo"],
            "Silhouette": r["metricas"]["silhouette"],
            "Calinski-Harabasz": r["metricas"]["calinski_harabasz"],
            "Davies-Bouldin": r["metricas"]["davies_bouldin"],
        } for r in resultados])

        return resultados, tabla

    def decidir(self, resultados):
        # resultados es lista de dicts con la clave "metricas"
        if not resultados:
            raise ValueError("No hay resultados para comparar.")
        for r in resultados:
            m = r.get("metricas", {})
            r["silhouette"] = m.get("silhouette", float("nan"))
            r["calinski_harabasz"] = m.get("calinski_harabasz", float("nan"))
            r["davies_bouldin"] = m.get("davies_bouldin", float("nan"))

        ch_values = [r["calinski_harabasz"] for r in resultados
                     if np.isfinite(r["calinski_harabasz"])]
        max_ch = max(ch_values) if ch_values else 1.0
        if max_ch == 0 or not np.isfinite(max_ch):
            max_ch = 1.0

        for r in resultados:
            sil = r["silhouette"] if np.isfinite(r["silhouette"]) else 0.0
            ch = r["calinski_harabasz"] if np.isfinite(r["calinski_harabasz"]) else 0.0
            db = r["davies_bouldin"] if np.isfinite(r["davies_bouldin"]) else 0.0
            r["score_compuesto"] = sil * 0.5 + (ch / max_ch) * 0.3 - db * 0.2

        return max(resultados, key=lambda r: r["score_compuesto"])
