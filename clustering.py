from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
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

from no_supervisado import NoSupervisado


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
            if not _HAS_KMEDOIDS:
                raise ImportError("sklearn-extra no está instalado.")
            model = KMedoids(n_clusters=n_clusters, metric=kwargs.get("metric", "cityblock"),
                             random_state=random_state, max_iter=kwargs.get("max_iter", 500))
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
