from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, ward, average, single, complete, fcluster
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from util_pca import ACPmain
from no_supervisado import NoSupervisado

class Clustering(NoSupervisado):

    def __init__(self, df):
        super().__init__(df)

    def ACP(self, n_componentes): 
        p_acp = ACPmain(self.df,n_componentes) 
        self.__ploteoGraficos(p_acp,1)
        self.__ploteoGraficos(p_acp,2)
        self.__ploteoGraficos(p_acp,3)
        
        
    def __ploteoGraficos(self,p_acp, tipo): 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8,4), dpi = 200)
        if tipo==1:
            p_acp.plot_plano_principal()
        elif tipo==2:
            p_acp.plot_circulo()
        elif tipo==3:
            p_acp.plot_sobreposicion()     
            plt.show()
        

    def HAC(self):
        p_hac = self.df
        ward_res = ward(self.df)      
        average_res = average(self.df)  
        single_res  = single(self.df)    
        complete_res = complete(self.df)
        self.__ploteoGraficosHAC(p_hac, ward_res, 1)
        self.__ploteoGraficosHAC(p_hac, average_res, 2)
        self.__ploteoGraficosHAC(p_hac, single_res, 3)
        self.__ploteoGraficosHAC(p_hac, complete_res, 4)
        self.__clusterHAC(1)
        self.__clusterHAC(2)

    # Silhouette Score
        metodos = ['ward', 'average', 'single', 'complete']
        for metodo in metodos:
            silhouette_scores = []
            for k in range(2, 11):
                modelo = AgglomerativeClustering(n_clusters=k, linkage=metodo, metric='euclidean')
                etiquetas = modelo.fit_predict(self.df)
                score = silhouette_score(self.df, etiquetas)
                silhouette_scores.append(score)
                
            mejor_k = np.argmax(silhouette_scores) + 2
            mejor_score = max(silhouette_scores)
            plt.figure(figsize=(10, 6))
            plt.plot(range(2, 11), silhouette_scores, marker='o')
            plt.title(f'Silhouette Score ({metodo})')
            plt.xlabel('Número de clusters')
            plt.ylabel('Silhouette Score')
            plt.grid(True)
            plt.show()
            self.agregar_modelo(f'HAC-{metodo}', mejor_k, mejor_score)

    def __ploteoGraficosHAC(self, p_hac, res, tipo):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
        dendrogram(res, labels=self.df.index.tolist(), ax=ax)

        mensajes = ['Metodo Ward:', 'Average:', 'Single:', 'Complete:']
        print(mensajes[tipo - 1])

        if tipo == 4:
            plt.savefig('dendrogram.png')
        ax.grid(False)
        plt.show()

    def __clusterHAC(self, tipo):
        grupos = fcluster(linkage(self.df, method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
        grupos = grupos-1 
        centros = np.array(pd.concat([self.centroide(0, self.df, grupos), 
                              self.centroide(1, self.df, grupos),
                              self.centroide(2, self.df, grupos)]))
        if tipo == 1:
            self.bar_plot(centros, self.df.columns, scale=True)
        plt.show()
        

# SECCION K-MEDIAS
    def KMedia(self):
        self.__ploteoGraficosKMEDIAS(1)
        self.__ploteoGraficosKMEDIAS(2)

    def __ploteoGraficosKMEDIAS(self, tipo):
        if tipo == 1:
            self.__ploteoKmedias()
        elif tipo == 2:
            self.__ploteoKmedoids()
    
    def __ploteoKmedias(self):
        kmedias = KMeans(n_clusters=3, max_iter=500, n_init=150)
        kmedias.fit(self.df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.df)
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedias.predict(self.df)]
        ax.scatter(componentes[:, 0], componentes[:, 1],c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medias')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedias.cluster_centers_)
        self.radar_plot(centros, self.df.columns)
        plt.show()

    def __ploteoKmedoids(self):
        kmedoids = KMedoids(n_clusters=3, max_iter=500, metric='cityblock')
        kmedoids.fit(self.df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.df)
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedoids.predict(self.df)]
        ax.scatter(componentes[:, 0], componentes[:, 1],c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medoids')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedoids.cluster_centers_)
        self.radar_plot(centros, self.df.columns)
        plt.show()
        
# UMAP
    def UMAP(self, n_componentes=2, n_neighbors=15):
        if self.df is None or self.df.empty:
            raise ValueError("El DataFrame está vacío o no ha sido definido.")
        if self.df.isnull().any().any():
            raise ValueError("El DataFrame contiene valores nulos.")

    # Normalización previa
        df_scaled = StandardScaler().fit_transform(self.df)

    # UMAP
        modelo_umap = umap.UMAP(n_components=n_componentes, n_neighbors=n_neighbors)
        componentes = modelo_umap.fit_transform(df_scaled)

    # Gráfico
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.7)
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title('UMAP')
        ax.grid(False)
        plt.show()

        return componentes

#T-SNE
    def TSNE(self):
        tsne = TSNE(n_components=2, random_state=42)
        df_tsne = tsne.fit_transform(self.df)
        
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(df_tsne)
            silhouette_avg = silhouette_score(df_tsne, cluster_labels)
            silhouette_scores.append(silhouette_avg)
    
        optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(df_tsne)
        
        print('Silhouette Score:', max(silhouette_scores))
        self.__plotTSNE(df_tsne, cluster_labels, silhouette_scores)
        
        self.agregar_modelo('TSNE', optimal_n_clusters, max(silhouette_scores))


    def __plotTSNE(self, df, cluster_labels, silhouette_scores):
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Puntuación de la Silueta')
        plt.title('Silhouette Score')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[:, 0], y=df[:, 1], hue=cluster_labels, palette='tab10', legend='full')
        plt.title('TSNE')
        plt.xlabel('TSNE Dimensión 1')
        plt.ylabel('TSNE Dimensión 2')
        plt.legend(title='Cluster')
        plt.show()