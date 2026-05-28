import pandas as pd

from eda import analisisEDA


class NoSupervisado(analisisEDA):
    def __init__(self, df):
        super().__init__(None, None)
        self.df = df
        self.__model = []

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, p_model):
        self.__model = p_model

    def benchmark(self):
        return pd.DataFrame(
            self.__model,
            columns=['Algoritmo', 'Num de Clusters', 'Silhouette Score']
        )

    def agregar_modelo(self, algoritmo, n_clusters, silhouette_score):
        self.__model.append([algoritmo, n_clusters, silhouette_score])
