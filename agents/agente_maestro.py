from agents.agente_eda import AgenteEDA
from agents.agente_preprocesamiento import AgentePreprocesamiento
from agents.agente_clustering import AgenteClustering
from agents.agente_clasificacion import AgenteClasificacion
from agents.agente_regresion import AgenteRegresion


class AgenteMaestro:

    def __init__(self, df):
        self.df = df
        self._eda = AgenteEDA(df)
        self._preprocesamiento = AgentePreprocesamiento(df)
        self._clustering = AgenteClustering(df)
        self._clasificacion = AgenteClasificacion(df)
        self._regresion = AgenteRegresion(df)

    # ── EDA ─────────────────────────────────────────────────────────────────
    def ejecutar_eda(self):
        return self._eda.analizar()

    def sugerir_tipo(self):
        return self._eda.detectar_tipo_problema()

    # ── Preprocesamiento ─────────────────────────────────────────────────────
    def preprocesar(self):
        """Retorna instancia de AgentePreprocesamiento para encadenar operaciones."""
        return self._preprocesamiento

    # ── Clustering ───────────────────────────────────────────────────────────
    def ejecutar_clustering(self, algoritmo, n_clusters=3, columnas=None,
                            normalizar=True, random_state=42, **kwargs):
        return self._clustering.ejecutar(
            algoritmo, n_clusters=n_clusters, columnas=columnas,
            normalizar=normalizar, random_state=random_state, **kwargs,
        )
    def decidir_clustering(self, resultados):
        return self._clustering.decidir(resultados)

    def comparar_clustering(self, n_clusters=3, columnas=None,
                            normalizar=True, random_state=42, **kwargs):
        """
        Ejecuta todos los algoritmos de clustering disponibles y devuelve
        una tabla comparativa con métricas (Silhouette, Calinski-Harabasz, Davies-Bouldin).
        """
        return self._clustering.comparar_todos(
            n_clusters=n_clusters, columnas=columnas,
            normalizar=normalizar, random_state=random_state, **kwargs,
        )

    # ── Clasificación ────────────────────────────────────────────────────────
    def ejecutar_clasificacion(self, target, algoritmo="Random Forest",
                               params=None, test_size=0.25, random_state=42):
        resultado = self._clasificacion.entrenar(
            algoritmo, target=target, params=params,
            test_size=test_size, random_state=random_state,
        )
        resultado["decision"] = self._clasificacion.decidir(resultado)
        return resultado

    def comparar_clasificacion(self, target, params=None, test_size=0.25, random_state=42):
        return self._clasificacion.comparar_todos(
            target, params=params, test_size=test_size, random_state=random_state,
        )

    # ── Regresión ────────────────────────────────────────────────────────────
    def ejecutar_regresion(self, target, algoritmo="Regresión Lineal Múltiple",
                           params=None, test_size=0.25, random_state=42,
                           umbral_r2=0.7, umbral_rmse=None):
        resultado = self._regresion.entrenar(
            algoritmo, target=target, params=params,
            test_size=test_size, random_state=random_state,
        )
        resultado["decision"] = self._regresion.decidir(
            resultado, umbral_r2=umbral_r2, umbral_rmse=umbral_rmse
        )
        return resultado

    def comparar_regresion(self, target, params=None, test_size=0.25, random_state=42):
        return self._regresion.comparar_todos(
            target, params=params, test_size=test_size, random_state=random_state,
        )
