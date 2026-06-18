from core.eda import analisisEDA


class AgenteEDA:

    def __init__(self, df):
        self._backend = analisisEDA(None, None)
        self._backend.df = df

    def analizar(self):
        """Retorna dict con métricas del dataset."""
        return self._backend.resumen_dict()

    def detectar_tipo_problema(self):
        """Heurística para sugerir tipo de problema (clasificación/regresión/no supervisado)."""
        return analisisEDA.detectar_tipo_problema(self._backend.df)

    def tipos_datos(self):
        """Imprime tipos de datos de todas las columnas."""
        self._backend.tipoDatos()

    def estadisticas(self):
        """Imprime estadísticas descriptivas completas."""
        self._backend.analisis()

    def valores_faltantes(self):
        """Imprime valores faltantes por columna."""
        self._backend.valores_faltantes()

    def valores_unicos(self, columna):
        """Imprime valores únicos de una columna con sus frecuencias."""
        self._backend.valores_unicos(columna)

    # ── Visualizaciones ──────────────────────────────────────────────────────
    def grafico_boxplot(self):
        """Genera boxplots para todas las columnas numéricas."""
        return self._backend.graficoBoxplot_fig()

    def histogramas(self):
        """Genera histogramas para todas las columnas numéricas."""
        return self._backend.histogramas_fig()

    def distribucion_variables(self):
        """Genera gráficos de distribución (KDE) para todas las columnas numéricas."""
        return self._backend.distribucionVariables_fig()

    def grafico_densidad(self):
        """Genera gráficos de densidad para todas las columnas numéricas."""
        return self._backend.datosDensidad_fig()

    def grafico_correlacion(self):
        """Genera heatmap de correlaciones."""
        return self._backend.graficoCorrelacion_fig()

    def graficos_dispersion(self, columnas=None):
        """Genera pair plot de dispersión."""
        return self._backend.graficosDispersion_fig(columnas=columnas)

    def histograma_clase(self, columna_objetivo):
        """Genera histograma de distribución de clases."""
        return self._backend.histogramaClase_fig(columna_objetivo)

    # ── Estadísticas avanzadas ────────────────────────────────────────────────
    def analisis_estadistico(self):
        """Retorna dict con media, mediana, moda, varianza, desv_std, min, max por columna."""
        return self._backend.analisis_estadistico_dict()

    # ── Registros inconsistentes ──────────────────────────────────────────────
    def detectar_inconsistencias(self):
        """Busca valores negativos en columnas positivas, columnas constantes, desbalance extremo."""
        return self._backend.detectar_inconsistencias()

    # ── Outliers ──────────────────────────────────────────────────────────────
    def detectar_outliers_iqr(self, columnas=None):
        """Detecta outliers usando el método IQR. Retorna dict con resultados por columna."""
        return self._backend.detectar_outliers_iqr(columnas=columnas)

    def detectar_outliers_zscore(self, columnas=None, umbral=3):
        """Detecta outliers usando Z-Score. Retorna dict con resultados por columna."""
        return self._backend.detectar_outliers_zscore(columnas=columnas, umbral=umbral)

    def grafico_outliers_iqr(self, resultados_iqr, max_cols=12):
        """Genera figura con boxplots y límites IQR."""
        return self._backend.grafico_outliers_iqr_fig(resultados_iqr, max_cols=max_cols)

    # ── Correlación Spearman ──────────────────────────────────────────────────
    def grafico_correlacion_spearman(self):
        """Genera heatmap de correlación de Spearman."""
        return self._backend.correlacion_spearman_fig()
