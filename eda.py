import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from seaborn import color_palette
from scipy.cluster.hierarchy import dendrogram, linkage
from math import ceil, floor, pi


class analisisEDA():

    def __init__(self, path, num):
        self.__df = self.__datosCargados(path, num)

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, p_df):
        self.__df = p_df

    def __datosCargados(self, path, num):
        if path is None:
            return None
        if num == 1:
            return pd.read_csv(path, sep=",", decimal=".", index_col=0)
        if num == 2:
            return pd.read_csv(path, sep=";", decimal=".")
        return None

    # ── Información del dataset ──────────────────────────────────────────────
    def tipoDatos(self):
        print(self.__df.dtypes)

    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include=["number"])

    def analisisCompleto(self):
        columnas_categoricas = self.__df.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"Columnas categóricas convertidas a dummies: {columnas_categoricas}")
        self.__df = pd.get_dummies(self.__df, columns=columnas_categoricas, drop_first=True).astype(int)

    def eliminarColumnas(self, columnas):
        self.__df.drop(columns=columnas, inplace=True)

    def renombrarColumnas(self, nuevos_nombres):
        self.__df.rename(columns=nuevos_nombres, inplace=True)

    def valores_unicos(self, v):
        unique_values = self.__df[v].unique()
        print("Valores unicos en", v, ":")
        for value in unique_values:
            count = (self.__df[v] == value).sum()
            print(f"{value}: {count}")

    def valores_faltantes(self):
        missing_values = self.__df.isna().sum()
        print("Missing values by column:")
        print(missing_values)
        print('\n')

    def eliminarDuplicados(self):
        antes = self.__df.shape[0]
        self.__df.drop_duplicates(inplace=True)
        despues = self.__df.shape[0]
        print(f"Se eliminaron {antes - despues} filas duplicadas. Total actual: {despues} filas.")

    def eliminarNulos(self):
        nulos_antes = self.__df.isnull().sum().sum()
        filas_antes = self.__df.shape[0]
        print(f"Valores nulos totales antes de eliminar: {nulos_antes}")
        self.__df.dropna(inplace=True)
        nulos_despues = self.__df.isnull().sum().sum()
        filas_despues = self.__df.shape[0]
        print(f"Filas eliminadas por contener nulos: {filas_antes - filas_despues}")
        print(f"Valores nulos restantes: {nulos_despues}")

    def analisis(self):
        print("Dimensiones:", self.__df.shape)
        print(self.__df.head())
        print("=" * 40)
        print("Estadisticas Descriptivas Generales")
        print("=" * 40)
        print("Media:\n", self.__df.mean(numeric_only=True))
        print("=" * 40)
        print("Mediana:\n", self.__df.median(numeric_only=True))
        print("=" * 40)
        print("Desviación estándar:\n", self.__df.std(numeric_only=True))
        print("=" * 40)
        print("Máximos:\n", self.__df.max(numeric_only=True))
        print("=" * 40)
        print("Mínimos:\n", self.__df.min(numeric_only=True))
        print("=" * 40)
        print("Cuantiles:\n", self.__df.quantile([0, 0.25, 0.5, 0.75, 1], numeric_only=True))

    def resumen_dict(self):
        """Devuelve un diccionario con métricas del dataset (para Streamlit)."""
        return {
            "shape": self.__df.shape,
            "num_cols": self.__df.select_dtypes(include="number").columns.tolist(),
            "cat_cols": self.__df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "nulos_totales": int(self.__df.isnull().sum().sum()),
            "duplicados": int(self.__df.duplicated().sum()),
            "dtypes": self.__df.dtypes.astype(str).to_dict(),
            "descripcion": self.__df.describe(include="all"),
        }

    # ── Helpers internos de grid ─────────────────────────────────────────────
    def _grid(self, n, columnas=3, size=(5, 4)):
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(size[0] * columnas, size[1] * filas), dpi=120)
        axes = np.array(axes).flatten() if n > 1 else np.array([axes]).flatten()
        return fig, axes

    # ── Gráficos: versiones print/show (compat) ──────────────────────────────
    def graficoBoxplot(self):
        fig = self.graficoBoxplot_fig()
        plt.show()

    def histogramas(self):
        fig = self.histogramas_fig()
        plt.show()

    def distribucionVariables(self):
        fig = self.distribucionVariables_fig()
        plt.show()

    def histogramaClase(self, columna_objetivo):
        fig = self.histogramaClase_fig(columna_objetivo)
        if fig is not None:
            plt.show()

    def datosDensidad(self):
        fig = self.datosDensidad_fig()
        plt.show()

    def correlaciones(self):
        corr = self.__df.corr(numeric_only=True)
        print("Matriz de correlaciones:\n")
        print(corr)
        return corr

    def graficoCorrelacion(self):
        fig = self.graficoCorrelacion_fig()
        plt.show()

    def graficosDispersion(self):
        fig = self.graficosDispersion_fig()
        if fig is not None:
            plt.show()

    # ── Gráficos: versiones que retornan figura (para Streamlit) ─────────────
    def graficoBoxplot_fig(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        fig, axes = self._grid(n)
        colores = sns.color_palette("Set3", n)
        i = -1
        for i, col in enumerate(columnas_numericas):
            sns.boxplot(y=self.__df[col], ax=axes[i], color=colores[i])
            axes[i].set_title(f"Boxplot de {col}", fontsize=10)
            axes[i].set_ylabel(col)
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def histogramas_fig(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        fig, axes = self._grid(n)
        colores = sns.color_palette("Set2", n)
        i = -1
        for i, col in enumerate(columnas_numericas):
            ax = axes[i]
            ax.hist(self.__df[col], bins=30, color=colores[i], edgecolor='black', alpha=0.7)
            ax.set_title(f"Histograma de {col}", fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            ax.grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def distribucionVariables_fig(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        fig, axes = self._grid(n)
        colores = sns.color_palette("coolwarm", n)
        i = -1
        for i, col in enumerate(columnas_numericas):
            sns.histplot(self.__df[col], kde=True, ax=axes[i], color=colores[i], bins=30)
            axes[i].set_title(f"Distribución de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def histogramaClase_fig(self, columna_objetivo):
        if columna_objetivo not in self.__df.columns:
            print(f"La columna '{columna_objetivo}' no existe en el DataFrame.")
            return None
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        colores = sns.color_palette("pastel")
        self.__df[columna_objetivo].value_counts().plot(kind='bar', color=colores, ax=ax)
        ax.set_title(f"Distribución de la Clase: {columna_objetivo}")
        ax.set_xlabel(columna_objetivo)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

    def datosDensidad_fig(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        fig, axes = self._grid(n)
        colores = sns.color_palette("husl", n)
        i = -1
        for i, col in enumerate(columnas_numericas):
            sns.kdeplot(data=self.__df, x=col, fill=True, ax=axes[i], color=colores[i], linewidth=2)
            axes[i].set_title(f"Densidad de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Densidad")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def graficoCorrelacion_fig(self):
        corr = self.__df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True, fmt=".2f",
                    linewidths=0.5, linecolor='white', square=True,
                    cbar_kws={"shrink": 0.8, "label": "Correlación"},
                    annot_kws={"size": 9}, ax=ax)
        ax.set_title("Mapa de Calor de Correlaciones", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig

    def graficosDispersion_fig(self, columnas=None):
        columnas_numericas = columnas if columnas is not None else self.__df.select_dtypes(include='number').columns
        if len(columnas_numericas) < 2:
            print("No hay suficientes variables numéricas para graficar dispersión.")
            return None
        g = sns.pairplot(self.__df[columnas_numericas])
        g.fig.suptitle("Gráficos de Dispersión por Pares", y=1.02)
        plt.tight_layout()
        return g.fig

    # ── Helpers estáticos usados por clustering ──────────────────────────────
    @staticmethod
    def centroide(num_cluster, datos, clusters):
        ind = clusters == num_cluster
        return pd.DataFrame(datos[ind].mean()).T

    @staticmethod
    def recodificar(col, nuevo_codigo):
        col_cod = pd.Series(col, copy=True)
        for llave, valor in nuevo_codigo.items():
            col_cod.replace(llave, valor, inplace=True)
        return col_cod

    @staticmethod
    def bar_plot(centros, labels, scale=False, cluster=None, var=None):
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=120)
        centros = np.copy(centros)
        if scale:
            for col in range(centros.shape[1]):
                col_max = max(centros[:, col]) or 1
                centros[:, col] = centros[:, col] / col_max
        colores = color_palette("Set2")
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0

        def inside_plot(valores, lbls, titulo):
            plt.barh(range(len(valores)), valores, 1 / 1.5, color=colores)
            plt.xlim(minimo, ceil(centros.max()))
            plt.title(titulo)

        if var is not None:
            centros = np.array([n[[x in var for x in labels]] for n in centros])
            labels = labels[[x in var for x in labels]]
        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, f'Cluster {i}')
                plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([])
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, f'Cluster {i}')
                plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([])
                pos += 1
        return fig

    @staticmethod
    def radar_plot(centros, labels):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=120)
        centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if max(n) != min(n)
                            else (n / n * 50) for n in centros.T])
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                   ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
                   color="grey", size=8)
        plt.ylim(-10, 100)
        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth=1, linestyle='solid', label=f'Cluster {i}')
            ax.fill(angulos, valores, alpha=0.3)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        return fig

    def __str__(self):
        return f'AnalisisDatosExploratorio shape={None if self.__df is None else self.__df.shape}'

    # ── Sugerencia de tipo de problema ───────────────────────────────────────
    @staticmethod
    def detectar_tipo_problema(df, umbral_clases=20):
        """
        Heurística para sugerir si el dataset es de clasificación, regresión
        o no supervisado, y cuál columna usar como target.

        Retorna:
            {
                "tipo": "clasificacion" | "regresion" | "no_supervisado",
                "target_sugerido": str | None,
                "razon": str,
                "candidatos": list[dict]  # otras columnas que podrían ser target
            }
        """
        if df is None or df.empty:
            return {"tipo": "no_supervisado", "target_sugerido": None,
                    "razon": "DataFrame vacío.", "candidatos": []}

        nombres_target = {"target", "label", "labels", "class", "classes", "clase",
                          "y", "objetivo", "outcome", "diagnosis", "result",
                          "resultado", "categoria", "category"}

        candidatos = []
        for col in df.columns:
            serie = df[col].dropna()
            if serie.empty:
                continue
            n_unicos = serie.nunique()
            es_numerica = pd.api.types.is_numeric_dtype(serie)
            es_categorica = (not es_numerica) or (n_unicos <= umbral_clases and serie.dtype.kind in "iub")
            nombre_match = col.lower().strip() in nombres_target

            candidatos.append({
                "columna": col,
                "n_unicos": int(n_unicos),
                "es_numerica": bool(es_numerica),
                "es_categorica_likely": bool(es_categorica),
                "nombre_match": bool(nombre_match),
                "posicion": list(df.columns).index(col),
            })

        # 1) Match por nombre típico de target
        por_nombre = [c for c in candidatos if c["nombre_match"]]
        if por_nombre:
            elegido = por_nombre[0]
            col = elegido["columna"]
            if elegido["es_categorica_likely"]:
                return {"tipo": "clasificacion", "target_sugerido": col,
                        "razon": f"Columna '{col}' tiene nombre típico de target y "
                                 f"{elegido['n_unicos']} valores únicos → clasificación.",
                        "candidatos": candidatos}
            return {"tipo": "regresion", "target_sugerido": col,
                    "razon": f"Columna '{col}' tiene nombre típico de target y es "
                             f"numérica continua → regresión.",
                    "candidatos": candidatos}

        # 2) Última columna como convención ML
        if candidatos:
            ultima = candidatos[-1]
            col = ultima["columna"]
            if not ultima["es_numerica"] or (ultima["n_unicos"] <= umbral_clases
                                              and ultima["n_unicos"] >= 2):
                tipo = "clasificacion"
                razon = (f"La última columna '{col}' tiene {ultima['n_unicos']} valores "
                         f"únicos → probable target categórico (clasificación).")
                return {"tipo": tipo, "target_sugerido": col,
                        "razon": razon, "candidatos": candidatos}
            if ultima["es_numerica"] and ultima["n_unicos"] > umbral_clases:
                return {"tipo": "regresion", "target_sugerido": col,
                        "razon": f"La última columna '{col}' es numérica continua con "
                                 f"{ultima['n_unicos']} valores únicos → regresión.",
                        "candidatos": candidatos}

        # 3) Sin candidato claro → no supervisado
        return {"tipo": "no_supervisado", "target_sugerido": None,
                "razon": "No se detectó una columna objetivo clara. Sugerencia: "
                         "análisis exploratorio o clustering.",
                "candidatos": candidatos}
