import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from seaborn import color_palette
from scipy.cluster.hierarchy import dendrogram, linkage
from math import ceil, floor, pi

# Clase EDA:
class analisisEDA():
# Aqui inicio y cargo el Dataset desde archivo CSV.
    def __init__(self,path, num):
        self.__df = self.__datosCargados(path, num)
    
    @property
    def df(self):
        return self.__df
    
    @df.setter
    def df(self, p_df):
        self.__df = p_df

# Cargar el dataset
    def __datosCargados(self, path, num):
        if num == 1:
            return pd.read_csv(path,
            sep = ",",
            decimal = ".",
            index_col = 0)
        if num == 2:
            return pd.read_csv(path,
            sep = ";",
            decimal = ".")
        
# Mostrar el tipo de datos
    def tipoDatos(self):
        print(self.__df.dtypes)

# Filtra las columnas num
    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include=["number"])

# Cambiar variables categoricas a dummies
    def analisisCompleto(self):
        columnas_categoricas = self.__df.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"Columnas categóricas convertidas a dummies: {columnas_categoricas}")
        self.__df = pd.get_dummies(self.__df, columns=columnas_categoricas, drop_first=True).astype(int)


# Eliminar las columnas que no sirven
    def eliminarColumnas(self, columnas):
        self.__df.drop(columns=columnas, inplace=True)

# Cambiar el nombre de las columnas
    def renombrarColumnas(self, nuevos_nombres):
        self.__df.rename(columns=nuevos_nombres, inplace=True)

# Valores unicos
    def valores_unicos(self, v):
        unique_values = self.__df[v].unique()
        print("Valores unicos en", v,":")
        for value in unique_values:
            count = (self.__df[v] == value).sum()
            print(f"{value}: {count}")

# Valores Faltantes
    def valores_faltantes(self):
        missing_values = self.__df.isna().sum()
        print("Missing values by column:")
        print(missing_values)
        print('\n')

# Eliminar los datos duplicados
    def eliminarDuplicados(self):
        antes = self.__df.shape[0]
        self.__df.drop_duplicates(inplace=True)
        despues = self.__df.shape[0]
        print(f"Se eliminaron {antes - despues} filas duplicadas. Total actual: {despues} filas.")

# Eliminar los datos Nulos
    def eliminarNulos(self):
        nulos_antes = self.__df.isnull().sum().sum()
        filas_antes = self.__df.shape[0]
        print(f"Valores nulos totales antes de eliminar: {nulos_antes}")
        self.__df.dropna(inplace=True)
        nulos_despues = self.__df.isnull().sum().sum()
        filas_despues = self.__df.shape[0]
        print(f"Filas eliminadas por contener nulos: {filas_antes - filas_despues}")
        print(f"Valores nulos restantes: {nulos_despues}")

# Analisis de los datos del dataset
    def analisis(self):
        print("Dimensiones:", self.__df.shape)
        print(self.__df.head())
        print("="*40)
        print("Estadisticas Descriptivas Generales")
        print("="*40)
        print("Media:\n", self.__df.mean(numeric_only=True))
        print("="*40)
        print("Mediana:\n", self.__df.median(numeric_only=True))
        print("="*40)
        print("Desviación estándar:\n", self.__df.std(numeric_only=True))
        print("="*40)
        print("Máximos:\n", self.__df.max(numeric_only=True))
        print("="*40)
        print("Mínimos:\n", self.__df.min(numeric_only=True))
        print("="*40)
        print("Cuantiles:\n", self.__df.quantile([0, 0.25, 0.5, 0.75, 1], numeric_only=True))

# Boxplot
    def graficoBoxplot(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()
        colores = sns.color_palette("Set3", n)
        for i, col in enumerate(columnas_numericas):
            sns.boxplot(y=self.__df[col], ax=axes[i], color=colores[i])
            axes[i].set_title(f"Boxplot de {col}", fontsize=10)
            axes[i].set_ylabel(col)
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

# Histograma (sin la clase predictoria)
    def histogramas(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()  # Para acceder como una lista plana
        colores = sns.color_palette("Set2", n)
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
        plt.show()

# Gráficos de distribución (histogramas) 
    def distribucionVariables(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()
        colores = sns.color_palette("coolwarm", n)
        for i, col in enumerate(columnas_numericas):
            sns.histplot(self.__df[col], kde=True, ax=axes[i], color=colores[i], bins=30)
            axes[i].set_title(f"Distribución de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

# Histograma de la clase predictora
    def histogramaClase(self, columna_objetivo):
        if columna_objetivo in self.__df.columns:
            plt.figure(figsize=(8, 5), dpi=150)
            colores = sns.color_palette("pastel")
            self.__df[columna_objetivo].value_counts().plot(kind='bar', color=colores)
            plt.title(f"Distribución de la Clase: {columna_objetivo}")
            plt.xlabel(columna_objetivo)
            plt.ylabel("Frecuencia")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print(f"La columna '{columna_objetivo}' no existe en el DataFrame.")

# Funcion de la Densidad de los datos
    def datosDensidad(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)  
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()
        colores = sns.color_palette("husl", n)
        for i, col in enumerate(columnas_numericas):
            sns.kdeplot(data=self.__df, x=col, fill=True, ax=axes[i], color=colores[i], linewidth=2)
            axes[i].set_title(f"Densidad de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Densidad")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

# Correlacion
    def correlaciones(self):
        corr = self.__df.corr(numeric_only=True)
        print("Matriz de correlaciones:\n")
        print(corr)

# Grafico de correlacion
    def graficoCorrelacion(self):
        corr = self.__df.corr(numeric_only=True)
        plt.figure(figsize=(12, 8), dpi=150)
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed() 
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True,fmt=".2f",linewidths=0.5,linecolor='white', square=True, cbar_kws={"shrink": 0.8, "label": "Correlación"},annot_kws={"size": 10, "color": "black"})
        plt.title("Mapa de Calor de Correlaciones", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# Graficos de dispersion
    def graficosDispersion(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        if len(columnas_numericas) >= 2:
            sns.pairplot(self.__df[columnas_numericas])
            plt.suptitle("Gráficos de Dispersión por Pares", y=1.02)
            plt.tight_layout()
            plt.show()
        else:
            print("No hay suficientes variables numéricas para graficar dispersión.")

    def centroide(num_cluster, datos, clusters):
        ind = clusters == num_cluster
        return pd.DataFrame(datos[ind].mean()).T

    def recodificar(col, nuevo_codigo):
        col_cod = pd.Series(col, copy=True)
        for llave, valor in nuevo_codigo.items():
            col_cod.replace(llave, valor, inplace=True)
        return col_cod

# EXTRA SECCION PARA LOS GRAFICOS EN HAC   
    def bar_plot(centros, labels, scale = False,cluster = None, var = None):
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        centros = np.copy(centros)
        if scale:
            for col in range(centros.shape[1]):
                centros[:,col] = centros[:,col] / max(centros[:,col])
        colores = color_palette("Set2")
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
        def inside_plot(valores, labels, titulo):
            plt.barh(range(len(valores)), valores, 1/1.5, color = colores)
            plt.xlim(minimo, ceil(centros.max()))
            plt.title(titulo)
        if var is not None:
            centros = np.array([n[[x in var for x in labels]] for n in centros])
            colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
            labels = labels[[x in var for x in labels]]
        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([]) 
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([]) 
                pos += 1
  
    def radar_plot(centros, labels):
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if 
                            max(n) != min(n) else (n/n * 50) for n in centros.T])
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar = True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], 
            color = "grey", size = 8)
        plt.ylim(-10, 100)
        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth = 1, linestyle = 'solid', 
                    label = 'Cluster ' + str(i))
            ax.fill(angulos, valores, alpha = 0.3)
        plt.legend(loc='upper right', bbox_to_anchor = (0.1, 0.1))

    def __str__(self):
        return f'AnalisisDatosExploratorio: {self.__df} {self.__path}'