import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class AgentePreprocesamiento:

    def __init__(self, df):
        self.df = df.copy()
        self._scaler = None
        self._label_encoders = {}

    def seleccionar_numericas(self):
        """Selecciona solo columnas numéricas."""
        self.df = self.df.select_dtypes(include="number")
        return self

    def eliminar_columnas(self, columnas):
        """Elimina columnas especificadas."""
        self.df = self.df.drop(columns=columnas, errors='ignore')
        return self

    def eliminar_nulos(self):
        """Elimina filas con valores nulos."""
        antes = self.df.shape[0]
        self.df = self.df.dropna()
        despues = self.df.shape[0]
        print(f"Eliminadas {antes - despues} filas con nulos. Total: {despues}")
        return self

    def eliminar_duplicados(self):
        """Elimina filas duplicadas."""
        antes = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        despues = self.df.shape[0]
        print(f"Eliminadas {antes - despues} filas duplicadas. Total: {despues}")
        return self

    def imputar_nulos(self, estrategia="mean", columnas=None):
        """
        Imputa valores nulos.
        
        Parámetros:
            estrategia: 'mean', 'median', 'mode', o un valor específico
            columnas: lista de columnas a imputar (None = todas las numéricas)
        """
        if columnas is None:
            columnas = self.df.select_dtypes(include="number").columns
        
        for col in columnas:
            if col not in self.df.columns:
                continue
            if estrategia == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif estrategia == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif estrategia == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                self.df[col] = self.df[col].fillna(estrategia)
        return self

    def escalar(self, columnas=None):
        """
        Escala variables numéricas con StandardScaler.
        
        Parámetros:
            columnas: lista de columnas a escalar (None = todas las numéricas)
        """
        if columnas is None:
            columnas = self.df.select_dtypes(include="number").columns.tolist()
        
        self._scaler = StandardScaler()
        self.df[columnas] = self._scaler.fit_transform(self.df[columnas])
        return self

    def codificar_categoricas(self, columnas=None, metodo="dummies"):
        """
        Codifica variables categóricas.
        
        Parámetros:
            columnas: lista de columnas a codificar (None = todas las categóricas)
            metodo: 'dummies' (one-hot) o 'label' (label encoding)
        """
        if columnas is None:
            columnas = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if metodo == "dummies":
            self.df = pd.get_dummies(self.df, columns=columnas, drop_first=True)
            # Convertir booleanos a int
            bool_cols = self.df.select_dtypes(include='bool').columns
            self.df[bool_cols] = self.df[bool_cols].astype(int)
        elif metodo == "label":
            for col in columnas:
                if col not in self.df.columns:
                    continue
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self._label_encoders[col] = le
        return self

    def renombrar_columnas(self, mapeo):
        """Renombra columnas según un diccionario de mapeo."""
        self.df = self.df.rename(columns=mapeo)
        return self

    def separar_features_target(self, target):
        """
        Separa variables predictoras y objetivo.
        
        Retorna:
            (X, y) donde X son las features y y es el target
        """
        if target not in self.df.columns:
            raise ValueError(f"Columna '{target}' no existe en el DataFrame")
        
        X = self.df.drop(columns=[target])
        y = self.df[target]
        return X, y

    def get_dataframe(self):
        """Retorna el DataFrame procesado."""
        return self.df

    def get_scaler(self):
        """Retorna el scaler entrenado (si se usó)."""
        return self._scaler

    def get_label_encoders(self):
        """Retorna los label encoders entrenados (si se usaron)."""
        return self._label_encoders

    def resumen(self):
        """Imprime resumen del dataset actual."""
        print(f"Shape: {self.df.shape}")
        print(f"Nulos: {self.df.isnull().sum().sum()}")
        print(f"Duplicados: {self.df.duplicated().sum()}")
        print(f"Columnas: {list(self.df.columns)}")
