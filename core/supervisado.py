import pandas as pd

from eda import analisisEDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Supervisado(analisisEDA):

    def __init__(self, df):
        super().__init__(None, None)
        self.df = df

    def preparar_datos(self, target='target', test_size=0.25, random_state=42, scale=True):
        if target not in self.df.columns:
            raise ValueError(f"No existe la columna '{target}' en el DataFrame")

        X = self.df.drop(columns=[target]).select_dtypes(include='number')
        if X.empty:
            raise ValueError("No hay columnas numéricas predictoras disponibles.")

        if scale:
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, y
