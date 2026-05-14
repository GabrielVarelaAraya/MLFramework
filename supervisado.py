import pandas as pd

from eda import analisisEDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Supervisado(analisisEDA):

    def __init__(self, df):
        super().__init__(None, None)
        self.df = df

    def preparar_datos(self, target='target'):
        if target not in self.df.columns:
            raise ValueError(f"No existe la columna {target}")

        X = self.df.drop(columns=[target])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        y = self.df[target]

        return train_test_split(X, y, test_size=0.25, random_state=42)