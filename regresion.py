from supervisado import Supervisado
import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Regresion(Supervisado):

    def __init__(self, df):
        super().__init__(df)

    # ── Métricas ─────────────────────────────────────────────────────────────
    @staticmethod
    def _errores(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        denom = np.sum(np.abs(y_true))
        er = np.sum(np.abs(y_true - y_pred)) / denom if denom else float('nan')
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae, "ER": er, "R2": r2}

    # ── Modelos individuales (X_train, X_test, y_train, y_test) ──────────────
    def RegLinSimp(self, X_train, X_test, y_train, y_test):
        X_train_2d = np.asarray(X_train).reshape(-1, 1)
        X_test_2d = np.asarray(X_test).reshape(-1, 1)
        modelo = LinearRegression().fit(X_train_2d, y_train)
        preds = modelo.predict(X_test_2d)
        return modelo, preds, self._errores(y_test, preds)

    def RegLinMult(self, X_train, X_test, y_train, y_test):
        modelo = LinearRegression().fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def RegLasso(self, X_train, X_test, y_train, y_test, alpha=0.1):
        modelo = Lasso(alpha=alpha, max_iter=5000).fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def RegLassoCV(self, X_train, X_test, y_train, y_test):
        cv = LassoCV(alphas=np.logspace(-6, 6, 200), cv=10, max_iter=5000).fit(X_train, y_train)
        modelo = Lasso(alpha=cv.alpha_, max_iter=5000).fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def RegRidge(self, X_train, X_test, y_train, y_test, alpha=1.0):
        modelo = Ridge(alpha=alpha).fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def RegRidgeCV(self, X_train, X_test, y_train, y_test):
        cv = RidgeCV(alphas=np.logspace(-10, 2, 200), fit_intercept=True).fit(X_train, y_train)
        modelo = Ridge(alpha=cv.alpha_).fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def SVM(self, X_train, X_test, y_train, y_test, kernel="rbf", C=100, epsilon=0.1, degree=3):
        modelo = make_pipeline(StandardScaler(),
                               SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree))
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def DecisionTreeReg(self, X_train, X_test, y_train, y_test, max_depth=3, random_state=42):
        modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def RandomForestReg(self, X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42):
        modelo = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state)
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    def XGBoostingReg(self, X_train, X_test, y_train, y_test, n_estimators=500,
                      max_depth=4, min_samples_split=5, random_state=42):
        modelo = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           random_state=random_state)
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        return modelo, preds, self._errores(y_test, preds)

    # ── API para Streamlit ───────────────────────────────────────────────────
    def entrenar(self, algoritmo, target, params=None, test_size=0.25, random_state=42):
        params = params or {}
        X_train, X_test, y_train, y_test, _ = self.preparar_datos(
            target=target, test_size=test_size, random_state=random_state
        )

        a = algoritmo.lower()
        if a in ("lineal multiple", "regresión lineal múltiple", "regresion lineal multiple", "linear"):
            mdl, preds, err = self.RegLinMult(X_train, X_test, y_train, y_test)
        elif a in ("lasso",):
            mdl, preds, err = self.RegLassoCV(X_train, X_test, y_train, y_test)
        elif a in ("ridge",):
            mdl, preds, err = self.RegRidgeCV(X_train, X_test, y_train, y_test)
        elif a in ("svm",):
            mdl, preds, err = self.SVM(X_train, X_test, y_train, y_test,
                                       kernel=params.get("kernel", "rbf"))
        elif a in ("árbol de decisión", "arbol de decision", "dt", "decision tree"):
            mdl, preds, err = self.DecisionTreeReg(X_train, X_test, y_train, y_test,
                                                   max_depth=params.get("max_depth", 4),
                                                   random_state=random_state)
        elif a in ("random forest", "rf"):
            mdl, preds, err = self.RandomForestReg(X_train, X_test, y_train, y_test,
                                                   n_estimators=params.get("n_estimators", 100),
                                                   max_depth=params.get("max_depth", 5),
                                                   random_state=random_state)
        elif a in ("gradient boosting", "xg", "xgboost"):
            mdl, preds, err = self.XGBoostingReg(X_train, X_test, y_train, y_test,
                                                 n_estimators=params.get("n_estimators", 100),
                                                 max_depth=params.get("max_depth", 4),
                                                 random_state=random_state)
        else:
            raise ValueError(f"Algoritmo desconocido: {algoritmo}")

        return {
            "modelo_nombre": algoritmo,
            "modelo": mdl,
            "y_test": np.asarray(y_test),
            "y_pred": np.asarray(preds),
            "metricas": err,
            "r2": err["R2"],
            "rmse": err["RMSE"],
            "mae": err["MAE"],
        }

    def comparar_todos(self, target, params=None, test_size=0.25, random_state=42):
        import traceback
        params = params or {}
        algoritmos = ["Regresión Lineal Múltiple", "Lasso", "Ridge", "SVM",
                      "Árbol de Decisión", "Random Forest", "Gradient Boosting"]
        resultados = []
        errores = []
        for alg in algoritmos:
            try:
                res = self.entrenar(alg, target=target, params=params,
                                    test_size=test_size, random_state=random_state)
                resultados.append(res)
            except Exception as e:
                err_info = {
                    "algoritmo": alg,
                    "tipo": type(e).__name__,
                    "mensaje": str(e),
                    "traceback": traceback.format_exc(),
                }
                errores.append(err_info)
                print(f"[ERROR] {alg} ({type(e).__name__}): {e}")

        tabla = pd.DataFrame([{
            "Modelo": r["modelo_nombre"],
            "R²": r["r2"],
            "RMSE": r["rmse"],
            "MAE": r["mae"],
        } for r in resultados]).sort_values("R²", ascending=False).reset_index(drop=True)
        return resultados, tabla, errores
