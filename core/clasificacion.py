from core.supervisado import Supervisado
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             classification_report, roc_auc_score)
import xgboost as xgb


class Clasificacion(Supervisado):

    def __init__(self, df):
        super().__init__(df)

    # ── Builders de modelos ──────────────────────────────────────────────────
    def modeloKNN(self, X_train, y_train, n_neighbors, algorithm='auto'):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
        model.fit(X_train, y_train)
        return model

    def modeloDT(self, X_train, y_train, min_samples_split=2, max_depth=None, random_state=42):
        model = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                       max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def modeloRF(self, X_train, y_train, n_estimators=100, min_samples_split=2,
                 max_depth=None, random_state=42):
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       min_samples_split=min_samples_split,
                                       max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def modeloXG(self, X_train, y_train, n_estimators=100, min_samples_split=2,
                 max_depth=3, random_state=42):
        model = GradientBoostingClassifier(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split,
                                           max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def modeloADA(self, X_train, y_train, estimator=None, n_estimators=50, random_state=42):
        model = AdaBoostClassifier(estimator=estimator, n_estimators=n_estimators,
                                   random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def modeloSVM(self, X_train, y_train, C=1.0, kernel='rbf', gamma='scale', random_state=42):
        model = CalibratedClassifierCV(
            SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state),
            ensemble=False
        )
        model.fit(X_train, y_train)
        return model

    def modeloLR(self, X_train, y_train, C=1.0, solver='lbfgs', max_iter=1000, random_state=42):
        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def modeloNB(self, X_train, y_train):
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model

    def modeloXGBoost(self, X_train, y_train, n_estimators=100, max_depth=3,
                      learning_rate=0.1, random_state=42):
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=learning_rate, random_state=random_state,
                                  eval_metric='logloss')
        model.fit(X_train, y_train)
        return model

    # ── Métodos legacy (con print) ───────────────────────────────────────────
    def predecir(self, model, X_test):
        return model.predict(X_test)

    def evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MC, list(np.unique(y)))
        for k in indices:
            print(f"\n{k}:\n{indices[k]}")

    def indices_general(self, MC, nombres=None):
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria = pd.DataFrame(MC.diagonal() / np.sum(MC, axis=1)).T
        if nombres is not None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión": MC,
                "Precisión Global": precision_global,
                "Error Global": error_global,
                "Precisión por categoría": precision_categoria}

    # ── API para Streamlit: retornan dicts ───────────────────────────────────
    def entrenar(self, algoritmo, target, params=None, test_size=0.25, random_state=42):
        """Entrena un modelo y retorna dict con métricas + matriz + predicciones."""
        params = params or {}
        X_train, X_test, y_train, y_test, y = self.preparar_datos(
            target=target, test_size=test_size, random_state=random_state
        )

        model = self._build_model(algoritmo, X_train, y_train, params, random_state)
        y_pred = model.predict(X_test)

        return self._construir_resultado(algoritmo, model, X_test, y_test, y_pred, y)

    def _build_model(self, algoritmo, X_train, y_train, params, random_state):
        a = algoritmo.lower()
        if a in ("knn",):
            return self.modeloKNN(X_train, y_train,
                                  n_neighbors=params.get("n_neighbors", 5),
                                  algorithm=params.get("algorithm", "auto"))
        if a in ("dt", "decision tree", "árbol de decisión", "arbol de decision"):
            return self.modeloDT(X_train, y_train,
                                 min_samples_split=params.get("min_samples_split", 2),
                                 max_depth=params.get("max_depth", None),
                                 random_state=random_state)
        if a in ("rf", "random forest"):
            return self.modeloRF(X_train, y_train,
                                 n_estimators=params.get("n_estimators", 100),
                                 min_samples_split=params.get("min_samples_split", 2),
                                 max_depth=params.get("max_depth", None),
                                 random_state=random_state)
        if a in ("xg", "gradient boosting", "gradientboosting"):
            return self.modeloXG(X_train, y_train,
                                 n_estimators=params.get("n_estimators", 100),
                                 min_samples_split=params.get("min_samples_split", 2),
                                 max_depth=params.get("max_depth", 3),
                                 random_state=random_state)
        if a in ("ada", "adaboost"):
            return self.modeloADA(X_train, y_train,
                                  estimator=params.get("estimator"),
                                  n_estimators=params.get("n_estimators", 50),
                                  random_state=random_state)
        if a in ("svm", "svc", "support vector machine"):
            return self.modeloSVM(X_train, y_train,
                                  C=params.get("C", 1.0),
                                  kernel=params.get("kernel", "rbf"),
                                  gamma=params.get("gamma", "scale"),
                                  random_state=random_state)
        if a in ("lr", "logistic regression", "regresión logística", "regresion logistica"):
            return self.modeloLR(X_train, y_train,
                                 C=params.get("C", 1.0),
                                 solver=params.get("solver", "lbfgs"),
                                 max_iter=params.get("max_iter", 1000),
                                 random_state=random_state)
        if a in ("nb", "naive bayes", "naive_bayes"):
            return self.modeloNB(X_train, y_train)
        if a in ("xgb", "xgboost"):
            return self.modeloXGBoost(X_train, y_train,
                                      n_estimators=params.get("n_estimators", 100),
                                      max_depth=params.get("max_depth", 3),
                                      learning_rate=params.get("learning_rate", 0.1),
                                      random_state=random_state)
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")

    def _construir_resultado(self, nombre, model, X_test, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        clases = sorted(set(c for c in (set(y_test) | set(y_pred))))
        indices = self.indices_general(MC, [str(c) for c in clases])

        n_clases = len(clases)
        if n_clases == 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        elif n_clases > 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        else:
            roc_auc = None

        return {
            "modelo_nombre": nombre,
            "modelo": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "clases": clases,
            "exactitud": accuracy_score(y_test, y_pred),
            "f1_ponderado": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc,
            "error_global": indices["Error Global"],
            "matriz_confusion": MC,
            "precision_por_categoria": indices["Precisión por categoría"],
            "reporte": classification_report(y_test, y_pred,
                                             target_names=[str(c) for c in clases],
                                             output_dict=True, zero_division=0),
        }

    def comparar_todos(self, target, params=None, test_size=0.25, random_state=42):
        """Entrena todos los algoritmos y retorna lista de resultados + tabla resumen."""
        params = params or {}
        algoritmos = ["KNN", "Decision Tree", "Random Forest", "SVM",
                      "Logistic Regression", "Naive Bayes", "XGBoost",
                      "Gradient Boosting", "AdaBoost"]
        resultados = []
        for alg in algoritmos:
            res = self.entrenar(alg, target=target, params=params,
                                test_size=test_size, random_state=random_state)
            resultados.append(res)

        tabla = pd.DataFrame([{
            "Modelo": r["modelo_nombre"],
            "Exactitud": r["exactitud"],
            "F1-Score": r["f1_ponderado"],
            "ROC-AUC": r["roc_auc"] if r["roc_auc"] is not None else "N/A",
            "Error Global": r["error_global"],
        } for r in resultados]).sort_values("Exactitud", ascending=False).reset_index(drop=True)
        return resultados, tabla

    # ── BM legacy genérico (sin Si/No hard-coded) ────────────────────────────
    def BM(self, target="target"):
        """Benchmark de los 5 algoritmos. Imprime y retorna DataFrame."""
        _, tabla = self.comparar_todos(target=target)
        print(tabla.to_string(index=False))
        return tabla
