from supervisado import Supervisado
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

class Clasificacion(Supervisado):

    def __init__(self, df):
        super().__init__(df)

    # KNN  
    def modeloKNN(self, X_train, y_train, n_neighbors, algorithm):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
        model.fit(X_train, y_train)
        return model
    
    def KNN(self, n_neighbors):
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for algorithm in algorithms:
            print(f"\nUsando el algoritmo: {algorithm}")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloKNN(X_train, y_train, n_neighbors, algorithm)
            y_pred = self.predecir(model, X_test)
            self.evaluar(y_test, y_pred, y)

    def prediccionesKNN(self):
        print("=== Predicciones KNN ===")
        for n in range(1, 11):
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                print(f"\nKNN - Vecinos: {n}, Algoritmo: {algorithm}")
                print("                                              ")
                X_train, X_test, y_train, y_test, y = self.preparar_datos()
                model = self.modeloKNN(X_train, y_train, n_neighbors=n, algorithm=algorithm)
                y_pred = self.predecir(model, X_test)
                print("Predicciones en Testing:", y_pred)
                print("---------------------------------------------------")
                print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
                print("______________________________________________")

# Decision Tree
    def modeloDT(self, X_train, y_train, min_samples_split, max_depth):
        model = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def DT(self, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        self.evaluar(y_test, y_pred, y)
  
    def prediccionesDT(self):
        print("=== Predicciones Árbol de Decisión ===")
        for depth in range(1, 11):
            print(f"\nDT - max_depth: {depth}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloDT(X_train, y_train, min_samples_split=2, max_depth=depth)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

# Random Forest
    def modeloRF(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def RF(self, n_estimators, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloRF(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        self.evaluar(y_test, y_pred, y)

    def prediccionesRF(self):
        print("=== Predicciones Random Forest ===")
        for depth in range(1, 11):
            print(f"\nRF - max_depth: {depth}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloRF(X_train, y_train, n_estimators=100, min_samples_split=2, max_depth=depth)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

# XGBoost
    def modeloXG(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        model = GradientBoostingClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def XG(self, n_estimators, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloXG(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        self.evaluar(y_test, y_pred, y)

    def prediccionesXG(self):
        print("=== Predicciones Gradient Boosting ===")
        for depth in range(1, 11):
            print(f"\nXGBoost - max_depth: {depth}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloXG(X_train, y_train, n_estimators=100, min_samples_split=2, max_depth=depth)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

# ADABoost
    def modeloADA(self, X_train, y_train, estimator, n_estimators):
        model = AdaBoostClassifier(estimator=estimator, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model

    def ADA(self, n_estimators):
        estimators = {"Decision Tree": DecisionTreeClassifier(min_samples_split=2, max_depth=4),
                    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=4),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, min_samples_split=2, max_depth=4)
                    }
        for name, estimator in estimators.items():
            print(f"\nUsando el metodo: {name}")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloADA(X_train, y_train, estimator, n_estimators)
            y_pred = self.predecir(model, X_test)
            self.evaluar(y_test, y_pred, y)

    def prediccionesADA(self):
        print("=== Predicciones AdaBoost ===")
        for n in range(10, 101, 10):
            print(f"\nAdaBoost - n_estimators: {n}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            estimator = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
            model = self.modeloADA(X_train, y_train, estimator=estimator, n_estimators=n)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

    def predecir(self, model, X_test):
        return model.predict(X_test)
    
    def evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MC,list(np.unique(y)))
        for k in indices:
            print("\n%s:\n%s"%(k,str(indices[k])))
    
    def indices_general(self, MC, nombres = None):
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global     = 1 - precision_global
        precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
        if nombres!=None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión":MC, 
                "Precisión Global":   precision_global, 
                "Error Global":       error_global, 
                "Precisión por categoría":precision_categoria}
        

    def __KNNBM(self, n_neighbors=3, algorithm="ball_tree"):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloKNN(X_train, y_train, n_neighbors, algorithm)
        y_pred = self.predecir(model, X_test)
        MCknn = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MCknn,list(np.unique(y)))
        return indices
    
    def __DTBM(self, min_samples_split=8, max_depth=1):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        MCdt = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MCdt,list(np.unique(y)))
        return indices
    
    def __RFBM(self, n_estimators=200, min_samples_split=9, max_depth=8):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloRF(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        MCdt = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MCdt,list(np.unique(y)))
        return indices

    def __XGBM(self, n_estimators=100, min_samples_split=5, max_depth=1):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloXG(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        MCdt = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MCdt,list(np.unique(y)))
        return indices
    
    def __ADABM(self, estimator= GradientBoostingClassifier(n_estimators=100, min_samples_split=5, max_depth=1), n_estimators=10):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloADA(X_train, y_train, estimator, n_estimators)
        y_pred = self.predecir(model, X_test)
        MCdt = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MCdt,list(np.unique(y)))
        return indices

    def BM(self):
        datos = {"PG": [0,0,0,0,0],"EG": [0,0,0,0,0],"PP": [0,0,0,0,0],"PN": [0,0,0,0,0]}
        Tdatos = pd.DataFrame(datos,index=["AlgKnn","AlgDT","AlgRF","AlgXGBoost","AlgADABoost"],columns=["PG","EG","PP","PN"])

        indices = self.__KNNBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgKnn", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgKnn", "EG"] = indices['Error Global']
        Tdatos.loc["AlgKnn", "PP"] = PP.loc[0,'Si']
        Tdatos.loc["AlgKnn", "PN"] = PN.loc[0,'No']
        
        indices = self.__DTBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgDT", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgDT", "EG"] = indices['Error Global']
        Tdatos.loc["AlgDT", "PP"] = PP.loc[0,'Si']
        Tdatos.loc["AlgDT", "PN"] = PN.loc[0,'No']

        indices = self.__RFBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgRF", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgRF", "EG"] = indices['Error Global']
        Tdatos.loc["AlgRF", "PP"] = PP.loc[0,'Si']
        Tdatos.loc["AlgRF", "PN"] = PN.loc[0,'No']

        indices = self.__XGBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgXGBoost", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgXGBoost", "EG"] = indices['Error Global']
        Tdatos.loc["AlgXGBoost", "PP"] = PP.loc[0,'Si']
        Tdatos.loc["AlgXGBoost", "PN"] = PN.loc[0,'No']

        indices = self.__ADABM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgADABoost", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgADABoost", "EG"] = indices['Error Global']
        Tdatos.loc["AlgADABoost", "PP"] = PP.loc[0,'Si']
        Tdatos.loc["AlgADABoost", "PN"] = PN.loc[0,'No']

        print(Tdatos)
