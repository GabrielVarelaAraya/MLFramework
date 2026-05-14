from supervisado import Supervisado
import numpy as np
import pandas as pd
import math
import statistics

from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class Regresion(Supervisado):

    def __init__(self, df):
        super().__init__(df)

    def __Errores(self, y_true, y_predicted):
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        MSE0 = np.sum(np.square(y_true - y_predicted))
        RMSE = math.sqrt(MSE0 / (len(y_true)))
        MAE0 = np.sum(np.abs(y_true - y_predicted))
        MAE = MAE0 / (len(y_true))
        EN = np.sum(np.abs(y_true - y_predicted))
        ED = np.sum(np.abs(y_true))
        ER = EN / ED
        error = pd.DataFrame()
        nombres = ['RMSE', 'MAE', 'ER']
        errores = [RMSE, MAE, ER]
        error['Tipo'] = nombres
        error['Error'] = errores
        return error
        
    def __ResVarPred(self, VarPred):
        Cuartil = statistics.quantiles(VarPred)
        val_max = np.max(VarPred)
        val_min = np.min(VarPred)
        return {"Máximo": val_max,
                        "Cuartiles": Cuartil,
                        "Mínimo": val_min}
        
    
    def RegLinSimp(self, X_train, X_test, y_train, y_test):
        modelo_RegSimp= LinearRegression().fit(X_train.reshape(-1, 1), y_train)
        print("Coeficiente w1:", modelo_RegSimp.coef_)
        print("Coeficiente w0:", modelo_RegSimp.intercept_)
        predicciones = modelo_RegSimp.predict(X=X_test)
        print(predicciones[0:3,])
        err_RegSimp = self.__Errores(y_test,predicciones)
        return err_RegSimp

    def RegLinMult(self, X_train, X_test, y_train, y_test):
        modelo_RegMul = LinearRegression()
        modelo_RegMul.fit(X=X_train, y=y_train)
        predicciones = modelo_RegMul.predict(X=X_test)
        print(predicciones[0:3,])
        err_RegMul = self.__Errores(y_test,predicciones)
        return err_RegMul

    def RegLasso(self, X_train, X_test, y_train, y_test):
        modelo_lasso = Lasso(alpha=0.1)
        modelo_lasso.fit(X=X_train, y=y_train)
        predicciones = modelo_lasso.predict(X=X_test)
        err_lasso = self.__Errores(y_test,predicciones)
        df_coeficientes = pd.DataFrame({'predictor': X_train.columns,'coef': modelo_lasso.coef_.flatten()})
        print(df_coeficientes)
        return err_lasso
    
    def RegLassoCV(self, X_train, X_test, y_train, y_test):
        modelo = LassoCV(alphas = np.logspace(-6, 6, 200), cv = 10)
        modelo.fit(X=X_train, y=y_train)
        modelo_lasso2 = Lasso(alpha=modelo.alpha_)
        modelo_lasso2.fit(X=X_train, y=y_train)
        predicciones = modelo_lasso2.predict(X_test)
        err_lasso2 = self.__Errores(y_test,predicciones)
        df_coeficientes = pd.DataFrame({'predictor': X_train.columns,'coef': modelo_lasso2.coef_.flatten()})
        print(df_coeficientes)
        return err_lasso2
    
    def ComparacionLasso(self, X_train, X_test, y_train, y_test):
        err_lasso = self.RegLasso(X_train, X_test, y_train, y_test)
        err_lasso2 = self.RegLassoCV(X_train, X_test, y_train, y_test)
        comparacion = pd.DataFrame([err_lasso.Error, err_lasso2.Error])
        comparacion.columns = ['RMSE','MAE','ER']
        comparacion.index = ['Lasso', 'Lasso Óptimo']
        return comparacion
    
    def RegRidge(self, X_train, X_test, y_train, y_test):
        modelo_Ridge = Ridge(alpha = 1.0)
        modelo_Ridge.fit(X=X_train,y=y_train)
        predicciones = modelo_Ridge.predict(X_test)
        err_Ridge = self.__Errores(y_test,predicciones)
        df_coeficientes = pd.DataFrame({'predictor': X_train.columns,'coef': modelo_Ridge.coef_.flatten()})
        print(df_coeficientes)
        return err_Ridge
    
    def RegRidgeCV(self, X_train, X_test, y_train, y_test):
        modelo = RidgeCV(alphas = np.logspace(-10, 2, 200), fit_intercept   = True, store_cv_values = True)
        modelo.fit(X=X_train, y=y_train)
        modelo_Ridge2 = Ridge(alpha = modelo.alpha_)
        modelo_Ridge2.fit(X=X_train, y=y_train)
        predicciones = modelo_Ridge2.predict(X=X_test)
        err_Ridge2 = self.__Errores(y_test,predicciones)
        df_coeficientes = pd.DataFrame({'predictor': X_train.columns,'coef': modelo_Ridge2.coef_.flatten()})
        print(df_coeficientes)
        return err_Ridge2
    
    def ComparacionRidge(self, X_train, X_test, y_train, y_test):
        err_Ridge = self.RegRidge(X_train, X_test, y_train, y_test)
        err_Ridge2 = self.RegRidgeCV(X_train, X_test, y_train, y_test)
        comparacion = pd.DataFrame([err_Ridge.Error, err_Ridge2.Error])
        comparacion.columns = ['RMSE','MAE','ER']
        comparacion.index = ['Ridge', 'Ridge Óptimo']
        return comparacion
    
    def SVM(self, X_train, X_test, y_train, y_test):
        modelo1 = make_pipeline(StandardScaler(),SVR(kernel="rbf", C=100, epsilon=0.1))
        modelo2 = make_pipeline(StandardScaler(),SVR(kernel="linear", C=100,  epsilon=0.1))
        modelo3 = make_pipeline(StandardScaler(),SVR(kernel="poly", C=100,  degree=3, epsilon=0.1))
        modelo1.fit(X=X_train, y=y_train)
        modelo2.fit(X=X_train, y=y_train)
        modelo3.fit(X=X_train, y=y_train)
        predicciones1 = modelo1.predict(X=X_test)
        predicciones2 = modelo2.predict(X=X_test)
        predicciones3 = modelo3.predict(X=X_test)
        err1 = self.__Errores(y_test,predicciones1)
        err2 = self.__Errores(y_test,predicciones2)
        err3 = self.__Errores(y_test,predicciones3)
        return err1, err2, err3
    
    def ComparacionSVM(self, X_train, X_test, y_train, y_test):
        err1, err2, err3 = self.SVM(X_train, X_test, y_train, y_test)
        comparacion = pd.DataFrame([err1.Error, err2.Error, err3.Error])
        comparacion.columns = ['RMSE','MAE','ER']
        comparacion.index  = ['SVM rbf','SVM linear','SVM poly']
        return comparacion
    
    def DecisionTreeReg(self, X_train, X_test, y_train, y_test):
        modelo_arbol = DecisionTreeRegressor(max_depth=3,random_state= 123)
        modelo_arbol.fit(X=X_train, y=y_train)
        predicciones = modelo_arbol.predict(X=X_test)
        err_arbol = self.__Errores(y_test,predicciones)
        return err_arbol
    
    def RandomForestReg(self, X_train, X_test, y_train, y_test):
        modelo_Bosque = RandomForestRegressor(max_depth=2, random_state=0)
        modelo_Bosque.fit(X=X_train, y=y_train)
        predicciones = modelo_Bosque.predict(X=X_test)
        err_Bosque = self.__Errores(y_test,predicciones)
        return err_Bosque
    
    def XGBoostingReg(self, X_train, X_test, y_train, y_test):
        modelo_Potenciacion = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 4, min_samples_split = 5)
        modelo_Potenciacion.fit(X=X_train, y=y_train)
        predicciones = modelo_Potenciacion.predict(X=X_test)
        err_Potenciacion = self.__Errores(y_test,predicciones)
        return err_Potenciacion
    
    def ComparacionALL(self, X_train, X_test, y_train, y_test):
        err_RegMul = self.RegLinMult(X_train, X_test, y_train, y_test)
        err_lasso2 = self.RegLassoCV(X_train, X_test, y_train, y_test)
        err_Ridge2 = self.RegRidgeCV(X_train, X_test, y_train, y_test)
        err1, err2, err3 = self.SVM(X_train, X_test, y_train, y_test)
        err_arbol = self.DecisionTreeReg(X_train, X_test, y_train, y_test)
        err_Bosque = self.RandomForestReg(X_train, X_test, y_train, y_test)
        err_Potenciacion = self.XGBoostingReg(X_train, X_test, y_train, y_test)
        comparacion = pd.DataFrame([err_RegMul.Error,err_lasso2.Error,err_Ridge2.Error, err1.Error, err2.Error, err3.Error, err_arbol.Error, err_Bosque.Error, err_Potenciacion.Error])
        comparacion.columns = ['RMSE','MAE','ER']
        comparacion.index  = ['Múltiple', 'Lasso','Ridge','SVM rbf','SVM linear','SVM poly','Árbol', 'Bosques', 'Potenciación']
        return comparacion
    
    def Max(self, y):
        print(self.__ResVarPred(y))