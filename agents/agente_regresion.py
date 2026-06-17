from core.regresion import Regresion


class AgenteRegresion:

    def __init__(self, df):
        self._backend = Regresion(df)

    def entrenar(self, algoritmo, target, params=None, test_size=0.25, random_state=42):
        return self._backend.entrenar(
            algoritmo=algoritmo,
            target=target,
            params=params,
            test_size=test_size,
            random_state=random_state,
        )

    def comparar_todos(self, target, params=None, test_size=0.25, random_state=42):
        return self._backend.comparar_todos(
            target=target,
            params=params,
            test_size=test_size,
            random_state=random_state,
        )

    def decidir(self, resultado, umbral_r2=0.7, umbral_rmse=None):
        r2   = resultado["r2"]
        rmse = resultado["rmse"]

        r2_ok   = r2 >= umbral_r2
        rmse_ok = (umbral_rmse is None) or (rmse <= umbral_rmse)

        if r2_ok and rmse_ok:
            return f"Modelo aceptado — R² {r2:.4f} | RMSE {rmse:.4f}"
        razones = []
        if not r2_ok:
            razones.append(f"R² {r2:.4f} < {umbral_r2}"
)
        if not rmse_ok:
            razones.append(f"RMSE {rmse:.4f} > {umbral_rmse}"
)
        return f"Modelo requiere mejora — {' | '.join(razones)}"
