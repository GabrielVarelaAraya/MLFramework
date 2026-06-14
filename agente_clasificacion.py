from clasificacion import Clasificacion


class AgenteClasificacion:

    def __init__(self, df):
        self._backend = Clasificacion(df)

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

    def decidir(self, resultado, umbral=0.75):
        exactitud = resultado["exactitud"]
        if exactitud >= umbral:
            return f"Modelo aceptado — exactitud {exactitud:.2%}"
        return f"Modelo requiere mejora — exactitud {exactitud:.2%} (umbral {umbral:.0%})"
