import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification

def optimizar_bosque_aleatorio(X, y):
    # 1. Instanciar el modelo base
    rf = RandomForestClassifier(random_state=42)

    # 2. Definir el espacio de búsqueda
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }

    # 3. Configurar la búsqueda aleatoria
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=3,
        cv=3,
        random_state=42
    )

    # 4. Ajustar con los datos
    search.fit(X, y)

    # 5. Retornar diccionario con los mejores parámetros
    return {"mejores_parametros": search.best_params_}
