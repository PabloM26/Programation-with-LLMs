import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification

def generar_caso_de_uso_optimizar_bosque_aleatorio(**kwargs):

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    

    rf = RandomForestClassifier(random_state=42)
    param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=3, cv=3, random_state=42)
    search.fit(X, y)
    

    caso_dict = {
        "input": {"X": X, "y": y},
        "output": {"mejores_parametros": search.best_params_}
    }
    

    return caso_dict, "Caso de uso: Optimización de hiperparámetros con Random Search"

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
