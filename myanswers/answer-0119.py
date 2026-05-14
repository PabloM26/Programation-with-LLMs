import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def construir_ensamble_stacking(X, y):
    # 1. Definir los modelos base (Nivel 0)
    base_estimators = [
        ('rf',  RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # 2. Definir el meta-modelo (Nivel 1)
    meta_modelo = LogisticRegression()

    # 3. Construir el ensamble
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_modelo,
        cv=5
    )

    # 4. Entrenar el ensamble completo
    stacking_clf.fit(X, y)

    return stacking_clf
