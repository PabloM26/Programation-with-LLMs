import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ─── Función a implementar ───────────────────────────────────────────
def detectar_data_drift(X_train, X_new):
    # 1. Escalar ambos conjuntos (fit solo en train, transform en ambos)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_new_scaled   = scaler.transform(X_new)

    # 2. Reducir dimensionalidad a 2 componentes principales
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_new_pca   = pca.transform(X_new_scaled)

    # 3. Calcular centroide de cada conjunto
    centroide_train = np.mean(X_train_pca, axis=0)
    centroide_new   = np.mean(X_new_pca,   axis=0)

    # 4. Calcular distancia entre centroides
    distancia = np.linalg.norm(centroide_train - centroide_new)

    # 5. Detectar drift (umbral = 0.5)
    drift_detectado = bool(distancia > 0.5)

    return {
        "distancia_centroides": float(distancia),
        "drift_detectado":      drift_detectado
    }
