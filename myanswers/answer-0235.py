import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def curva_k_distances(X, k=5):
    # 1. Estandarizar X
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 2. Ajustar NearestNeighbors con k vecinos
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(Xs)

    # 3. Obtener distancias → shape (n_samples, k)
    distances, _ = nn.kneighbors(Xs)

    # 4. Distancia al k-ésimo vecino de cada punto
    kdist = distances[:, k - 1]

    # 5. Ordenar ascendente y redondear a 6 decimales
    kdist_sorted = np.round(np.sort(kdist.astype(float)), 6)

    # 6. Retornar diccionario
    return {
        "k": int(k),
        "k_distances_sorted": kdist_sorted  # shape (n_samples,)
    }
