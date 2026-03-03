import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

def generar_caso_entrenar_elasticnet():
    rng = np.random.default_rng()
    n = int(rng.integers(50, 100))
    n_features = int(rng.integers(8, 15))
    X = rng.uniform(0, 10, (n, n_features)).round(3)
    n_constantes = int(rng.integers(2, 4))
    for i in range(n_constantes):
        X[:, rng.integers(0, n_features)] = rng.uniform(0, 0.001, n)
    y = X @ rng.uniform(1, 3, n_features) + rng.normal(0, 0.5, n)
    umbral = round(float(rng.uniform(0.01, 0.1)), 3)
    input_dict = {"X": X.copy(), "y": y.copy(), "umbral_varianza": umbral}
    selector = VarianceThreshold(threshold=umbral)
    X_sel = selector.fit_transform(X)
    modelo = ElasticNet(random_state=42)
    modelo.fit(X_sel, y)
    y_pred = modelo.predict(X_sel)
    output = {
        "n_features_originales":    n_features,
        "n_features_seleccionadas": X_sel.shape[1],
        "mse": round(mean_squared_error(y, y_pred), 4),
        "r2":  round(r2_score(y, y_pred), 4)
    }
    return input_dict, output

inp1, out1 = generar_caso_entrenar_elasticnet()
print("=== Input ===")
print(f"X shape: {inp1['X'].shape}")
print(f"umbral_varianza: {inp1['umbral_varianza']}")
print("\n=== Output esperado ===")
print(out1)
