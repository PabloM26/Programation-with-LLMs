import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score

def generar_caso_predecir_bayesiano():
    rng = np.random.default_rng()
    n = int(rng.integers(40, 80))
    niveles = ["junior", "semi-senior", "senior"]
    categorias = [niveles]
    X_cat = rng.choice(niveles, size=(n, 1))
    X_num = rng.uniform(0, 10, (n, int(rng.integers(3, 6)))).round(2)
    encoder = OrdinalEncoder(categories=categorias)
    X_cat_enc = encoder.fit_transform(X_cat)
    X_all = np.hstack([X_cat_enc, X_num])
    y = X_all @ rng.uniform(0.5, 2, X_all.shape[1]) + rng.normal(0, 0.3, n)
    input_dict = {
        "X_cat": X_cat.copy(),
        "X_num": X_num.copy(),
        "y": y.copy(),
        "categorias": categorias
    }
    modelo = BayesianRidge()
    modelo.fit(X_all, y)
    y_pred = modelo.predict(X_all)
    output = {
        "r2":         round(r2_score(y, y_pred), 4),
        "coef":       modelo.coef_.round(4),
        "n_features": X_all.shape[1]
    }
    return input_dict, output

inp2, out2 = generar_caso_predecir_bayesiano()
print("\n\n=== Input ===")
print(f"X_cat (primeras 5):\n{inp2['X_cat'][:5]}")
print(f"X_num shape: {inp2['X_num'].shape}")
print(f"categorias: {inp2['categorias']}")
print("\n=== Output esperado ===")
print(f"r2: {out2['r2']}")
print(f"n_features: {out2['n_features']}")
print(f"coef: {out2['coef']}")
