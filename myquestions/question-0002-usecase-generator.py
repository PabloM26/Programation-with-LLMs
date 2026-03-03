import pandas as pd
import numpy as np

def generar_caso_de_uso_imputar_temperatura():
    rng = np.random.default_rng()
    n = int(rng.integers(8, 20))
    fechas = pd.date_range("2024-01-01", periods=n, freq="h")
    temps  = rng.uniform(18, 35, n).round(1)
    null_idx = rng.choice(range(1, n-1), size=rng.integers(2, 4), replace=False)
    temps[null_idx] = np.nan
    df = pd.DataFrame({"timestamp": fechas, "temperatura": temps})
    input_dict = {"df": df.copy(), "col": "temperatura"}
    y = temps.copy()
    for i in range(len(y)):
        if np.isnan(y[i]):
            ant = next(y[j] for j in range(i-1, -1, -1) if not np.isnan(y[j]))
            sig = next(y[j] for j in range(i+1, len(y))  if not np.isnan(y[j]))
            y[i] = round((ant + sig) / 2, 1)
    df_out = df.copy()
    df_out["temperatura"] = y
    return input_dict, df_out

inp3, out3 = generar_caso_de_uso_imputar_temperatura()
print("\n\n=== DataFrame de sensores (con NaN) ===")
print(inp3["df"].to_string(index=False))
print("\n=== Output esperado ===")
print(out3.to_string(index=False))
