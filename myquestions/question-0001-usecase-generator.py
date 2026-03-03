import numpy as np
import pandas as pd

def generar_caso_filtrar_canciones():
    rng = np.random.default_rng()
    n = int(rng.integers(10, 25))
    canciones = [f"cancion_{i}" for i in range(n)]
    reproducciones = rng.integers(100, 5000, n)
    duracion_seg = rng.integers(120, 360, n)
    df = pd.DataFrame({
        "cancion":        canciones,
        "reproducciones": reproducciones,
        "duracion_seg":   duracion_seg
    })
    min_rep = int(rng.integers(500, 2000))
    dur_min = int(rng.integers(120, 200))
    dur_max = int(rng.integers(250, 340))
    input_dict = {
        "df":                 df.copy(),
        "min_reproducciones": min_rep,
        "duracion_min":       dur_min,
        "duracion_max":       dur_max
    }
    output = df[
        (df["reproducciones"] >= min_rep) &
        (df["duracion_seg"]   >= dur_min) &
        (df["duracion_seg"]   <= dur_max)
    ]
    return input_dict, output

inp, out = generar_caso_filtrar_canciones()
print("=== Base de datos original ===")
print(inp["df"].to_string(index=False))
print(f"\nmin_reproducciones={inp['min_reproducciones']}, "
      f"duracion_min={inp['duracion_min']}, duracion_max={inp['duracion_max']}")
print("\n=== Output esperado ===")
print(out.to_string(index=False))
