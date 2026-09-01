import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.preprocessing import CreditPreprocessor


# 1. Cargar datos
df = pd.read_parquet(
    "datos_credito.parquet"
)


# 2. Variables de entrada
variables_entrada = [
    "personas_a_cargo",
    "ingreso_mensual",
    "cuota_estimada",
    "score_crediticio",
    "dti_previo"
]

X = df[variables_entrada]
y = df["aprobado"]


# 3. Construir pipeline final
pipeline_final = Pipeline([
    (
        "preprocessing",
        CreditPreprocessor()
    ),
    (
        "classifier",
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    )
])


# 4. Entrenar modelo
pipeline_final.fit(
    X,
    y
)


# 5. Umbral seleccionado en el notebook
UMBRAL_APROBACION = 0.70


# 6. Crear carpeta models si no existe
os.makedirs(
    "models",
    exist_ok=True
)


# 7. Guardar modelo y umbral
artefacto = {
    "modelo": pipeline_final,
    "umbral": UMBRAL_APROBACION
}

joblib.dump(
    artefacto,
    "models/modelo_credito.joblib"
)


print(
    "Modelo entrenado y guardado correctamente."
)