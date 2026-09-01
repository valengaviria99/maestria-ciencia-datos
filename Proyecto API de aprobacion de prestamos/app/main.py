import joblib
import pandas as pd

from fastapi import FastAPI

from app.schemas import SolicitudCredito
from src.preprocessing import CreditPreprocessor


app = FastAPI(
    title="API de Aprobación de Créditos"
)


artefacto = joblib.load(
    "models/modelo_credito.joblib"
)

modelo = artefacto["modelo"]
umbral = artefacto["umbral"]


@app.get("/")
def inicio():
    return {
        "mensaje": "API de aprobación de créditos funcionando"
    }


@app.post("/predict")
def predict(solicitud: SolicitudCredito):

    datos = pd.DataFrame([
        solicitud.model_dump()
    ])

    probabilidad = modelo.predict_proba(
        datos
    )[0, 1]

    decision = (
        "Aprobado"
        if probabilidad >= umbral
        else "No aprobado"
    )

    return {
        "decision": decision,
        "probabilidad_aprobacion": round(
            float(probabilidad),
            4
        )
    }