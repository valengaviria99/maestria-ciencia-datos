from pydantic import BaseModel


class SolicitudCredito(BaseModel):
    personas_a_cargo: int
    ingreso_mensual: float
    cuota_estimada: float
    score_crediticio: int
    dti_previo: float