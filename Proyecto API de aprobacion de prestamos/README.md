# API de Inferencia para Aprobación de Préstamos

Proyecto de clasificación para estimar la aprobación de solicitudes de crédito y exponer el modelo mediante una API construida con **FastAPI** y preparada para su ejecución en **Docker**.

El proyecto incluye análisis exploratorio, comparación de modelos de clasificación, selección de un modelo final, pipeline de preprocesamiento, serialización con `joblib` y un endpoint `POST /predict` para realizar inferencias.

## Objetivo

Construir una solución reproducible que permita:

- Analizar las variables asociadas con la aprobación de créditos.
- Entrenar y comparar diferentes modelos de clasificación.
- Seleccionar un modelo final y un umbral de decisión.
- Automatizar el preprocesamiento necesario para la inferencia.
- Exponer el modelo mediante una API REST.
- Preparar la aplicación para ejecutarse dentro de un contenedor Docker.

## Estructura del proyecto

```text
Taller_2_PD/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── models/
│   └── modelo_credito.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modelos.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── train.py
│
├── datos_credito.parquet
├── Dockerfile
├── requirements.txt
└── README.md
```

## Datos

La base contiene **15.000 solicitudes de crédito** y una variable objetivo binaria:

- `aprobado = 1`: crédito aprobado.
- `aprobado = 0`: crédito no aprobado.

Entre las variables analizadas se encuentran:

- `edad`
- `personas_a_cargo`
- `ingreso_mensual`
- `monto_solicitado`
- `plazo_meses`
- `cuota_estimada`
- `score_crediticio`
- `dti_previo`
- `dti_total_proyectado`

Durante el modelado también se creó la variable:

```text
cuota_ingreso = cuota_estimada / ingreso_mensual
```

## Análisis exploratorio

El análisis exploratorio se encuentra en:

```text
notebooks/01_eda.ipynb
```

Entre los principales hallazgos:

- No se identificaron valores nulos ni registros duplicados.
- La variable objetivo presenta una distribución aproximada de 66 % aprobados y 34 % no aprobados.
- El `score_crediticio` presenta una relación positiva con la aprobación.
- El `dti_total_proyectado` y el `dti_previo` presentan una relación negativa con la aprobación.
- El monto solicitado por sí solo tiene una relación débil con la decisión.
- Algunas variables presentan redundancia, especialmente las relacionadas con ingreso, cuota y DTI.

## Modelado

El proceso de modelado se encuentra en:

```text
notebooks/02_modelos.ipynb
```

Se evaluaron los siguientes modelos:

1. Regresión Logística con todas las variables.
2. Regresión Logística con un conjunto reducido de variables.
3. Random Forest.
4. Gradient Boosting.

Los modelos fueron comparados mediante métricas como:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Matriz de confusión

También se evaluaron diferentes umbrales de clasificación para analizar el balance entre la identificación de solicitudes aprobadas y no aprobadas.

## Modelo seleccionado

El modelo seleccionado para el despliegue fue **Gradient Boosting**.

Configuración utilizada:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

Variables utilizadas por el modelo final:

```text
personas_a_cargo
ingreso_mensual
score_crediticio
dti_previo
cuota_ingreso
```

Se utiliza un umbral de aprobación de:

```text
0.70
```

Por tanto:

```text
probabilidad >= 0.70  -> Aprobado
probabilidad < 0.70   -> No aprobado
```

En el ejercicio de modelado, Gradient Boosting mostró un mejor equilibrio entre las clases y una mayor capacidad para identificar solicitudes no aprobadas frente a la regresión logística con el umbral seleccionado.

> Nota metodológica: el conjunto de prueba también fue utilizado para explorar el umbral de clasificación, por lo que los resultados deben interpretarse como parte de un ejercicio comparativo de modelado y selección del punto de corte.

## Pipeline de preprocesamiento

El archivo:

```text
src/preprocessing.py
```

contiene la clase `CreditPreprocessor`.

Su función es calcular automáticamente:

```python
cuota_ingreso = cuota_estimada / ingreso_mensual
```

y seleccionar las cinco variables requeridas por el modelo.

Esto permite que la API reciba variables originales y realice internamente el preprocesamiento antes de generar una predicción.

## Entrenamiento del modelo final

El archivo:

```text
src/train.py
```

se encarga de:

- cargar los datos;
- construir el pipeline;
- entrenar Gradient Boosting;
- definir el umbral de aprobación;
- guardar el modelo y el umbral en un archivo `.joblib`.

Para entrenar nuevamente el modelo desde la raíz del proyecto:

```bash
python -m src.train
```

El artefacto generado se guarda en:

```text
models/modelo_credito.joblib
```

## API con FastAPI

La API se encuentra en:

```text
app/main.py
```

El esquema de entrada está definido en:

```text
app/schemas.py
```

### Ejecutar la API localmente

Desde la raíz del proyecto:

```bash
python -m uvicorn app.main:app --reload
```

Luego abrir:

```text
http://127.0.0.1:8000/docs
```

FastAPI genera automáticamente una interfaz Swagger para probar los endpoints.

## Endpoints

### GET /

Permite verificar que la API se encuentra funcionando.

Respuesta esperada:

```json
{
  "mensaje": "API de aprobación de créditos funcionando"
}
```

### POST /predict

Recibe los datos de una solicitud de crédito y devuelve la decisión y la probabilidad estimada de aprobación.

Ejemplo de entrada:

```json
{
  "personas_a_cargo": 1,
  "ingreso_mensual": 3500000,
  "cuota_estimada": 650000,
  "score_crediticio": 720,
  "dti_previo": 0.25
}
```

Ejemplo de respuesta:

```json
{
  "decision": "Aprobado",
  "probabilidad_aprobacion": 0.82
}
```

La probabilidad mostrada en el ejemplo es ilustrativa y puede variar según los datos enviados.

## Dependencias

Las principales dependencias del proyecto son:

```text
fastapi==0.141.1
uvicorn==0.52.4
pandas==2.3.1
scikit-learn==1.9.0
joblib==1.6.0
```

Para instalarlas:

```bash
python -m pip install -r requirements.txt
```

## Docker

El proyecto incluye un `Dockerfile` para ejecutar la API en un entorno aislado y reproducible.

Construir la imagen:

```bash
docker build -t prestamos-api .
```

Ejecutar el contenedor:

```bash
docker run -p 8000:8000 prestamos-api
```

Después se puede acceder nuevamente a:

```text
http://127.0.0.1:8000/docs
```

El dataset no necesita ser copiado al contenedor de inferencia, ya que la API utiliza directamente el modelo previamente entrenado almacenado en `models/modelo_credito.joblib`.

## Flujo general

```text
Datos
  |
  v
EDA
  |
  v
Comparación de modelos
  |
  v
Gradient Boosting
  |
  v
Pipeline de preprocesamiento
  |
  v
modelo_credito.joblib
  |
  v
FastAPI
  |
  v
POST /predict
  |
  v
Docker
```

## Tecnologías utilizadas

- Python
- pandas
- scikit-learn
- FastAPI
- Pydantic
- joblib
- Uvicorn
- Docker
- Jupyter Notebook

## Consideraciones

Este proyecto corresponde a un ejercicio académico de clasificación e inferencia. El modelo no debe interpretarse como un sistema real de decisión crediticia ni utilizarse para tomar decisiones financieras sobre personas sin una validación técnica, regulatoria y de negocio adicional.
