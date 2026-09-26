# Insurance Cost Prediction Web App

A machine learning web app that predicts annual insurance charges based on patient information. Built with [PyCaret](https://pycaret.org/) for model training and Flask for serving predictions.

This repo was cloned from https://github.com/pycaret/deployment-heroku but with updated versions of packages.

## What it does

- Trains a regression model using PyCaret on an insurance dataset with features: `age`, `sex`, `bmi`, `children`, `smoker`, and `region`
- Serves the model via a Flask web app with a form-based UI and a REST API endpoint
- Packages everything in a Docker container for easy deployment

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI with input form |
| GET | `/health` | Health check |
| POST | `/predict` | Form-based prediction (returns HTML) |
| POST | `/predict_api` | JSON-based prediction (returns JSON) |

### Example request to `/predict_api`

```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "sex": "male", "bmi": 26.5, "children": 2, "smoker": "no", "region": "northeast"}'
```

Response:
```json
{"output": 12345}
```

## Build and run with Docker

```bash
docker build -t insurance-app .
docker run -p 5000:5000 -it insurance-app
```

Then open http://localhost:5000 in your browser.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Tech stack

- **PyCaret 3.3.1** — model training and inference
- **Flask 3.0.0** — web framework
- **pandas 1.5.1** — data handling
- **Docker** — containerization