# Lead Scoring Deployment Package

This folder is an isolated deployment package and does not modify existing project code.

## What this package provides

- A custom MLflow `pyfunc` model that includes:
  - feature lists
  - `processor.pkl`
  - trained model file
  - inference preprocessing logic
- A FastAPI app that serves predictions from a registered MLflow model.
- A standard Python package (`pyproject.toml`) that can be built as a wheel.

## Folder layout

- `src/lead_scoring_service/pyfunc_model.py`: Custom MLflow model wrapper.
- `src/lead_scoring_service/api.py`: FastAPI app.
- `scripts/register_mlflow_model.py`: Logs and registers packaged model in MLflow.

## Register model in MLflow

From project root:

```powershell
d:/Project/myenv/Scripts/python.exe d:/Project/deployment_package/scripts/register_mlflow_model.py
```

Optional environment variables:

- `MODEL_FILE_PATH` (default: `d:/Project/training_pipeline/models/lightgbm_best_model.pkl`)
- `PROCESSOR_FILE_PATH` (default: `d:/Project/utils/processor.pkl`)
- `TRACKING_PATH` (default: `d:/Project/experiments/mlruns`)
- `REGISTERED_MODEL_NAME` (default: `LeadScoringService`)

## Run FastAPI

```powershell
d:/Project/myenv/Scripts/python.exe -m uvicorn lead_scoring_service.api:app --reload --app-dir d:/Project/deployment_package/src
```

Optional environment variables for API:

- `MLFLOW_TRACKING_URI` (default points to local `experiments/mlruns`)
- `MODEL_URI` (default: `models:/LeadScoringService/latest`)

## Build package wheel

```powershell
cd d:/Project/deployment_package
d:/Project/myenv/Scripts/python.exe -m pip install build
d:/Project/myenv/Scripts/python.exe -m build
```

The wheel will be generated under `deployment_package/dist`.
