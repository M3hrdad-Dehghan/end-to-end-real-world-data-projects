# SaaS Lead Scoring

## Business Problem & Objective

SaaS companies invest heavily in marketing and customer success, but not every lead converts to a repeat buyer. Without a reliable way to identify which customers are likely to repurchase, sales and marketing teams waste resources on low-probability leads while high-value ones go under-served.

The objective of this project is to build a binary classification model that predicts whether a SaaS customer will repurchase, enabling the business to prioritize outreach, personalize campaigns, and reduce churn.

---

## Data & Inputs

**Source:** `data/raw/saas_lead.csv` — a dataset of SaaS customer records with 29 features per lead.

**Feature groups:**

| Group | Examples |
|---|---|
| Demographics | `age`, `gender`, `country` |
| Company profile | `company_size`, `industry`, `job_title`, `annual_company_revenue`, `plan_type` |
| Purchase behavior | `months_as_customer`, `total_spend`, `num_purchases`, `last_purchase_days_ago`, `has_churned_before`, `discount_used` |
| Product usage | `monthly_active_days`, `avg_session_duration_min`, `features_used_count`, `api_calls_last_30_days`, `product_tours_completed`, `onboarding_completed` |
| Support & marketing | `support_tickets_total`, `avg_ticket_resolution_hours`, `sat_score`, `campaigns_received`, `last_marketing_touch_days_ago`, `attended_webinar`, `social_media_engaged` |

**Target:** `will_repurchase` (binary: 0 = will not repurchase, 1 = will repurchase)

---

## Technical Approach

The project follows a reproducible ML pipeline with a deployed REST API and browser-based UI.

### 1. Exploratory Data Analysis
Notebooks in `notebooks/` cover distribution analysis, class balance, and feature correlations (`eda.ipynb`, `preprocessing.ipynb`, `balancing.ipynb`).

### 2. Training Pipeline (DVC-orchestrated)
Three sequential stages defined in `dvc.yaml`:

- **Data ingestion** — copies and validates the raw CSV into the pipeline's working directory.
- **Preprocessing** — applies stratified train/test split, scales numerical features, and encodes categoricals via a fitted `PreprocessorManager` saved to `utils/processor.pkl`.
- **Model training** — trains a **LightGBM** classifier with Optuna-tuned hyperparameters, logs params and per-class metrics to **MLflow**, registers the model, and saves the artifact to `training_pipeline/models/`.

**Final model performance (test set):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Will NOT repurchase (0) | 0.665 | 0.617 | 0.640 |
| Will repurchase (1) | 0.781 | 0.815 | 0.798 |

### 3. Inference
`inference/inference.py` loads new leads from `new_data.csv`, applies the saved preprocessor, loads the latest registered MLflow model, and writes predictions with class probabilities to `predictions.csv`.

### 4. Deployment
A **FastAPI** service (`deployment_package/src/lead_scoring_service/api.py`) wraps the model as a REST endpoint (`POST /predict`). The model is packaged as an MLflow `pyfunc` for environment-independent serving. A plain-HTML/JS frontend (`frontend/index.html`) provides a form-based UI that calls the API and displays the prediction verdict with probability bars.

---

## Key Skills Demonstrated

- **Machine learning** — binary classification with LightGBM; hyperparameter tuning with Optuna; class-level evaluation using precision, recall, and F1.
- **MLOps** — end-to-end experiment tracking and model registry with MLflow; pipeline reproducibility with DVC.
- **Data engineering** — modular preprocessing with a serializable `PreprocessorManager`; stratified splits to preserve class balance.
- **API development** — production-ready REST API with FastAPI and Pydantic validation; CORS-enabled for browser clients.
- **Software engineering** — clean separation of training, inference, and serving concerns; reusable utility modules; deployment package structure.
- **Frontend** — lightweight interactive UI for real-time single-lead scoring without a framework dependency.
