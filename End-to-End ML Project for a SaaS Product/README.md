<div align="center">
  <h1>
    End-to-End SaaS Lead Scoring ML Pipeline with MLOps & Deployment
  </h1>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-LightGBM%20Model-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/MLOps-DVC%20%26%20MLflow-lightblue?style=flat-square"/>
  <img src="https://img.shields.io/badge/API-FastAPI%20Deployment-success?style=flat-square"/>
</p>

---

## 🧠 Business Problem

SaaS companies invest heavily in marketing and customer success, but not every lead converts to a repeat buyer. Without a reliable way to identify which customers are likely to repurchase, sales and marketing teams waste resources on low-probability leads while high-value ones go under-served.

---

## 🎯 Objective 

The objective of this project is to build a binary classification model that predicts whether a SaaS customer will repurchase, enabling the business to prioritize outreach, personalize campaigns, and reduce churn.

---

## 📊 Data & Inputs

- A dataset of SaaS customer records with 29 features per lead.
- Demographics, Company profile, Purchase behavior, Product usage, Support & marketing

---

## ⚙️ Technical Approach

- Exploratory Data Analysis: Notebooks cover distribution analysis, class balance, and feature correlations (`eda.ipynb`, `preprocessing.ipynb`, `balancing.ipynb`).
- Three-stage DVC pipeline: data ingestion (copy & validate CSV), preprocessing (stratified split + scaling + encoding via saved PreprocessorManager), and model training (Optuna-tuned LightGBM with MLflow logging, registry, and saved artifacts).
- `inference/inference.py` loads new leads from `new_data.csv`, applies the saved preprocessor, loads the latest registered MLflow model, and writes predictions with class probabilities to `predictions.csv`.
- A **FastAPI** service (`deployment_package/src/lead_scoring_service/api.py`) wraps the model as a REST endpoint (`POST /predict`). The model is packaged as an MLflow `pyfunc` for environment-independent serving. A plain-HTML/JS frontend (`frontend/index.html`) provides a form-based UI that calls the API and displays the prediction verdict with probability bars.

---

## 🛠 Key Skills Demonstrated

- Machine learning: binary classification with LightGBM; hyperparameter tuning with Optuna; class-level evaluation using precision, recall, and F1.
- MLOps: end-to-end experiment tracking and model registry with MLflow; pipeline reproducibility with DVC.
- Data engineering: modular preprocessing with a serializable `PreprocessorManager`; stratified splits to preserve class balance.
- API development: production-ready REST API with FastAPI and Pydantic validation; CORS-enabled for browser clients.
- Software engineering: clean separation of training, inference, and serving concerns; reusable utility modules; deployment package structure.
- Frontend: lightweight interactive UI for real-time single-lead scoring without a framework dependency.

---

## 🎥 YouTube Walkthrough
 
https://www.youtube.com/watch?v=k5c1FpPhhf0
