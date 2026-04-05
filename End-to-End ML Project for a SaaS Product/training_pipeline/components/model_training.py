
from pathlib import Path
import sys
import json
import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


# -----------------------------------------------------------------
# Set base dir using this script location
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------
# Set tracking URI to experiments folder 
tracking_path = PROJECT_ROOT / "experiments" / "mlruns"
tracking_uri = "file:///" + str(tracking_path.as_posix()).lstrip('/')
mlflow.set_tracking_uri(tracking_uri)

# -----------------------------------------------------------------
# Paths to processed train data
X_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_train.csv"
y_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_train.csv"

# -----------------------------------------------------------------
# Paths to test data 
X_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_test.csv"
y_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_test.csv"

# -----------------------------------------------------------------
# Define experiment name
experiment_name = "Training Script"
mlflow.set_experiment(experiment_name)

# -----------------------------------------------------------------
# Define Numerical and categorical columns
from utils.features import numerical_cols, categorical_cols
NUM_COL = numerical_cols
CAT_COL = categorical_cols






# =============================================
# Best Metric for LightGBM Training after Tunning
# =============================================
RANDOM_STATE = 42
VERBOSITY = -1
N_ESTIMATORS = 279
LEARNING_RATE = 0.033823816473221295
MAX_DEPTH = 4
NUM_LEAVES = 100
SUBSAMPLE = 0.5892263863671638
COLSAMPLE_BYTREE = 0.8583502884521798



# =============================================
# LightGBM Training with MLflow
# =============================================
if __name__ == "__main__":
    # Check data paths
    missing_files = []
    for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not path.exists():
            missing_files.append(str(path))
    if missing_files:
        raise FileNotFoundError(
            "The following data files are missing:\n" + "\n".join(missing_files) +
            "\nRun preprocessing first or check paths."
        )
    
    # -------------------------
    # Load train & test data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # -------------------------
    # Train model with MLflow using best hyperparameters
    RUN_NAME = "LightGBM_BestParams"
    model = LGBMClassifier(
        random_state = RANDOM_STATE,
        verbosity = VERBOSITY,
        n_estimators = N_ESTIMATORS,
        learning_rate = LEARNING_RATE,
        max_depth = MAX_DEPTH,
        num_leaves = NUM_LEAVES,
        subsample = SUBSAMPLE,
        colsample_bytree = COLSAMPLE_BYTREE,
    )

    metrics_dir = PROJECT_ROOT / "training_pipeline" / "metrics"
    models_dir = PROJECT_ROOT / "training_pipeline" / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lightgbm_best_model.pkl"
    metrics_path = metrics_dir / "training_metrics.json"

    with mlflow.start_run(run_name=RUN_NAME) as run:
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        labels = sorted(y_train.unique())

        train_f1 = f1_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        train_precision = precision_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        train_recall = recall_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        test_precision = precision_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)

        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("num_leaves", NUM_LEAVES)
        mlflow.log_param("subsample", SUBSAMPLE)
        mlflow.log_param("colsample_bytree", COLSAMPLE_BYTREE)
        mlflow.log_param("random_state", RANDOM_STATE)

        for label, f1_val, precision_val, recall_val in zip(labels, train_f1, train_precision, train_recall):
            mlflow.log_metric(f"train_f1_macro_{label}", float(f1_val))
            mlflow.log_metric(f"train_precision_{label}", float(precision_val))
            mlflow.log_metric(f"train_recall_{label}", float(recall_val))
        for label, f1_val, precision_val, recall_val in zip(labels, test_f1, test_precision, test_recall):
            mlflow.log_metric(f"test_f1_macro_{label}", float(f1_val))
            mlflow.log_metric(f"test_precision_{label}", float(precision_val))
            mlflow.log_metric(f"test_recall_{label}", float(recall_val))

        mlflow.sklearn.log_model(model, artifact_path="model")
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model_name = "LightGBM_BestModel"
            mlflow.register_model(model_uri, registered_model_name)
            print(f"Registered model under MLflow name: {registered_model_name}")
        except Exception as register_error:
            print(f"Model registration skipped: {register_error}")

    joblib.dump(model, str(model_path))

    metrics = {
        "experiment_id": run.info.experiment_id,
        "run_id": run.info.run_id,
        "params": {
            "random_state": RANDOM_STATE,
            "verbosity": VERBOSITY,
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
            "num_leaves": NUM_LEAVES,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
        },
    }
    for label, val in zip(labels, train_f1):
        metrics[f"train_f1_macro_{int(label)}"] = float(val)
    for label, val in zip(labels, train_precision):
        metrics[f"train_precision_{int(label)}"] = float(val)
    for label, val in zip(labels, train_recall):
        metrics[f"train_recall_{int(label)}"] = float(val)
    for label, val in zip(labels, test_f1):
        metrics[f"test_f1_macro_{int(label)}"] = float(val)
    for label, val in zip(labels, test_precision):
        metrics[f"test_precision_{int(label)}"] = float(val)
    for label, val in zip(labels, test_recall):
        metrics[f"test_recall_{int(label)}"] = float(val)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Run completed. Run name: {RUN_NAME}. Metrics saved to {metrics_path}")
    print(f"Model saved to {model_path}")