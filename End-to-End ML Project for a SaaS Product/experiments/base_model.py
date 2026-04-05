# mlflow ui --backend-store-uri experiments/mlruns

from pathlib import Path
import sys
import os
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


# -----------------------------------------------------------------
# Set base dir using this script location
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------
# Set tracking URI to experiments folder 
tracking_path = PROJECT_ROOT / "experiments" / "mlruns"
tracking_uri = "file:///" + str(tracking_path.as_posix()).lstrip('/')
mlflow.set_tracking_uri(tracking_uri)

# -----------------------------------------------------------------
# Paths to processed train data
X_train_path = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
y_train_path = PROJECT_ROOT / "data" / "processed" / "y_train.csv"

# -----------------------------------------------------------------
# Paths to test data 
X_test_path = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
y_test_path = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

# -----------------------------------------------------------------
# Define experiment name
experiment_name = "BaselineModels"
mlflow.set_experiment(experiment_name)

# -----------------------------------------------------------------
# Define Numerical and categorical columns
from utils.features import numerical_cols, categorical_cols
NUM_COL = numerical_cols
CAT_COL = categorical_cols








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
    # Train model with MLflow
    RUN_NAME = "LightGBM"
    with mlflow.start_run(run_name=RUN_NAME):
        model = LGBMClassifier(
            random_state=42, verbosity=-1, n_estimators=100, learning_rate=0.1
        )
        model.fit(X_train, y_train)

        # Training metrics
        y_pred_train = model.predict(X_train)
        labels = sorted(y_train.unique())
        precision_train = precision_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)

        # Test metrics
        y_pred_test = model.predict(X_test)
        precision_test = precision_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)

        # Log parameters
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("random_state", 42)

        # Log metrics
        for label, p, r in zip(labels, precision_train, recall_train):
            mlflow.log_metric(f"train_precision_{label}", float(p))
            mlflow.log_metric(f"train_recall_{label}", float(r))
        for label, p, r in zip(labels, precision_test, recall_test):
            mlflow.log_metric(f"test_precision_{label}", float(p))
            mlflow.log_metric(f"test_recall_{label}", float(r))

        print(f"Run completed. Run name: {RUN_NAME}. Metrics logged to MLflow experiment: {experiment_name}")





# =============================================
# XGBoost
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
    # Train model with MLflow
    RUN_NAME = "XGBoost"
    with mlflow.start_run(run_name=RUN_NAME):
        model = XGBClassifier(random_state=42, verbosity=0, n_estimators=100, learning_rate=0.3, max_depth=6)
        model.fit(X_train, y_train)

        # Training metrics
        y_pred_train = model.predict(X_train)
        labels = sorted(y_train.unique())
        precision_train = precision_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)

        # Test metrics
        y_pred_test = model.predict(X_test)
        precision_test = precision_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)

        # Log parameters
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.3)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("random_state", 42)

        # Log metrics
        for label, p, r in zip(labels, precision_train, recall_train):
            mlflow.log_metric(f"train_precision_{label}", float(p))
            mlflow.log_metric(f"train_recall_{label}", float(r))
        for label, p, r in zip(labels, precision_test, recall_test):
            mlflow.log_metric(f"test_precision_{label}", float(p))
            mlflow.log_metric(f"test_recall_{label}", float(r))

        print(f"Run completed. Run name: {RUN_NAME}. Metrics logged to MLflow experiment: {experiment_name}")


# =============================================
# Logistic Regression
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
    # Train model with MLflow
    RUN_NAME = "LogisticRegression"
    with mlflow.start_run(run_name=RUN_NAME):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Training metrics
        y_pred_train = model.predict(X_train)
        labels = sorted(y_train.unique())
        precision_train = precision_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)

        # Test metrics
        y_pred_test = model.predict(X_test)
        precision_test = precision_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)

        # Log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)

        # Log training metrics
        for label, p, r in zip(labels, precision_train, recall_train):
            mlflow.log_metric(f"train_precision_{label}", float(p))
            mlflow.log_metric(f"train_recall_{label}", float(r))

        # Log test metrics
        for label, p, r in zip(labels, precision_test, recall_test):
            mlflow.log_metric(f"test_precision_{label}", float(p))
            mlflow.log_metric(f"test_recall_{label}", float(r))

        print(f"Run completed. Run name: {RUN_NAME}. Metrics logged to MLflow experiment: {experiment_name}")




# =============================================
# Random Forest
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
    # Train model with MLflow
    RUN_NAME = "RandomForest"
    with mlflow.start_run(run_name=RUN_NAME):
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)

        # Training metrics
        y_pred_train = model.predict(X_train)
        labels = sorted(y_train.unique())
        precision_train = precision_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)

        # Test metrics
        y_pred_test = model.predict(X_test)
        precision_test = precision_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)

        # Log parameters
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log training metrics
        for label, p, r in zip(labels, precision_train, recall_train):
            mlflow.log_metric(f"train_precision_{label}", float(p))
            mlflow.log_metric(f"train_recall_{label}", float(r))

        # Log test metrics
        for label, p, r in zip(labels, precision_test, recall_test):
            mlflow.log_metric(f"test_precision_{label}", float(p))
            mlflow.log_metric(f"test_recall_{label}", float(r))

        print(f"Run completed. Run name: {RUN_NAME}. Metrics logged to MLflow experiment: {experiment_name}")