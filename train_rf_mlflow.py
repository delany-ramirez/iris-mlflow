# train_rf_mlflow.py  ───────────────────────────────────────────────────────────
import argparse
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

# -------------------- CLI ------------------------------------------------------
parser = argparse.ArgumentParser(description="Random Forest + MLflow (Iris)")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--experiment_name", type=str, default="Iris_RF")
parser.add_argument("--run_name", type=str, default=None)
args = parser.parse_args()

# -------------------- Tracking URI (edit if needed) ---------------------------
# Defaults to sqlite:///mlflow.db in the working directory.
# mlflow.set_tracking_uri("sqlite:///C:/_mlflow_runs/mlflow.db")  # optional

# -------------------- Datos ----------------------------------------------------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
)

# -------------------- Experimento y run ----------------------------------------
mlflow.set_experiment(args.experiment_name)
with mlflow.start_run(run_name=args.run_name) as run:
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # ---- loggeo ----------------------------------------------------------------
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)

    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="model",  
        signature=signature,
        input_example=X_train[:5],
        # skops rejects this type by default; safe here because this script trains
        # the model itself. Do not extend this trust to models from unknown sources.
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("=" * 60)
    print("Run ID:", run.info.run_id)
    print("Experiment ID:", run.info.experiment_id)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print(f"accuracy={acc:.3f}  f1_macro={f1:.3f}")
    print("=" * 60)
