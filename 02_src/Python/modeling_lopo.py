#!/usr/bin/env python


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import label_binarize
import mlflow
import mlflow.sklearn




# ============================================================
# Setup
# ============================================================

RESULTS_DIR = os.getenv("HOSTSCOPE_RESULTS_DIR", "03_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures", "modeling"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)

# ============================================================
# MLflow setup (optional but recommended)
# ============================================================
# Configure via environment variables:
#   MLFLOW_TRACKING_URI        (e.g., http://localhost:5000 or file:./mlruns)
#   MLFLOW_EXPERIMENT_NAME     (default: HostScope)
#   MLFLOW_RUN_NAME            (default: HostScope_LOPO_Modeling)
#
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "HostScope")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def _log_artifact_if_exists(path: str):
    if path and os.path.exists(path):
        mlflow.log_artifact(path)



# ============================================================
# Utility functions
# ============================================================

def select_feature_columns(df, exclude):
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def lopo_splits(df, donor_col="ID"):
    logo = LeaveOneGroupOut()
    groups = df[donor_col].astype(str).values
    X_dummy = np.zeros((len(df), 1))
    return list(logo.split(X_dummy, groups=groups))


def plot_confusion_matrix(
    cm,
    labels,
    title,
    out_path,
    normalize=False,
    figsize=(6, 6),
    cmap="Blues"
):
    """
    Plot a readable confusion matrix with large annotations.

    Parameters
    ----------
    cm : array-like (n_classes, n_classes)
        Confusion matrix
    labels : list of str
        Class labels in display order
    title : str
        Plot title
    out_path : str
        Where to save PNG
    normalize : bool
        If True, normalize rows to proportions
    figsize : tuple
        Figure size
    cmap : str
        Matplotlib colormap
    """

    cm = np.array(cm)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=1.5,
        linecolor="white",
        annot_kws={
            "size": 18,
            "weight": "bold",
            "color": "black"
        }
    )

    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14, rotation=0)

    ax.set_xlabel("Predicted", fontsize=16, labelpad=12)
    ax.set_ylabel("True", fontsize=16, labelpad=12)

    ax.set_title(title, fontsize=18, pad=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



# ============================================================
# Pairwise Logistic Regression
# ============================================================


def run_pairwise_logistic(df):

    exclude_cols = {"donor_id", "timepoint", "y_binary", "y_stage"}
    feature_cols = select_feature_columns(df, exclude_cols)

    splits = lopo_splits(df)

    oof_rows = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splits):

        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df["y_binary"].values

        X_test  = test_df[feature_cols].values
        y_test  = test_df["y_binary"].values

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=1000,
                )),
            ]
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Store OOF predictions
        for yt, yp, yp_prob in zip(y_test, y_pred, y_prob):
            oof_rows.append({
                "fold": fold,
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_prob": float(yp_prob),
            })

        fold_metrics.append({
            "model": "pairwise_logistic",
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "mae": float(np.mean(np.abs(y_pred - y_test))),
        })

    # =========================
    # OVERALL LOPO METRICS
    # =========================
    oof_df = pd.DataFrame(oof_rows)

    overall_metrics = {
        "model": "pairwise_logistic",
        "balanced_accuracy": float(
            balanced_accuracy_score(oof_df["y_true"], oof_df["y_pred"])
        ),
        "auc": float(
            roc_auc_score(oof_df["y_true"], oof_df["y_prob"])
        ),
        "mae": float(
            np.mean(np.abs(oof_df["y_pred"] - oof_df["y_true"]))
        ),
    }


    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(oof_df["y_true"], oof_df["y_pred"])

    plot_confusion_matrix(
        cm,
        labels=["D28", "D0"],
        title="Pairwise Logistic (D0 vs D28)",
        out_path=f"{RESULTS_DIR}/figures/modeling/confusion_pairwise.png",
        normalize=False
    )

    plot_confusion_matrix(
        cm,
        labels=["D28", "D0"],
        title="Pairwise Logistic (D0 vs D28) — Normalized",
        out_path=f"{RESULTS_DIR}/figures/modeling/confusion_pairwise_normalized.png",
        normalize=True
    )

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    report = classification_report(
        oof_df["y_true"],
        oof_df["y_pred"],
        target_names=["D28", "D0"],
        output_dict=True
    )

    pd.DataFrame(report).T.to_csv(
        f"{RESULTS_DIR}/tables/classification_report_pairwise.csv"
    )

    return fold_metrics, overall_metrics

# ============================================================
# Ordinal Logistic Regression
# ============================================================


def run_ordinal_logistic(df, return_artifacts=False):

    exclude_cols = {"ID", "timepoint", "y_binary", "y_stage"}
    feature_cols = select_feature_columns(df, exclude_cols)

    splits = lopo_splits(df)

    oof_rows = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splits):

        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        X_test  = scaler.transform(test_df[feature_cols].values)

        y_train = train_df["y_stage"].values
        y_test  = test_df["y_stage"].values

        model = OrderedModel(y_train, X_train, distr="logit")
        result = model.fit(method="bfgs", disp=False)

        probs = result.predict(X_test)        # (n, 3)
        y_pred = np.argmax(probs, axis=1)

        for yt, yp, pr in zip(y_test, y_pred, probs):
            oof_rows.append({
                "model": "ordinal_logistic",
                "fold": fold,
                "y_true": int(yt),
                "y_pred": int(yp),
                "probs": pr,
            })

        fold_metrics.append({
            "model": "ordinal_logistic",
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "auc": np.nan,  # computed globally
            "mae": float(np.mean(np.abs(y_pred - y_test))),
        })

    oof_df = pd.DataFrame(oof_rows)

    # --- Overall metrics ---
    y_true_bin = label_binarize(oof_df["y_true"], classes=[0, 1, 2])
    y_prob_mat = np.vstack(oof_df["probs"].values)

    overall_metrics = {
        "model": "ordinal_logistic",
        "balanced_accuracy": float(
            balanced_accuracy_score(oof_df["y_true"], oof_df["y_pred"])
        ),
        "auc": float(
            roc_auc_score(
                y_true_bin,
                y_prob_mat,
                average="macro",
                multi_class="ovr",
            )
        ),
        "mae": float(
            np.mean(np.abs(oof_df["y_pred"] - oof_df["y_true"]))
        ),
    }

    # --- Confusion matrix ---
    cm = confusion_matrix(
    oof_df["y_true"],
    oof_df["y_pred"],
    labels=[0, 1, 2]
)

    plot_confusion_matrix(
    cm,
    labels=["D0", "D7", "D28"],
    title="Ordinal Logistic (D0–D7–D28)",
    out_path=f"{RESULTS_DIR}/figures/modeling/confusion_ordinal.png",
    normalize=False,
    figsize=(7, 7)
)

    plot_confusion_matrix(
    cm,
    labels=["D0", "D7", "D28"],
    title="Ordinal Logistic (D0–D7–D28) — Normalized",
    out_path=f"{RESULTS_DIR}/figures/modeling/confusion_ordinal_normalized.png",
    normalize=True,
    figsize=(7, 7)
)

    # --- Classification report ---
    report = classification_report(
        oof_df["y_true"],
        oof_df["y_pred"],
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(
        f"{RESULTS_DIR}/tables/classification_report_ordinal.csv"
    )

    if return_artifacts:
        return {
            "fold_metrics": fold_metrics,
            "overall_metrics": overall_metrics,
            "X": X_all,
            "y": y_all,
            "feature_names": feature_cols,
            "model": final_model,
            "result": final_result,
            "scaler": scaler_final,
        }

    return fold_metrics, overall_metrics

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df_pairwise_path = os.getenv(
        "HOSTSCOPE_PAIRWISE_CSV",
        os.path.join("01_data", "processed", "hostscope_features_pairwise_D0_vs_D28.csv"),
    )
    df_ordinal_path = os.getenv(
        "HOSTSCOPE_ORDINAL_CSV",
        os.path.join("01_data", "processed", "hostscope_features_ordinal_D0_D7_D28.csv"),
    )

    df_pairwise = pd.read_csv(df_pairwise_path)
    df_ordinal = pd.read_csv(df_ordinal_path)

    
    # ============================================================
    # MLflow: top-level run for this execution
    # ============================================================
    run_name = os.getenv("MLFLOW_RUN_NAME", "HostScope_LOPO_Modeling")
    with mlflow.start_run(run_name=run_name):

        # Log high-level context
        mlflow.set_tag("project", "HostScope")
        mlflow.set_tag("cv_scheme", "LOPO")
        mlflow.set_tag("task_pairwise", "Day0_vs_Day28")
        mlflow.set_tag("task_ordinal", "Day0_lt_Day7_lt_Day28")

        # Log dataset locations (lightweight + reproducible)
        mlflow.log_param(
            "features_pairwise_path",
            os.path.relpath(df_pairwise_path) if os.path.exists(df_pairwise_path) else df_pairwise_path
        )
        mlflow.log_param(
            "features_ordinal_path",
            os.path.relpath(df_ordinal_path) if os.path.exists(df_ordinal_path) else df_ordinal_path
        )

        # ---------------------------
        # Pairwise model run
        # ---------------------------
        with mlflow.start_run(run_name="pairwise_logistic", nested=True):
            mlflow.set_tag("model", "logistic_regression")
            mlflow.set_tag("task", "pairwise")
            mlflow.log_params({
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000
            })

            pairwise_folds, pairwise_overall = run_pairwise_logistic(df_pairwise)

            # Log overall metrics
            mlflow.log_metrics({
                "balanced_accuracy": float(pairwise_overall.get("balanced_accuracy", np.nan)),
                "auc": float(pairwise_overall.get("auc", np.nan)),
                "mae": float(pairwise_overall.get("mae", np.nan)),
            })

            # Log key artifacts produced by this run
            _log_artifact_if_exists(f"{RESULTS_DIR}/tables/classification_report_pairwise.csv")
            _log_artifact_if_exists(f"{RESULTS_DIR}/figures/modeling/confusion_pairwise.png")
            _log_artifact_if_exists(f"{RESULTS_DIR}/figures/modeling/confusion_pairwise_normalized.png")

        # ---------------------------
        # Ordinal model run
        # ---------------------------
        with mlflow.start_run(run_name="ordinal_logistic", nested=True):
            mlflow.set_tag("model", "ordered_logit")
            mlflow.set_tag("task", "ordinal")
            mlflow.log_params({
                "distr": "logit",
                "fit_method": "bfgs"
            })

            ordinal_folds, ordinal_overall = run_ordinal_logistic(df_ordinal)

            mlflow.log_metrics({
                "balanced_accuracy": float(ordinal_overall.get("balanced_accuracy", np.nan)),
                "auc": float(ordinal_overall.get("auc", np.nan)),
                "mae": float(ordinal_overall.get("mae", np.nan)),
            })

            _log_artifact_if_exists(f"{RESULTS_DIR}/tables/classification_report_ordinal.csv")
            _log_artifact_if_exists(f"{RESULTS_DIR}/figures/modeling/confusion_ordinal.png")
            _log_artifact_if_exists(f"{RESULTS_DIR}/figures/modeling/confusion_ordinal_normalized.png")

        # ---------------------------
        # Combined summary tables
        # ---------------------------
        metrics_df = pd.DataFrame(pairwise_folds + ordinal_folds)
        metrics_path = f"{RESULTS_DIR}/tables/hostscope_fold_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        overall_df = pd.DataFrame([pairwise_overall, ordinal_overall])
        overall_path = f"{RESULTS_DIR}/tables/hostscope_overall_metrics.csv"
        overall_df.to_csv(overall_path, index=False)

        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(overall_path)


