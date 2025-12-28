import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap


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




# ============================================================
# Setup
# ============================================================

RESULTS_DIR = "/Users/lashandawilliams/Immune_Recovery_Ordinal_Model_Malaria_scRNAseq/03_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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
# Ordinal Logistic Regression
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

def run_ordinal_logistic(df, donor_col="ID", return_artifacts=False):
    """
    Ordinal LOPO evaluation + final model for SHAP.
    Produces:
      - fold_metrics
      - overall_metrics (balanced_accuracy, auc (OVR macro), mae)
      - oof_predictions with stable row_id (one row per sample)
      - final model artifacts for SHAP (X_all in the same row order as df_work)
    """

    df_work = df.copy()
    df_work = df_work.reset_index(drop=False).rename(columns={"index": "row_id"})  # stable key
    if df_work["row_id"].duplicated().any():
        raise ValueError("row_id is not unique after reset_index(drop=False).")

    exclude_cols = {"row_id", "donor_id", "timepoint", "y_binary", "y_stage"}
    feature_cols = select_feature_columns(df_work, exclude_cols)

    splits = lopo_splits(df_work, donor_col=donor_col)

   
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    oof_rows = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_df = df_work.iloc[train_idx].copy()
        test_df  = df_work.iloc[test_idx].copy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        X_test  = scaler.transform(test_df[feature_cols].values)

        y_train = train_df["y_stage"].values
        y_test  = test_df["y_stage"].values

        model = OrderedModel(y_train, X_train, distr="logit")
        result = model.fit(method="bfgs", disp=False)

        probs = result.predict(X_test)            # (n_test, n_classes)
        y_pred = np.argmax(probs, axis=1)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_prob_all.append(probs)

        for rid, yt, yp in zip(test_df["row_id"].values, y_test, y_pred):
            oof_rows.append({
                "row_id": int(rid),
                "fold": int(fold),
                "y_true": int(yt),
                "y_pred": int(yp),
            })

        fold_metrics.append({
            "model": "ordinal_logistic",
            "fold": int(fold),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "auc": np.nan,  # fill later at overall
            "mae": float(np.mean(np.abs(y_pred - y_test))),
        })


    y_true_all = np.asarray(y_true_all, dtype=int)
    y_pred_all = np.asarray(y_pred_all, dtype=int)
    y_prob_all = np.vstack(y_prob_all)


    oof_df = pd.DataFrame(oof_rows)


    if oof_df.shape[0] != df_work.shape[0]:

        missing = set(df_work["row_id"]) - set(oof_df["row_id"])
        dupes = oof_df["row_id"][oof_df["row_id"].duplicated()].unique().tolist()
        raise ValueError(
            f"OOF rows ({oof_df.shape[0]}) != n_samples ({df_work.shape[0]}). "
            f"Missing row_ids: {sorted(list(missing))[:10]}... "
            f"Duplicated row_ids: {dupes[:10]}..."
        )

    if oof_df["row_id"].duplicated().any():
        raise ValueError("OOF row_id duplicates detected (should be impossible).")

  
    y_true_bin = label_binarize(y_true_all, classes=[0, 1, 2])

    overall_metrics = {
        "model": "ordinal_logistic",
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "auc": float(roc_auc_score(y_true_bin, y_prob_all, multi_class="ovr", average="macro")),
        "mae": float(np.mean(np.abs(y_pred_all - y_true_all))),
    }

    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(df_work[feature_cols].values)
    y_all = df_work["y_stage"].values.astype(int)

    final_model = OrderedModel(y_all, X_all, distr="logit")
    final_result = final_model.fit(method="bfgs", disp=False)

    if return_artifacts:
        return {
            "df_work": df_work,                 
            "feature_names": feature_cols,
            "fold_metrics": fold_metrics,
            "overall_metrics": overall_metrics,
            "oof_predictions": oof_df.sort_values("row_id").reset_index(drop=True),
            "X": X_all,
            "y": y_all,
            "model": final_model,
            "result": final_result,
            "scaler": scaler_final,
        }

    return fold_metrics, overall_metrics



df_ordinal = pd.read_csv("/Users/lashandawilliams/Immune_Recovery_Ordinal_Model_Malaria_scRNAseq/01_data/processed/hostscope_features_ordinal_D0_D7_D28.csv")


ordinal_outputs = run_ordinal_logistic(df_ordinal, donor_col="ID", return_artifacts=True)

df_work = ordinal_outputs["df_work"]                # has row_id
oof     = ordinal_outputs["oof_predictions"]        # LOPO preds keyed by row_id
X       = ordinal_outputs["X"]
y_all   = ordinal_outputs["y"]
feature_names = ordinal_outputs["feature_names"]
result  = ordinal_outputs["result"]


def ordinal_expected_stage(X_input):
    """
    Compute expected ordinal stage:
    E[y] = sum_k P(y=k) * k
    """
    probs = result.predict(X_input)   # ✅ correct API
    stages = np.arange(probs.shape[1])
    return np.dot(probs, stages)


rng = np.random.default_rng(0)
background_idx = rng.choice(X.shape[0], size=min(30, X.shape[0]), replace=False)
X_background = X[background_idx]


X_explain = X


explainer = shap.KernelExplainer(
    ordinal_expected_stage,
    X_background
)

shap_values = explainer.shap_values(
    X_explain,
    nsamples=200
)

program_features = [
    "CD4_T_Immune_Checkpoint",
    "CD8_T_Immune_Checkpoint",
    "NK_Immune_Checkpoint",
    "CD4_T_IFN_I_Response",
    "CD8_T_IFN_I_Response",
    "NK_IFN_I_Response",
    "Mono_CD14_Inflammatory_Response",
    "Mono_CD16_Inflammatory_Response",
    "GammaDelta_T_Cytotoxic_Effector",
    "NK_Cytotoxic_Effector",
    "CD8_T_Cytotoxic_Effector"
]

composition_features = [
    "comp_CD4_T",
    "comp_CD8_T",
    "comp_NK",
    "comp_Plasma",
    "comp_Platelet",
    "comp_Mono_CD14",
    "comp_Mono_CD16",
    "comp_GammaDelta_T",
    "comp_Proliferating"
]
program_idx = [feature_names.index(f) for f in program_features]
composition_idx = [feature_names.index(f) for f in composition_features]

feature_idx = {f: i for i, f in enumerate(feature_names)}

prog_idx = [feature_idx[f] for f in program_features]
comp_idx = [feature_idx[f] for f in composition_features]

mean_abs_shap_prog = np.mean(np.abs(shap_values[:, prog_idx]))
mean_abs_shap_comp = np.mean(np.abs(shap_values[:, comp_idx]))

T_real = mean_abs_shap_prog - mean_abs_shap_comp

print(T_real)


def permute_program_indices(prog_idx, comp_idx, n_features):
    all_idx = np.arange(n_features)
    remaining = np.setdiff1d(all_idx, comp_idx)

    perm_prog = np.random.choice(
        remaining,
        size=len(prog_idx),
        replace=False
    )
    return perm_prog


N_PERM = 1000
T_perm = np.zeros(N_PERM)

for i in range(N_PERM):
    perm_prog_idx = rng.choice(
        np.setdiff1d(np.arange(len(feature_names)), comp_idx),
        size=len(prog_idx),
        replace=False
    )

    perm_prog_shap = np.mean(np.abs(shap_values[:, perm_prog_idx]))
    T_perm[i] = perm_prog_shap - mean_abs_shap_comp

T_perm = np.array(T_perm)


p_value = (1 + np.sum(T_perm >= T_real)) / (1 + len(T_perm))

print(p_value)



plt.hist(T_perm, bins=40, alpha=0.7)
plt.axvline(T_real, color="red", linestyle="--", linewidth=2)
plt.xlabel("Δ mean |SHAP| (Programs − Composition)")
plt.ylabel("Count")
plt.title("Permutation Test: Program vs Composition Reliance")
plt.savefig(f"{RESULTS_DIR}/figures/permutation/Permutation Test: Program vs Composition Reliance.png")
plt.show()
