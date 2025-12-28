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

import joypy


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

    # --- Work on a copy with a stable row_id ---
    df_work = df.copy()
    df_work = df_work.reset_index(drop=False).rename(columns={"index": "row_id"})  # stable key
    if df_work["row_id"].duplicated().any():
        raise ValueError("row_id is not unique after reset_index(drop=False).")

    exclude_cols = {"row_id", "donor_id", "timepoint", "y_binary", "y_stage"}
    feature_cols = select_feature_columns(df_work, exclude_cols)

    splits = lopo_splits(df_work, donor_col=donor_col)

    # Collect overall (OOF)
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

        # ✅ OOF rows keyed by row_id (this is the critical part)
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

    # --- Stack overall ---
    y_true_all = np.asarray(y_true_all, dtype=int)
    y_pred_all = np.asarray(y_pred_all, dtype=int)
    y_prob_all = np.vstack(y_prob_all)

    # --- Build OOF DF and enforce 1 row per sample ---
    oof_df = pd.DataFrame(oof_rows)

    # HARD GUARANTEES
    if oof_df.shape[0] != df_work.shape[0]:
        # This should never happen now; if it does, show diagnostics
        missing = set(df_work["row_id"]) - set(oof_df["row_id"])
        dupes = oof_df["row_id"][oof_df["row_id"].duplicated()].unique().tolist()
        raise ValueError(
            f"OOF rows ({oof_df.shape[0]}) != n_samples ({df_work.shape[0]}). "
            f"Missing row_ids: {sorted(list(missing))[:10]}... "
            f"Duplicated row_ids: {dupes[:10]}..."
        )

    if oof_df["row_id"].duplicated().any():
        raise ValueError("OOF row_id duplicates detected (should be impossible).")

    # --- Overall metrics ---
    y_true_bin = label_binarize(y_true_all, classes=[0, 1, 2])

    overall_metrics = {
        "model": "ordinal_logistic",
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "auc": float(roc_auc_score(y_true_bin, y_prob_all, multi_class="ovr", average="macro")),
        "mae": float(np.mean(np.abs(y_pred_all - y_true_all))),
    }

    # --- Final model on ALL data for SHAP (same order as df_work) ---
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(df_work[feature_cols].values)
    y_all = df_work["y_stage"].values.astype(int)

    final_model = OrderedModel(y_all, X_all, distr="logit")
    final_result = final_model.fit(method="bfgs", disp=False)

    if return_artifacts:
        return {
            "df_work": df_work,                 # includes row_id and original columns
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



shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title("Global SHAP Importance\n(Ordinal Malaria Recovery)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/shap/shap_global_bar.png", dpi=300)
plt.close()


shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Summary\nImmune Programs Driving Recovery")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/shap/shap_beeswarm.png", dpi=300)
plt.close()


mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[-5:][::-1]
top_features = [feature_names[i] for i in top_idx]

print("Top programs for dependence plots:")
for f in top_features:
    print("  ", f)


for feat in top_features:
    shap.dependence_plot(
        feat,
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"SHAP Dependence: {feat}")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figures/shap/shap_dependence_{feat}.png", dpi=300)
    plt.close()


shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df["row_id"] = df_work["row_id"].values
shap_df["true_stage"] = y_all

df_plot = shap_df.merge(oof[["row_id", "fold", "y_true", "y_pred"]], on="row_id", how="left", validate="one_to_one")

if df_plot["y_pred"].isna().any():
    bad = df_plot[df_plot["y_pred"].isna()][["row_id", "true_stage"]].head()
    raise ValueError(f"Some rows did not match OOF predictions. Example:\n{bad}")

df_plot["abs_error"] = (df_plot["y_true"] - df_plot["y_pred"]).abs()
df_plot["error_type"] = np.where(
    df_plot["abs_error"] == 0, "correct",
    np.where(df_plot["abs_error"] == 1, "adjacent_error", "extreme_error")
)

df_plot["mean_abs_shap"] = df_plot[feature_names].abs().mean(axis=1)


print(df_work.shape, oof.shape)
print(df_plot.shape)
print(df_plot["error_type"].value_counts())



plt.figure(figsize=(6, 4))

sns.violinplot(
    data=df_plot,
    x="error_type",
    y="mean_abs_shap",
    inner="quartile",
    cut=0
)

sns.stripplot(
    data=df_plot,
    x="error_type",
    y="mean_abs_shap",
    color="black",
    size=4,
    alpha=0.6
)

plt.title("Model reliance (mean |SHAP|) by prediction outcome")
plt.xlabel("")
plt.ylabel("Mean |SHAP|")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/shap/Model reliance (mean |SHAP|) by prediction outcome.png")
plt.show()


plt.figure(figsize=(6, 4))

sns.swarmplot(
    data=df_plot,
    x="true_stage",
    y="mean_abs_shap",
    hue="error_type",
    dodge=True,
    size=5
)

plt.xlabel("True recovery stage")
plt.ylabel("Mean |SHAP|")
plt.title("Model reliance across recovery")
plt.legend(title="")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/shap/Model reliance across recovery.png")
plt.show()



program = "CD4_T_Immune_Checkpoint"

plt.figure(figsize=(6, 4))

sns.violinplot(
    data=df_plot,
    x="true_stage",
    y=program,
    inner="quartile",
    cut=0
)

sns.stripplot(
    data=df_plot,
    x="true_stage",
    y=program,
    color="black",
    alpha=0.6
)

plt.title(f"{program}: |SHAP| across recovery")
plt.xlabel("True stage")
plt.ylabel("|SHAP| contribution")
plt.savefig(f"{RESULTS_DIR}/figures/shap/CD4_T_Immune_Checkpoint |SHAP| across recovery.png")
plt.tight_layout()
plt.show()


program = "NK_Immune_Checkpoint"

plt.figure(figsize=(6, 4))

sns.violinplot(
    data=df_plot,
    x="true_stage",
    y=program,
    inner="quartile",
    cut=0
)

sns.stripplot(
    data=df_plot,
    x="true_stage",
    y=program,
    color="black",
    alpha=0.6
)

plt.title(f"{program}: |SHAP| across recovery")
plt.xlabel("True stage")
plt.ylabel("|SHAP| contribution")
plt.savefig(f"{RESULTS_DIR}/figures/shap/NK_Immune_Checkpoint |SHAP| across recovery.png")
plt.tight_layout()
plt.show()


program = "CD8_T_Immune_Checkpoint"

plt.figure(figsize=(6, 4))

sns.violinplot(
    data=df_plot,
    x="true_stage",
    y=program,
    inner="quartile",
    cut=0
)

sns.stripplot(
    data=df_plot,
    x="true_stage",
    y=program,
    color="black",
    alpha=0.6
)

plt.title(f"{program}: |SHAP| across recovery")
plt.xlabel("True stage")
plt.ylabel("|SHAP| contribution")
plt.savefig(f"{RESULTS_DIR}/figures/shap/CD8_T_Immune_Checkpoint |SHAP| across recovery.png")
plt.tight_layout()
plt.show()


program = "comp_Plasma"

plt.figure(figsize=(6, 4))

sns.violinplot(
    data=df_plot,
    x="true_stage",
    y=program,
    inner="quartile",
    cut=0
)

sns.stripplot(
    data=df_plot,
    x="true_stage",
    y=program,
    color="black",
    alpha=0.6
)

plt.title(f"{program}: |SHAP| across recovery")
plt.xlabel("True stage")
plt.ylabel("|SHAP| contribution")
plt.savefig(f"{RESULTS_DIR}/figures/shap/Composition Plasma |SHAP| across recovery.png")
plt.tight_layout()
plt.show()


program = "comp_Platelet"

plt.figure(figsize=(6, 4))

sns.violinplot(
    data=df_plot,
    x="true_stage",
    y=program,
    inner="quartile",
    cut=0
)

sns.stripplot(
    data=df_plot,
    x="true_stage",
    y=program,
    color="black",
    alpha=0.6
)

plt.title(f"{program}: |SHAP| across recovery")
plt.xlabel("True stage")
plt.ylabel("|SHAP| contribution")
plt.savefig(f"{RESULTS_DIR}/figures/shap/Composition Platelet |SHAP| across recovery.png")
plt.tight_layout()
plt.show()




joypy.joyplot(
    df_plot,
    by="true_stage",
    column="mean_abs_shap",
    figsize=(6, 4),
    overlap=1.5
)

plt.xlabel("Mean |SHAP|")
plt.title("Distribution of model reliance across recovery")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/shap/Distribution of model reliance across recovery.png")
plt.show()

