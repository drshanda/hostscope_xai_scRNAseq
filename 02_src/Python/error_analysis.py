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
from scipy.stats import mannwhitneyu


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


ordinal_outputs = run_ordinal_logistic(df_ordinal, return_artifacts=True)

oof = ordinal_outputs["oof_predictions"].copy()

oof["abs_error"] = np.abs(oof["y_true"] - oof["y_pred"])
oof["error_type"] = np.where(
    oof["abs_error"] == 0, "correct",
    np.where(oof["abs_error"] == 1, "adjacent_error", "extreme_error")
)

oof["true_label"] = oof["y_true"].map({0: "D0", 1: "D7", 2: "D28"})
oof["pred_label"] = oof["y_pred"].map({0: "D0", 1: "D7", 2: "D28"})

oof["error_type"].value_counts()


# Create a features table with matching sample_id
df_feat = df_ordinal.copy()
df_feat = df_feat.reset_index(drop=False).rename(columns={"index": "row_id"})

X_df = df_feat.set_index("row_id")[ordinal_outputs["feature_names"]]

merged = oof.merge(X_df, on="row_id", how="inner")




# Compare means
merged.groupby("error_type")[ordinal_outputs["feature_names"]].mean()


program_cols = feature_names

summary = merged.groupby("error_type")[program_cols].mean()
summary





plt.figure(figsize=(10, 4))
sns.heatmap(
    summary,
    cmap="coolwarm",
    center=0
)
plt.title("Mean Immune Program Scores by Error Type")
plt.savefig(f"{RESULTS_DIR}/figures/errors/Mean Immune Program Scores by Error Type.png")

programs_to_test = [
    "CD8_T_Immune_Checkpoint",
    "CD4_T_Immune_Checkpoint",
    "NK_Immune_Checkpoint",
    "comp_Plasma",
    "comp_Platelet"
]

def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    if nx == 0 or ny == 0:
        return np.nan

    gt = sum(xi > yj for xi in x for yj in y)
    lt = sum(xi < yj for xi in x for yj in y)

    return (gt - lt) / (nx * ny)

mw_results = {}

for prog in programs_to_test:
    correct = merged.loc[merged.error_type == "correct", prog]
    adj = merged.loc[merged.error_type == "adjacent_error", prog]

    # Mann–Whitney U
    U, p = mannwhitneyu(correct, adj, alternative="two-sided")

    # Cliff’s delta
    delta = cliffs_delta(correct, adj)

    mw_results[prog] = {
        "U": U,
        "p_value": p,
        "cliffs_delta": delta,
        "median_correct": np.median(correct),
        "median_adjacent": np.median(adj),
        "n_correct": len(correct),
        "n_adjacent": len(adj),
    }

mw_df = (
    pd.DataFrame.from_dict(mw_results, orient="index")
    .reset_index()
    .rename(columns={"index": "program"})
)

def cliffs_label(delta):
    if pd.isna(delta):
        return "NA"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"

mw_df["delta_magnitude"] = mw_df["cliffs_delta"].apply(cliffs_label)

mw_df.to_csv(
    f"{RESULTS_DIR}/tables/mann_whitney_cliffs_delta_summary.csv",
    index=False
)

mw_plot = mw_df.copy()
mw_plot = mw_plot.sort_values("cliffs_delta")


plt.figure(figsize=(6, 0.4 * len(mw_plot)))

plt.hlines(
    y=mw_plot["program"],
    xmin=0,
    xmax=mw_plot["cliffs_delta"],
    color="black",
    linewidth=2
)

plt.scatter(
    mw_plot["cliffs_delta"],
    mw_plot["program"],
    s=80,
    zorder=3
)

# Reference line at no effect
plt.axvline(0, color="gray", linestyle="--")

plt.xlabel("Cliff’s δ (effect size)")
plt.ylabel("")
plt.title("Effect sizes (Cliff’s δ): Correct vs Adjacent Errors")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/errors/cliffs_delta_forest.png", dpi=300)
plt.show()

def build_df_plot(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    sample_id_col_pred: str,
    sample_id_col_feat: str,
    true_label_col: str,
    pred_label_col: str,
    stage_label_map: dict
) -> pd.DataFrame:
    """
    Build plotting dataframe for trajectory and ordinal residual analysis.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain sample ID, true label, predicted label
    features_df : pd.DataFrame
        Must contain sample ID and immune program scores
    sample_id_col_pred : str
        Column name of sample ID in predictions_df
    sample_id_col_feat : str
        Column name of sample ID in features_df
    true_label_col : str
        Column name of true ordinal label (numeric)
    pred_label_col : str
        Column name of predicted ordinal label (numeric)
    stage_label_map : dict
        Mapping from numeric stage -> label (e.g. {0:"D0",1:"D7",2:"D28"})

    Returns
    -------
    df_plot : pd.DataFrame
    """

    df = predictions_df.copy()

    # --- Standardize column names ---
    df = df.rename(
        columns={
            sample_id_col_pred: "row_id",
            true_label_col: "true_stage_num",
            pred_label_col: "pred_stage_num",
        }
    )

    # --- Compute ordinal residual ---
    df["ordinal_residual"] = (
        df["pred_stage_num"] - df["true_stage_num"]
    )

    # --- Validate ordinality ---
    if (df["ordinal_residual"].abs() > 1).any():
        raise ValueError("Extreme ordinal errors detected (|residual| > 1)")

    # --- Error type ---
    df["error_type"] = np.where(
        df["ordinal_residual"] == 0,
        "correct",
        "adjacent_error",
    )

    # --- Stage labels ---
    df["true_stage"] = df["true_stage_num"].map(stage_label_map)


    feat = features_df.copy()

    if sample_id_col_feat != "row_id":
        if "row_id" in feat.columns:
            raise ValueError(
                "features_df already contains 'sample_id', "
                "but a different column was requested for renaming."
        )
        feat = feat.rename(columns={sample_id_col_feat: "row_id"})

    df_plot = pd.concat([df, feat], axis=1)

    if df_plot.shape[0] != df.shape[0]:
        raise ValueError("Sample mismatch after merge")


    df_plot = pd.concat([df, feat], axis=1)

    if df_plot.shape[0] != df.shape[0]:
        raise ValueError("Sample mismatch after merge")

    return df_plot


def plot_program_trajectory(
    df_plot: pd.DataFrame,
    program_col: str,
    stage_col: str = "true_stage_num",
    error_col: str = "error_type",
    ax=None,
    title: str = None
):
    """
    Plot immune program trajectory across ordinal stages.
    """

    if ax is None:
        ax = plt.gca()

    sns.stripplot(
        data=df_plot,
        x=stage_col,
        y=program_col,
        hue=error_col,
        jitter=0.2,
        dodge=True,
        size=7,
        palette={
            "correct": "#1b9e77",
            "adjacent_error": "#d95f02",
        },
        ax=ax,
    )

    sns.regplot(
        data=df_plot,
        x=stage_col,
        y=program_col,
        lowess=True,
        scatter=False,
        color="black",
        line_kws={"linewidth": 2},
        ax=ax,
    )

    ax.set_title(title or f"{program_col} trajectory")
    ax.set_xlabel("True stage")
    ax.set_ylabel(f"{program_col} score")

    ax.legend_.remove()


def plot_ordinal_residuals(
    df_plot: pd.DataFrame,
    program_col: str,
    residual_col: str = "ordinal_residual",
    ax=None,
    title: str = None
):
    """
    Plot ordinal residuals vs immune program score.
    """

    if ax is None:
        ax = plt.gca()

    sns.stripplot(
        data=df_plot,
        x=program_col,
        y=residual_col,
        jitter=0.15,
        size=7,
        color="#444444",
        ax=ax,
    )

    ax.axhline(0, linestyle="--", color="gray")

    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(
        ["-1 (Pred earlier)", "0 (Correct)", "+1 (Pred later)"]
    )
    ax.set_xticks([])
    ax.set_xlabel("")

    ax.set_title(title or f"{program_col} residuals")
    ax.set_xlabel(f"{program_col} score")
    ax.set_ylabel("Ordinal residual")


def plot_program_error_panel(
    df_plot: pd.DataFrame,
    program_cols: list,
    figsize=(14, 10)
):
    """
    Create multi-panel figure:
    rows = programs
    cols = trajectory | residual
    """

    sns.set(style="whitegrid", context="talk")

    n_prog = len(program_cols)
    fig, axes = plt.subplots(
        n_prog, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.3, 1]},
    )

    if n_prog == 1:
        axes = np.array([axes])

    for i, prog in enumerate(program_cols):
        plot_program_trajectory(
            df_plot,
            program_col=prog,
            ax=axes[i, 0],
            title=f"{prog} across recovery",
        )

        plot_ordinal_residuals(
            df_plot,
            program_col=prog,
            ax=axes[i, 1],
            title=f"{prog} ordinal residuals",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig




def create_predictions_df_from_oof(
    oof_predictions: pd.DataFrame,
    row_id_col: str,
    true_label_col: str,
    pred_label_col: str
) -> pd.DataFrame:
    """
    Create predictions_df from LOPO out-of-fold predictions
    using row-stable identifiers.
    """

    required = {row_id_col, true_label_col, pred_label_col}
    missing = required - set(oof_predictions.columns)
    if missing:
        raise KeyError(f"Missing required columns in OOF predictions: {missing}")

    predictions_df = (
        oof_predictions
        .loc[:, [row_id_col, true_label_col, pred_label_col]]
        .rename(columns={
            row_id_col: "row_id",
            true_label_col: "y_true",
            pred_label_col: "y_pred"
        })
        .copy()
    )

    # Sanity checks
    if predictions_df.isnull().any().any():
        raise ValueError("Missing values detected in predictions_df")

    if not set(predictions_df["y_true"].unique()).issubset({0, 1, 2}):
        raise ValueError("y_true must be ordinal integers {0,1,2}")

    if not set(predictions_df["y_pred"].unique()).issubset({0, 1, 2}):
        raise ValueError("y_pred must be ordinal integers {0,1,2}")

    return predictions_df


predictions_df = create_predictions_df_from_oof(
    oof_predictions=oof,
    row_id_col="row_id",
    true_label_col="y_true",
    pred_label_col="y_pred"
)


def create_feature_df_from_csv(
    csv_path: str,
    id_col: str,
    timepoint_col: str
) -> pd.DataFrame:
    """
    Create feature_df with explicit sample_id construction.
    """

    feature_df = pd.read_csv(csv_path)

    # ---- construct sample_id ----
    feature_df["row_id"] = (
        feature_df[id_col].astype(str)
        + "_"
        + feature_df[timepoint_col].astype(str)
    )

    # ---- sanity checks ----
    if feature_df["row_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values detected")

    return feature_df


features_df = create_feature_df_from_csv(
    csv_path="/Users/lashandawilliams/Immune_Recovery_Ordinal_Model_Malaria_scRNAseq/01_data/processed/hostscope_features_ordinal_D0_D7_D28.csv",
    id_col="ID",
    timepoint_col="timepoint"
)


# Example mappings
stage_map = {0: "Day0", 1: "Day7", 2: "Day28"}

df_plot = build_df_plot(
    predictions_df=predictions_df,
    features_df=features_df,
    sample_id_col_pred="ID",
    sample_id_col_feat="row_id",  
    true_label_col="y_true",
    pred_label_col="y_pred",
    stage_label_map=stage_map,
)


fig = plot_program_error_panel(
    df_plot,
    program_cols=[
        "comp_Plasma",
        "comp_Platelet",
    ],
)
plt.savefig(f"{RESULTS_DIR}/figures/errors/error_panel1.png")
plt.show()


stage_map = {0: "Day0", 1: "Day7", 2: "Day28"}

df_plot = build_df_plot(
    predictions_df=predictions_df,
    features_df=features_df,
    sample_id_col_pred="ID",
    sample_id_col_feat="row_id",  
    true_label_col="y_true",
    pred_label_col="y_pred",
    stage_label_map=stage_map,
)


fig = plot_program_error_panel(
    df_plot,
    program_cols=[
        "NK_Immune_Checkpoint",
        "CD8_T_Immune_Checkpoint",
        "CD4_T_Immune_Checkpoint"
    ],
)
plt.savefig(f"{RESULTS_DIR}/figures/errors/error_panel2.png")
plt.show()
