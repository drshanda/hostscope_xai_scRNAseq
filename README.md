# HostScope: Interpretable, Program-Level Modeling of Immune Recovery After Malaria

**AI/ML • Bioinformatics • Single-Cell RNA-seq • Interpretable ML • Computational Immunology**

---

## 1. Overview

This project presents an end-to-end **interpretable machine learning pipeline for longitudinal single-cell RNA-seq data**, integrating:

* Single-cell immune profiling (scRNA-seq)
* Program-level feature engineering (functional immune gene sets)
* Patient-aware supervised machine learning
* Explainable AI (SHAP)
* Error and uncertainty analysis
* MLOps tooling (MLflow, DVC, Docker, AWS S3)

The goal of HostScope is **not to maximize predictive accuracy**, but to model **immune recovery after malaria as an ordered, biologically ambiguous process** while enforcing strict patient-level evaluation. Rather than relying on gene-level features or cell-type composition alone, HostScope represents immune state using **functional immune programs** that capture coordinated biological activity across lineages.

---

## 3. Project Objectives

HostScope aims to:

* Model immune recovery at the patient level, not the cell level.
* Represent immune state using functional immune programs.
* Treat recovery as an ordinal, continuous process.
* Use interpretable ML to understand both predictions and uncertainty.
* Demonstrate how biological ambiguity manifests in model errors.
---

## 3. Repository Structure

```
01_data/
└── processed/
    ├── hostscope_features_all.csv
    ├── hostscope_features_infected.csv
    ├── hostscope_features_ordinal_D0_D7_D28.csv
    ├── hostscope_features_pairwise_D0_vs_D28.csv
02_src/
├── Python/
│   ├── modeling_lopo.py
│   ├── shap_analysis.py
│   ├── error_analysis.py
│   └── permutation_analysis.py
│
└── R/
    ├── data_processing.R
    ├── aggregation_patient_level.R
    ├── composition_analysis.R
    └── nes_analysis.R
03_results/
├── figures/
│   ├── composition/
│   ├── modeling/
│   ├── shap/
│   ├── errors/
│   ├── permutation/
│   └── functional_analysis_plots/
│
└── tables/
docker/
mlruns/

```

---

## 4. Methods

* **Data Source**: `annotated_Sabah_data_21Oct2022.rds` can be downloaded here: https://zenodo.org/records/6973241

### 4.1 Immune Program Feature Engineering

Immune state is represented using **curated functional immune programs**—gene sets capturing coordinated biological processes such as:

* Immune checkpoint activity
* Cytotoxic effector function
* Interferon signaling
* Plasma cell biosynthesis
* Innate inflammatory sensing

Program scores are computed at the **single-cell level** and aggregated to the **patient × timepoint** level. Programs are lineage-restricted where biologically appropriate.

### Cell-Type Composition Features

Cell-type composition (lineage proportions per patient/timepoint) is computed separately and **explicitly isolated from program features**, enabling direct comparison between:

* abundance-driven signal
* function-driven signal

This separation prevents conflation of composition effects with functional immune remodeling.

---

### 4.2 Machine Learning Pipeline (Python)

Models trained:

* **Pairwise Logistic Regression**
  * Distinguishes acute infection (Day0) from recovery (Day28)
  * Serves as a comparator for simpler recovery framing

* **Ordinal Logistic Regression**
  * Predicts ordered recovery stage: Day0 < Day7 < Day28
  * Captures recovery as a trajectory rather than a categorical switch

| Model             | Task                | Balanced Accuracy | AUC   | MAE   |
| ----------------- | ------------------- | ----------------- | ----- | ----- |
| Pairwise Logistic | Day0 vs Day28       | 0.70              | 0.667 | 0.273 |
| Ordinal Logistic  | Day0 < Day7 < Day28 | 0.70              | 0.860 | 0.294 |


* Performance is intentionally modest but stable, reflecting true inter-patient heterogeneity.
* Ordinal MAE ≈ 0.29 (on a 0–2 scale) indicates predictions are typically within one-third of a stage.
* Errors are overwhelmingly adjacent-stage, with no catastrophic Day0 ↔ Day28 swaps.
* All models are evaluated using leave-one-patient-out (LOPO) cross-validation.
---

### 4.3 Interpretability and Error Analysis

SHAP analyses reveal that:
* Composition features (e.g., plasma cells, platelets) capture broad recovery shifts.
* Program-level features dominate within-sample explanatory weight, especially for intermediate states.
* Immune checkpoint programs consistently push predictions toward earlier stages, while humoral programs push toward recovery.
* Day7 often shows mean SHAP ≈ 0, reflecting heterogeneous intermediate biology rather than absence of signal.

Program vs composition reliance
* Paired Wilcoxon test: programs > composition within samples (p = 7.6 × 10⁻⁶)
* Permutation calibration: effect size modest but robust, consistent with correlated immune structure and small n

Misclassification and effect size analysis
* Errors are categorized as correct, adjacent, or extreme
* Adjacent errors dominate and align with transitional immune states
* Program-level immune checkpoint features show moderate Cliff’s δ, distinguishing correct from ambiguous predictions more effectively than composition

Key conclusion:
Model ambiguity reflects overlapping immune states during recovery—not instability or noise.

---

### 4.4 Functional Enrichment Analysis (R)

Lineage-specific pseudobulk differential expression (Day28 vs Day0) is analyzed using GSEA (GO Biological Processes).

Key findings:
* Recovery is dominated by RNA processing, ribosome biogenesis, and translational reconstitution.
* Acute infection is dominated by immune activation, chemotaxis, and inflammatory signaling.
* Intermediate states show overlapping distributions, explaining ordinal ambiguity and SHAP compression.

SHAP and GSEA converge on a coherent mechanistic narrative of immune recovery.
---

## 5. MLOps Components

### 5.1 DVC (Data Version Control)

DVC is used to track and version:

* Processed patient × timepoint feature matrices
* Intermediate analysis artifacts

Artifacts are remote-tracked on AWS S3.

---

### 5.2 AWS S3 Integration

AWS S3 is used for storing:

* Versioned feature matrices
* DVC-tracked artifacts

This ensures reproducibility, centralized storage, and long-term auditability.

---

### 5.3 Docker Containerization

A Docker image standardizes the Python ML environment, including:

* Base image
* Dependency installation
* Serialized trained model
* SHAP-compatible runtime

Benefits include:

* Reproducibility across machines
* Portable deployment
* Future compatibility with AWS SageMaker or Kubernetes

---

### 5.4 MLflow Tracking

MLflow is used to log:

* Model parameters
* LOPO cross-validation metrics
* Final model performance
* SHAP explanation artifacts

The MLflow UI enables experiment comparison and model lineage tracking.

---

## 6. Results Summary

### Machine Learning Results

* Ordinal Balanced accuracy = 0.70 (LOPO)
* Ordinal MAE ≈ 0.29 
* Ordinal AUC = 0.86
* Errors are overwhelmingly **adjacent-stage**
* No catastrophic Day0 ↔ Day28 misclassifications

### Interpretability Results

* Within-sample SHAP reliance favors immune programs over composition
* Directionality aligns with known malaria immunology:

  * checkpoint activity → acute-like
  * plasma biosynthesis → recovery-like

* Day7 shows SHAP compression reflecting biological heterogeneity, not signal loss

### Functional Biology Results

* Recovery dominated by RNA processing, ribosome biogenesis, and translation
* Immune activation resolves asynchronously across lineages
* Functional overlap explains ordinal ambiguity and structured misclassification

---

## 7. Discussion

This project demonstrates how **interpretability-first, patient-aware ML** can extract meaningful biological insight from scRNA-seq data even when predictive accuracy is constrained by biological reality.

Rather than hiding uncertainty behind inflated performance, HostScope explicitly models and interprets ambiguity as biological signal. The convergence of ML explanations, functional enrichment, and structured error analysis supports a coherent view of immune recovery as a continuous, lineage-specific process.

---

## 8. Conclusion

HostScope provides a reproducible, transparent framework for:

* Modeling immune trajectories under uncertainty
* Avoiding common scRNA-seq ML pitfalls
* Interpreting model behavior mechanistically
* Demonstrating principled ML design in biomedical settings

This project is best understood not as a recovery classifier, but as a **framework for reasoning about immune recovery using interpretable machine learning**.

---

## 9. Running the Project

Here is a **clean, accurate rewrite** of that README section using your **actual file paths, folder structure, and Docker layout**, written for clarity and skim-readability.

You can drop this in verbatim.

---

## 9. Running the Project

HostScope can be run either **natively** (Python/R) or via **Docker**. Processed feature matrices are retrieved using DVC.

---

### Fetch Versioned Processed Data

Processed feature matrices are versioned with DVC and stored in Amazon S3.

To retrieve the exact data used for the analysis:

```bash
dvc pull
```

This restores all files in:

```text
01_data/processed/
```

---

### Docker Build (Optional)

Docker configuration is located in the `docker/` directory.

Build the Docker image from the repository root:

```bash
docker build -t hostscope -f docker/Dockerfile .
```

---

### Run Analysis in Docker

Execute the modeling script inside the container (DVC data is pulled automatically at startup):

```bash
docker run --rm \
  -v $(pwd)/mlruns:/app/mlruns \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  hostscope \
  python 02_src/Python/modeling_lopo_mlflow.py
```

MLflow runs can then be viewed locally using the MLflow UI.

---

