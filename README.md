# Student Depression Prediction Using Machine Learning

SENG 352 Data Analysis term project — a fully reproducible, end-to-end binary classification pipeline that predicts student depression risk from academic, lifestyle, demographic, and mental-health survey features.

> **Disclaimer:** This model is not a clinical diagnostic tool. It is a classroom machine-learning project and should not be used for real clinical decisions.

---

## Dataset

| | |
|---|---|
| **Name** | Student Depression Dataset |
| **Source** | [Kaggle: hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) |
| **Raw size** | 27,901 rows × 18 columns |
| **Clean size** | 27,859 rows × 18 columns |
| **Target** | `Depression` — binary (0 = No Depression, 1 = Depression) |
| **Class balance** | 0: %41.4 / 1: %58.6 |

Place the raw CSV at:
```
data/Student Depression Dataset.csv
```

---

## Project Structure

```
student-depression-prediction/
├── data/
│   ├── Student Depression Dataset.csv   # raw data (not committed)
│   └── cleaned.csv                      # output of DQA step (not committed)
├── notebooks/
│   └── student_depression_colab.ipynb  # main notebook delivery (Colab-ready)
├── src/
│   ├── dqa.py              # Data Quality Assessment
│   ├── eda.py              # Exploratory Data Analysis
│   ├── features.py         # Feature engineering + preprocessing pipeline
│   ├── train.py            # Model training + MLflow tracking
│   ├── evaluate.py         # Evaluation + error analysis
│   ├── shap_analysis.py    # SHAP explainability
│   ├── lime_analysis.py    # LIME explainability
│   ├── fairness.py         # Subgroup & fairness evaluation
│   ├── significance.py     # Statistical significance testing
│   ├── ensemble.py         # Stacking ensemble (BONUS)
│   ├── ann_model.py        # Artificial Neural Network (MLP)
│   ├── bayesian_opt.py     # Bayesian hyperparameter optimization
│   ├── embedding_viz.py    # UMAP & t-SNE visualization
│   ├── seed_sensitivity.py # Seed sensitivity analysis
│   └── counterfactual.py   # Counterfactual explanations
├── models/
│   ├── best_model.pkl        # LogisticRegression_Tuned
│   ├── svm_model.pkl
│   ├── ann_model.pkl
│   ├── stacking_model.pkl
│   └── bayesopt_model.pkl
├── reports/
│   ├── figures/              # All saved plots (not committed)
│   ├── analysis_log.md       # Step-by-step findings & decisions
│   └── model_card.md         # Model card (Mitchell et al. 2019)
├── main.py                   # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### Full pipeline (recommended)
```bash
python main.py
```

Runs all steps sequentially:
1. Data Quality Assessment → `data/cleaned.csv`
2. EDA → `reports/figures/`
3. Feature Engineering → 43 → 15 features via RFE
4. Model Training → MLflow experiment `student-depression-prediction`
5. Evaluation → confusion matrix, ROC, PR curves
6. SHAP & LIME explainability
7. Subgroup/fairness evaluation
8. Embedding visualization (UMAP + t-SNE)
9. Statistical significance testing
10. Counterfactual explanations

### Notebook-first run (recommended for presentation/submission)
```bash
# open notebooks/student_depression_colab.ipynb in Google Colab
```

The notebook includes:
- full analysis narrative (DQA -> EDA -> modeling -> explainability -> fairness)
- checklist evidence section for SENG 352 evaluation
- Gradio demo section for interactive risk analysis

### View MLflow experiment UI
```bash
mlflow ui
# open http://localhost:5000
```

---

## Model Results

| Model | CV F1-macro | CV ROC-AUC | Notes |
|---|---|---|---|
| **LogisticRegression_Tuned** ✅ | **0.8397** | **0.9192** | Best model |
| SVM_RBF_Tuned | 0.8397 | 0.9190 | Statistically equivalent to LR |
| LightGBM | 0.8369 | 0.9140 | |
| RandomForest | 0.8304 | 0.9102 | |
| ANN (MLP) | 0.8388 | 0.9183 | (128, 64, 32) hidden layers |
| XGBoost | 0.8233 | 0.9010 | |
| Stacking Ensemble | 0.8395 | 0.9197 | LR+RF+XGB+LGB |
| DecisionTree | 0.8156 | 0.8946 | |

### Test Set (held-out, n=5,572)

| Metric | Value |
|---|---|
| F1-macro | **0.8353** |
| ROC-AUC | **0.9208** |
| Average Precision | **0.9401** |
| Accuracy | **0.84** |
| Bootstrap 95% CI (F1) | [0.8288, 0.8475] |

---

## Top Features (SHAP)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Have you ever had suicidal thoughts? | 1.079 |
| 2 | Academic Pressure | 0.893 |
| 3 | Financial Stress | 0.659 |
| 4 | Dietary Habits (Healthy) | 0.435 |
| 5 | Work/Study Hours | 0.355 |

---

## Key Design Decisions

- **No data leakage** — preprocessing fitted on train set only
- **Class imbalance** — `class_weight='balanced'` / `scale_pos_weight`
- **Feature selection** — RFE with LogisticRegression (43 → 15 features)
- **Experiment tracking** — MLflow (local), experiment: `student-depression-prediction`
- **Random seed** — 42 throughout (seed sensitivity std=0.0005, model is stable)
- **Statistical validation** — McNemar + paired t-test confirm LR ≈ SVM (p>0.05)
- **Fairness** — Subgroup evaluation flags low Academic Pressure group (recall %58.6)
