# Student Depression Prediction

Binary classification pipeline to predict depression in students using academic, lifestyle, and demographic features.

**Dataset:** [hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) — 27,901 rows × 18 features.

---

## Project Structure

```
student-depression-prediction/
├── data/
│   ├── Student Depression Dataset.csv   # raw data
│   └── cleaned.csv                      # output of DQA step
├── notebooks/
│   ├── 01_dqa.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── dqa.py         # Data Quality Assessment
│   ├── eda.py         # Exploratory Data Analysis
│   ├── features.py    # Feature engineering pipeline
│   ├── train.py       # Model training + MLflow tracking
│   └── evaluate.py    # Evaluation + error analysis
├── models/
│   └── best_model.pkl
├── reports/
│   └── figures/       # All saved plots
├── requirements.txt
├── README.md
└── main.py            # End-to-end runner
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

This runs all 5 steps sequentially and prints a final summary.

### Step-by-step

```bash
# 1. Data Quality Assessment
python -c "from src.dqa import run_dqa; run_dqa('data/Student Depression Dataset.csv', 'data/cleaned.csv')"

# 2. EDA (saves figures to reports/figures/)
python -c "import pandas as pd; from src.eda import run_eda; run_eda(pd.read_csv('data/cleaned.csv'))"

# 3–5. Via Jupyter notebooks
jupyter notebook notebooks/
```

### View MLflow UI

```bash
mlflow ui
# open http://localhost:5000
```

---

## Model Results

| Model | CV F1-macro | CV ROC-AUC |
|---|---|---|
| Logistic Regression | — | — |
| Decision Tree | — | — |
| Random Forest | — | — |
| XGBoost | — | — |
| LightGBM | — | — |
| Best (Tuned) | — | — |

*Run `python main.py` to populate this table.*

---

## Key Design Decisions

- **No data leakage**: preprocessing fitted on train set only.
- **Class imbalance**: handled via `class_weight='balanced'` / `scale_pos_weight`.
- **Feature selection**: RFE with LogisticRegression selects top 15 features.
- **Tracking**: all experiments logged to local MLflow server.
- **Random seed**: 42 throughout.
