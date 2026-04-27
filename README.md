# Student Depression Prediction Using Machine Learning

SENG 352 Data Analysis term project for building a reproducible binary classification pipeline that predicts student depression risk from academic, lifestyle, demographic, and mental-health related survey features.

## Dataset

- **Name:** Student Depression Dataset
- **Source:** [Kaggle: hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- **Expected size:** approximately 27,901 rows and 18 columns
- **Target variable:** `Depression`
- **Target type:** binary classification
  - `0` = No Depression
  - `1` = Depression

Place the CSV file at:

```text
data/raw/Student Depression Dataset.csv
```

The workspace also keeps the original downloaded CSV at `data/Student Depression Dataset.csv`.

## Project Structure

```text
student-depression-prediction/
├── data/
│   ├── raw/Student Depression Dataset.csv
│   └── processed/
├── figures/
├── models/
├── notebooks/
│   └── student_depression_prediction_analysis.ipynb
├── reports/
│   └── final_report_notes.md
├── src/
│   └── project_pipeline.py
├── README.md
└── requirements.txt
```

## How to Run

Use Python 3. A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/student_depression_prediction_analysis.ipynb
```

On macOS, XGBoost may require the OpenMP runtime (`libomp`). If XGBoost cannot be imported, the notebook skips it and still compares the required sklearn models plus `MLPClassifier`.

Run the notebook from top to bottom after restarting the kernel. The notebook uses `random_state = 42` throughout and saves:

- EDA plots to `figures/`
- cleaned data to `data/processed/student_depression_clean.csv`
- model comparison results to `model_comparison_results.csv`
- feature engineering comparison to `feature_engineering_comparison.csv`
- tuning results to `best_parameters.csv`
- final test metrics to `final_test_metrics.csv`
- final model pipeline to `models/final_model.pkl`
- error analysis outputs to `reports/`

## Objective

The project compares multiple supervised machine learning models, evaluates them with stratified 5-fold cross-validation and a held-out test set, and discusses false negatives as a critical risk in depression early-warning settings.

This model is **not** a clinical diagnostic tool. It is only suitable as a classroom machine-learning project or, with substantial expert review and governance, a decision-support prototype.
