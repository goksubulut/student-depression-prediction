# Student Depression Prediction Using Machine Learning: Final Report Notes

## 1. Project Objective

The objective is to build a reproducible binary classification machine learning pipeline that predicts whether a student is likely to report depression based on academic, lifestyle, demographic, and related survey variables.

## 2. Dataset Description

The dataset is the Student Depression Dataset from Kaggle (`hopesb/student-depression-dataset`). It contains approximately 27,901 records and 18 columns. The target variable is `Depression`, where `0` represents no depression and `1` represents depression.

## 3. Data Quality Assessment

The notebook validates schema, target integrity, missingness, duplicates, and outlier-related edge cases before modeling. The raw dataset has 27,901 rows and 18 columns. `Financial Stress` has 3 missing rows; those rows are removed.

After checks, 9 rows with `CGPA = 0` are dropped as likely data-entry artifacts. Rare categories `Sleep Duration = Others` and `Dietary Habits = Others` are also removed to keep encoding consistent and reduce sparse noise in low-frequency bins. The final cleaned dataset has 27,859 rows.

`Work Pressure` and `Job Satisfaction` are near-zero-variance and are excluded from modeling.

## 4. EDA Summary

The analysis saves target distribution plots, numeric histograms, numeric boxplots, depression-grouped boxplots, a correlation heatmap, categorical bar charts, categorical distributions by depression, and a selected-feature pairplot. The EDA focuses on academic pressure, financial stress, sleep duration, work/study hours, CGPA, suicidal thoughts, and family history of mental illness.

## 5. Preprocessing

The preprocessing workflow uses `ColumnTransformer` with stratified 80/20 split (`random_state = 42`). Numeric features are scaled with `MinMaxScaler`. Nominal categorical variables are one-hot encoded with `handle_unknown="ignore"`.

`Sleep Duration` is ordinal-encoded (1-5) and used both as a standalone feature and inside an interaction term.

## 6. Feature Engineering

Core feature engineering includes:
- dropping non-informative columns (`id`, `Profession`, `Work Pressure`, `Job Satisfaction`, `City`)
- binary mapping for suicidal-thoughts and family-history fields
- interaction feature `AP_x_Sleep`
- RFE-based feature selection (43 to 15 features)

## 7. Models Used

The project compares Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM under stratified 5-fold CV. Additional analyses include SVM, ANN, stacking, Bayesian optimization, fairness analysis, significance tests, SHAP, LIME, and counterfactual explanations.

## 8. Evaluation Results

The tuned Logistic Regression model is selected as final. Held-out test metrics:
- Accuracy: 0.84
- F1-macro: 0.8353
- ROC-AUC: 0.9208
- Average Precision: 0.9401

Evaluation artifacts include confusion matrix, ROC/PR curves, calibration, threshold analysis, and detailed FP/FN profiling.

## 9. Best Model and Justification

The best model is selected primarily by macro F1, with ROC-AUC as secondary support. The final model is tuned class-balanced Logistic Regression (`C = 0.1`, `penalty = l2`, `solver = lbfgs`). It is preferred over statistically equivalent alternatives (such as SVM) due to interpretability and deployment simplicity.

## 10. Error Analysis

False positives and false negatives are inspected separately. A false negative means a student may be at risk, but the model predicts no depression. In an early-warning setting, false negatives are more serious than many false positives because a missed at-risk student may not receive timely support.

## 11. Limitations

The dataset is observational and may contain self-reporting bias, sampling bias, survey wording effects, and unmeasured confounders. Model performance on this dataset does not guarantee performance in a different university, country, year, or support-service setting.

## 12. Ethical Considerations

This model is not a clinical diagnostic tool. It can only be considered as a decision-support or early-warning system. Any real-world use would require expert review, privacy protection, informed governance, bias monitoring, and institutional approval.

## 13. Conclusion

The project demonstrates a complete supervised learning workflow: data loading, quality assessment, EDA, preprocessing, feature engineering, model comparison, tuning, held-out evaluation, error analysis, model saving, and ethical reflection.
