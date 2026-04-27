# Student Depression Prediction Using Machine Learning: Final Report Notes

## 1. Project Objective

The objective is to build a reproducible binary classification machine learning pipeline that predicts whether a student is likely to report depression based on academic, lifestyle, demographic, and related survey variables.

## 2. Dataset Description

The dataset is the Student Depression Dataset from Kaggle (`hopesb/student-depression-dataset`). It contains approximately 27,901 records and 18 columns. The target variable is `Depression`, where `0` represents no depression and `1` represents depression.

## 3. Data Quality Assessment

The notebook verifies dataset shape, column names, data types, summary statistics, the existence of the `Depression` target, and binary target values. The loaded dataset has 27,901 rows and 18 columns. Missing values are checked for every column, with specific attention to `Financial Stress`. `Financial Stress` has 3 missing values, so those rows are dropped.

Duplicate rows, `CGPA = 0`, high ages such as 59, and IQR-based numeric outliers are inspected. The dataset has 0 duplicate rows, 9 rows with `CGPA = 0`, and 1 row with age 59 or higher. Outliers are not automatically removed because unusual values may reflect valid student circumstances or survey coding. Rare categories such as `Sleep Duration = Others` and `Dietary Habits = Others` are kept and handled through one-hot encoding because they are valid survey responses and removing them would discard real observations.

`Work Pressure` and `Job Satisfaction` are inspected for near-zero variance. `Work Pressure = 0` for about 99.99% of rows and `Job Satisfaction = 0` for about 99.97% of rows, so they are dropped from modeling.

## 4. EDA Summary

The analysis saves target distribution plots, numeric histograms, numeric boxplots, depression-grouped boxplots, a correlation heatmap, categorical bar charts, categorical distributions by depression, and a selected-feature pairplot. The EDA focuses on academic pressure, financial stress, sleep duration, work/study hours, CGPA, suicidal thoughts, and family history of mental illness.

## 5. Preprocessing

The preprocessing workflow uses `sklearn` `Pipeline` and `ColumnTransformer`. Numeric features are scaled using `StandardScaler`, and nominal categorical variables are one-hot encoded with `handle_unknown="ignore"`. The train-test split is stratified, uses an 80/20 split, and uses `random_state = 42`.

`Sleep Duration` is not ordinal encoded as a primary categorical feature because the dataset includes an `Others` response and the categories are survey intervals rather than exact measurements. For the engineered interaction feature only, sleep categories are converted to approximate numeric hour estimates.

## 6. Feature Engineering

The pipeline tests engineered features including `Academic_Pressure_x_Sleep`, `Stress_Load`, `Academic_Lifestyle_Load`, `Sleep_Hours_Estimate`, and `CGPA_Category`. Model performance is compared with and without engineered features using stratified 5-fold cross-validation.

## 7. Models Used

The notebook compares Logistic Regression, class-balanced Logistic Regression, class-balanced Decision Tree, class-balanced Random Forest, MLPClassifier, and XGBoost when the package is importable. In the completed local run, XGBoost was skipped because the installed package could not load the macOS OpenMP runtime (`libomp`). Cross-validation records accuracy, precision, recall, F1-score, macro F1-score, and ROC-AUC.

## 8. Evaluation Results

The saved outputs are `model_comparison_results.csv`, `best_parameters.csv`, and `final_test_metrics.csv`. In the completed run, the final held-out test metrics are accuracy = 0.8423, precision = 0.8793, recall = 0.8470, F1 = 0.8628, macro F1 = 0.8387, and ROC-AUC = 0.9189. The final evaluation includes a classification report, confusion matrix, ROC curve, precision-recall curve, feature importance, and misclassification analysis.

## 9. Best Model and Justification

The best model is selected primarily using macro F1-score during cross-validation, with ROC-AUC considered as a secondary ranking metric. Macro F1 is appropriate because it gives both classes meaningful weight and is less forgiving when one class performs poorly. The tuned class-balanced Logistic Regression model was selected, with `C = 0.01`, `penalty = l2`, and `solver = lbfgs`.

## 10. Error Analysis

False positives and false negatives are inspected separately. A false negative means a student may be at risk, but the model predicts no depression. In an early-warning setting, false negatives are more serious than many false positives because a missed at-risk student may not receive timely support.

## 11. Limitations

The dataset is observational and may contain self-reporting bias, sampling bias, survey wording effects, and unmeasured confounders. Model performance on this dataset does not guarantee performance in a different university, country, year, or support-service setting.

## 12. Ethical Considerations

This model is not a clinical diagnostic tool. It can only be considered as a decision-support or early-warning system. Any real-world use would require expert review, privacy protection, informed governance, bias monitoring, and institutional approval.

## 13. Conclusion

The project demonstrates a complete supervised learning workflow: data loading, quality assessment, EDA, preprocessing, feature engineering, model comparison, tuning, held-out evaluation, error analysis, model saving, and ethical reflection.
