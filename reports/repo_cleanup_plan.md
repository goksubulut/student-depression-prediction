# Repository Cleanup Categorization

## Keep (Active + Evidence)
- `notebooks/student_depression_colab.ipynb`
- `src/`
- `reports/analysis_log.md`
- `reports/model_card.md`
- `reports/final_report_notes.md`
- `README.md`
- `requirements.txt`
- `main.py`
- `.gitignore`

## Archive (Result Artifacts / Optional in final submission zip)
- `model_comparison_results.csv`
- `final_test_metrics.csv`
- `best_parameters.csv`
- `feature_engineering_comparison.csv`
- `mlflow.db`
- `models/`
- `mlruns/`
- `figures/`

## Delete (Temporary / Non-project noise)
- `__pycache__/`
- `*.pyc`
- `C?UsersASUSAppDataLocalTempchecklist_out.txt`

## Notes
- Keep the notebook as the main delivery artifact.
- Keep `src/` and `reports/` as academic traceability evidence (how and why decisions were made).
- Archive large experiment artifacts if course submission size is constrained.
