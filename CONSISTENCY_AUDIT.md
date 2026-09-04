# Streamlit consistency audit

The deployment package uses the executed outputs in `Final_churn_project.ipynb`
as its numerical source of truth.

## Verified and retained

- Dataset: 7,043 rows and 21 raw columns
- Model matrix: 36 predictors
- Development/test split: 5,634 / 1,409 rows
- Deployed model: class-weighted Logistic Regression, `C=0.01`, `lbfgs`
- Development-selected threshold: 0.56
- Development threshold results: precision 0.5561, recall 0.7458, F1 0.6371
- Held-out results at 0.56: precision 0.5401, recall 0.7380, F1 0.6237,
  ROC-AUC 0.8400, PR-AUC 0.6278, MCC 0.4693
- Held-out confusion matrix: TN 800, FP 235, FN 98, TP 276
- Cost-sensitivity thresholds: 0.53 at 3:1, 0.34 at 5:1, 0.26 at 10:1

The model artefact is rebuilt from the supplied `telcos.csv` and is accepted
only if it reproduces held-out F1 0.6237288135593221 and confusion matrix
`[800, 235, 98, 276]`.

## Excluded as incompatible

`extension_results(2).json`, `threshold_sweep(2).png`, `cost_curve(2).png` and
`cv_boxplot(2).png` belong to a different experimental run. They report a
0.43 F1 threshold and different cross-validation and cost-optimal values. They
must not be displayed beside the final notebook results because doing so would
create contradictory evidence.

The older Streamlit metadata also contained Random Forest and MLP values that
did not match the executed final notebook. Those values were replaced with the
final notebook outputs.

## Methodological caveat

The notebook performs dummy encoding before the stratified split. The encoding
does not use the target, but the report should not claim that every preprocessing
step was fitted inside cross-validation. A stricter future version should fit a
`OneHotEncoder(handle_unknown="ignore")` on development rows only.
