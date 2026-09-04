# Telecom Churn Research Prototype

This folder contains the verified Streamlit deployment for the MSc telecom
churn project. It loads a frozen class-weighted Logistic Regression pipeline and
never retrains when the application starts.

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Verify

```bash
pip install pytest
pytest -q
python -m compileall app.py src tests build_deployment.py
```

To regenerate the frozen model from the included dataset:

```bash
python build_deployment.py
```

The build stops if the held-out F1 or confusion matrix does not reproduce the
verified final-notebook values.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder, not the
   enclosing ZIP file.
2. Sign in to Streamlit Community Cloud and select **Create app**.
3. Choose the repository, branch and `app.py` as the entry point.
4. Select Python 3.12 in advanced settings.
5. Deploy and share the generated `*.streamlit.app` URL with the supervisor and
   viva panel.

No secrets, API keys or external database are required.

## Evidence policy

The app reports values from `Final_churn_project.ipynb`. See
`CONSISTENCY_AUDIT.md` for the exact retained values and why the separate
extension figures were excluded.

This is an academic prototype, not a production decision system.
