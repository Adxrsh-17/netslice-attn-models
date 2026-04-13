# 5G Traffic Forecasting Project

This repository contains a 5G network slice traffic forecasting project with:

- `backend/` for API and inference services
- `frontend/` for the web interface
- `scripts/` for GitHub-ready Python script versions of notebook workflows

## Notebook Conversion

The following notebooks were converted into script files for source control:

- `embb-slice.ipynb` -> `scripts/embb_slice.py`
- `mmtc-slice.ipynb` -> `scripts/mmtc_slice.py`
- `urllc-slice.ipynb` -> `scripts/urllc_slice.py`

## Run Backend

Use PowerShell:

```powershell
cd backend
pip install -r requirements.txt
python main.py
```

## Notes

- Large datasets, trained model binaries, and generated metrics are git-ignored.
- Keep generated outputs local unless you intentionally want to version them.
