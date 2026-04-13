# Kaggle Instructions

## Running the Notebooks

1. Upload the dataset to Kaggle
2. Open each slice notebook (embb-slice.ipynb, mmtc-slice.ipynb, urllc-slice.ipynb)
3. Run all cells in order
4. Download the output files (models, metrics, plots)

## Dataset

The 36 KPI dataset is located in `36kpi_parquet/36kpi/` directory.

## Output Files

Each notebook produces:
- Model files (.h5) in `{slice}/models/`
- Prediction files (.npy) in `{slice}/metrics/`
- Plot images (.png) in `{slice}/plots/`
- Metrics JSON in `{slice}/metrics/`
