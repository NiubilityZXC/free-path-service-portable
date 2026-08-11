# Artifact Policy

This module is used by the `POST /api/zpinch/predict` endpoint in the free-path web app.

## Runtime Artifact

- `ai_xlsx_all_in_training_outputs/best_combined_model_artifact.pkl` (~304 MB)

The file is required for the live prediction endpoint, but it is too large for regular GitHub Git storage.
Store it with Git LFS, a GitHub Release asset, or external object storage.

## Keep In Regular Git

- training and inference scripts
- `README.md`
- compact JSON and text summaries
- the source Excel workbook if repository policy allows it

## Ignore

- large `.pkl` or `.joblib` trained model files unless Git LFS is installed and configured
- local training logs and cache directories
