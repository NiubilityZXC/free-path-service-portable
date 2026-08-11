# Artifact Policy

The web app mixes source code, compact runtime data, and very large trained model artifacts.
GitHub regular Git storage should keep source and small active files. Large model files need Git LFS, GitHub Releases, or external object storage.

## Keep In Regular Git

- `free_path_web_app.py`
- `unified_free_path_core.py`
- `high_z_free_path_model.py`
- `build_single_point_simulators.py`
- training, prediction, test, and report scripts
- `data_Al.txt`, `data_Be.txt`, `data_Au2.txt`, `data.txt`
- `free_path_model_outputs/unified_free_path_model.npz`
- `free_path_model_outputs/unified_free_path_metrics.json`
- `free_path_model_outputs/unified_standard_training_data.txt`
- `free_path_model_outputs/unified_permanent_benchmark_*.txt`
- `free_path_model_outputs/unified_permanent_benchmark_*.npz`
- `free_path_model_outputs/unified_permanent_extrapolation_*.txt`
- `free_path_model_outputs/unified_permanent_extrapolation_*.npz`
- simulator source files and small runtime executables under `free_path_model_outputs/simulators/`

## Needs LFS Or External Storage

- `free_path_model_outputs/high_z_au_catboost_model.cbm` (~204 MB)
- `free_path_model_outputs/high_z_au_catboost_model_member02.cbm` (~204 MB)
- `free_path_model_outputs/high_z_au_catboost_model_member03.cbm` (~267 MB)
- `free_path_model_outputs/high_z_au_catboost_model_member04.cbm` (~267 MB)
- `/home/user/zpinch10000/ai_xlsx_all_in_training_outputs/best_combined_model_artifact.pkl` (~304 MB)

## Ignore

- multi-GB `.joblib` experiments and backups
- `.bak_*`, `.backup_*`, and `.before_*` files
- temporary model-search directories
- training logs and CatBoost local logs
- compiled simulator object files
