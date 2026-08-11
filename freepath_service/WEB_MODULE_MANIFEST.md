# Web Module Manifest

This directory serves the intranet app at `http://192.168.110.64:8790/`.

## Runtime Entry

- Process: `/home/user/miniconda3/envs/ai/bin/python /home/user/Rosseland_opa_new_hetero/equally/free_path_web_app.py --host 0.0.0.0 --port 8790 --model /home/user/Rosseland_opa_new_hetero/equally/free_path_model_outputs/unified_free_path_model.npz`
- Working directory: `/home/user/Rosseland_opa_new_hetero/equally`
- Health endpoint: `GET /health`

## Page Tabs

- Free-path prediction: single-point prediction, batch prediction, source policy, FLASH free-path control, training import/export, evaluation, and live benchmark.
- FLASH console: project status, project creation, parameter editing, build/run/stop, command execution, log/file viewing, and plot rendering.
- Pulse capacitor lifetime: embeds the app running from `/home/user/pulse_capacitor_online_eval` on port `8890`.
- Z-pinch kinetic energy: calls the model artifact under `/home/user/zpinch10000`.

## API Surface

- `GET /`
- `GET /health`
- `GET /api/datasets`
- `GET /api/flash/control`
- `GET /api/flash/status`
- `GET /api/flash/file`
- `GET /api/flash/plot`
- `GET /api/predict/status`
- `GET /api/train/status`
- `GET /api/training-data/export`
- `POST /api/predict`
- `POST /api/predict/start`
- `POST /api/predict/batch`
- `POST /api/flash/control`
- `POST /api/flash/params`
- `POST /api/flash/run`
- `POST /api/flash/stop`
- `POST /api/flash/command`
- `POST /api/flash/project`
- `POST /api/train/start`
- `POST /api/train/confirm`
- `POST /api/training-data/import`
- `POST /api/evaluate/upload`
- `POST /api/benchmark/file`
- `POST /api/zpinch/predict`

## Local Module Roots

- CLI harness: `/home/user/cli-anything-flash`
- Main free-path web app: `/home/user/Rosseland_opa_new_hetero/equally`
- Pulse capacitor iframe app: `/home/user/pulse_capacitor_online_eval`
- Z-pinch model app assets: `/home/user/zpinch10000`
- FLASH tree used by the web console: `/home/user2/Haigerloch/FLASH4.8`

## Verified Checks

- `GET /health` returned `{"ok": true}`.
- `POST /api/predict` returned a free-path prediction for `Z=13`.
- `GET http://127.0.0.1:8890/` returned the pulse capacitor UI.
- `POST /api/zpinch/predict` returned a RandomForest prediction using the Z-pinch artifact.
