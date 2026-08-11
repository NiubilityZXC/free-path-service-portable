FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN=/usr/local/bin/python \
    FREE_PATH_BUNDLE_ROOT=/app \
    CAPACITOR_APP_DIR=/app/pulse_capacitor_online_eval \
    ZPINCH_ROOT=/app/zpinch \
    FREE_PATH_PYTHON=/usr/local/bin/python \
    FLASH_ROOT=/app/FLASH4.8

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-runtime.txt /app/requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-runtime.txt
COPY . /app
RUN chmod +x /app/setup.sh /app/start_all.sh /app/stop_all.sh /app/status.sh /app/pack_archive.sh \
    /app/scripts/wait_http.py /app/scripts/portable_smoke_test.py \
    /app/scripts/flash_env.sh /app/scripts/rebuild_flash_project.py \
    /app/scripts/repair_flash_symlinks.py /app/scripts/regenerate_manifests.sh \
    /app/zpinch/predict_current.py

EXPOSE 8790 8890
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python /app/scripts/wait_http.py http://127.0.0.1:8790/health --timeout 3 --quiet
CMD ["/app/start_all.sh", "--foreground"]
