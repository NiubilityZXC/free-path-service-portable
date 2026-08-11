"""Client helpers for the local Rosseland free-path web model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "http://127.0.0.1:8790"


class FreePathError(RuntimeError):
    """Raised when the free-path web service is unavailable or rejects a request."""


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise FreePathError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except OSError as exc:
        raise FreePathError(
            f"Cannot reach free-path service at {url}. Start free_path_web_app.py first or pass --url."
        ) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise FreePathError(f"Non-JSON response from {url}: {body[:200]}") from exc


def control(url: str = DEFAULT_URL, update: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    """Get or update FLASH free-path web control state."""

    endpoint = url.rstrip("/") + "/api/flash/control"
    if update is None:
        return _request_json("GET", endpoint, timeout=timeout)
    return _request_json("POST", endpoint, payload=update, timeout=timeout)


def probe(
    z: float,
    rod: float,
    tep: float,
    tgama: float,
    mode: str = "hybrid",
    model_elements: str | list[float] | None = None,
    url: str = DEFAULT_URL,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Predict one free-path point through the web service."""

    payload: dict[str, Any] = {
        "Z": z,
        "rod": rod,
        "tep": tep,
        "tgama": tgama,
        "predictionMode": mode,
    }
    if model_elements is not None:
        payload["modelElements"] = model_elements
    return _request_json("POST", url.rstrip("/") + "/api/predict", payload=payload, timeout=timeout)


def batch(
    text: str,
    mode: str = "hybrid",
    model_elements: str | list[float] | None = None,
    default_z: float | None = None,
    url: str = DEFAULT_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Predict many free-path points through the web service."""

    payload: dict[str, Any] = {"text": text, "predictionMode": mode}
    if model_elements is not None:
        payload["modelElements"] = model_elements
    if default_z is not None:
        payload["defaultZ"] = default_z
    return _request_json("POST", url.rstrip("/") + "/api/predict/batch", payload=payload, timeout=timeout)
