#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Probe FLASH opacity replacement values through the local free-path web API.

The web model expects standard coordinates:
    Z log10(rho[g/cm^3]) log10(Tele[eV]) log10(Trad[eV])

For a pure material cell, the Rosseland transport opacity in FLASH units is
approximately:
    kappa_R[cm^2/g] = 1 / (rho[g/cm^3] * lnu[cm])

and the corresponding inverse mean free path is:
    alpha[1/cm] = rho * kappa_R = 1 / lnu
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request


K_TO_EV = 8.617333262145e-5


def post_predict(url: str, payload: dict[str, float], timeout: float) -> dict:
    api_url = url.rstrip("/") + "/api/predict"
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_flash_control(url: str, timeout: float) -> dict:
    api_url = url.rstrip("/") + "/api/flash/control"
    with urllib.request.urlopen(api_url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FLASH cell quantities to free-path model opacity values."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8790")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--require-flash-enabled",
        action="store_true",
        help="read /api/flash/control and fail unless FLASH model use is enabled",
    )
    parser.add_argument(
        "--reject-model-range-warnings",
        action="store_true",
        help="fail when the selected value comes from model extrapolation outside the training range",
    )
    parser.add_argument(
        "--plain-kappa",
        action="store_true",
        help="print only kappa_R[cm^2/g], for simple Fortran parsing",
    )
    parser.add_argument("--Z", type=float, required=True)
    parser.add_argument("--rho", type=float, help="cell mass density in g/cm^3")
    parser.add_argument("--mass-fraction", type=float, default=1.0)
    parser.add_argument("--tele-k", type=float, help="electron temperature in K")
    parser.add_argument("--trad-k", type=float, help="radiation temperature in K")
    parser.add_argument("--tep-ev", type=float, help="electron temperature in eV")
    parser.add_argument("--tgama-ev", type=float, help="radiation temperature in eV")
    parser.add_argument(
        "--model-coordinates",
        action="store_true",
        help="use --rod/--tep/--tgama directly instead of converting physical units",
    )
    parser.add_argument("--rod", type=float, help="model coordinate log10(rho)")
    parser.add_argument("--tep", type=float, help="model coordinate log10(Tele[eV])")
    parser.add_argument("--tgama", type=float, help="model coordinate log10(Trad[eV])")
    args = parser.parse_args()

    if args.model_coordinates:
        if args.rod is None or args.tep is None or args.tgama is None:
            raise ValueError("--model-coordinates requires --rod --tep --tgama")
        payload = {
            "Z": float(args.Z),
            "rod": float(args.rod),
            "tep": float(args.tep),
            "tgama": float(args.tgama),
        }
        rho_for_kappa = None
    else:
        if args.rho is None:
            raise ValueError("physical conversion requires --rho")
        mass_fraction = positive(float(args.mass_fraction), "mass-fraction")
        rho_for_model = positive(float(args.rho) * mass_fraction, "rho * mass_fraction")

        if args.tep_ev is None:
            if args.tele_k is None:
                raise ValueError("provide --tep-ev or --tele-k")
            tep_ev = positive(float(args.tele_k), "tele-k") * K_TO_EV
        else:
            tep_ev = positive(float(args.tep_ev), "tep-ev")

        if args.tgama_ev is None:
            if args.trad_k is None:
                raise ValueError("provide --tgama-ev or --trad-k")
            tgama_ev = positive(float(args.trad_k), "trad-k") * K_TO_EV
        else:
            tgama_ev = positive(float(args.tgama_ev), "tgama-ev")

        payload = {
            "Z": float(args.Z),
            "rod": math.log10(rho_for_model),
            "tep": math.log10(tep_ev),
            "tgama": math.log10(tgama_ev),
        }
        rho_for_kappa = rho_for_model

    control = None
    if args.require_flash_enabled:
        control = get_flash_control(args.url, args.timeout)
        if not control.get("enabled", False):
            raise RuntimeError("FLASH free-path model control is disabled")
        enabled_elements = [float(value) for value in control.get("modelElements", [])]
        if not any(math.isclose(float(args.Z), value, rel_tol=0.0, abs_tol=1e-8) for value in enabled_elements):
            raise RuntimeError(f"Z={float(args.Z):g} is not enabled for FLASH free-path model use")
        payload["predictionMode"] = control.get("predictionMode", "hybrid")
        payload["modelElements"] = enabled_elements

    prediction = post_predict(args.url, payload, args.timeout)
    if (
        args.reject_model_range_warnings
        and prediction.get("selected_source") == "model"
        and prediction.get("range_warnings")
    ):
        raise RuntimeError("free-path model prediction is outside the training range")
    lnu = positive(float(prediction["lnu"]), "lnu")
    kappa = None
    if rho_for_kappa is not None:
        kappa = 1.0 / (rho_for_kappa * lnu)
        if args.plain_kappa:
            print(f"{kappa:.17e}")
            return
    elif args.plain_kappa:
        raise ValueError("--plain-kappa requires physical conversion with --rho")

    output = {
        "request": {
            "url": args.url.rstrip("/") + "/api/predict",
            "payload": payload,
        },
        "prediction": {
            "lnu_cm": lnu,
            "log10_lnu": float(prediction["log10_lnu"]),
            "mode": prediction.get("mode"),
            "range_warnings": prediction.get("range_warnings", []),
            "model_route": prediction.get("model_route", {}),
        },
        "opacity": {
            "alpha_cm_inv": 1.0 / lnu,
        },
    }
    if control is not None:
        output["flash_control"] = control
    if rho_for_kappa is not None:
        output["opacity"]["rho_for_kappa_g_cm3"] = rho_for_kappa
        output["opacity"]["kappa_cm2_g"] = kappa

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
