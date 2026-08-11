#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_learning.strict_waveform_models import (  # noqa: E402
    LOOKBACK_CANDIDATES,
    save_report,
    search_lookbacks_and_evaluate,
    train_and_evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and test deep models on strict waveform features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "ceshishuju_patent_features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "models" / "strict_deep_report.json",
    )
    parser.add_argument("--lookback", type=int)
    parser.add_argument(
        "--lookbacks",
        default=",".join(str(value) for value in LOOKBACK_CANDIDATES),
        help="comma-separated lookback candidates; ignored when --lookback is set",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--life-forecast-steps", type=int, default=5000)
    args = parser.parse_args()

    if args.lookback is not None:
        report = train_and_evaluate(
            args.input,
            lookback=args.lookback,
            horizon=args.horizon,
            max_epochs=args.max_epochs,
            patience=args.patience,
            future_steps=args.life_forecast_steps,
        )
    else:
        lookbacks = [int(value.strip()) for value in args.lookbacks.split(",") if value.strip()]
        report = search_lookbacks_and_evaluate(
            args.input,
            lookbacks=lookbacks,
            horizon=args.horizon,
            max_epochs=args.max_epochs,
            patience=args.patience,
            future_steps=args.life_forecast_steps,
        )
    save_report(report, args.output)
    print(f"report: {args.output}")
    print(f"selected model: {report.get('selectedModelName', report['selectedDeepModelName'])}")
    print(f"best test model: {report['bestTestModelName']}")
    print(f"best lookback: {report['lookback']}")
    print(f"test NMAE: {report['summary']['testNmaePercent']:.4f}%")
    print(f"test normalized accuracy: {report['summary']['testNormalizedAccuracyPercent']:.4f}%")
    print(f"training seconds: {report['environment']['trainingSeconds']:.2f}")


if __name__ == "__main__":
    main()
