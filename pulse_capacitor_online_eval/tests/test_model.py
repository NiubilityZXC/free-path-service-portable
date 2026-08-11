import math
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    compare_prediction_models,
    evaluate_dataset,
    gm11,
    holt_damped_forecast,
    infer_failure,
    level_ratio_check,
    linear_trend_forecast,
    list_available_datasets,
    load_strict_deep_report,
    smooth_sequence,
    validate_dataset,
)


class GreyModelTests(unittest.TestCase):
    def test_gm11_tracks_exponential_growth(self):
        values = [10.0 * (1.04**idx) for idx in range(12)]
        result = gm11(values, horizon=40)
        self.assertGreater(result.accuracy_percent, 99.0)
        self.assertGreater(result.forecast[20], values[-1])
        self.assertTrue(math.isfinite(result.a))
        self.assertTrue(math.isfinite(result.b))

    def test_level_ratio_check_has_expected_bounds(self):
        values = [100.0, 99.5, 99.0, 98.5, 98.0, 97.5]
        ratios, bounds, valid = level_ratio_check(values)
        self.assertEqual(len(ratios), len(values) - 1)
        self.assertLess(bounds[0], 1.0)
        self.assertGreater(bounds[1], 1.0)
        self.assertTrue(valid)

    def test_smoothing_preserves_first_value_and_length(self):
        values = [1.0, 3.0, 2.0, 4.0, 3.0]
        smoothed = smooth_sequence(values)
        self.assertEqual(smoothed[0], values[0])
        self.assertEqual(len(smoothed), len(values))
        self.assertLess(smoothed[2], values[1])

    def test_alternative_forecasts_include_requested_future_steps(self):
        values = [5.0 + 0.25 * idx for idx in range(20)]
        linear = linear_trend_forecast(values, horizon=12)
        holt = holt_damped_forecast(values, horizon=12)
        self.assertEqual(len(linear.forecast), 32)
        self.assertEqual(len(holt.forecast), 32)
        self.assertAlmostEqual(linear.forecast[-1], 5.0 + 0.25 * 31, places=8)
        self.assertGreater(holt.forecast[-1], values[-1])


class EvaluationTests(unittest.TestCase):
    def test_strict_waveform_defaults_follow_confirmed_aging_directions(self):
        self.assertEqual(infer_failure("VoltageMaxRaw", 100.0, {}), ("increase", 106.98))
        self.assertEqual(
            infer_failure("VoltageFirstZeroTimeUs", 1.0, {}),
            ("decrease", 0.7668),
        )

    def test_real_csv_evaluation_returns_summary(self):
        report = evaluate_dataset(
            {
                "dataset": "cap1-1.csv",
                "columns": ["C1kHz", "ESR1kHz"],
                "window": 80,
                "horizon": 600,
                "capLossPercent": 5,
                "esrGrowthPercent": 100,
                "accuracyThreshold": 90,
                "lifeRatioThreshold": 90,
            }
        )
        self.assertEqual(report["dataset"], "cap1-1.csv")
        self.assertEqual(len(report["results"]), 2)
        self.assertIn("summary", report)
        self.assertGreater(report["summary"]["finalAccuracyPercent"], 80)
        for item in report["results"]:
            self.assertIn(item["engineeringStatus"], {"正常", "预警"})
            self.assertIn(item["warningStatus"], {"正常", "预警"})

    def test_rolling_validation_compares_forecast_to_future_truth(self):
        report = validate_dataset(
            {
                "dataset": "cap1-1.csv",
                "columns": ["C1kHz", "ESR1kHz"],
                "window": 80,
                "validationHorizon": 20,
                "validationErrorThreshold": 5,
            }
        )
        self.assertEqual(report["window"], 80)
        self.assertEqual(report["horizon"], 20)
        self.assertEqual(len(report["results"]), 2)
        self.assertGreater(report["summary"]["finalMapePercent"], 0)
        self.assertLess(report["summary"]["finalMapePercent"], 10)
        for item in report["results"]:
            self.assertGreater(item["metrics"]["runs"], 0)
            first_point = item["points"][0]
            self.assertEqual(first_point["trainStartIndex"], 1)
            self.assertEqual(first_point["trainEndIndex"], 80)
            self.assertEqual(first_point["targetIndex"], 100)
            self.assertIn(item["status"], {"通过", "误差偏高"})

    def test_rolling_validation_range_and_stride_are_configurable(self):
        report = validate_dataset(
            {
                "dataset": "cap1-1.csv",
                "columns": ["C1kHz"],
                "window": 80,
                "validationHorizon": 10,
                "validationStartIndex": 5,
                "validationEndIndex": 140,
                "validationStride": 2,
            }
        )
        self.assertEqual(report["validationStartIndex"], 5)
        self.assertEqual(report["validationEndIndex"], 140)
        self.assertEqual(report["validationStride"], 2)
        points = report["results"][0]["points"]
        self.assertEqual(points[0]["trainStartIndex"], 5)
        self.assertEqual(points[0]["trainEndIndex"], 84)
        self.assertEqual(points[0]["targetIndex"], 94)
        self.assertLessEqual(points[-1]["targetIndex"], 140)
        if len(points) > 1:
            self.assertEqual(points[1]["trainStartIndex"] - points[0]["trainStartIndex"], 2)

    def test_time_based_horizons_are_converted_to_steps(self):
        report = evaluate_dataset(
            {
                "dataset": "cap1-1.csv",
                "columns": ["C1kHz"],
                "window": 80,
                "predictionTime": 12.5,
            }
        )
        expected = math.ceil(12.5 / report["period"])
        self.assertEqual(report["horizon"], expected)
        self.assertEqual(report["predictionTime"], 12.5)

        validation = validate_dataset(
            {
                "dataset": "cap1-1.csv",
                "columns": ["C1kHz"],
                "window": 80,
                "validationTime": 12.5,
                "validationStartIndex": 1,
                "validationEndIndex": 120,
            }
        )
        self.assertEqual(validation["horizon"], expected)
        self.assertEqual(validation["validationTime"], 12.5)

    def test_patent_waveform_features_dataset_runs(self):
        datasets = list_available_datasets()["datasets"]
        converted = next(
            item for item in datasets if item["name"] == "ceshishuju_patent_features.csv"
        )
        self.assertEqual(converted["rowCount"], 606)
        self.assertEqual(
            converted["defaultColumns"],
            [
                "VoltageMaxRaw",
                "VoltageFirstZeroTimeUs",
                "VoltageReversePeakCoefficient",
                "VoltageMinAbsRaw",
                "DischargePeriodSec",
            ],
        )

        report = evaluate_dataset(
            {
                "dataset": "ceshishuju_patent_features.csv",
                "columns": ["VoltageMaxRaw", "VoltageFirstZeroTimeUs"],
                "window": 80,
                "horizon": 200,
                "accuracyThreshold": 90,
                "lifeRatioThreshold": 90,
            }
        )
        self.assertEqual(report["source"]["name"], "用户提供 OWON 示波器脉冲放电波形")
        self.assertGreater(report["summary"]["finalAccuracyPercent"], 90)
        self.assertIn("电流换算", "".join(report["uncertainties"]))
        by_column = {item["column"]: item for item in report["results"]}
        self.assertAlmostEqual(by_column["VoltageMaxRaw"]["criticalValue"], 31768.7808)
        self.assertAlmostEqual(
            by_column["VoltageFirstZeroTimeUs"]["criticalValue"], 0.2407752
        )

        validation = validate_dataset(
            {
                "dataset": "ceshishuju_patent_features.csv",
                "columns": ["VoltageMaxRaw", "VoltageFirstZeroTimeUs"],
                "window": 80,
                "validationHorizon": 20,
                "validationErrorThreshold": 10,
            }
        )
        self.assertEqual(validation["summary"]["status"], "通过")

    def test_parameter_rules_override_shared_thresholds(self):
        report = evaluate_dataset(
            {
                "dataset": "ceshishuju_patent_features.csv",
                "columns": ["VoltageMaxRaw", "VoltageFirstZeroTimeUs"],
                "window": 80,
                "horizon": 200,
                "parameterRules": {
                    "VoltageMaxRaw": {"direction": "decrease", "percent": 7.5},
                    "VoltageFirstZeroTimeUs": {"direction": "increase", "percent": 12.5},
                },
            }
        )
        by_column = {item["column"]: item for item in report["results"]}
        self.assertEqual(by_column["VoltageMaxRaw"]["direction"], "decrease")
        self.assertEqual(by_column["VoltageFirstZeroTimeUs"]["direction"], "increase")
        self.assertEqual(by_column["VoltageMaxRaw"]["baselineRow"], 1)
        self.assertEqual(by_column["VoltageFirstZeroTimeUs"]["baselineRow"], 1)
        self.assertEqual(by_column["VoltageMaxRaw"]["baselineValue"], 29696.0)
        self.assertEqual(by_column["VoltageFirstZeroTimeUs"]["baselineValue"], 0.314)
        self.assertAlmostEqual(
            by_column["VoltageMaxRaw"]["criticalValue"],
            by_column["VoltageMaxRaw"]["baselineValue"] * 0.925,
        )
        self.assertAlmostEqual(
            by_column["VoltageFirstZeroTimeUs"]["criticalValue"],
            by_column["VoltageFirstZeroTimeUs"]["baselineValue"] * 1.125,
        )

    def test_model_comparison_selects_lowest_rolling_mape(self):
        report = compare_prediction_models(
            {
                "dataset": "cap1-1.csv",
                "columns": ["C1kHz"],
                "window": 80,
                "horizon": 120,
                "validationHorizon": 20,
                "validationStartIndex": 1,
                "validationEndIndex": 180,
                "validationStride": 5,
            }
        )
        self.assertEqual(len(report["methods"]), 5)
        item = report["results"][0]
        candidate_mapes = [
            candidate["metrics"]["mapePercent"]
            for candidate in item["candidateMetrics"]
        ]
        self.assertAlmostEqual(item["bestMetrics"]["mapePercent"], min(candidate_mapes))
        self.assertEqual(len(item["forecast"]), 80 + 120)
        self.assertEqual(
            item["validationPoints"][-1]["targetIndex"],
            report["validationEndIndex"],
        )

    def test_strict_deep_report_uses_chronological_holdout(self):
        report = load_strict_deep_report()
        self.assertEqual(report["dataset"], "ceshishuju_patent_features.csv")
        self.assertEqual(len(report["features"]), 5)
        self.assertLess(report["split"]["trainEndRow"], report["split"]["validationStartRow"])
        self.assertLess(
            report["split"]["validationEndRow"], report["split"]["testStartRow"]
        )
        deep_models = [
            model for model in report["models"] if model["modelType"] == "deep_learning"
        ]
        selected = next(
            model for model in deep_models if model["id"] == report["selectedDeepModel"]
        )
        self.assertEqual(
            selected["validationMetrics"]["meanNmaePercent"],
            min(model["validationMetrics"]["meanNmaePercent"] for model in deep_models),
        )
        self.assertEqual(
            selected["testMetrics"]["samples"], report["split"]["testSamples"]
        )
        best_test = min(
            report["models"],
            key=lambda model: model["testMetrics"]["meanNmaePercent"],
        )
        selected_model = min(
            report["models"],
            key=lambda model: model["validationMetrics"]["meanNmaePercent"],
        )
        self.assertEqual(report["selectedModel"], selected_model["id"])
        self.assertEqual(report["bestTestModel"], best_test["id"])
        self.assertIn("gm11", {model["id"] for model in report["models"]})
        self.assertIn("quantized_hybrid", {model["id"] for model in report["models"]})
        self.assertEqual(report["selectedModel"], report["lookbackSearch"]["bestModelId"])
        self.assertEqual(report["lookback"], report["lookbackSearch"]["bestLookback"])
        self.assertGreaterEqual(len(report["lookbackSearch"]["candidates"]), 2)
        per_model_search = {
            item["modelId"]: item for item in report["lookbackSearch"]["perModel"]
        }
        for model in report["models"]:
            self.assertEqual(model["lookback"], per_model_search[model["id"]]["bestLookback"])
        focus_candidates = report["lookbackSearch"]["candidates"]
        self.assertTrue(
            all(item["focusModelId"] == report["selectedModel"] for item in focus_candidates)
        )
        minimum_validation = min(
            item["validationNmaePercent"] for item in focus_candidates
        )
        tied_windows = sorted(
            item["lookback"]
            for item in focus_candidates
            if abs(item["validationNmaePercent"] - minimum_validation) <= 1e-6
        )
        self.assertEqual(report["lookback"], tied_windows[len(tied_windows) // 2])
        self.assertEqual(len(report["modelTestSeries"]), len(report["models"]))
        for model_series in report["modelTestSeries"]:
            self.assertEqual(len(model_series["series"]), len(report["features"]))
            self.assertEqual(
                len(model_series["series"][0]["points"]),
                report["split"]["testSamples"],
            )
            self.assertIn("actual", model_series["series"][0]["points"][0])
            self.assertIn("predicted", model_series["series"][0]["points"][0])
        life_forecast = report["lifeForecast"]
        self.assertEqual(life_forecast["modelId"], report["selectedModel"])
        self.assertEqual(life_forecast["currentRow"], report["rowCount"])
        self.assertEqual(life_forecast["healthyBaselineRows"], 1)
        self.assertEqual(life_forecast["failureBaselineValues"]["VoltageMaxRaw"], 29696.0)
        self.assertEqual(
            life_forecast["failureBaselineValues"]["VoltageFirstZeroTimeUs"],
            0.314,
        )
        self.assertEqual(life_forecast["baselineRowsByFeature"]["DischargePeriodSec"], 2)
        self.assertEqual(life_forecast["failureBaselineValues"]["DischargePeriodSec"], 5.968)
        self.assertFalse(
            any("最初 32 次健康阶段中位数" in note for note in report["notes"])
        )
        self.assertTrue(
            any("原始数据第1次放电" in note for note in report["notes"])
        )
        calibration = life_forecast["failureCalibration"]
        self.assertEqual(calibration["observedRows"], 606)
        self.assertEqual(calibration["additionalFailureSteps"], 1500)
        self.assertEqual(calibration["totalFailureRow"], 2106)
        self.assertAlmostEqual(
            calibration["rules"]["VoltageMaxRaw"]["thresholdValue"], 31768.7808
        )
        self.assertAlmostEqual(
            calibration["rules"]["VoltageFirstZeroTimeUs"]["thresholdValue"],
            0.2407752,
        )
        self.assertEqual(len(life_forecast["series"]), len(report["features"]))
        for feature_series in life_forecast["series"]:
            self.assertEqual(len(feature_series["observedValues"]), report["lookback"])
            self.assertEqual(
                len(feature_series["values"]), life_forecast["maxFutureSteps"]
            )
            self.assertTrue(all(value >= 0 for value in feature_series["values"]))
            self.assertTrue(all(math.isfinite(value) for value in feature_series["values"]))
        self.assertTrue(math.isfinite(report["summary"]["testNmaePercent"]))


if __name__ == "__main__":
    unittest.main()
