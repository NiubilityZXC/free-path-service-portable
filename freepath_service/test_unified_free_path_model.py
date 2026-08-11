#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import train_unified_free_path_model as trainer
from build_single_point_simulators import discover_data_elements
from free_path_web_app import FreePathWebApp
from unified_free_path_core import (
    ACTIVE_UNIFIED_ARTIFACT,
    ACTIVE_UNIFIED_METRICS,
    BENCHMARK_DATA_PATH,
    BENCHMARK_KEYS_PATH,
    EXTRAPOLATION_DATA_PATH,
    EXTRAPOLATION_KEYS_PATH,
    STANDARD_INPUT_COLUMNS,
    STANDARD_TRAINING_COLUMNS,
    ROOT,
    UnifiedModel,
)


class UnifiedFreePathTests(unittest.TestCase):
    def test_permanent_benchmark_exists(self):
        self.assertTrue(BENCHMARK_KEYS_PATH.exists())
        self.assertTrue(BENCHMARK_DATA_PATH.exists())
        self.assertTrue(EXTRAPOLATION_KEYS_PATH.exists())
        self.assertTrue(EXTRAPOLATION_DATA_PATH.exists())
        keys = BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines()
        extrap_keys = EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(keys), 20000)
        self.assertGreater(len(extrap_keys), 20000)

    def test_model_metadata_records_benchmark(self):
        metadata = json.loads(ACTIVE_UNIFIED_METRICS.read_text(encoding="utf-8"))
        self.assertEqual(metadata["artifact_type"], "unified_free_path_local_regression")
        self.assertEqual(metadata["input_features"], STANDARD_INPUT_COLUMNS)
        self.assertEqual(metadata["standard_training_format"], " ".join(STANDARD_TRAINING_COLUMNS))
        self.assertGreater(metadata["n_benchmark"], 20000)
        self.assertGreater(metadata["n_extrapolation_benchmark"], 20000)
        self.assertLess(metadata["benchmark_metrics"]["overall"]["smape_percent"], 5.0)
        self.assertIn("extrapolation_benchmark_metrics", metadata)
        self.assertIn("Permanent benchmark", metadata["holdout_policy"])
        for item in metadata["training_files"].values():
            self.assertEqual(
                item["total_rows"],
                item["training_rows"] + item["benchmark_excluded"] + item["extrapolation_excluded"],
            )

    def test_existing_holdout_keys_are_reused_without_adding_new_rows(self):
        original_paths = {
            "OUTPUT_DIR": trainer.OUTPUT_DIR,
            "BENCHMARK_KEYS_PATH": trainer.BENCHMARK_KEYS_PATH,
            "BENCHMARK_DATA_PATH": trainer.BENCHMARK_DATA_PATH,
            "EXTRAPOLATION_KEYS_PATH": trainer.EXTRAPOLATION_KEYS_PATH,
            "EXTRAPOLATION_DATA_PATH": trainer.EXTRAPOLATION_DATA_PATH,
        }
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            trainer.OUTPUT_DIR = tmp
            trainer.BENCHMARK_KEYS_PATH = tmp / "benchmark_keys.txt"
            trainer.BENCHMARK_DATA_PATH = tmp / "benchmark_data.npz"
            trainer.EXTRAPOLATION_KEYS_PATH = tmp / "extrapolation_keys.txt"
            trainer.EXTRAPOLATION_DATA_PATH = tmp / "extrapolation_data.npz"
            try:
                trainer.BENCHMARK_KEYS_PATH.write_text("locked-random-key\n", encoding="utf-8")
                trainer.EXTRAPOLATION_KEYS_PATH.write_text("locked-extrapolation-key\n", encoding="utf-8")
                payload = {
                    "x_model": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float),
                    "y_log": np.array([0.0], dtype=float),
                    "element": np.array(["Z_1"]),
                }
                np.savez_compressed(trainer.BENCHMARK_DATA_PATH, **payload)
                np.savez_compressed(trainer.EXTRAPOLATION_DATA_PATH, **payload)
                new_dataset = {
                    "keys": np.array(
                        [
                            "new-low-boundary",
                            "locked-random-key",
                            "new-interior",
                            "locked-extrapolation-key",
                            "new-high-boundary",
                        ],
                        dtype=object,
                    ),
                    "x_model": np.array(
                        [
                            [999.0, 0.0, 2.0, 3.0],
                            [999.0, 1.0, 2.0, 3.0],
                            [999.0, 2.0, 2.0, 3.0],
                            [999.0, 3.0, 2.0, 3.0],
                            [999.0, 4.0, 2.0, 3.0],
                        ],
                        dtype=float,
                    ),
                    "y_log": np.zeros(5, dtype=float),
                    "element_labels": np.array(["Z_999"] * 5, dtype=object),
                }

                random_keys, extrapolation_keys = trainer.ensure_benchmarks([new_dataset], 1.0, 0.2)

                self.assertEqual(random_keys, {"locked-random-key"})
                self.assertEqual(extrapolation_keys, {"locked-extrapolation-key"})
                self.assertEqual(
                    set(trainer.BENCHMARK_KEYS_PATH.read_text(encoding="utf-8").splitlines()),
                    {"locked-random-key"},
                )
                self.assertEqual(
                    set(trainer.EXTRAPOLATION_KEYS_PATH.read_text(encoding="utf-8").splitlines()),
                    {"locked-extrapolation-key"},
                )
            finally:
                for name, value in original_paths.items():
                    setattr(trainer, name, value)

    def test_cli_predicts_known_grid_nodes(self):
        completed = subprocess.run(
            [
                sys.executable,
                "predict_unified_free_path.py",
                "13",
                "-1.870",
                "1.771",
                "1.755",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(np.isfinite(result["lnu"]))
        self.assertGreater(result["lnu"], 0.0)

    def test_web_prediction_reports_training_membership_and_simulator_state(self):
        app = FreePathWebApp(ACTIVE_UNIFIED_ARTIFACT)
        result = app.predict({"Z": 13, "rod": 0.720, "tep": 1.771, "tgama": 1.755})
        self.assertIn("timing", result)
        self.assertGreaterEqual(result["timing"]["model_ms"], 0.0)
        self.assertTrue(result["data_membership"]["exists_in_standard_data"])
        self.assertTrue(result["data_membership"]["in_training_set"])
        self.assertFalse(result["reference_truth"]["found"])
        self.assertIn("未勾选", result["reference_truth"]["reason"])
        self.assertTrue(app._simulator_executable(13).exists())

    def test_data_element_discovery_includes_current_files(self):
        discovered = discover_data_elements()
        self.assertIn(4, discovered)
        self.assertIn(13, discovered)
        self.assertIn(79, discovered)

    def test_prediction_job_reports_progress_and_result(self):
        app = FreePathWebApp(ACTIVE_UNIFIED_ARTIFACT)
        status = app.start_prediction({"Z": 13, "rod": -1.870, "tep": 1.771, "tgama": 1.755})
        self.assertIn(status["status"], {"queued", "running", "succeeded"})
        deadline = time.time() + 10.0
        while status["status"] in {"queued", "running"} and time.time() < deadline:
            time.sleep(0.05)
            status = app.prediction_status({"id": [status["job_id"]]})
        self.assertEqual(status["status"], "succeeded")
        self.assertEqual(status["progress"], 100.0)
        self.assertTrue(np.isfinite(status["result"]["lnu"]))
        self.assertGreater(status["result"]["lnu"], 0.0)

    def test_simulator_request_state_is_explicit_when_missing(self):
        app = FreePathWebApp(ACTIVE_UNIFIED_ARTIFACT)
        result = app.predict({"Z": 999, "rod": -1.870, "tep": 1.771, "tgama": 1.755, "runSimulation": True})
        self.assertTrue(result["reference_truth"]["requested"])
        self.assertFalse(result["reference_truth"]["invoked"])
        self.assertIn("不支持", result["reference_truth"]["reason"])

    def test_simulator_inventory_lists_all_supported_elements(self):
        app = FreePathWebApp(ACTIVE_UNIFIED_ARTIFACT)
        inventory = app._simulator_inventory()
        for z in [4, 6, 13, 22, 26, 29, 42, 50, 56, 63, 74, 79, 82, 92]:
            self.assertIn(z, inventory["supported_z"])

    def test_web_training_data_import_replace_append_and_export(self):
        app = FreePathWebApp(ACTIVE_UNIFIED_ARTIFACT)
        path = ROOT / "free_path_model_outputs" / "unit_training_import.txt"
        for old in path.parent.glob("unit_training_import*"):
            old.unlink()
        try:
            replaced = app.import_training_data(
                {
                    "path": str(path),
                    "mode": "replace",
                    "text": "Z rod tep tgama lnu\n13 1 2 3 4\n13 1 2 3 5\n",
                }
            )
            self.assertEqual(replaced["imported_rows"], 2)
            self.assertEqual(replaced["final_rows"], 1)
            self.assertEqual(replaced["duplicate_rows_removed"], 1)
            self.assertIsNone(replaced["backup_path"])
            arr = np.loadtxt(path, dtype=float)
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            self.assertEqual(arr.shape, (1, 5))
            self.assertEqual(float(arr[0, 4]), 5.0)

            appended = app.import_training_data(
                {
                    "path": str(path),
                    "mode": "append",
                    "text": "13 1 2 4 6\n",
                }
            )
            self.assertEqual(appended["previous_rows"], 1)
            self.assertEqual(appended["imported_rows"], 1)
            self.assertEqual(appended["final_rows"], 2)
            self.assertTrue(Path(appended["backup_path"]).exists())
            self.assertEqual(app.export_training_data_path({"path": [str(path)]}), path)

            element_replaced = app.import_training_data(
                {
                    "path": str(path),
                    "mode": "replace_element",
                    "legacyZ": 79,
                    "removeNonpositive": True,
                    "text": "rod tep tgama lnu\n1 2 3 7\n1 2 4 -1\n",
                }
            )
            self.assertEqual(element_replaced["imported_rows"], 1)
            self.assertEqual(element_replaced["nonpositive_rows_removed"], 1)
            self.assertEqual(element_replaced["final_rows"], 3)
            arr = np.loadtxt(path, dtype=float)
            self.assertEqual(int(np.sum(np.isclose(arr[:, 0], 79.0))), 1)
            self.assertTrue(np.all(arr[:, 4] > 0.0))
        finally:
            for old in path.parent.glob("unit_training_import*"):
                old.unlink()

    def test_batch_prediction_shape(self):
        model = UnifiedModel.load(ACTIVE_UNIFIED_ARTIFACT)
        points = np.array(
            [
                [13.0, -1.870, 1.771, 1.755],
                [79.0, 2.7781513213184033, 3.6020631833806736, 2.778158343802252],
            ],
            dtype=float,
        )
        pred = model.predict_log(points)
        self.assertEqual(pred.shape, (2,))
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
