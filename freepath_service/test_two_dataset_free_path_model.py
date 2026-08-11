#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from train_two_dataset_free_path_model import DEFAULT_OUTPUT, load_dataset


ROOT = Path(__file__).resolve().parent
ARTIFACT = DEFAULT_OUTPUT / "free_path_two_dataset_hybrid_model.npz"


class TwoDatasetFreePathTests(unittest.TestCase):
    def test_both_datasets_load_as_regular_grids(self):
        data_al = load_dataset("data_Al", ROOT / "data_Al.txt")
        data = load_dataset("data", ROOT / "data.txt")
        self.assertEqual([len(axis) for axis in data_al.axes], [50, 50, 50])
        self.assertEqual([len(axis) for axis in data.axes], [50, 50, 50])
        self.assertEqual(data_al.coordinate_transform, "identity")
        self.assertEqual(data.coordinate_transform, "log10")
        self.assertEqual(data_al.repaired_lnu_count, 0)
        self.assertEqual(data.repaired_lnu_count, 137)

    def test_cli_lists_datasets(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "predict_two_dataset_free_path.py"), "--list-datasets"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("data_Al", completed.stdout)
        self.assertIn("data:", completed.stdout)

    def test_cli_predicts_known_data_al_grid_node(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "predict_two_dataset_free_path.py"),
                "--dataset",
                "data_Al",
                "-1.870",
                "1.771",
                "1.755",
                "--json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["mode"], "interpolation")
        self.assertAlmostEqual(result["lnu"], 0.0498257, delta=1e-9)

    def test_cli_predicts_known_data_grid_node_with_auto_log_transform(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "predict_two_dataset_free_path.py"),
                "--dataset",
                "data",
                "600.000098",
                "4000.0294",
                "600.0098",
                "--json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["mode"], "interpolation")
        self.assertEqual(result["input"]["applied_transform"], "log10")
        self.assertAlmostEqual(result["lnu"], 0.000006668470, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
