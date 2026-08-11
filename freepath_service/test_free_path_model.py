#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from train_free_path_model import DEFAULT_DATA, DEFAULT_OUTPUT, load_grid_data, make_interpolator


ROOT = Path(__file__).resolve().parent


class FreePathModelTests(unittest.TestCase):
    def test_data_is_complete_positive_grid(self):
        grid = load_grid_data(DEFAULT_DATA)
        self.assertEqual(grid.data.shape, (125000, 4))
        self.assertEqual([len(axis) for axis in grid.axes], [50, 50, 50])
        self.assertTrue(np.all(grid.lnu_grid > 0.0))

    def test_interpolator_reproduces_known_grid_nodes(self):
        grid = load_grid_data(DEFAULT_DATA)
        interpolator = make_interpolator(grid.axes, grid.log_lnu_grid, "cubic")
        indices = np.array(
            [
                [0, 0, 0],
                [3, 7, 11],
                [20, 25, 30],
                [49, 49, 49],
            ],
            dtype=int,
        )
        points = np.column_stack([grid.axes[dim][indices[:, dim]] for dim in range(3)])
        expected_log = grid.log_lnu_grid[tuple(indices[:, dim] for dim in range(3))]
        predicted_log = interpolator(points)
        np.testing.assert_allclose(predicted_log, expected_log, rtol=0.0, atol=1e-10)

    def test_prediction_cli_known_first_row(self):
        model_candidates = sorted(DEFAULT_OUTPUT.glob("free_path_model_data_Al_*.npz"))
        self.assertTrue(
            model_candidates,
            "No saved model artifact found. Run `python train_free_path_model.py` before tests.",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "predict_free_path.py"),
                "-1.870",
                "1.771",
                "1.755",
                "--model",
                str(model_candidates[-1]),
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        result = json.loads(completed.stdout)
        self.assertAlmostEqual(result["lnu"], 0.0498257, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
