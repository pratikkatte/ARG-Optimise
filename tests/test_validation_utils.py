import unittest

try:
    import pandas as pd
    from validation.scripts.point_accuracy.inputs import iter_pairs
    from validation.scripts.point_accuracy.plots import fill_grid
    from validation.scripts.point_accuracy.reporting import common_metric_values
except ImportError:
    pd = None
    fill_grid = None
    iter_pairs = None
    common_metric_values = None


class ValidationUtilityTests(unittest.TestCase):
    @unittest.skipIf(iter_pairs is None, "validation dependencies are not installed")
    def test_pairs_are_unique(self):
        pairs = iter_pairs(4, 1)
        self.assertEqual(pairs, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        self.assertEqual(len(pairs), len(set(pairs)))

    @unittest.skipIf(pd is None, "pandas is an optional validation dependency")
    def test_weighted_metrics(self):
        frame = pd.DataFrame({
            "Simulated": [1.0, 2.0],
            "PosteriorMean": [2.0, 2.0],
            "len": [1.0, 3.0],
        })
        metrics = common_metric_values(frame, legacy_mse=9.0)
        self.assertAlmostEqual(metrics["weighted_mse"], 0.25)
        self.assertAlmostEqual(metrics["weighted_mae"], 0.25)
        self.assertEqual(metrics["legacy_mseall"], 9.0)

    @unittest.skipIf(fill_grid is None, "validation dependencies are not installed")
    def test_heatmap_grid_accumulates_rounded_bins(self):
        frame = pd.DataFrame({
            "Simulated": [0.04, 0.06, 0.06, 0.4],
            "PosteriorMean": [0.04, 0.06, 0.06, 0.1],
            "x": [1.0, 2.0, 3.0, 10.0],
        })
        _, _, grid, _ = fill_grid(frame, (0.0, 0.3), (0.0, 0.3))
        self.assertEqual(grid[0, 0], 1.0)
        self.assertEqual(grid[1, 1], 5.0)
        self.assertTrue(pd.isna(grid[0, 1]))


if __name__ == "__main__":
    unittest.main()
