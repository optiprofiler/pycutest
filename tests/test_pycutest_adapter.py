from __future__ import annotations

from datetime import date
from pathlib import Path
import math
import os
import random
import sys
import unittest

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]

op_candidates = [
    REPO_DIR / "optiprofiler" / "python",
    REPO_DIR.parents[1] / "optiprofiler" / "python",
]
for op_path in op_candidates:
    if (op_path / "optiprofiler").is_dir():
        sys.path.insert(0, str(op_path))
        break

sys.path.insert(0, str(REPO_DIR))

IMPORT_ERROR = None
try:
    from pycutest_tools import pycutest_clear_all_cache, pycutest_load, pycutest_select
except Exception as exc:  # pragma: no cover - exercised on machines without CUTEst.
    IMPORT_ERROR = exc
    pycutest_clear_all_cache = pycutest_load = pycutest_select = None


def _as_array(value):
    if value is None:
        return np.empty(0)
    return np.asarray(value)


@unittest.skipIf(IMPORT_ERROR is not None, f"PyCUTEst is not available: {IMPORT_ERROR}")
class PyCUTEstAdapterTests(unittest.TestCase):
    def tearDown(self):
        pycutest_clear_all_cache()

    def assert_problem_contract(self, problem_name):
        problem = pycutest_load(problem_name)
        self.assertGreaterEqual(problem.n, 1)
        self.assertEqual(problem.x0.size, problem.n)

        fx0 = problem.fun(problem.x0)
        self.assertTrue(math.isfinite(float(fx0)) or math.isnan(float(fx0)))

        cub0 = _as_array(problem.cub(problem.x0))
        ceq0 = _as_array(problem.ceq(problem.x0))
        self.assertEqual(cub0.ndim, 1)
        self.assertEqual(ceq0.ndim, 1)

    def test_select_and_load_rosenbrock(self):
        selected = pycutest_select({"ptype": "u", "maxdim": 5})
        self.assertIn("ROSENBR", selected)
        self.assert_problem_contract("ROSENBR")

    def test_daily_random_small_problem_sample(self):
        seed = int(os.environ.get("OP_RANDOM_SEED", date.today().strftime("%Y%m%d")))
        candidates = pycutest_select(
            {
                "ptype": "ubln",
                "maxdim": 5,
                "maxb": 20,
                "maxlcon": 20,
                "maxnlcon": 20,
                "maxcon": 20,
            }
        )
        self.assertGreaterEqual(len(candidates), 2)

        rng = random.Random(seed)
        sample = rng.sample(candidates, k=min(2, len(candidates)))
        print(f"PyCUTEst random sample seed={seed}: {sample}")
        for problem_name in sample:
            with self.subTest(problem=problem_name):
                self.assert_problem_contract(problem_name)


if __name__ == "__main__":
    unittest.main()
