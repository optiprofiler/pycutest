from __future__ import annotations

from datetime import date
from pathlib import Path
from contextlib import contextmanager
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
    from pycutest_tools import (
        pycutest_clear_all_cache,
        pycutest_check_available,
        pycutest_get_default_options,
        pycutest_load,
        pycutest_select,
        pycutest_validate_options,
    )
    pycutest_check_available()
except Exception as exc:  # pragma: no cover - exercised on machines without CUTEst.
    IMPORT_ERROR = exc
    pycutest_check_available = pycutest_clear_all_cache = None
    pycutest_get_default_options = None
    pycutest_load = pycutest_select = pycutest_validate_options = None


@contextmanager
def _temporary_env(**updates):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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

    def test_config_environment_overrides_variable_size(self):
        options = {
            "ptype": "u",
            "maxdim": 10,
            "maxb": 20,
            "maxlcon": 20,
            "maxnlcon": 20,
            "maxcon": 20,
        }
        with _temporary_env(
            PYCUTEST_VARIABLE_SIZE=None,
            PYCUTEST_TEST_FEASIBILITY_PROBLEMS=None,
        ):
            default_names = pycutest_select(dict(options))
        with _temporary_env(
            PYCUTEST_VARIABLE_SIZE="all",
            PYCUTEST_TEST_FEASIBILITY_PROBLEMS="0",
        ):
            all_names = pycutest_select(dict(options))

        self.assertGreater(len(all_names), len(default_names))
        self.assertTrue(any(name.startswith("HILBERTA_N_") for name in all_names))
        self.assertFalse(any(name.startswith("HILBERTA_N_") for name in default_names))

    def test_config_environment_accepts_feasibility_modes_and_rejects_invalid_values(self):
        options = {"ptype": "ubln", "maxdim": 10}
        with _temporary_env(PYCUTEST_TEST_FEASIBILITY_PROBLEMS="0"):
            baseline = pycutest_select(dict(options))
        with _temporary_env(PYCUTEST_TEST_FEASIBILITY_PROBLEMS="2"):
            all_names = pycutest_select(dict(options))
        self.assertGreaterEqual(len(all_names), len(baseline))

        with _temporary_env(PYCUTEST_VARIABLE_SIZE="not-a-mode"):
            with self.assertRaises(ValueError):
                pycutest_select(dict(options))
        with _temporary_env(PYCUTEST_TEST_FEASIBILITY_PROBLEMS="3"):
            with self.assertRaises(ValueError):
                pycutest_select(dict(options))

    def test_explicit_options_override_environment_and_reach_load(self):
        library_options = {
            "variable_size": "default",
            "test_feasibility_problems": 0,
        }
        with _temporary_env(PYCUTEST_VARIABLE_SIZE="not-a-mode"):
            selected = pycutest_select(
                {"ptype": "u", "mindim": 2, "maxdim": 2},
                library_options,
            )
        self.assertIn("ROSENBR", selected)
        problem = pycutest_load("ROSENBR", library_options=library_options)
        self.assertEqual(problem.n, 2)

    def test_library_option_callbacks_are_strict(self):
        self.assertEqual(
            set(pycutest_get_default_options()),
            {"variable_size", "test_feasibility_problems"},
        )
        with self.assertRaises(ValueError):
            pycutest_validate_options({"variable_size": "all", "unknown": 1})
        with self.assertRaises(ValueError):
            pycutest_validate_options({
                "variable_size": [],
                "test_feasibility_problems": 0,
            })


if __name__ == "__main__":
    unittest.main()
