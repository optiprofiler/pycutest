from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
import unittest

from pycutest_tools import (
    _assert_loaded_instance,
    _expected_instance_metadata,
    _load_reviewed_sif_defaults,
    _parse_parameter_configurations,
    _resolve_problem_instance,
)


REPO_DIR = Path(__file__).resolve().parents[1]


class VariableMetadataTests(unittest.TestCase):
    def test_bare_variable_problem_uses_reviewed_default_parameters(self):
        base_name, params, expected = _resolve_problem_instance("OSCIPATH", {})

        self.assertEqual(base_name, "OSCIPATH")
        self.assertEqual(params, {"N": 10, "RHO": 500.0})
        self.assertEqual(expected["dim"], 10)

    def test_encoded_variable_problem_resolves_matching_instance(self):
        base_name, params, expected = _resolve_problem_instance(
            "OSCIPATH_N_25_RHO_1",
            {},
        )

        self.assertEqual(base_name, "OSCIPATH")
        self.assertEqual(params, {"N": 25, "RHO": 1})
        self.assertEqual(expected["dim"], 25)

    def test_custom_parameters_remain_available_without_false_metadata_claim(self):
        base_name, params, expected = _resolve_problem_instance(
            "OSCIPATH",
            {"N": 7, "RHO": 3},
        )

        self.assertEqual(base_name, "OSCIPATH")
        self.assertEqual(params, {"N": 7, "RHO": 3})
        self.assertIsNone(expected)

    def test_post_load_invariant_rejects_dimension_drift(self):
        problem = SimpleNamespace(n=500, mb=0, mlcon=0, mnlcon=0, mcon=0)
        expected = {
            "dim": 10,
            "mb": 0,
            "mlcon": 0,
            "mnlcon": 0,
            "mcon": 0,
        }

        with self.assertRaisesRegex(RuntimeError, "expected dim=10.*loaded dim=500"):
            _assert_loaded_instance("OSCIPATH", problem, expected)

    def test_reviewed_defaults_cover_every_variable_metadata_row(self):
        with (REPO_DIR / "probinfo_pycutest.csv").open(newline="") as metadata_file:
            rows = list(csv.DictReader(metadata_file))
        variable_names = {row["problem_name"] for row in rows if row["argins"]}
        reviewed_names = set(_load_reviewed_sif_defaults()["problems"])

        self.assertEqual(reviewed_names, variable_names)

    def test_every_parameterized_instance_resolves_its_recorded_shape(self):
        with (REPO_DIR / "probinfo_pycutest.csv").open(newline="") as metadata_file:
            rows = list(csv.DictReader(metadata_file))

        checked = 0
        for row in rows:
            configurations = _parse_parameter_configurations(row["argins"])
            dims = [int(float(value)) for value in row["dims"].split()]
            self.assertEqual(len(configurations), len(dims), row["problem_name"])
            for configuration, dimension in zip(configurations, dims):
                expected = _expected_instance_metadata(
                    row["problem_name"],
                    configuration,
                )
                self.assertIsNotNone(expected, (row["problem_name"], configuration))
                self.assertEqual(expected["dim"], dimension)
                checked += 1

        self.assertEqual(checked, 2588)


if __name__ == "__main__":
    unittest.main()
