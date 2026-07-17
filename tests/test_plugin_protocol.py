from __future__ import annotations

from importlib import metadata
import multiprocessing
import unittest
from unittest import mock

IMPORT_ERROR = None
try:
    import optiprofiler_pycutest
    from optiprofiler.problem_libraries import (
        PROBLEM_LIBRARY_ENTRY_POINT_GROUP,
        ProblemLibraryRef,
        _resolve_problem_library_options,
        load_problem_library,
    )
except Exception as exc:  # pragma: no cover - exercised before editable install.
    IMPORT_ERROR = exc


def _spawn_select_pycutest(problem_options, library_options):
    import optiprofiler_pycutest

    return optiprofiler_pycutest.get_problem_library().select(
        problem_options,
        library_options,
    )


@unittest.skipIf(IMPORT_ERROR is not None, f"Plugin package is not installed: {IMPORT_ERROR}")
class PyCUTEstPluginProtocolTests(unittest.TestCase):
    def test_entry_point_metadata_is_installed(self):
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=PROBLEM_LIBRARY_ENTRY_POINT_GROUP)
        else:
            selected = entry_points.get(PROBLEM_LIBRARY_ENTRY_POINT_GROUP, [])
        values = {entry_point.name: entry_point.value for entry_point in selected}
        self.assertEqual(
            values.get("pycutest"),
            "optiprofiler_pycutest:get_problem_library",
        )

    def test_distribution_does_not_offer_runtime_installation_extra(self):
        distribution = metadata.distribution("optiprofiler-pycutest")
        extras = set(distribution.metadata.get_all("Provides-Extra") or ())
        self.assertNotIn("runtime", extras)
        self.assertIn("tests", extras)

    def test_factory_returns_api_v1_plugin_without_importing_pycutest_runtime(self):
        from optiprofiler_pycutest import pycutest_tools

        previous_module = pycutest_tools._PYCUTEST_MODULE
        previous_cache = pycutest_tools._PYCUTEST_CACHE_DIR
        plugin = optiprofiler_pycutest.get_problem_library()
        self.assertEqual(plugin.name, "pycutest")
        self.assertEqual(plugin.api_version, 1)
        self.assertIsNotNone(plugin.check_available)
        self.assertIsNotNone(plugin.get_default_options)
        self.assertIsNotNone(plugin.validate_options)

        self.assertIs(pycutest_tools._PYCUTEST_MODULE, previous_module)
        self.assertEqual(pycutest_tools._PYCUTEST_CACHE_DIR, previous_cache)

    def test_check_available_wraps_runtime_import_errors(self):
        from optiprofiler_pycutest import pycutest_tools

        previous_module = pycutest_tools._PYCUTEST_MODULE
        pycutest_tools._PYCUTEST_MODULE = None
        try:
            with mock.patch.object(
                pycutest_tools,
                "_configure_pycutest_cache",
                return_value="/tmp/pycutest-test-cache",
            ), mock.patch.object(
                pycutest_tools.importlib,
                "import_module",
                side_effect=ModuleNotFoundError("No module named 'pycutest'"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "upstream PyCUTEst installation instructions",
                ):
                    pycutest_tools.pycutest_check_available()
        finally:
            pycutest_tools._PYCUTEST_MODULE = previous_module

    def test_entry_point_reference_loads_plugin(self):
        reference = ProblemLibraryRef(
            "pycutest",
            "entry_point",
            "optiprofiler_pycutest:get_problem_library",
            distribution="optiprofiler-pycutest",
        )
        plugin = load_problem_library(reference)
        effective = _resolve_problem_library_options(
            plugin,
            {"variable_size": "default", "test_feasibility_problems": 0},
        )
        self.assertEqual(
            effective,
            {"variable_size": "default", "test_feasibility_problems": 0},
        )
        selected = plugin.select({"ptype": "u", "mindim": 2, "maxdim": 2}, effective)
        self.assertIn("ROSENBR", selected)

    def test_collect_info_reads_committed_metadata(self):
        rows = optiprofiler_pycutest.pycutest_collect_info()
        self.assertGreater(len(rows), 0)
        self.assertIn("problem_name", rows[0])

    def test_api_v1_callbacks_work_in_spawned_process(self):
        options = {"ptype": "u", "mindim": 2, "maxdim": 2}
        library_options = {
            "variable_size": "default",
            "test_feasibility_problems": 0,
        }
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            selected = pool.apply(
                _spawn_select_pycutest,
                (options, library_options),
            )
        self.assertIn("ROSENBR", selected)


if __name__ == "__main__":
    unittest.main()
