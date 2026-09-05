"""Preserve paper instance identities and reject known metadata drift."""
import os
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from test_pycutest_adapter import IMPORT_ERROR, _temporary_env
if IMPORT_ERROR is None:
    import pycutest_tools as adapter


@unittest.skipIf(IMPORT_ERROR is not None, f'PyCUTEst unavailable: {IMPORT_ERROR}')
class InstanceInvariantTests(unittest.TestCase):
    def test_default_oscipath_or_explicit_runtime_drift_error(self):
        with _temporary_env(PYCUTEST_VARIABLE_SIZE='default'):
            names = adapter.pycutest_select({'mindim':8, 'maxdim':10})
        self.assertIn('OSCIPATH', names)
        if os.environ.get('OP_TEST_DRIFT_RUNTIME') == '1':
            with self.assertRaisesRegex(RuntimeError, 'expected dim=10, loaded dim=500') as caught:
                adapter.pycutest_load('OSCIPATH')
            print(f'Drift rejected: {caught.exception}')
        else:
            problem = adapter.pycutest_load('OSCIPATH')
            self.assertEqual(problem.n, 10)
            self.assertTrue(np.isfinite(problem.fun(problem.x0)))

    def test_known_encoded_and_keyword_parameters(self):
        for name, params, dimension in [
            ('OSCIPATH_N_10_RHO_1', {}, 10),
            ('OSCIPATH', {'N':10, 'RHO':1}, 10),
            ('OSCIPATH_N_25_RHO_1', {}, 25),
        ]:
            with self.subTest(name=name):
                problem = adapter.pycutest_load(name, **params)
                self.assertEqual(problem.n, dimension)
                self.assertTrue(np.isfinite(problem.fun(problem.x0)))

    def test_encoded_parameters_keep_legacy_precedence(self):
        # The paper API recursively uses encoded params and discards keywords.
        problem = adapter.pycutest_load('OSCIPATH_N_10_RHO_1', N=500, RHO=500)
        self.assertEqual(problem.n, 10)
        self.assertAlmostEqual(problem.fun(problem.x0), 1.0)

    def test_unlisted_custom_parameters_remain_callable(self):
        for name, params, dimension in [
            ('OSCIPATH_N_10_RHO_500', {}, 10),
            ('OSCIPATH', {'N':10, 'RHO':500}, 10),
            ('OSCIPATH', {'N':25}, 25),
        ]:
            with self.subTest(name=name, params=params):
                problem = adapter.pycutest_load(name, **params)
                self.assertEqual(problem.n, dimension)
                self.assertTrue(np.isfinite(problem.fun(problem.x0)))

    def test_unlisted_parameters_keep_upstream_rejection(self):
        with self.assertRaisesRegex(RuntimeError, 'SIFDECODE'):
            adapter.pycutest_load('OSCIPATH', N=7)

    def test_all_recorded_counts_are_checked_after_construction(self):
        for attribute in ('n', 'mb', 'mlcon', 'mnlcon', 'mcon'):
            actual = dict(n=10, mb=0, mlcon=0, mnlcon=0, mcon=0)
            actual[attribute] += 1
            with self.subTest(attribute=attribute):
                with patch.object(adapter, 'Problem', return_value=SimpleNamespace(**actual)):
                    with self.assertRaisesRegex(RuntimeError, 'instance metadata mismatch'):
                        adapter.pycutest_load('OSCIPATH_N_10_RHO_1')


if __name__ == '__main__':
    unittest.main()
