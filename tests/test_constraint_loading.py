"""Exercise fail-closed constraint conversion through the actual adapter."""

from contextlib import contextmanager
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import numpy as np

import pycutest_tools as adapter


class NativeFixture:
    def __init__(self, linear=True, equality=False, failure=False):
        self.n, self.m = 2, 1
        self.x0 = np.array([1.0, 2.0])
        self.bl, self.bu = np.full(2, -1e20), np.full(2, 1e20)
        self.cl, self.cu = np.array([-1e20]), np.array([0.0])
        self.is_linear_cons = np.array([linear])
        self.is_eq_cons = np.array([equality])
        self.failure = failure
        self.gradient_calls = 0

    def obj(self, x):
        return float(x @ x)

    def grad(self, x):
        return 2 * x

    def ihess(self, x, **kwargs):
        return 2 * np.eye(2)

    def cons(self, x, gradient=False):
        if gradient:
            self.gradient_calls += 1
            if self.failure:
                raise ArithmeticError('injected native Jacobian failure')
        c = np.array([x[0] + 2 * x[1] - 1])
        return (c, np.array([[1.0, 2.0]])) if gradient else c


@contextmanager
def native_input(problem):
    runtime = SimpleNamespace(import_problem=lambda *args, **kwargs: problem)
    if hasattr(adapter, '_get_pycutest'):
        with patch.object(adapter, '_get_pycutest', return_value=runtime):
            yield
    else:
        with patch.object(adapter, 'pycutest', runtime):
            yield


class ConstraintLoadingTests(unittest.TestCase):
    def test_nan_bounds_cannot_drop_an_unreviewed_constraint(self):
        for linear in (False, True):
            for side in ('cl', 'cu'):
                with self.subTest(linear=linear, side=side):
                    p = NativeFixture(linear=linear)
                    setattr(p, side, np.array([np.nan]))
                    with native_input(p), self.assertRaisesRegex(ValueError, 'constraint bounds'):
                        adapter.pycutest_load('ADAPTERFAILCLOSED')

    def test_failed_linear_inequality_probe_cannot_drop_constraints(self):
        with native_input(NativeFixture(failure=True)):
            with self.assertRaisesRegex(RuntimeError, 'linear constraints') as caught:
                adapter.pycutest_load('ADAPTERFAILCLOSED')
        self.assertIsInstance(caught.exception.__cause__, ArithmeticError)

    def test_failed_linear_equality_probe_has_context_and_cause(self):
        p = NativeFixture(equality=True, failure=True)
        p.cl[:] = p.cu
        with native_input(p):
            with self.assertRaisesRegex(RuntimeError, 'linear constraints') as caught:
                adapter.pycutest_load('ADAPTERFAILCLOSED')
        self.assertIsInstance(caught.exception.__cause__, ArithmeticError)

    def test_nonlinear_only_load_does_not_probe_a_jacobian(self):
        p = NativeFixture(linear=False, failure=True)
        with native_input(p), patch.object(adapter.np, 'zeros', wraps=np.zeros) as zeros:
            problem = adapter.pycutest_load('ADAPTERFAILCLOSED')
        self.assertEqual(p.gradient_calls, 0)
        self.assertNotIn((p.m, p.n), [call.args[0] for call in zeros.call_args_list])
        np.testing.assert_allclose(problem.cub(problem.x0), [4.0])

    def test_valid_linear_constraint_keeps_offset_and_sign(self):
        p = NativeFixture()
        p.cl[:], p.cu[:] = -3.0, 4.0
        with native_input(p):
            problem = adapter.pycutest_load('ADAPTERFAILCLOSED')
        np.testing.assert_allclose(problem.aub, [[1, 2], [-1, -2]])
        np.testing.assert_allclose(problem.bub, [5, 2])

    def test_valid_linear_equality_keeps_nonzero_offset(self):
        p = NativeFixture(equality=True)
        p.cl[:], p.cu[:] = 4.0, 4.0
        with native_input(p):
            problem = adapter.pycutest_load('ADAPTERFAILCLOSED')
        np.testing.assert_allclose(problem.aeq, [[1, 2]])
        np.testing.assert_allclose(problem.beq, [5])

    def test_invalid_bound_shapes_fail_before_conversion(self):
        for values in (np.array([]), np.array([[0.0]])):
            p = NativeFixture()
            p.cu = values
            with native_input(p), self.assertRaisesRegex(ValueError, 'constraint bounds'):
                adapter.pycutest_load('ADAPTERFAILCLOSED')

    def test_invalid_linear_jacobian_fails_with_context(self):
        for jac in (np.array([1.0, 2.0]), np.array([[np.nan, 2.0]])):
            p = NativeFixture()
            p.cons = lambda x, gradient=False: (np.array([4.0]), jac)
            with native_input(p), self.assertRaisesRegex(RuntimeError, 'linear constraints'):
                adapter.pycutest_load('ADAPTERFAILCLOSED')


@unittest.skipUnless(os.environ.get('CUTEST'), 'Requires an installed CUTEst runtime')
class RealCUTEstConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = (adapter._get_pycutest() if hasattr(adapter, '_get_pycutest')
                       else adapter.pycutest)

    def test_valid_unconstrained_bound_linear_nonlinear_problems(self):
        for name, kind in [('ROSENBR', 'u'), ('HS1', 'b'), ('HS21', 'l'),
                           ('HS28', 'l'), ('HS13', 'n')]:
            with self.subTest(problem=name):
                p = adapter.pycutest_load(name)
                self.assertEqual(p.ptype, kind)
                self.assertTrue(np.isfinite(p.fun(p.x0)))
                self.assertTrue(np.all(np.isfinite(p.cub(p.x0))))
                self.assertTrue(np.all(np.isfinite(p.ceq(p.x0))))

    def test_native_nan_metadata_is_rejected_without_csv_identity(self):
        native = self.runtime.import_problem('HS21')
        with patch.object(native, 'cl', np.full(native.m, np.nan)), native_input(native):
            with self.assertRaisesRegex(ValueError, 'constraint bounds'):
                adapter.pycutest_load('ADAPTERFAILCLOSED')

    def test_native_failed_linear_probe_is_rejected_without_csv_identity(self):
        native = self.runtime.import_problem('HS21')
        with patch.object(native, 'cons', side_effect=ArithmeticError('injected CUTEst failure')):
            with native_input(native), self.assertRaisesRegex(RuntimeError, 'linear constraints'):
                adapter.pycutest_load('ADAPTERFAILCLOSED')

    def test_reviewed_identity_also_rejects_failed_probe(self):
        native = self.runtime.import_problem('HS21')
        with patch.object(native, 'cons', side_effect=ArithmeticError('injected CUTEst failure')):
            with native_input(native), self.assertRaisesRegex(RuntimeError, 'linear constraints') as caught:
                adapter.pycutest_load('HS21')
        self.assertIsInstance(caught.exception.__cause__, ArithmeticError)


if __name__ == '__main__':
    unittest.main()
