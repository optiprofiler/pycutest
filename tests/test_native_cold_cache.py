"""Real cold-cache concurrency against an installed CUTEst toolchain.

Independent interpreters import problems that are not compiled yet, released
together once PyCUTEst is imported. Without the adapter lock this fails natively
(partly built modules, SIFDECODE errors, FileExistsError). The workers are plain
subprocesses, not multiprocessing children: those would inherit this process's
sys.path, including any cache used earlier, and could import an already compiled
module from there instead of racing. The tests are skipped without a runtime.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_VARIABLES = ('ARCHDEFS', 'SIFDECODE', 'CUTEST', 'MASTSIF', 'MYARCH')
RUNTIME_AVAILABLE = (importlib.util.find_spec('pycutest') is not None
                     and all(name in os.environ for name in RUNTIME_VARIABLES))
TIMEOUT = 300
WORKER = """
import pathlib, sys, time
root, ready, go, name = sys.argv[1:5]
sys.path.insert(0, root)
import pycutest_tools as adapter
adapter._get_pycutest()
pathlib.Path(ready).touch()
deadline = time.monotonic() + 120
while not pathlib.Path(go).exists():
    assert time.monotonic() < deadline, 'no start signal'
    time.sleep(0.001)
problem = adapter.pycutest_load(name)
print(name, problem.n, repr(float(problem.fun(problem.x0))))
"""


@unittest.skipUnless(RUNTIME_AVAILABLE, 'Requires an installed PyCUTEst/CUTEst runtime.')
class NativeColdCacheTests(unittest.TestCase):
    def race(self, names):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'cache'
            cache.mkdir()
            env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            env['PYCUTEST_CACHE'] = str(cache)
            go = root / 'go'
            ready = [root / f'ready-{index}' for index in range(len(names))]
            workers = [
                subprocess.Popen(
                    [sys.executable, '-c', WORKER, str(ROOT), str(ready[index]), str(go), name],
                    env=env, cwd=temporary, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for index, name in enumerate(names)
            ]
            try:
                deadline = time.monotonic() + TIMEOUT
                while not all(path.exists() for path in ready):
                    self.assertLess(time.monotonic(), deadline, 'Workers did not start.')
                    time.sleep(0.01)
                go.touch()
                outputs = [worker.communicate(timeout=TIMEOUT)[0] for worker in workers]
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.kill()
                        worker.wait()
            for worker, output in zip(workers, outputs):
                self.assertEqual(worker.returncode, 0, output)
            # The race is only meaningful if this cache really was compiled.
            holder = cache / 'pycutest_cache_holder'
            self.assertEqual(sorted(path.name for path in holder.iterdir() if path.is_dir()
                                    and path.name != '__pycache__'), sorted(set(names)))
            self.assertTrue((cache / '.optiprofiler-import.lock').is_file())
            return sorted(tuple(output.strip().splitlines()[-1].split()) for output in outputs)

    def test_processes_share_one_compilation_of_an_uncompiled_problem(self):
        outcome = self.race(['ROSENBR'] * 3)
        self.assertEqual(set(outcome), {('ROSENBR', '2', '24.199999999999996')})

    def load(self, cache, name):
        env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
        env['PYCUTEST_CACHE'] = str(cache)
        code = ('import sys; sys.path.insert(0, sys.argv[1]); import pycutest_tools as adapter; '
                'problem = adapter.pycutest_load(sys.argv[2]); print(problem.n, problem.fun(problem.x0))')
        return subprocess.run([sys.executable, '-c', code, str(ROOT), name], env=env, cwd=str(cache),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT)

    def test_entry_of_an_interrupted_build_recovers(self):
        with tempfile.TemporaryDirectory() as cache:
            first = self.load(cache, 'ROSENBR')
            self.assertEqual(first.returncode, 0, first.stdout)
            # Upstream writes __init__.py last: this is what a killed build leaves.
            (Path(cache) / 'pycutest_cache_holder' / 'ROSENBR' / '__init__.py').unlink()
            second = self.load(cache, 'ROSENBR')
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertEqual(second.stdout.split()[-2:], first.stdout.split()[-2:])

    def test_fractional_default_parameter_loads(self):
        with tempfile.TemporaryDirectory() as cache:
            result = self.load(cache, 'INDEF')
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout.split()[-2], '5000')

    def test_different_problems_initialize_one_cold_cache(self):
        outcome = self.race(['ROSENBR', 'HS21', 'ALLINITU'])
        self.assertEqual([item[0] for item in outcome], ['ALLINITU', 'HS21', 'ROSENBR'])


if __name__ == '__main__':
    unittest.main()
