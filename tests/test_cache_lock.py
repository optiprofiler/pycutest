"""POSIX process-lock and actual adapter call-boundary regression tests.

The upstream substitute checks coordination only; native cold-cache probes
must additionally run against an installed CUTEst toolchain on Linux.
"""
from contextlib import contextmanager
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pycutest_tools as adapter


LOCK_NAME = '.optiprofiler-import.lock'


def lock_is_free(root):
    with (Path(root) / LOCK_NAME).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
    return True


@contextmanager
def runtime(root, import_call=None, clear_call=None):
    substitute = SimpleNamespace(
        system_paths=SimpleNamespace(get_cache_path=lambda: str(root)),
        build_interface=SimpleNamespace(get_problem_directory=lambda name, sifParams=None:
                                        str(Path(root) / 'pycutest_cache_holder' / name)),
        import_problem=import_call, clear_cache=clear_call)
    if hasattr(adapter, '_get_pycutest'):
        with patch.object(adapter, '_get_pycutest', return_value=substitute), \
                patch.object(adapter, '_PYCUTEST_MODULE', substitute):
            yield substitute
    else:
        with patch.object(adapter, 'pycutest', substitute):
            yield substitute


def hold_lock(root, entered, release):
    with adapter._pycutest_cache_lock(root):
        entered.set()
        if not release.wait(15):
            raise TimeoutError('Lock owner did not receive a release signal.')


def timed_lock(root, barrier, queue):
    barrier.wait(15)
    with adapter._pycutest_cache_lock(root):
        start = time.monotonic()
        time.sleep(0.02)
        end = time.monotonic()
    queue.put((start, end, os.getpid()))


def clear_worker(root, started, done, all_entries):
    def clear(*args, **kwargs):
        assert not lock_is_free(root)
        (Path(root) / 'cleared').touch()
    with runtime(root, clear_call=clear):
        started.set()
        if all_entries:
            adapter.pycutest_clear_all_cache()
        else:
            adapter.pycutest_clear_cache('ROSENBR')
        done.set()


class CacheLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = mp.get_context('spawn')

    def start(self, target, *args):
        process = self.context.Process(target=target, args=args)
        process.start()
        self.addCleanup(self.stop, process)
        return process

    @staticmethod
    def stop(process):
        if process.is_alive():
            process.terminate()
        process.join(10)

    def finish(self, process):
        process.join(15)
        self.assertFalse(process.is_alive(), 'Worker hung.')
        self.assertEqual(process.exitcode, 0)

    def test_eight_independently_spawned_processes_exclude_each_other(self):
        barrier = self.context.Barrier(8)
        queue = self.context.Queue()
        processes = [self.start(timed_lock, str(self.root), barrier, queue) for _ in range(8)]
        intervals = sorted(queue.get(timeout=30) for _ in processes)
        for process in processes:
            self.finish(process)
        self.assertEqual(len({item[2] for item in intervals}), 8)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])))
        queue.close()
        queue.join_thread()

    def test_exception_releases_lock_and_preserves_inode(self):
        failure = ArithmeticError('construction failed')
        with self.assertRaises(ArithmeticError) as caught:
            with adapter._pycutest_cache_lock(self.root):
                inode = (self.root / LOCK_NAME).stat().st_ino
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertTrue(lock_is_free(self.root))
        with adapter._pycutest_cache_lock(self.root):
            self.assertEqual((self.root / LOCK_NAME).stat().st_ino, inode)

    def test_terminated_owner_does_not_leave_a_stale_lock(self):
        entered, release = self.context.Event(), self.context.Event()
        process = self.start(hold_lock, str(self.root), entered, release)
        self.assertTrue(entered.wait(15))
        self.assertFalse(lock_is_free(self.root))
        process.terminate()
        process.join(10)
        self.assertFalse(process.is_alive())
        self.assertTrue(lock_is_free(self.root))

    def test_unrelated_cache_roots_are_independent(self):
        with adapter._pycutest_cache_lock(self.root):
            with adapter._pycutest_cache_lock(self.root / 'other'):
                self.assertFalse(lock_is_free(self.root))
                self.assertFalse(lock_is_free(self.root / 'other'))

    def test_alias_paths_lock_the_same_inode(self):
        alias = self.root / 'alias'
        target = self.root / 'target'
        target.mkdir()
        alias.symlink_to(target, target_is_directory=True)
        with adapter._pycutest_cache_lock(alias):
            self.assertFalse(lock_is_free(target))

    def test_invalid_cache_path_fails(self):
        invalid = self.root / 'not-a-directory'
        invalid.write_text('retained')
        with self.assertRaises(OSError):
            with adapter._pycutest_cache_lock(invalid):
                self.fail('Invalid cache root was accepted.')
        self.assertEqual(invalid.read_text(), 'retained')

    @unittest.skipIf(os.geteuid() == 0, 'Permission check requires an unprivileged user.')
    def test_permission_error_is_not_suppressed(self):
        self.root.chmod(0o500)
        try:
            with self.assertRaises(PermissionError):
                with adapter._pycutest_cache_lock(self.root):
                    self.fail('Read-only root was accepted.')
        finally:
            self.root.chmod(0o700)

    def test_import_exception_is_unchanged_and_lock_is_released(self):
        failure = RuntimeError('native import failed')
        def import_call(*args, **kwargs):
            self.assertFalse(lock_is_free(self.root))
            raise failure
        with runtime(self.root, import_call=import_call):
            with self.assertRaises(RuntimeError) as caught:
                adapter.pycutest_load('CACHELOCKPROBE')
        self.assertIs(caught.exception, failure)
        self.assertTrue(lock_is_free(self.root))

    def test_import_uses_runtime_root_not_stale_configured_root(self):
        stale = self.root / 'stale'
        actual = self.root / 'actual'
        actual.mkdir()
        failure = RuntimeError('stop before conversion')
        def import_call(*args, **kwargs):
            self.assertFalse(lock_is_free(actual))
            self.assertFalse((stale / LOCK_NAME).exists())
            raise failure
        with patch.dict(os.environ, {'PYCUTEST_CACHE': str(stale)}), \
                runtime(actual, import_call=import_call):
            with self.assertRaisesRegex(RuntimeError, 'stop before conversion'):
                adapter.pycutest_load('CACHELOCKPROBE')

    def test_lock_ends_before_problem_conversion(self):
        native = SimpleNamespace(n=1, m=0, bl=np.array([-1e20]), bu=np.array([1e20]),
                                 cl=None, cu=None, x0=np.array([0.0]))
        def import_call(*args, **kwargs):
            self.assertFalse(lock_is_free(self.root))
            return native
        def converted(*args, **kwargs):
            self.assertTrue(lock_is_free(self.root))
            raise RuntimeError('reached conversion without lock')
        with runtime(self.root, import_call=import_call), patch.object(adapter, 'Problem', converted):
            with self.assertRaisesRegex(RuntimeError, 'reached conversion without lock'):
                adapter.pycutest_load('CACHELOCKPROBE')

    def test_entry_of_a_killed_build_is_cleared_under_the_lock(self):
        entry = self.root / 'pycutest_cache_holder' / 'CACHELOCKPROBE'
        entry.mkdir(parents=True)
        cleared = []

        def clear_call(name, sifParams=None):
            self.assertFalse(lock_is_free(self.root))
            cleared.append(name)

        def import_call(*args, **kwargs):
            self.assertEqual(cleared, ['CACHELOCKPROBE'])
            raise RuntimeError('stop before conversion')

        with runtime(self.root, import_call=import_call, clear_call=clear_call):
            with self.assertRaisesRegex(RuntimeError, 'stop before conversion'):
                adapter.pycutest_load('CACHELOCKPROBE')

    def test_complete_entry_is_not_cleared(self):
        entry = self.root / 'pycutest_cache_holder' / 'CACHELOCKPROBE'
        entry.mkdir(parents=True)
        (entry / '__init__.py').touch()

        def clear_call(*args, **kwargs):
            self.fail('A complete entry was cleared.')

        def import_call(*args, **kwargs):
            raise RuntimeError('stop before conversion')

        with runtime(self.root, import_call=import_call, clear_call=clear_call):
            with self.assertRaisesRegex(RuntimeError, 'stop before conversion'):
                adapter.pycutest_load('CACHELOCKPROBE')

    def test_clear_all_never_touches_the_cache_of_the_working_directory(self):
        # PyCUTEst imported elsewhere defaults to the current folder.
        working, home = self.root / 'working', self.root / 'home'
        kept = working / 'pycutest_cache_holder'
        kept.mkdir(parents=True)
        own = home / 'optiprofiler' / 'pycutest' / 'pycutest_cache_holder'
        own.mkdir(parents=True)
        upstream = SimpleNamespace(system_paths=SimpleNamespace(
            get_cache_path=lambda: os.environ.get('PYCUTEST_CACHE', str(working))))
        environment = {key: value for key, value in os.environ.items() if key != 'PYCUTEST_CACHE'}
        environment['XDG_CACHE_HOME'] = str(home)
        with patch.dict(os.environ, environment, clear=True), \
                patch.dict(sys.modules, {'pycutest': upstream}), \
                patch.object(adapter, '_PYCUTEST_MODULE', None), \
                patch.object(adapter, '_PYCUTEST_CACHE_DIR', None):
            adapter.pycutest_clear_all_cache()
        self.assertTrue(kept.exists())
        self.assertFalse(own.exists())

    def test_cache_root_comes_first_on_the_path(self):
        cache = self.root / 'cache'
        path = [str(cache), 'somewhere']
        with patch.dict(os.environ, {'PYCUTEST_CACHE': str(cache)}), \
                patch.object(sys, 'path', path), \
                patch.object(adapter, '_PYCUTEST_CACHE_DIR', None):
            root = adapter._configure_pycutest_cache()
        self.assertEqual(path[0], root)
        self.assertEqual(path.count(root), 1)

    def test_holder_of_another_cache_is_refused(self):
        holder = self.root / 'other' / 'pycutest_cache_holder'
        holder.mkdir(parents=True)
        module = SimpleNamespace(__file__=str(holder / '__init__.py'))
        with patch.dict(sys.modules, {'pycutest_cache_holder': module}):
            adapter._assert_cache_holder(self.root / 'other')
            with self.assertRaisesRegex(ImportError, 'cache holder'):
                adapter._assert_cache_holder(self.root)

    def test_fractional_default_parameters_are_not_passed(self):
        problems = adapter._load_reviewed_sif_defaults()['problems']
        self.assertIsNone(adapter._runtime_sif_params('INDEF', dict(problems['INDEF'])))
        self.assertEqual(adapter._runtime_sif_params('INDEF', {'N': 50, 'ALPHA': 0.5}),
                         {'N': 50, 'ALPHA': 0.5})
        self.assertEqual(adapter._runtime_sif_params('ARGLINA', dict(problems['ARGLINA'])),
                         problems['ARGLINA'])
        self.assertIsNone(adapter._runtime_sif_params('ROSENBR', {}))

    def test_clear_operations_wait_for_import_lock_and_keep_inode(self):
        for all_entries in (False, True):
            with self.subTest(clear_all=all_entries):
                holder = self.root / 'pycutest_cache_holder'
                holder.mkdir(exist_ok=True)
                (holder / 'building').touch()
                started, done = self.context.Event(), self.context.Event()
                with adapter._pycutest_cache_lock(self.root):
                    inode = (self.root / LOCK_NAME).stat().st_ino
                    process = self.start(clear_worker, str(self.root), started, done, all_entries)
                    self.assertTrue(started.wait(15))
                    self.assertFalse(done.wait(0.15))
                    self.assertTrue((holder / 'building').exists())
                self.assertTrue(done.wait(15))
                self.finish(process)
                self.assertEqual((self.root / LOCK_NAME).stat().st_ino, inode)
                if all_entries:
                    self.assertFalse(holder.exists())


if __name__ == '__main__':
    unittest.main()
