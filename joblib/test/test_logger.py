"""
Tests for joblib logging utilities
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import re
import subprocess
import sys

import pytest

from joblib.logger import PrintTime

try:
    import numpy as np
except ImportError:
    np = None


def test_print_time(tmpdir, capsys):
    # A simple smoke test for PrintTime.
    logfile = tmpdir.join('test.log').strpath
    with capsys.disabled():  # avoid FutureWarning
        print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    # Create a second time, to smoke test log rotation.
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    # And a third time
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')

    out_printed_text, err_printed_text = capsys.readouterr()
    # Use regexps to be robust to time variations
    match = r"Foo: 0\..s, 0\..min\nFoo: 0\..s, 0..min\nFoo: " + \
            r".\..s, 0..min\n"
    if not re.match(match, err_printed_text):
        raise AssertionError('Excepted %s, got %s' %
                             (match, err_printed_text))


def test_parallel_verbosity():
    verbose_levels = [0, 1, 5, 10, 50]
    num_lines = [0, 2, 3, 4, 11]
    for verbose, num_lines in zip(verbose_levels, num_lines):
        cmd = '''if 1:
            from joblib import Parallel, delayed
            if __name__ == '__main__':
                slice_of_data = Parallel(n_jobs=2, verbose={})(
                    delayed(id)(i) for i in range(10))
        '''.format(verbose)
        p = subprocess.Popen([sys.executable, '-c', cmd],
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
        p.wait()
        out, err = p.communicate()
        lines = err.decode().split('\n')[:-1]
        assert out == b''
        assert len(lines) == num_lines
        if len(lines) > 0:
            assert lines[0].endswith(
                'Using backend LokyBackend with 2 concurrent workers.'
            )
            assert all(
                l.startswith('[Parallel(n_jobs=2)]: Done') for l in lines[1:]
            )
            assert all("memory" not in l for l in lines)
            assert all("batching" not in l for l in lines)


def test_parallel_reduction_verbosity():
    if np is None:
        pytest.skip('Reduction tests only apply for numpy arrays')

    verbose_levels = [0, 49, 51, 100]
    for verbose in verbose_levels:
        cmd = '''if 1:
            import numpy as np
            from joblib import Parallel, delayed

            data = np.ones(1000).astype(np.int8)
            if __name__ == '__main__':
                result = Parallel(
                    n_jobs=2, verbose={}, max_nbytes=100)(
                        delayed(len)(data) for _ in range(10))
        '''.format(verbose)
        p = subprocess.Popen([sys.executable, '-c', cmd],
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
        p.wait()
        out, err = p.communicate()
        lines = err.decode().split('\n')[:-1]
        assert out == b''
        if verbose in (0, 49):
            assert all('reduction' not in l for l in lines)
        if verbose == 51:
            assert any(l.startswith('joblib.reduction:INFO') for l in lines)
            assert all(
                not l.startswith('joblib.reduction:DEBUG') for l in lines
            )
        if verbose == 100:
            assert any(l.startswith('joblib.reduction:INFO') for l in lines)
            assert any(l.startswith('joblib.reduction:DEBUG') for l in lines)


def test_parallel_memory_verbosity(tmpdir):
    verbose_levels = [0, 1, 11]
    expected_lines = [1, 3, 7]
    for verbose, expected_lines in zip(verbose_levels, expected_lines):
        cmd = '''if 1:
            from joblib import Memory


            memory = Memory(location='{}', verbose={})

            @memory.cache
            def f(a):
                pass


            f(1)
            f(1)
            memory.clear()
        '''.format(tmpdir.strpath, verbose)
        p = subprocess.Popen([sys.executable, '-c', cmd],
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
        p.wait()
        out, err = p.communicate()
        lines = err.decode().split('\n')[:-1]
        assert out == b''
        assert all(l.startswith('joblib.memory') for l in lines)
        assert len(lines) == expected_lines
        if verbose == 0:
            assert all('INFO' not in l for l in lines)
        if verbose == 1:
            assert all('DEBUG' not in l for l in lines)
        if verbose == 11:
            assert any('DEBUG' not in l for l in lines)


def test_parallel_batching_verbosity():
    verbose_levels = [0, 59, 61]
    for verbose in verbose_levels:
        cmd = '''if 1:
            import numpy as np
            from joblib import Parallel, delayed

            if __name__ == '__main__':
                result = Parallel(n_jobs=2, verbose={})(
                    delayed(id)(i) for i in range(10))
        '''.format(verbose)
        p = subprocess.Popen([sys.executable, '-c', cmd],
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
        p.wait()
        out, err = p.communicate()
        lines = err.decode().split('\n')[:-1]
        assert out == b''
        if verbose in (0, 59):
            assert all('batching' not in l for l in lines)
        if verbose == 61:
            assert any(l.startswith('joblib.batching') for l in lines)


def test_logging_filtering():
    cmd = '''if 1:
        import logging

        from joblib import Parallel, delayed

        logging.getLogger('joblib.parallel').handlers[0].setLevel(logging.ERROR)
        logging.getLogger('joblib.reduction').handlers[0].setLevel(logging.ERROR)
        logging.getLogger('joblib.batching').handlers[0].setLevel(logging.ERROR)
        logging.getLogger('joblib.memory').handlers[0].setLevel(logging.ERROR)

        if __name__ == '__main__':
            result = Parallel(n_jobs=2, verbose=100)(
                delayed(id)(i) for i in range(100))
    '''
    p = subprocess.Popen([sys.executable, '-c', cmd],
                         stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    p.wait()
    out, err = p.communicate()
    assert out == b''
    assert err == b''
