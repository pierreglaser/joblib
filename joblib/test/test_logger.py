"""
Test the logger module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import re
import subprocess
import sys

from joblib.logger import PrintTime


def test_print_time(tmpdir, capsys):
    # A simple smoke test for PrintTime.
    logfile = tmpdir.join('test.log').strpath
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
    """Check that mmap_mode is respected even at the first call"""
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
