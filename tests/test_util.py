"""
Created on Sat Jun  6 10:48:14 2020

@author: johnmiller

Test util.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest testys/test_util.py -—cov=pybor.util

Or to save as html report:
$ pytest tests/test_util.py -—cov=pybor.util --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_util.py --cov=pybor.util --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_util.py::test_prediction2 --cov=pybor.util --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""

import statistics
import math

import pybor.util as util
from pybor.dev.data import testing1, training1

import pytest

# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================d
def test_find_acc_cut_point():
    native = [0.5, 1.0, 1.25, 1.5, 2.0, 3.0]
    loan = [1.35, 2.5, 3.5, 4.0]

    # Native: 0.5   1.0   1.25         1.5   2.0        3.0
    # Loan:                     1.35               2.5        3.5   4.0
    # score  4    5     6     7      6     7     8     7    8     7     6
    # max score is 8, cut_point is 2.0

    cut_point = util.find_acc_cut_point(native, loan)
    print('cut_point', cut_point)
    assert cut_point == 2.0

def test_find_frac_cut_point():
    native = [0.5, 1.0, 1.25, 1.5, 2.0, 3.0]

    cut_point = util.find_frac_cut_point(native, fraction=0.5)
    print('cut_point', cut_point)
    assert cut_point == 1.375

    cut_point = util.find_frac_cut_point(native, fraction=0.95)
    print('cut_point', cut_point)
    assert cut_point == 2.5  # Bias to within the distribution

    cut_point = util.find_frac_cut_point(native, fraction=0.05)
    print('cut_point', cut_point)
    assert cut_point == 0.75  # Bias to within the distribution

    cut_point = util.find_frac_cut_point(native, fraction=1.0)
    print('cut_point', cut_point)
    assert cut_point == 3.0

    cut_point = util.find_frac_cut_point(native, fraction=0.0)
    print('cut_point', cut_point)
    assert cut_point == 0.5


def test_find_acc_cut_point_deltas():
    # Just shift location to simulate deltas.
    # Native: 0.5   1.0   1.25         1.5   2.0        3.0
    # Loan:                     1.35               2.5        3.5   4.0
    # score  4    5     6     7      6     7     8     7    8     7     6
    # max score is 8, cut_point is 2.0

    native = [-0.5, 0.0, 0.25, 0.5, 1.0, 2.0]
    loan = [0.35, 1.5, 2.5, 3.0]

    cut_point = util.find_acc_cut_point_deltas(native, loan)
    print('cut_point', cut_point)
    assert cut_point == 1.0

    native = [-1.5, -1.0, -0.75, -0.5, 0.0, 1.0]
    loan = [-0.65, 0.5, 1.5, 2.0]

    cut_point = util.find_acc_cut_point_deltas(native, loan)
    print('cut_point', cut_point)
    assert cut_point == 0.0

def test_find_fscore_cut_point_deltas():
    native = [-1.5, -1.0, -0.75, -0.5, 0.0, 1.0]
    loan = [-0.65, 0.5, 1.5, 2.0]
    print(native)
    print(loan)

    f = util.find_fscore_cut_point_deltas(native, loan)
    print(f'cut_point {f:.2f}.')
    assert f == 0.0

def test_calculate_fscore():
    n_true_pos = 3
    n_true = 5
    n_pos = 8

    f = util.calculate_fscore(n_true_pos, n_true, n_pos)
    print(f'fscore {f:.4f}.')
    assert abs(f - 0.4615) < 0.0001

    f = util.calculate_fscore(n_true_pos, n_true, 0)
    assert f == 0.0
    f = util.calculate_fscore(n_true_pos, 0, n_pos)
    assert f == 0.0
    f = util.calculate_fscore(0, n_true, n_pos)
    assert f == 0

def test_train_test_split():

    train, test = util.train_test_split(training1, split=0.20)
    assert len(train) == int(0.80*len(training1))
    assert len(test) == int(math.ceil(0.20*len(training1)))


def test_logger():
    logger = util.get_logger(__name__)

if __name__ == "__main__":
    test_find_acc_cut_point()
    test_find_frac_cut_point()
    test_find_acc_cut_point_deltas()
    test_calculate_fscore()
    test_find_fscore_cut_point_deltas()