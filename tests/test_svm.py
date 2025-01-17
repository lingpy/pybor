"""
To test and report coverage:

% pytest tests/test_svm.py --cov=pybor.svm --cov-report term-missing

"""
import pytest
from pybor.svm import BagOfSounds
from pybor.dev.data import testing, training


def test_svm():
    bags = BagOfSounds(training)
    tests = bags.predict_data([[a, b] for a, b, c in testing])
    assert bags.predict(["m", "a", "t", "ə", "r"]) == 0
