#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Markov.py module of detection of loan words.


All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest testy/test_markov.py -—cov=pybor.markov

Or to save as html report:
$ pytest tests/test_markov.py -—cov=pybor.markov --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_markov.py --cov=pybor.markov --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_markov.py::test_prediction2 --cov=pybor.markov --cov-report term-missing

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""

import pybor.config as config
from pybor.markov import DualMarkov, NativeMarkov
import pybor.evaluate as evaluate

from pybor.dev.data import testing1, training1

def test_DualMarkov_predict_data():
    print('\nTest dual Markov predict data')
    dm = DualMarkov(training1)

    print('Training set predictions')
    predictions = dm.predict_data(training1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, training1, True)

    results =  evaluate.evaluate_model(predictions, training1)

    print('Testing set predictions')
    predictions = dm.predict_data(testing1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, testing1, True)
    results =  evaluate.evaluate_model(predictions, testing1)

    row = testing1[42]
    print('Check out token row', row)
    token = row[1]
    print('Entropy for', token, dm.native.calculate_entropy(token))
    print('Entropy for', token, dm.loan.calculate_entropy(token))
    print('Prediction for',token, dm.predict(token))

def test_NativeMarkov_predict_data():
    print('\nTest native Markov predict data')
    nm = NativeMarkov(training1)

    print('Training set predictions')
    predictions = nm.predict_data(training1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, training1, True)

    results =  evaluate.evaluate_model(predictions, training1)

    print('Testing set predictions')
    predictions = nm.predict_data(testing1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, testing1, True)
    results =  evaluate.evaluate_model(predictions, testing1)

    row = testing1[42]
    print('Check out token row', row)
    token = row[1]
    print('Entropy for', token, nm.nativemodel.calculate_entropy(token))
    print('Prediction for',token, nm.predict(token))

def get_tokens(data):
    return [token for _, token, _ in data]

def test_settings():
    settings = config.MarkovSettings()
    dm = DualMarkov(training1, settings=settings)
    assert dm.model == 'kni'
    assert dm.smoothing == 0.1
    assert dm.order == 3
    assert dm.native.model == 'kni'
    assert dm.native.smoothing == 0.1
    assert dm.native.order == 3
    assert dm.loan.model == 'kni'
    assert dm.loan.smoothing == 0.1
    assert dm.loan.order == 3
    assert dm.settings.order == 3
    assert dm.settings.smoothing == 0.1
    assert dm.settings.model == 'kni'

    dm = DualMarkov(training1, order=2, smoothing=0.1, model='ls', settings=settings)
    assert dm.model == 'ls'
    assert dm.smoothing == 0.1
    assert dm.order == 2
    assert dm.native.model == 'ls'
    assert dm.native.smoothing == 0.1
    assert dm.native.order == 2
    assert dm.loan.model == 'ls'
    assert dm.loan.smoothing == 0.1
    assert dm.loan.order == 2
    assert dm.settings.order == 3
    assert dm.settings.smoothing == 0.1
    assert dm.settings.model == 'kni'

    dm = DualMarkov(training1, model='ls', settings=settings)
    assert dm.model == 'ls'
    assert dm.smoothing == 0.1
    assert dm.order == 3
    assert dm.native.model == 'ls'
    assert dm.native.smoothing == 0.1
    assert dm.native.order == 3
    assert dm.loan.model == 'ls'
    assert dm.loan.smoothing == 0.1
    assert dm.loan.order == 3
    assert dm.settings.order == 3
    assert dm.settings.smoothing == 0.1
    assert dm.settings.model == 'kni'

    settings = config.MarkovSettings(order=2, smoothing=0.3, model='lp')
    dm = DualMarkov(training1, model='ls', settings=settings)
    assert dm.model == 'ls'
    assert dm.smoothing == 0.3
    assert dm.order == 2
    assert dm.native.model == 'ls'
    assert dm.native.smoothing == 0.3
    assert dm.native.order == 2
    assert dm.loan.model == 'ls'
    assert dm.loan.smoothing == 0.3
    assert dm.loan.order == 2
    assert dm.settings.order == 2
    assert dm.settings.smoothing == 0.3
    assert dm.settings.model == 'lp'

    settings = config.MarkovSettings(smoothing=0.3, model='lp')
    nm = NativeMarkov(training1, model='ls', p=0.90, settings=settings)
    assert nm.model == 'ls'
    assert nm.smoothing == 0.3
    assert nm.order == 3
    assert nm.p == 0.90
    assert nm.nativemodel.model == 'ls'
    assert nm.nativemodel.smoothing == 0.3
    assert nm.nativemodel.order == 3
    assert nm.settings.order == 3
    assert nm.settings.smoothing == 0.3
    assert nm.settings.model == 'lp'
    assert nm.settings.p == 0.995

    settings = config.MarkovSettings(smoothing=0.2, model='lp', p=0.95)
    nm = NativeMarkov(training1, settings=settings)
    assert nm.model == 'lp'
    assert nm.smoothing == 0.2
    assert nm.order == 3
    assert nm.p == 0.95
    assert nm.nativemodel.model == 'lp'
    assert nm.nativemodel.smoothing == 0.2
    assert nm.nativemodel.order == 3
    assert nm.settings.order == 3
    assert nm.settings.smoothing == 0.2
    assert nm.settings.model == 'lp'
    assert nm.settings.p == 0.95

    settings = config.MarkovSettings(smoothing=0.2)
    nm = NativeMarkov(training1, settings=settings)
    assert nm.model == 'kni'
    assert nm.smoothing == 0.2
    assert nm.order == 3
    assert nm.p == 0.995
    assert nm.nativemodel.model == 'kni'
    assert nm.nativemodel.smoothing == 0.2
    assert nm.nativemodel.order == 3
    assert nm.settings.order == 3
    assert nm.settings.smoothing == 0.2
    assert nm.settings.model == 'kni'
    assert nm.settings.p == 0.995

    nm = NativeMarkov(training1)
    assert nm.model == 'kni'
    assert nm.smoothing == 0.1
    assert nm.order == 3
    assert nm.p == 0.995
    assert nm.nativemodel.model == 'kni'
    assert nm.nativemodel.smoothing == 0.1
    assert nm.nativemodel.order == 3
    assert nm.settings.order == 3
    assert nm.settings.smoothing == 0.1
    assert nm.settings.model == 'kni'
    assert nm.settings.p == 0.995

if __name__ == "__main__":
    #test_DualMarkov_predict_data()
    #test_NativeMarkov_predict_data()
    test_settings()