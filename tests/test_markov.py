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


from pybor.markov import DualMarkov
from pybor.markov import NativeMarkov
from pybor.dev.data import testing1, training1
import pybor.evaluate as evaluate


def test_DualMarkov_predict_data():
    print('\nTest dual Markov predict data')
    dm = DualMarkov(training1)

    print('Training set predictions')
    predictions = dm.predict_data(training1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, training1, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')

    results =  evaluate.evaluate_model(predictions, training1)
    evaluate.print_evaluation(results)

    print('Testing set predictions')
    predictions = dm.predict_data(testing1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, testing1, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')
    results =  evaluate.evaluate_model(predictions, testing1)
    evaluate.print_evaluation(results)

    row = testing1[42]
    print('Check out token row', row)
    token = row[1]
    print('Prediction for',token, dm.predict(token))

def test_NativeMarkov_predict_data():
    print('\nTest native Markov predict data')
    nm = NativeMarkov(training1)

    print('Training set predictions')
    predictions = nm.predict_data(training1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, training1, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')

    results =  evaluate.evaluate_model(predictions, training1)
    evaluate.print_evaluation(results)

    print('Testing set predictions')
    predictions = nm.predict_data(testing1)
    tp, tn, fp, fn = evaluate.false_positive(predictions, testing1, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')
    results =  evaluate.evaluate_model(predictions, testing1)
    evaluate.print_evaluation(results)

    row = testing1[42]
    print('Check out token row', row)
    token = row[1]
    print('Prediction for',token, nm.predict(token))

def get_tokens(data):
    return [token for _, token, _ in data]

