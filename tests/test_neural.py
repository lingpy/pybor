#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:55:36 2020

@author: johnmiller

Test neural_tf.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest testys/test_neural.py -—cov=pybor.neural

Or to save as html report:
$ pytest tests/test_neural.py -—cov=pybor.neural --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_neural.py --cov=pybor.neural --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_neural.py::test_prediction2 --cov=pybor.neural --cov-report term-missing

File names can come last instead of first in the argument list.
"""

import pybor.neural_cfg as ncfg
from pybor.data_tf import NeuralData
from pybor.neural_tf import NeuralWord
from pybor.neural import Neural
import pybor.evaluate as evaluate

from pybor.dev.data import testing1, training1

import pytest

# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================d
def test_instantiation():
    # Reduce time for test with fewer epochs
    ncfg.attention['epochs'] = 10
    ncfg.recurrent['epochs'] = 10
    neural = Neural(training1, testing1)
    assert neural.language == ncfg.neural['language']
    assert neural.series == ncfg.neural['series']
    assert neural.detect_type == ncfg.neural['detect_type']
    assert neural.model_type == ncfg.neural['model_type']
    assert neural.vocab is not None
    assert neural.native_data is not None
    assert neural.native_data.test_data is not None
    assert neural.native_model is not None
    assert neural.loan_data is not None
    assert neural.loan_data.test_data is not None

def test_instantiation1():
    ncfg.attention['epochs'] = 10
    ncfg.recurrent['epochs'] = 10
    language='A very very very very very long long long ... language name'
    neural = Neural(training1, detect_type='native', model_type='recurrent', language = language)
    assert neural.language == language
    assert neural.model_type == 'recurrent'
    assert neural.detect_type == 'native'
    assert neural.native_data.test_data is None
    assert neural.loan_data.test_data is None
    assert neural.loan_model is None

def test_instantiation2():
    ncfg.attention['epochs'] = 10
    ncfg.recurrent['epochs'] = 10
    neural = Neural(training1, testing1,
                    detect_type='dual', model_type='recurrent', language = 'German')
    assert neural.language == 'German'
    assert neural.model_type == 'recurrent'
    assert neural.detect_type == 'dual'
    assert neural.native_data.test_data is not None
    assert neural.native_model is not None
    assert neural.loan_data.test_data is not None
    assert neural.loan_model is not None

def test_instantiation3():
    ncfg.attention['epochs'] = 10
    ncfg.recurrent['epochs'] = 10
    neural = Neural(training1, testing1,
                    detect_type='native', model_type='attention', language = 'German')
    assert neural.model_type == 'attention'
    assert neural.native_data.test_data is not None
    assert neural.loan_data.test_data is not None
    assert neural.loan_model is None

def test_prediction():
    ncfg.attention['epochs'] = 30
    ncfg.recurrent['epochs'] = 30
    neural = Neural(training1, testing1,
                    detect_type='dual', model_type='recurrent', language = 'German')
    assert neural.loan_model is not None
    print("Evaluate train dataset.")
    predictions = neural.predict_data(training1)
    train_metrics = evaluate.evaluate_model(predictions, training1)
    evaluate.print_evaluation(train_metrics)
    evaluate.false_positive(predictions, training1)

    print("Evaluate test dataset.")
    predictions = neural.predict_data(testing1)
    test_metrics = evaluate.evaluate_model(predictions, testing1)
    evaluate.print_evaluation(test_metrics)
    evaluate.false_positive(predictions, testing1)

def test_prediction1():
    ncfg.attention['epochs'] = 30
    ncfg.recurrent['epochs'] = 30
    neural = Neural(training1, testing1,
                    detect_type='native', model_type='recurrent', language = 'German')
    assert neural.loan_model is None
    print("Evaluate train dataset.")
    predictions = neural.predict_data(training1)
    train_metrics = evaluate.evaluate_model(predictions, training1)
    evaluate.print_evaluation(train_metrics)
    evaluate.false_positive(predictions, training1)

    print("Evaluate test dataset.")
    predictions = neural.predict_data(testing1)
    test_metrics = evaluate.evaluate_model(predictions, testing1)
    evaluate.print_evaluation(test_metrics)
    evaluate.false_positive(predictions, testing1)

def test_prediction2():
    ncfg.attention['epochs'] = 30
    ncfg.recurrent['epochs'] = 30
    neural = Neural(training1, testing1,
                    detect_type='native', model_type='attention', language = 'German')
    assert neural.loan_model is None
    print("Evaluate train dataset.")
    predictions = neural.predict_data(training1)
    train_metrics = evaluate.evaluate_model(predictions, training1)
    evaluate.print_evaluation(train_metrics)
    evaluate.false_positive(predictions, training1)

    print("Evaluate test dataset.")
    predictions = neural.predict_data(testing1)
    test_metrics = evaluate.evaluate_model(predictions, testing1)
    evaluate.print_evaluation(test_metrics)
    evaluate.false_positive(predictions, testing1)

    print("Individual prediction")
    #    [53, ['z', 'u', 'm', 'p͡f'], 0],
    #    [54, ['m', 'oː', 'r', 'a', 's', 't'], 1],
    token = ['z', 'u', 'm', 'p͡f']
    print(f'id: 53, token: {token}, prediction: {neural.predict(token)}, truth: 0')
    token = ['m', 'oː', 'r', 'a', 's', 't']
    print(f'id: 54, token: {token}, prediction: {neural.predict(token)}, truth: 1')

def test_prediction3():
    ncfg.attention['epochs'] = 30
    ncfg.recurrent['epochs'] = 30
    neural = Neural(training1, testing1,
                    detect_type='dual', model_type='attention', language = 'German')
    assert neural.loan_model is not None
    print("Evaluate train dataset.")
    predictions = neural.predict_data(training1)
    train_metrics = evaluate.evaluate_model(predictions, training1)
    evaluate.print_evaluation(train_metrics)
    evaluate.false_positive(predictions, training1)

    print("Evaluate test dataset.")
    predictions = neural.predict_data(testing1)
    test_metrics = evaluate.evaluate_model(predictions, testing1)
    evaluate.print_evaluation(test_metrics)
    evaluate.false_positive(predictions, testing1)

    print("Individual prediction")
    #    [53, ['z', 'u', 'm', 'p͡f'], 0],
    #    [54, ['m', 'oː', 'r', 'a', 's', 't'], 1],
    token = ['z', 'u', 'm', 'p͡f']
    print(f'id: 53, token: {token}, prediction: {neural.predict(token)}, truth: 0')
    token = ['m', 'oː', 'r', 'a', 's', 't']
    print(f'id: 54, token: {token}, prediction: {neural.predict(token)}, truth: 1')

# =============================================================================
# Helper functions for testing
#
# =============================================================================



# =============================================================================
# Execute tests as standalone file
# =============================================================================

if __name__ == "__main__":
    # test_instantiation()
    # test_instantiation1()
    # test_instantiation2()
    # test_instantiation3()
    # test_prediction()
    # test_prediction1()
    # test_prediction2()
    test_prediction3()