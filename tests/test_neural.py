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

from pybor.config import *
from pybor.neural import NeuralData
from pybor.entropies import NeuralWord
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
    attention = AttentionSettings(epochs=10)
    recurrent = RecurrentSettings(epochs=10)
    neural = Neural(testing=testing1, training=training1)
    assert neural.vocab is not None
    assert neural.native_data is not None
    assert neural.native_data.testing is not None
    assert neural.native_model is not None
    assert neural.loan_data is not None
    assert neural.loan_data.testing is not None

def test_instantiation1():
    attention = AttentionSettings(epochs=10)
    recurrent = RecurrentSettings(epochs=10)
    language='A very very very very very long long long ... language name'
    neural = Neural(training=training1, testing=[], detect_type='native', model_type='recurrent', 
            settings=recurrent, language=language)
    assert neural.language == language
    assert neural.model_type == 'recurrent'
    assert neural.detect_type == 'native'
    assert not neural.native_data.testing 
    assert not neural.loan_data.testing
    #assert neural.loan_model is None

def test_instantiation2():
    attention = AttentionSettings(epochs=10)
    recurrent = RecurrentSettings(epochs=10)
    neural = Neural(training1, testing1,
                    detect_type='dual', model_type='recurrent',
                    settings=recurrent, language='German')
    assert neural.language == 'German'
    assert neural.model_type == 'recurrent'
    assert neural.detect_type == 'dual'
    assert neural.native_data.testing is not None
    assert neural.native_model is not None
    assert neural.loan_data.testing is not None
    #assert neural.loan_model is not None

def test_instantiation3():
    attention = AttentionSettings(epochs=10)
    recurrent = RecurrentSettings(epochs=10)
    neural = Neural(training1, testing1,
                    detect_type='native', model_type='attention',
                    language='German',
                    settings=attention)
    assert neural.model_type == 'attention'
    assert neural.native_data.testing is not None
    assert neural.loan_data.testing is not None
    #assert neural.loan_model is None

# =============================================================================
#
# After this point, tests primarily see whether the methods execute.
#
# =============================================================================

def test_prediction():
    attention = AttentionSettings(epochs=30)
    recurrent = RecurrentSettings(epochs=30)
    neural = Neural(training1, testing1,
                    detect_type='dual', model_type='recurrent',
                    settings=recurrent, language='German')
    assert neural.loan_model 
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
#
def test_prediction1():
    attention = AttentionSettings(epochs=30)
    recurrent = RecurrentSettings(epochs=30)
    neural = Neural(training1, testing1,
                    detect_type='native', settings=recurrent, model_type='recurrent', 
                    language='German')
    #assert not neural.loan_model
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
#
def test_prediction2():
    attention = AttentionSettings(epochs=30)
    recurrent = RecurrentSettings(epochs=30)
    neural = Neural(training1, testing1,
                    detect_type='native', settings=attention, model_type='attention', language = 'German')
    #assert neural.loan_model is None
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

#def test_prediction3():
#    ncfg.attention['epochs'] = 30
#    ncfg.recurrent['epochs'] = 30
#    neural = Neural(training1, testing1,
#                    detect_type='dual', model_type='attention', language = 'German')
#    assert neural.loan_model is not None
#    print("Evaluate train dataset.")
#    predictions = neural.predict_data(training1)
#    train_metrics = evaluate.evaluate_model(predictions, training1)
#    evaluate.print_evaluation(train_metrics)
#    evaluate.false_positive(predictions, training1)
#
#    print("Evaluate test dataset.")
#    predictions = neural.predict_data(testing1)
#    test_metrics = evaluate.evaluate_model(predictions, testing1)
#    evaluate.print_evaluation(test_metrics)
#    evaluate.false_positive(predictions, testing1)
#
#    print("Individual prediction")
#    #    [53, ['z', 'u', 'm', 'p͡f'], 0],
#    #    [54, ['m', 'oː', 'r', 'a', 's', 't'], 1],
#    token = ['z', 'u', 'm', 'p͡f']
#    print(f'id: 53, token: {token}, prediction: {neural.predict(token)}, truth: 0')
#    token = ['m', 'oː', 'r', 'a', 's', 't']
#    print(f'id: 54, token: {token}, prediction: {neural.predict(token)}, truth: 1')

