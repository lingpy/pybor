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
Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""

from pybor.config import AttentionSettings, RecurrentSettings, EntropiesSettings, BaseSettings
from pybor.config import NeuralSettings #, NeuralNativeSettings, NeuralDualSettings
from pybor.neural import NeuralData
from pybor.entropies import NeuralWord, NeuralWordRecurrent, NeuralWordAttention
from pybor.neural import NeuralNative, NeuralDual
import pybor.evaluate as evaluate

from pybor.dev.data import testing1, training1

import pytest


# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================
def test_vocab():
    data = NeuralData(training1)

    print(f'len(vocab)={len(data.vocab)}.')
    assert len(data.vocab) == 114
    print(f'vocab.size={data.vocab.size}.')
    assert data.vocab.size == len(data.vocab.vocab)//2
    print(f'vocab[15]={data.vocab[15]}.')
    assert data.vocab[15] == data.vocab.vocab[15]
    print(f'vocab.get(15)={data.vocab.get(15)}.')
    assert data.vocab.get('u') == data.vocab.vocab['u']
    print(f'vocab.get(16)={data.vocab.get(16)}.')
    print(f'vocab.get(15,"<unk>")={data.vocab.get(15,"<unk>")}.')
    assert data.vocab.get(15, "<unk>") == data.vocab.vocab[15]
    print(f'vocab.get(255,"<unk>")={data.vocab.get(255,"<unk>")}.')
    assert data.vocab.get(255, "<unk>") == "<unk>"

def test_data():
    data = NeuralData(training1, testing1)
    tokens = data.get_data_tokens(data.fit)
    assert len(tokens) == len(data.fit)
    tokens_ids = data.get_tokens_ids(tokens)
    assert len(tokens_ids) == len(tokens)
    assert data.translate(tokens) == tokens_ids
    tokens_ids_ = data.get_data_tokens_ids(data.fit)
    assert tokens_ids == tokens_ids_

    trainer = data.trainer
    validator = data.validator
    tester = data.tester
    assert trainer.data_len == len(data.fit)
    assert validator.data_len == len(data.val)
    assert tester.data_len == len(data.testing)
    assert trainer == data.trainer
    assert validator == data.validator
    assert tester == data.tester

    # Iterate at least 1ce through generator.
    gen = trainer.generate()
    x, y = next(gen)
    print('x=', x)
    print('y=', y)
    steps = trainer.data_len//trainer.batch_size
    for i, (x, y) in enumerate(gen):
        if i > steps: break

def test_instantiation():
    # Reduce time for test with fewer epochs
    settings = RecurrentSettings(epochs=10, tf_verbose=1)
    neural = NeuralDual(testing=testing1, training=training1, settings=settings)
    assert neural.vocab is not None
    assert neural.model_type == 'recurrent'
    assert neural.native_data is not None
    assert neural.native_data.testing is not None
    assert neural.native_model is not None
    assert neural.loan_data is not None
    assert neural.loan_data.testing is not None
    assert isinstance(neural, NeuralDual)
    assert isinstance(neural.native_model, NeuralWordRecurrent)
    assert isinstance(neural.loan_model, NeuralWordRecurrent)
    assert isinstance(neural.native_model.settings, RecurrentSettings)
    assert isinstance(neural.loan_model.settings, RecurrentSettings)

def test_instantiation1():
    language='A very very very very very long long long ... language name'
    neural = NeuralNative(training=training1, testing=[],
                          model_type='attention', language=language)
    assert neural.language == language
    assert neural.model_type == 'attention'
    assert not neural.native_data.testing
    assert isinstance(neural, NeuralNative)
    assert isinstance(neural.native_model, NeuralWordAttention)
    assert isinstance(neural.native_model.settings, AttentionSettings)

def test_instantiation2():
    neural = NeuralDual(training1, testing1, model_type='recurrent', language='German')
    assert neural.language == 'German'
    assert neural.model_type == 'recurrent'
    assert neural.native_data.testing is not None
    assert neural.native_model is not None
    assert neural.loan_data.testing is not None
    assert neural.loan_model is not None
    assert isinstance(neural, NeuralDual)
    assert isinstance(neural.native_model, NeuralWordRecurrent)
    assert isinstance(neural.loan_model, NeuralWordRecurrent)
    assert isinstance(neural.native_model.settings, RecurrentSettings)
    assert isinstance(neural.loan_model.settings, RecurrentSettings)

def test_instantiation3():
    settings = AttentionSettings(epochs=10, tf_verbose=1)
    neural = NeuralNative(training1, testing1, model_type='attention',
                          language='German', settings=settings)
    assert settings.epochs == 10
    assert settings.tf_verbose == 1
    assert neural.settings.epochs == 10
    assert neural.settings.tf_verbose == 1

    neural.train()
    assert neural.model_type == 'attention'
    assert neural.native_data.testing is not None
    assert isinstance(neural.native_model, NeuralWordAttention)
    assert isinstance(neural.settings, NeuralSettings)
    assert isinstance(neural.native_model.settings, AttentionSettings)

# =============================================================================
#
# After this point, tests primarily see whether the methods execute.
#
# =============================================================================

def test_prediction():
    settings = RecurrentSettings(epochs=50)
    neural = NeuralDual(training1, testing1, model_type='recurrent',
                        language='German', settings=settings)
    assert neural.loan_model
    neural.train()

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
    settings = RecurrentSettings(epochs=30)
    neural = NeuralNative(training1, testing1, model_type='recurrent',
                          language='German', settings=settings)
    neural.train()

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
    neural = NeuralNative(training1, testing1, model_type='attention', language = 'German')
    neural.train()

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

    neural = NeuralDual(training1, testing1, model_type='attention', language='German')
    neural.train()
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

if __name__ == "__main__":
    #test_vocab()
    #test_data()
    test_instantiation3()
    #test_prediction()