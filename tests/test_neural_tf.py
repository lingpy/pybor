#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:20:30 2020

@author: johnmiller

Test neural_tf.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest testys/test_neural_tf.py -—cov=pybor.neural_tf

Or to save as html report:
$ pytest tests/test_neural_tf.py -—cov=pybor.neural_tf --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_neural_tf.py --cov=pybor.neural_tf --cov-report term-missing

"""

import pybor.neural_cfg as ncfg
from pybor.data_tf import NeuralData
from pybor.neural_tf import NeuralWord
from pybor.dev.data import testing1, training1

import pytest

# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================d
def test_instantiation():
    data = NeuralData(training1, testing1)

    ncfg.recurrent['print_summary'] = True
    ncfg.recurrent['plot_model'] = True

    word = NeuralWord(len(data.vocab), model_type='recurrent')
    assert word.model is not None
    assert word.vocab_len == len(data.vocab)
    assert word.language == ''
    assert word.basis == 'all'
    assert word.series == ''
    assert word.name == ''
    assert word.model_type == 'recurrent'

    ncfg.attention['print_summary'] = True
    ncfg.attention['plot_model'] = True
    word = NeuralWord(len(data.vocab), model_type='attention',
                      language='German', series='Test', name='attention 1')
    assert word.model is not None
    assert word.vocab_len == len(data.vocab)
    assert word.language == 'German'
    assert word.basis == 'all'
    assert word.series == 'Test'
    assert word.name == 'attention 1'
    assert word.model_type == 'attention'

    word = NeuralWord(len(data.vocab), model_type='super-model')
    assert word.model_type == 'recurrent', 'model type is %r, should be recurrent' %word.model_type

def test_non_positive_vocab_len():
    with pytest.raises(Exception):
        assert NeuralWord(0, model_type='recurrent')
    with pytest.raises(Exception):
        assert NeuralWord(model_type='recurrent')

def test_train_model():
    data = NeuralData(training1, testing1)
    train_gen, val_gen, test_gen = data.make_generators()

    word = NeuralWord(len(data.vocab), model_type='recurrent', language='German')
    word.train(train_gen=train_gen, val_gen=val_gen)
    word.evaluate_test(test_gen=test_gen)

    word = NeuralWord(len(data.vocab), model_type='attention', language='German')
    word.train(train_gen=train_gen, val_gen=val_gen)
    word.evaluate_test(test_gen=test_gen)

    entropies = word.calculate_entropies(data.test_tokens_ids)
    entropy = word.calculate_entropy(data.test_tokens_ids[17])
    assert abs(entropies[17] - entropy) < 1.0e-5


def test_train_model_post_cfg_change():
    data = NeuralData(training1, testing1)
    train_gen, val_gen, test_gen = data.make_generators()

    ncfg.recurrent['rnn_cell_type'] = 'LSTM'
    ncfg.recurrent['embedding_dropout'] = 0.1
    word = NeuralWord(len(data.vocab), model_type='recurrent', language='German')
    word.train(train_gen=train_gen, val_gen=val_gen)
    word.evaluate_test(test_gen=test_gen)

    ncfg.attention['rnn_cell_type'] = 'GRU'
    ncfg.attention['embedding_dropout'] = 0.1
    word = NeuralWord(len(data.vocab), model_type='attention', language='German')
    word.train(train_gen=train_gen, val_gen=val_gen)
    word.evaluate_test(test_gen=test_gen)

# =============================================================================
# Helper functions for testing
#
# =============================================================================



# =============================================================================
# Execute tests as standalone file
# =============================================================================

if __name__ == "__main__":
    test_instantiation()
    test_non_positive_vocab_len()
    test_train_model()
    test_train_model_post_cfg_change()