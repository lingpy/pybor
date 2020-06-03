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
$ pytest testys/test_entropies.py -—cov=pybor.entropies

Or to save as html report:
$ pytest tests/test_entropies.py -—cov=pybor.entropies --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_entropies.py --cov=pybor.test_entropies --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_entropies.py::test_prediction2 --cov=pybor.entropies --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""
import statistics

from pybor.config import BaseSettings, EntropiesSettings, RecurrentSettings, AttentionSettings
from pybor.neural import Vocab, NeuralData, KerasBatchGenerator
from pybor.entropies import NeuralWord, NeuralWordRecurrent, NeuralWordAttention
from pybor.dev.data import testing1, training1

import pytest

# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================d
def test_instantiation():


    data = NeuralData(training1, testing1)

    word = NeuralWordRecurrent(data.vocab.size)
    assert word.vocab_len == len(data.vocab)//2
    assert word.language == ''
    assert word.basis == 'all'
    assert word.series == ''
    assert word.model is not None

    word = NeuralWordAttention(data.vocab.size, language='German', series='Test')
    assert word.vocab_len == len(data.vocab)//2
    assert word.language == 'German'
    assert word.basis == 'all'
    assert word.series == 'Test'
    assert word.model is not None

    settings = RecurrentSettings(print_summary=True, batch_size=64)
    word = NeuralWordRecurrent(data.vocab.size, settings=settings)
    assert word.settings.print_summary==True
    assert word.settings.batch_size==64

def test_non_positive_vocab_len():
    with pytest.raises(Exception):
        assert NeuralWordRecurrent(0)

def test_no_vocab_len():
    with pytest.raises(Exception):
        assert NeuralWordAttention()

def test_train_model():

    data = NeuralData(training1, testing1)

    word = NeuralWordRecurrent(data.vocab.size, language='German')
    word.train(train_gen=data.trainer, val_gen=data.validator)
    word.evaluate_test(test_gen=data.tester)

    word = NeuralWordAttention(data.vocab.size, language='German')
    word.train(train_gen=data.trainer, val_gen=data.validator)
    word.evaluate_test(test_gen=data.tester)

    tokens = data.get_data_tokens(data.testing)
    # For German data, max token size is 10, so 12 with start and stop symbols.
    tokens_ids = data.get_data_tokens_ids(data.testing)
    #BaseSettings.token_maxlen = 10
    entropies = word.calculate_entropies(tokens_ids)
    entropy = word.calculate_entropy(tokens_ids[17])
    print('Token:', tokens[17], tokens_ids[17], ' entropy=', entropy)
    assert abs(entropies[17] - entropy) < 1.0e-5

def get_mean_entropies(data, word):
    fit_tokens_ids = data.get_data_tokens_ids(data.fit)
    val_tokens_ids = data.get_data_tokens_ids(data.val)
    test_tokens_ids = data.get_data_tokens_ids(data.testing)
    fit_mean_entropy = statistics.mean(word.calculate_entropies(fit_tokens_ids))
    val_mean_entropy = statistics.mean(word.calculate_entropies(val_tokens_ids))
    test_mean_entropy = statistics.mean(word.calculate_entropies(test_tokens_ids))
    print('***')
    print(f'Entropies: fit {fit_mean_entropy:.4f}, ' +
          f'val {val_mean_entropy:.4f}, ' +
          f'test {test_mean_entropy:.4f}.')
    return (fit_mean_entropy, val_mean_entropy, test_mean_entropy)

def test_train_model_with_cfg():
    data = NeuralData(training1, testing1)

    settings = RecurrentSettings(rnn_cell_type='LSTM', epochs=10, embedding_dropout=0.1)

    word = NeuralWordRecurrent(data.vocab.size, language='German', settings=settings)
    word.train(train_gen=data.trainer, val_gen=data.validator)
    word.evaluate_test(test_gen=data.tester)
    get_mean_entropies(data, word)

    settings = AttentionSettings(rnn_cell_type='GRU', embedding_dropout=0.1)

    word = NeuralWordAttention(data.vocab.size, language='German', settings=settings)
    word.train(train_gen=data.trainer, val_gen=data.validator)
    word.evaluate_test(test_gen=data.tester)
    get_mean_entropies(data, word)


# =============================================================================
# Helper functions for testing
#
# =============================================================================



# =============================================================================
# Execute tests as standalone file
# =============================================================================

if __name__ == "__main__":
    test_instantiation()
    # test_non_positive_vocab_len()
    # test_no_vocab_len()
    # test_train_model()
    test_train_model_with_cfg()