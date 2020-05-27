#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test markov_nltk.py implementation of Markov word model for calclating entropies.


All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest testy/test_markov_nltk.py -—cov=pybor.markov_nltk

Or to save as html report:
$ pytest tests/test_markov_nltk.py -—cov=pybor.markov_nltk --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_markov_nltk.py --cov=pybor.markov_nltk --cov-report term-missing

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""
import statistics
import pytest

from pybor.markov_nltk import MarkovWord
from pybor.dev.data import testing1, training1


def test_with_training():
    print('Train with training1, alternative with testing1.')
    tokens = get_tokens(training1)
    assert len(training1) == len(tokens)
    alt_tokens = get_tokens(testing1)
    run_generic_test(tokens, alt_tokens)

def test_with_testing():
    print('Train with testing1, alternative with training1.')
    tokens = get_tokens(testing1)
    assert len(testing1) == len(tokens)
    alt_tokens = get_tokens(training1)
    run_generic_test(tokens, alt_tokens)


def test_with_various_models():
    print('Train various models with training1, alternative with testing1.')
    tokens = get_tokens(training1)
    alt_tokens = get_tokens(testing1)

    run_generic_test(tokens, alt_tokens, model='kni', smoothing=0.1)
    run_generic_test(tokens, alt_tokens, model='kni', smoothing=0.0)
    run_generic_test(tokens, alt_tokens, model='wbi')
    run_generic_test(tokens, alt_tokens, model='lp')
    run_generic_test(tokens, alt_tokens, model='ls', smoothing=0.5)
    run_generic_test(tokens, alt_tokens, model='ls', smoothing=0.1)
    run_generic_test(tokens, alt_tokens, model='ls', smoothing=0.0)

def test_with_mle_model_fails() :
    print('Train mle with training1, alternative with testing1.')
    tokens = get_tokens(training1)
    alt_tokens = get_tokens(testing1)

    with pytest.raises(Exception) as e_info:
        run_generic_test(tokens, alt_tokens, model='mle')


def run_generic_test(tokens, alt_tokens, model='kni', smoothing=0.5):
    print()
    print(f"Train model {model}.")
    markov = MarkovWord(tokens, order=3, model=model, smoothing=smoothing)
    entropies = markov.calculate_entropies(tokens)
    assert len(entropies) == len(tokens)
    check_statistics(entropies)

    token = tokens[35]
    entropy = markov.calculate_entropy(token)
    assert entropy > 0.0 and entropy < 12.0
    print(f'token={token}, entropy={entropy:.3f}')

    print("Try model alternative data")
    alt_entropies = markov.calculate_entropies(alt_tokens)
    assert len(alt_entropies) == len(alt_tokens)
    check_statistics(alt_entropies)

def check_statistics(entropies):
    n = len(entropies)
    mean = statistics.mean(entropies)
    stdev = statistics.stdev(entropies)
    print(f'entropies: n={n},  mean={mean:.3f},  stdev={stdev:.3f}')
    # Base 2 entropy calculation.
    assert n > 0 and n < 10000  # unlikely to have so many in testing.
    assert mean > 1.0 and mean < 7.0  # all human languages should conform to this for Markov.
    assert stdev > 0.0 and stdev < 4.0
    print(f'max entropy={max(entropies):.2f}')




def get_tokens(data):
    return [token for _, token, _ in data]

if __name__ == "__main__":
    test_with_training()
    test_with_testing()
    test_with_various_models()