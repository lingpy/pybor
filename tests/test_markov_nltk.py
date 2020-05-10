#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test markov_nltk implementation of Markov word model for calclating entropies.
"""
import statistics
import random
import math

from pybor.markov_nltk import MarkovWord
from pybor.dev.data import testing, training
from pybor.data import LexibankDataset


def test_with_training():
    tokens = get_tokens(training)
    assert len(training) == len(tokens)
    alt_tokens = get_tokens(testing)
    run_generic_test(tokens, alt_tokens)

def test_with_testing():
    tokens = get_tokens(testing)
    assert len(testing) == len(tokens)
    alt_tokens = get_tokens(training)
    run_generic_test(tokens, alt_tokens)


def test_with_English_table():
    ds = LexibankDataset('wold', transform=
        {"Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0})
    table = ds.get_table(
        language='English', form='FormChars', classification='Loan')
    assert table

    tokens = [token for _, token, _ in table]
    tokens = random.sample(tokens, len(tokens))
    idx = math.ceil(len(tokens)*0.75)
    train = tokens[:idx]
    test = tokens[idx+1:]

    run_generic_test(train, test)

def run_generic_test(tokens, alt_tokens):
    print("Train model")
    markov = MarkovWord(tokens, order=3)
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
    assert n > 0 and n < 10000  # unlikely to have so many in testing.
    assert mean > 1.0 and mean < 5.0  # all human languages should conform to this.
    assert stdev > 0.0 and stdev < 4.0
    print(f'max entropy={max(entropies):.2f}')




def get_tokens(data):
    return [token for _, token, _ in data]

if __name__ == "__main__":
    test_with_training()
    test_with_testing()
    test_with_English_table()