#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test entropies.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest tests/test_data.py -—cov=pybor.data

Or to save as html report:
$ pytest tests/test_data.py -—cov=pybor.data --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_data.py --cov=pybor.data --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_data.py::test_gettable --cov=pybor.data --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""

from pybor.data import LexibankDataset
from pybor.dev.data import testing, training
import pybor.data as data


def test_instantiate_LexibankDateset():
    lex = data.get_lexibank_access()
    assert lex

def test_gettable():

    lex = data.get_lexibank_access()
    table = lex.get_table(
        language='English', form='FormChars', classification='Borrowed')
    assert table

    print(f'table size={len(table)}')

    print('English table sample:')
    print(table[10:25])
    num_loan = sum([row[2] for row in table])
    print(f'Sum loan={num_loan}')
    assert num_loan > 0
    print('Compare devel table:')
    print(training[10:25])

def test_language_table_gen():
    gen = data.language_table_gen('English')
    language, table = next(gen)
    assert language == 'English'
    assert len(table) == 1516  # As of June 18, 2020
    assert len(table[0]) == 3
    print(table[:4])

    gen = data.language_table_gen(['English', 'Hup'])
    language, table = next(gen)
    assert language == 'English'
    assert len(table) == 1516  # As of June 18, 2020
    assert len(table[0]) == 3
    print(table[:2])
    for language, table in gen:
        assert language == 'Hup'
        assert len(table) > 500
        assert len(table[0]) == 3

    gen = data.language_table_gen()
    for language, table in gen:
        print(language, table[0])

def test_access_functions():
    lex = data.get_lexibank_access()
    language='Hup'
    languages = data.check_languages_with_lexibank(lex, language)
    print(languages)
    assert languages[0] == language


if __name__ == "__main__":
    #test_instantiate_LexibankDateset()
    test_gettable()
    test_language_table_gen()
    test_access_functions()


