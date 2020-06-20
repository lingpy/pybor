#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from pybor.data import LexibankDataset
from pybor.dev.data import testing, training
import pybor.data as data


def test_instantiate_LexibankDateset():
    ds = LexibankDataset('wold', transform=
            {"Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0})

    assert ds

def test_gettable():
    ds = LexibankDataset('wold', transform=
        {"Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0})

    table = ds.get_table(
        language='English', form='FormChars', classification='Loan')
    assert table

    print(f'table size={len(table)}')

    print('English table sample:')
    print(table[10:15])

    print('Compare devel table:')
    print(training[10:15])

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
    #test_gettable()
    #test_language_table_gen()
    test_access_functions()


