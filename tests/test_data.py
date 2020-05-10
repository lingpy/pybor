#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from pybor.data import LexibankDataset
from pybor.dev.data import testing, training


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



if __name__ == "__main__":
    #test_instantiate_LexibankDateset()
    test_gettable()
