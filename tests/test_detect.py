#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:09:21 2020

@author: johnmiller

Run command:
    $pytest tests/test_detect.py --cov=pybor.nltk -s --disable-pytest-warnings

Warning disabled because most are from libraries out of my control.

import sys

sys.path.append("src")

"""
import pytest
import pybor.nltk


@pytest.fixture(scope="session")
def data_setup(request):
    print("Setting up wordlist.")

    # Load data
    # From mobor since load_data not in pybor
    wordlist = mobor.data.load_data('wold')

    # Subset data and select only borrowed items (suited for WOLD)
    # TODO: replace hardcoded selector, allow no selector
    # could include criterion as argument to permit other than 0.375
    # could allow for lambda function as well
    subset = wordlist.get_language(
        "English", ["formchars", "borrowed"], dtypes=[list, float]
    )
    tokens = [row["formchars"] for row in subset]
    borrowedscore = [row["borrowed"] for row in subset]
    return tokens, borrowedscore

    def fin():
        print("Teardown")
        #Do nothing

def test_dual_model(data_setup):
    print("Test dual native-loan detect model")
    tokens, borrowedscore = data_setup
    dm, _ = pybor.nltk.detect_native_loan_dual_basis(
            tokens=tokens,
            borrowedscore=borrowedscore,
            method='kni',
            smoothing=0.5,
            order=3,
            trainfrac=0.8,
    )

    print('single word prediction:', '\nword=', tokens[37],
          dm.predict(tokens[37]), borrowedscore[37])

    print('multiple word prediction:','\nword=', tokens[37:40],
          dm.predict_data(tokens[37:40]), borrowedscore[37:40])


def test_native_model(data_setup):
    print("Test native detect model.")
    tokens, borrowedscore = data_setup
    nm, _ = pybor.nltk.detect_native_loan_native_basis(
            tokens=tokens,
            borrowedscore=borrowedscore,
            method='kni',
            smoothing=0.5,
            order=3,
            p=0.995,
            trainfrac=0.8)

    print('single word prediction:', '\nword=', tokens[37],
          nm.predict(tokens[37]), borrowedscore[37])

    print('multiple word prediction:','\nword=', tokens[37:40],
          nm.predict_data(tokens[37:40]), borrowedscore[37:40])

