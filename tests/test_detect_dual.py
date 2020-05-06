#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:09:21 2020

@author: johnmiller
"""

import sys
sys.path.append('src')

import pybor, mobor
import mobor.data
import pybor.nltk

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

dm, _ = pybor.nltk.detect_native_loan_dual_basis(
        tokens=tokens,
        borrowedscore=borrowedscore,
        method='kni',
        smoothing=0.5,
        order=3,
        trainfrac=0.8,
)

print('word=', tokens[37], dm.predict(tokens[37]), borrowedscore[37])

print('word=', tokens[37:40], dm.predict_data(tokens[37:40]), borrowedscore[37:40])

