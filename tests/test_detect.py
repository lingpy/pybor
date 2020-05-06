#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:09:21 2020

@author: johnmiller
"""

import sys

sys.path.append("src")

import mobor
import mobor.data
import mobor.detect_borrowing_ngram

# Load data
wordlist = mobor.data.load_data("wold")

# Subset data and select only borrowed items (suited for WOLD)
# TODO: replace hardcoded selector, allow no selector
# could include criterion as argument to permit other than 0.375
# could allow for lambda function as well
subset = wordlist.get_language(
    "English", ["formchars", "borrowed"], dtypes=[list, float]
)
tokens = [row["formchars"] for row in subset]
borrowedscore = [row["borrowed"] for row in subset]

mobor.detect_borrowing_ngram.detect_native_loan_dual_basis(
    tokens=tokens,
    borrowedscore=borrowedscore,
    method="kni",
    smoothing=0.5,
    order=3,
    trainfrac=0.8,
)
