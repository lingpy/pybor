#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

sys.path.append("src")

# NOTE: keeping code in place, but commented out, until rewritten
# following the new architecture

COMMENT_OUT = """
from mobor.data import Wordlist
from mobor.markov import Markov
from mobor.plot import plot_word_distributions
import math

wl = Wordlist.from_lexibank(
        'wold',
        fields=['borrowed'],
        fieldfunctions={
            "borrowed": lambda x: (int(x[0])*-1+5)/4
            })

print('loaded markov')

mk = Markov(
        wl,
        'English',
        ['concept', 'form', 'formchars', 'tokens', 'sca', 'borrowed'],
        dtypes = [str, str, str, str, str, float],
        post_order=2,
        pre_order=0
        )
mk.add_sequences(
        [row['formchars'] for row in mk.now['dicts']])
mk.train(normalize='kneserney', smoothing=0.5)

# retrieve distribution for loan words
loan, native = [], []
for row in mk.now['dicts']:
    if row['borrowed']>=0.5:
        loan += [mk.entropy(row['formchars'], base=math.e)]
    else:
        native += [mk.entropy(row['formchars'], base=math.e)]

plot_word_distributions(native, loan, filename='test.pdf',
                    title="English language native and loan entropies",
                    graphlimit=max([max(loan), max(native)])+1)

print('emitted graph to test.pdf')
"""
