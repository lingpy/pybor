#!/usr/bin/env python3

from pybor.dev.data import training, testing
from pybor.svm import *
import pybor.data as data
from pybor.evaluate import prf
import pickle
from statistics import mean
from sys import argv
from pybor.ngram import NgramModel
from pybor.markov import DualMarkov

# check different methods against each other
if 'ngram' in argv:
    model = NgramModel
elif 'dual' in argv:
    model = DualMarkov
else:
    model = BagOfSounds

def bigrams(sequence):
    return list(zip(['^']+sequence[:-1], sequence[1:]+['$']))

def trigrams(sequence):
    return list(zip(['^', '^']+sequence[:-1], ['^']+sequence+['$'],
        sequence[1:]+['$', '$']))

lex = data.get_lexibank_access()
table = []
stats = []
for language in lex.languages.values():
    table = lex.get_table(
            language=language['Name'],
            form='Tokens',
            classification='Borrowed'
            )
    train, test = table[:len(table)//2], table[len(table)//2:]
    
    train2, test2 = [[a, bigrams(b), c] for a, b, c in train], \
            [[a, bigrams(b), c] for a, b, c in test]
    train3, test3 = [[a, trigrams(b), c] for a, b, c in train], \
            [[a, trigrams(b), c] for a, b, c in test]

    
    if 'bg' in argv:
        bag = model(train2)
        guess = bag.predict_data([[a, b] for a, b, c in test2])
    elif 'tg' in argv:
        bag = model(train3)
        guess = bag.predict_data([[a, b] for a, b, c in test3])
    else:
        bag = model(train)
        guess = bag.predict_data([[a, b] for a, b, c in test])

    p, r, f, a = prf(test, guess)
    print('{4:30} | {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}'.format(
        p, r, f, a, language['Name']))
    stats += [[p, r, f, a]]

print('{4:30} | {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}'.format(
    mean([line[0] for line in stats]),
    mean([line[1] for line in stats]),
    mean([line[2] for line in stats]),
    mean([line[3] for line in stats]), 'TOTAL/MEAN'))
