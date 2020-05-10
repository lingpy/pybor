from pybor.dev.data import training, testing
from pybor.svm import *
from pybor.data import LexibankDataset
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

try:
    with open('wold.bin', 'rb') as f:
        lex = pickle.load(f)
except:
    lex = LexibankDataset(
            'wold', 
            transform={
                "Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0}
            )
    with open('wold.bin', 'wb') as f:
        pickle.dump(lex, f)

table = []
for language in lex.languages.values():
    table = lex.get_table(
            language=language['Name'], 
            form='FormChars', 
            classification='Loan'
            )
    train, test = table[:len(table)//2], table[len(table)//2:]
    
    bag = model(train)
    guess = model.predict_data([[a, b] for a, b, c in test])

    p, r, f, a = prf(test, guess)
    print('{4:30} | {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}'.format(
        p, r, f, a, language['Name']))
    table += [[p, r, f, a]]

print('{4:30} | {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}'.format(
    [line[0] for line in table], 
    [line[1] for line in table], 
    [line[2] for line in table], 
    [line[3] for line in table], 'TOTAL'))
