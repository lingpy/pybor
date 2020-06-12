from pybor.dev.data import training, testing
from pybor.svm import *
from pybor.data import LexibankDataset
from pybor.evaluate import prf
import pickle
from statistics import mean
from sys import argv
from pybor.ngram import NgramModel
from pybor.markov import DualMarkov
import random



# check different methods against each other
if 'ngram' in argv:
    model = NgramModel
elif 'dual' in argv:
    model = DualMarkov
else:
    model = BagOfSounds

# check for borrowing rate
if '-b' in argv:
    brate = int(argv[argv.index('-b')+1])
else:
    brate = 20

def bigrams(sequence):
    return list(zip(['^']+sequence[:-1], sequence[1:]+['$']))

def trigrams(sequence):
    return list(zip(['^', '^']+sequence[:-1], ['^']+sequence+['$'],
        sequence[1:]+['$', '$']))


# load binary wold
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

fakes = []
for a, b, c in training+testing:
    if c != 1:
        fakes += [[a, b, 1]]


table = []
stats = []
for language in lex.languages.values():

    table = lex.get_table(
            language=language['Name'],
            form='FormChars',
            classification='Loan'
            )
    train, test = [], []
    random.shuffle(table)
    fakeidx = list(range(len(fakes)))
    for i in range(len(table)//2):
        if random.randint(0, brate) == 0:
            fake = fakes[fakeidx.pop(random.randint(0, len(fakeidx)-1))]
            train += [fake]
        else:
            train += [[table[i][0], table[i][1], 0]]
    for i in range(len(table)//2, len(table)):
        if random.randint(0, brate) == 0:
            fake = fakes[fakeidx.pop(random.randint(0, len(fakeidx)-1))]
            test += [fake]
        else:
            test += [[table[i][0], table[i][1], 0]]
    
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
