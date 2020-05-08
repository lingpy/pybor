from pybor.dev.data import training, testing
from pybor.svm import *
from pybor.data import LexibankDataset
from pybor.evaluate import prf

lex = LexibankDataset(
        'wold', 
        transform={
            "Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0}
        )
for language in lex.languages.values():
    table = lex.get_table(
            language=language['Name'], 
            form='FormChars', 
            classification='Loan'
            )
    train, test = table[:len(table)//2], table[len(table)//2:]
    bag = BagOfSounds(train)
    guess = bag.predict_data([[a, b] for a, b, c in test])
    p, r, f, a = prf(test, guess)
    print('{4:30} | {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}'.format(
        p, r, f, a, language['Name']))

