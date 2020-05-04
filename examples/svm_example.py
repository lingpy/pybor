from pybor.svm import BagOfSounds
from pybor.dev.data import testing, training
from tabulate import tabulate

bags = BagOfSounds(training)
tp, tn, fp, fn = 0, 0, 0, 0
for idx, word, status in testing:
    res = bags.predict(word)
    if res > 0.5:
        if status == 1:
            tp += 1
        else:
            fp += 1
    else:
        if status == 0:
            tn += 1
        else:
            fn += 1

table = [
        ['', 'Borrowing', 'No Borrowing', 'Score'],
        ['True', tp, tn, (tp+tn)/(tp+tn+fp+fn)],
        ['False', fp, fn, (fp+fn)/(tp+tn+fp+fn)],
        ['Score', (tp+fp)/(tp+tn+fp+fn), (tn+fn)/(tp+tn+fp+fn)]
        ]

print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.2f'))
