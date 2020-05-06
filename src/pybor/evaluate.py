"""
Evaluate accuracy of borrowing detection.
"""
from tabulate import tabulate

def false_positive(test, gold, pprint=True):
    """
    Simple stats on false positives.
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    total = len(test)
    for (idxA, wordA, judgmentA), (idxB, wordB, judgmentB) in zip(
            test, gold):
        assert idxA == idxB
        if judgmentA == judgmentB:
            if judgmentB == 0:
                tn += 1
            elif judgmentB == 1:
                tp += 1
        else:
            if judgmentB == 0:
                fn += 1
            else:
                fp += 1
    if pprint:
        table = [
                ['', 'True', 'False', 'Total'],
                ['Positives', tp, fp, tp+fp],
                ['Negatives', tn, fn, tn+fn],
                ['Total', (tp+tn)/total, (fp+fn)/total, total]
                ]
        print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.2f'))
    return tp, tn, fp, fn


def prf(test, gold):
    """
    Compute precision, recall, and f-score for test and gold.
    """
    
    tp, tn, fp, fn = false_positive(test, gold, pprint=False)
    try:
        precision = tp/(tp+fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp/(tp+fn)
    except ZeroDivisionError:
        recall = 0
    if not precision and not recall:
        fs = 0
    else:
        fs = 2*(precision*recall)/(precision+recall)
    
    return precision, recall, fs
