"""
Evaluate accuracy of borrowing detection.
"""
from collections import namedtuple
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
                fp += 1
            else:
                fn += 1
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

    total = tp+tn+fp+fn
    accuracy = (tp+tn)/total if total > 0 else 0

    return precision, recall, fs, accuracy


# =============================================================================
#
# Quality measures for model reporting
#
# Drops sklearn.metrics in favor of prf at some cost in data preparation.
#
# =============================================================================
Bin_eval = namedtuple('Bin_eval', ['prec', 'recall', 'f1', 'acc', 'maj'])

def evaluate_model(test_data, data):
    """
    Evaluation loan word detection on precision, recall, F1, accuracy basis.

    Parameters
    ----------
    tst_data : [[str, [str], int]]
        List of language tokens in row format:
            [id, [char segments], predicted loanword status.

    data : [[str, [str], int]]
        List of language tokens in row format:
            [id, [segments], gold-status].

    Returns
    -------
    Bin_eval
        Evaluation as named tuple of precision, recall, F1, accuracy, majority.

    """

    (prec, recall, f1, acc) = prf(test_data, data)

    gold = [status for _, _, status in data]
    maj = max(sum(gold),len(gold)-sum(gold))/len(gold)

    return Bin_eval(prec=prec, recall=recall, f1=f1, acc=acc, maj=maj)


def print_evaluation(evaluation: Bin_eval):

    table = [['Precision', 'Recall', 'F-score', 'Accuracy', "Majority"],
             [evaluation.prec, evaluation.recall, evaluation.f1,
              evaluation.acc, evaluation.maj]]
    print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.3f'))
