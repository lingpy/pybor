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




from collections import namedtuple
import sklearn.metrics as metrics

Test_metrics = namedtuple('Binary_prediction', ['Acc', 'Maj_acc', 'Prec', 'Recall', 'F1'])

def calculate_metrics(ground, forecast):
    assert(len(ground) == len(forecast))
    prec_recall_f1 = metrics.precision_recall_fscore_support(
        ground, forecast, average='binary')[:-1]

    acc = metrics.accuracy_score(ground, forecast)
    maxpredict = max(sum(ground),len(ground)-sum(ground))
    maj_acc = maxpredict/len(ground)

    return Test_metrics(Acc=acc, Maj_acc=maj_acc, Prec=prec_recall_f1[0],
                           Recall=prec_recall_f1[1], F1=prec_recall_f1[2])


def print_test_prediction(test_quality: Test_metrics):
    print('\nQuality metrics:')
    print(test_quality)