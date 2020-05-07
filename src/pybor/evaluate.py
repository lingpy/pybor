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

    n_obs = tp+tn+fp+fn
    accuracy = (tp+tn)/n_obs if n_obs > 0 else 0

    return precision, recall, fs, accuracy


# =============================================================================
#
# Quality measures for model reporting
#
# Drops sklearn.metrics in favor of prf at some cost in data preparation.
#
# =============================================================================
Bin_eval = namedtuple('Bin_eval', ['prec', 'recall', 'f1', 'acc', 'maj'])

def evaluate_model(model, tokens, ground):

    forecast = model.predict_data(tokens)

    # Prepare in prf expected format.
    ground = ground.astype(int)
    forecast = forecast.astype(int)
    groundtable = [[idx, ground, tokens] for idx, ground, tokens
                   in zip(range(len(ground)),tokens,ground)]
    forecasttable = [[idx, forecast, tokens] for idx, forecast, tokens
                   in zip(range(len(forecast)),tokens,forecast)]

    (prec, recall, f1, acc) = prf(forecasttable, groundtable)
    #(prec, recall, f1, acc) = calculate_metrics(forecast, ground)

    maj = max(sum(ground),len(ground)-sum(ground))/len(ground)
    return Bin_eval(prec=prec, recall=recall, f1=f1, acc=acc, maj=maj)


def print_evaluation(evaluation: Bin_eval):

    table = [['Precision', 'Recall', 'F-score', 'Accuracy', "Majority"],
             [evaluation.prec, evaluation.recall, evaluation.f1,
              evaluation.acc, evaluation.maj]]
    print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.2f'))
