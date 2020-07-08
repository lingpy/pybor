"""
Evaluate accuracy of borrowing detection.
"""

# Import 3rd-party libraries
from tabulate import tabulate


def false_positive(test, gold, pprint=True):
    """
    Simple stats on false positives.
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    total = len(test)
    for (idx_a, _, judgment_a), (idx_b, _, judgment_b) in zip(test, gold):
        assert idx_a == idx_b
        if judgment_a == judgment_b:
            if judgment_b == 0:
                tn += 1
            elif judgment_b == 1:
                tp += 1
        else:
            if judgment_b == 0:
                fp += 1
            else:
                fn += 1
    if pprint:
        table = [
            ["", "True", "False", "Total"],
            ["Positives", tp, fp, tp + fp],
            ["Negatives", tn, fn, tn + fn],
            ["Total", (tp + tn) / total, (fp + fn) / total, total],
        ]
        print(tabulate(table, tablefmt="pipe", headers="firstrow", floatfmt=".2f"))

    return tp, tn, fp, fn


def prf(test, gold):
    """
    Compute precision, recall, and f-score for test and gold.
    """

    tp, tn, fp, fn = false_positive(test, gold, pprint=False)

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    if not precision and not recall:
        fs = 0
    else:
        fs = 2 * (precision * recall) / (precision + recall)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    return precision, recall, fs, accuracy


# =============================================================================
#
# Convenience function for evaluating and printing prediction quality.
#
# =============================================================================


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

    Returns (float, float, float, float)
    -------
    (prec, recall, f1, acc)
        Evaluation as tuple of precision, recall, F1, accuracy, majority.

    """

    results = prf(test_data, data)
    table = [
        ["Precision", "Recall", "F-score", "Accuracy"],
        [results[0], results[1], results[2], results[3]],
    ]
    print(tabulate(table, tablefmt="pipe", headers="firstrow", floatfmt=".3f"))
    return results
