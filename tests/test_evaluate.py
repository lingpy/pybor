"""
Created on Thu May 21 18:20:30 2020

@author: johnmiller

Test evaluate.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest tests/test_evaluate.py -—cov=pybor.evaluate

Or to save as html report:
$ pytest tests/test_evaluate.py -—cov=pybor.evaluate --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_evaluate.py --cov=pybor.evaluate --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_evaluate.py::test_prediction2 --cov=pybor.evaluate --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""
from pybor.evaluate import false_positive, prf, evaluate_model
from pybor.dev.data import testing, training


def test_false_positive():
    tp, tn, fp, fn = false_positive(testing, testing, pprint=True)
    assert fp == 0
    new_data = [[idx, word, 1] for idx, word, _ in testing]
    new_data2 = [[idx, word, 0] for idx, word, _ in testing]
    tp, tn, fp, fn = false_positive(new_data, new_data2)
    assert tp == 0


def test_prf():
    p, r, f, a = prf(testing, testing)
    assert p == 1.0
    assert a == calc_acc(testing, testing)
    new_data = [[idx, word, 1] for idx, word, _ in testing]
    new_data2 = [[idx, word, 0] for idx, word, _ in testing]
    p, r, f, a = prf(new_data, new_data2)
    assert p == 0
    assert a == calc_acc(new_data2, new_data)
    p, r, f, a = prf(new_data2, new_data)
    assert p == 0
    assert a == calc_acc(new_data2, new_data)

def calc_acc(test, gold):
    correct = sum([t==g for t, g in zip(test, gold)])
    return correct/len(test)



def test_prf_benchmark():
    import sklearn.metrics as metrics

    gold = [1]*50 + [0]*50
    test = [1, 1, 0, 1]*25

    p_, r_, f_ = metrics.precision_recall_fscore_support(gold, test, average='binary')[:-1]
    acc_ = metrics.accuracy_score(gold, test)
    print(f'benchmark: prec, recall, f1, acc = {(p_, r_, f_, acc_)}')

    # Expects datsets of [id, [form], loan] where loan is either test or gold indicators
    t100 = testing[:100]
    ids100 = [row[0] for row in t100]
    forms100 = [row[1] for row in t100]

    test_ds = [[ident, form, test_] for ident, form, test_ in zip(ids100, forms100, test)]
    gold_ds = [[ident, form, gold_] for ident, form, gold_ in zip(ids100, forms100, gold)]

    p, r, f, a = prf(test_ds, gold_ds)
    print(f'code: prec, recall, f1, acc = {(p, r, f, a)}')

    assert abs(p - p_) < 1e-7
    assert abs(r - r_) < 1e-7
    assert abs(f - f_) < 1e-7
    assert abs(a - acc_) < 1e-7


def test_evaluate_model():
    import sklearn.metrics as metrics

    gold = [1]*50 + [0]*50
    test = [1, 1, 0, 1]*25

    p_, r_, f_ = metrics.precision_recall_fscore_support(gold, test, average='binary')[:-1]
    acc_ = metrics.accuracy_score(gold, test)
    print(f'benchmark: prec, recall, f1, acc = {(p_, r_, f_, acc_)}')

    # Expects datsets of [id, [form], loan] where loan is either test or gold indicators
    t100 = testing[:100]
    ids100 = [row[0] for row in t100]
    forms100 = [row[1] for row in t100]

    test_ds = [[ident, form, test_] for ident, form, test_ in zip(ids100, forms100, test)]
    gold_ds = [[ident, form, gold_] for ident, form, gold_ in zip(ids100, forms100, gold)]

    results = evaluate_model(test_ds, gold_ds)

    assert abs(results[0] - p_) < 1e-7
    assert abs(results[1] - r_) < 1e-7
    assert abs(results[2] - f_) < 1e-7
    assert abs(results[3] - acc_) < 1e-7


if __name__ == "__main__":
    test_false_positive()
    test_prf()
    test_prf_benchmark()
    test_evaluate_model()