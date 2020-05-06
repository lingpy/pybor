from pybor.evaluate import false_positive, prf
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