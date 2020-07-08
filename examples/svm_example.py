from pybor.svm import BagOfSounds
from pybor.dev.data import testing, training
from pybor.evaluate import prf, false_positive


def bigrams(sequence):
    return list(zip(["^"] + sequence[:-1], sequence[1:] + ["$"]))


def trigrams(sequence):
    return list(
        zip(
            ["^", "^"] + sequence[:-1],
            ["^"] + sequence + ["$"],
            sequence[1:] + ["$", "$"],
        )
    )


training2 = [[a, bigrams(b), c] for a, b, c in training]
testing2 = [[a, bigrams(b), c] for a, b, c in testing]

training3 = [[a, trigrams(b), c] for a, b, c in training]
testing3 = [[a, trigrams(b), c] for a, b, c in testing]


bags = BagOfSounds(training, kernel="linear")
tests = bags.predict_data([[a, b] for a, b, c in testing])

false_positive(tests, testing, pprint=True)

prec, rec, fs, acc = prf(tests, testing)
if all([prec, rec, fs, acc]):
    print("Precision: {0:.2f}".format(prec))
    print("Recall:    {0:.2f}".format(rec))
    print("F-Score:   {0:.2f}".format(fs))
    print("Accuracy:  {0:.2f}".format(acc))


bags = BagOfSounds(training2, kernel="linear")
tests = bags.predict_data([[a, b] for a, b, c in testing2])

false_positive(tests, testing2, pprint=True)

prec, rec, fs, acc = prf(tests, testing)
if all([prec, rec, fs, acc]):
    print("Precision: {0:.2f}".format(prec))
    print("Recall:    {0:.2f}".format(rec))
    print("F-Score:   {0:.2f}".format(fs))
    print("Accuracy:  {0:.2f}".format(acc))

bags = BagOfSounds(training3, kernel="linear")
tests = bags.predict_data([[a, b] for a, b, c in testing3])

false_positive(tests, testing3, pprint=True)

prec, rec, fs, acc = prf(tests, testing3)
if all([prec, rec, fs, acc]):
    print("Precision: {0:.2f}".format(prec))
    print("Recall:    {0:.2f}".format(rec))
    print("F-Score:   {0:.2f}".format(fs))
    print("Accuracy:  {0:.2f}".format(acc))
