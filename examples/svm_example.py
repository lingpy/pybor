from pybor.svm import BagOfSounds
from pybor.dev.data import testing, training
from pybor.evaluate import prf, false_positive

bags = BagOfSounds(training, kernel='rbf')
tests = bags.predict_data([[a, b] for a, b, c in testing])

false_positive(tests, testing, pprint=True)

prec, rec, fs, acc = prf(tests, testing)
if all([prec, rec, fs, acc]):
    print('Precision: {0:.2f}'.format(prec))
    print('Recall:    {0:.2f}'.format(rec))
    print('F-Score:   {0:.2f}'.format(fs))
    print('Accuracy:  {0:.2f}'.format(acc))
