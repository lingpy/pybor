from pybor.ngram import NgramModel
from pybor.dev.data import testing, training
from pybor.evaluate import prf, false_positive

ngram_model = NgramModel(training)
tests = ngram_model.predict_data([[a, b] for a, b, c in testing])

false_positive(tests, testing, pprint=True)

p, r, f = prf(tests, testing)
if all([p, r, f]):
    print('Precision: {0:.2f}'.format(p))
    print('Recall:    {0:.2f}'.format(r))
    print('F-Score:   {0:.2f}'.format(f))
