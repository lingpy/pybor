from pybor.svm import BagOfSounds
from pybor.dev.data import testing, training
from pybor.evaluate import false_positive

bags = BagOfSounds(training, kernel='rbf')
tests = bags.predict_data([[a, b] for a, b, c in testing])
false_positive(tests, testing, pprint=True)

