"""
Utility functions for the package.
"""
from pathlib import Path
from collections import namedtuple
import sklearn.metrics as metrics

def test_data(*comps):
    return Path(__file__).parent.parent.parent.joinpath('tests', 'data',
            *comps).as_posix()


Test_prediction = namedtuple('Binary_prediction', ['Acc', 'Maj_acc', 'Prec', 'Recall', 'F1'])

def calculate_metrics(ground, forecast):
    assert(len(ground) == len(forecast))
    prec_recall_f1 = metrics.precision_recall_fscore_support(
        ground, forecast, average='binary')[:-1]
    acc = metrics.accuracy_score(ground, forecast)
    maxpredict = max(sum(ground),len(ground)-sum(ground))
    maj_acc = maxpredict/len(ground)

    return Test_prediction(Acc=acc, Maj_acc=maj_acc, Prec=prec_recall_f1[0],
                           Recall=prec_recall_f1[1], F1=prec_recall_f1[2])
