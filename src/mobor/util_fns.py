#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:52:35 2020

@author: johnmiller
"""
from collections import namedtuple

Test_prediction = namedtuple('Binary_prediction', ['Acc', 'Maj_acc', 'Prec', 'Recall', 'F1'])

def report_metrics(ground, forecast):
    import sklearn.metrics as metrics
    # ground and entropies are on token basis.
    assert(len(ground) == len(forecast)), f'ground size ({len(ground)}) not equal forecast size ({len(forecast)})'
    # forecast is versus the upper ref_limit.  Most should be < the limit and so native words.
    #forecast = [e < self._ref_limit for e in self._entropies]
    #print('Test for entropy <', self._ref_limit)
    prec_recall_f1 = metrics.precision_recall_fscore_support(ground, forecast, average='binary')[:-1]
    print('precision, recall, F1 =', prec_recall_f1)
    acc = metrics.accuracy_score(ground, forecast)
    print('n =', len(ground), ' accuracy =', acc)
    print('confusion matrix: tn, fp, fn, tp', metrics.confusion_matrix(ground, forecast).ravel())
    maxpredict = max(sum(ground),len(ground)-sum(ground))
    maj_acc = maxpredict/len(ground)
    print('Predict majority: accuracy=', maj_acc)

    return Test_prediction(Acc=acc, Maj_acc=maj_acc, Prec=prec_recall_f1[0],
                       Recall=prec_recall_f1[1], F1=prec_recall_f1[2])
