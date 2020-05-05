#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:52:35 2020

@author: johnmiller
"""
import numpy as np
from collections import namedtuple
import sklearn.metrics as metrics
import statistics
import math

Test_prediction = namedtuple('Binary_prediction', ['Acc', 'Maj_acc', 'Prec', 'Recall', 'F1'])

def report_metrics(ground, forecast, quiet=False):
    # ground and entropies are on token basis.
    assert(len(ground) == len(forecast)), f'ground size ({len(ground)}) not equal forecast size ({len(forecast)})'
    # forecast is versus the upper ref_limit.  Most should be < the limit and so native words.
    #forecast = [e < self._ref_limit for e in self._entropies]
    #print('Test for entropy <', self._ref_limit)
    prec_recall_f1 = metrics.precision_recall_fscore_support(ground, forecast, average='binary')[:-1]
    acc = metrics.accuracy_score(ground, forecast)
    maxpredict = max(sum(ground),len(ground)-sum(ground))
    maj_acc = maxpredict/len(ground)
    if not quiet:
        print('precision, recall, F1 =', prec_recall_f1)
        print('n =', len(ground), ' accuracy =', acc)
        print('confusion matrix: tn, fp, fn, tp', metrics.confusion_matrix(ground, forecast).ravel())
        print('Predict majority: accuracy=', maj_acc)

    return Test_prediction(Acc=acc, Maj_acc=maj_acc, Prec=prec_recall_f1[0],
                       Recall=prec_recall_f1[1], F1=prec_recall_f1[2])


def calculate_empirical_ref_limit(entropies, frac=0.99, quiet=False):
    # Entropies are not power law distributed, but neither are they Gaussian.
    # Better to use fraction of distribution to use as cut-point for discriminating between native and loan.
    idx = min((len(entropies)-1)*frac, len(entropies)-1)
    entropies = np.sort(entropies)
    ref_limit = (entropies[math.floor(idx)]+entropies[math.ceil(idx)])/2
    avg = statistics.mean(entropies)
    stdev = statistics.stdev(entropies)
    if not quiet:
        print(f'Native avg={avg:.3f}, stdev={stdev:.3f}')
        print(f'fraction {frac:.3f}, idx {idx:.2f}, ref limit={ref_limit:.3f}')
    return ref_limit
