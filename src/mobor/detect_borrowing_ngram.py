#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:06:49 2020

@author: johnmiller
"""


#### ************************************************
####
#### Functions for discrimination on individual words.
####
#### ************************************************

## Dual model approach - native and loan

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from mobor import markov
import mobor.util_fns as util_fns


def fit_native_loan_models(
        tokens=None,
        ground=None,
        method="kni",
        smoothing=0.5,
        order=3,
        quiet=False,
):

    if ground is None:
        return None  # Must have ground to fit.

    tokens_native = [
        token for token, select in zip(tokens, ground) if select
    ]
    nativemodel = markov.MarkovCharLM(
        tokens_native, model=method, order=order, smoothing=smoothing
    )
    nativeentropies = nativemodel.analyze_tokens(tokens)

    tokens_loan = [
        token for token, select in zip(tokens, ground) if not select
    ]
    loanmodel = markov.MarkovCharLM(
        tokens_loan, model=method, order=order, smoothing=smoothing
    )
    loanentropies = loanmodel.analyze_tokens(tokens)

    forecast = np.less(nativeentropies, loanentropies)

    if not quiet:
        print()
        print("* TRAIN RESULTS *")
    util_fns.report_metrics(ground, forecast, quiet)

    return nativemodel, loanmodel


def evaluate_models_for_test(native_model=None, loan_model=None,
                             val_tokens=None, val_ground=None, quiet=False,
):
    # Calculate entropies for test set.
    native_entropies = native_model.analyze_tokens(val_tokens)
    loan_entropies = loan_model.analyze_tokens(val_tokens)
    forecast = np.less(native_entropies, loan_entropies)
    if not quiet:
        print()
        print("* TEST RESULTS *")
    metrics = util_fns.report_metrics(val_ground, forecast, quiet)
    return metrics


def detect_native_loan_dual_basis(
        tokens=None,
        borrowedscore=None,
        method='kni',
        smoothing=0.5,
        order=3,
        trainfrac=0.8,
):

    ground = np.array(borrowedscore) < 0.5  # native versus loan
    train_idx, val_idx = train_test_split(range(len(tokens)), test_size=1-trainfrac)
    train_tokens = np.array(tokens)[train_idx]
    val_tokens = np.array(tokens)[val_idx]
    train_ground = np.array(ground)[train_idx]
    val_ground = np.array(ground)[val_idx]

    native_model, loan_model = fit_native_loan_models(
        tokens=train_tokens, ground=train_ground, method=method,
        smoothing=smoothing, order=order
    )
    metrics = evaluate_models_for_test(
        native_model, loan_model, val_tokens, val_ground, quiet=False,
    )
    return metrics


## Native model approach

def fit_native_model(tokens=None, ground=None, method="kni",
                     order=3, smoothing=0.5, p=0.995, quiet=False
):
    if ground is None:
        return None  # Must have ground to fit.

    tokens_native = [
        token for token, select in zip(tokens, ground) if select == True
    ]
    native_model = markov.MarkovCharLM(
        tokens_native, model=method, order=order, smoothing=smoothing
    )
    # Calculate empirical distribution limit of native only entropies.
    native_entropies = native_model.analyze_tokens(tokens_native)
    ref_limit = util_fns.calculate_empirical_ref_limit(
                    native_entropies, frac=p, quiet=quiet)
    # Then test  versus all entropies.
    trainentropies = native_model.analyze_tokens(tokens)
    forecast = [e < ref_limit for e in trainentropies]
    if not quiet:
        print()
        print("* TRAIN RESULTS *")
    util_fns.report_metrics(ground, forecast, quiet)

    return native_model, ref_limit


def detect_native_basis(
        tokens=None,
        borrowedscore=None,
        method='kni',
        smoothing=0.5,
        order=3,
        trainfrac=0.8,
        p=0.995,
        quiet=False
):

    ground = np.array(borrowedscore) < 0.5  # native versus loan
    train_idx, val_idx = train_test_split(range(len(tokens)), test_size=1-trainfrac)
    train_tokens = np.array(tokens)[train_idx]
    val_tokens = np.array(tokens)[val_idx]
    train_ground = np.array(ground)[train_idx]
    val_ground = np.array(ground)[val_idx]

    native_model, ref_limit = fit_native_model(train_tokens, train_ground,
                    method=method, order=order, smoothing=smoothing, p=p)

    # Evaluate on test set.
    val_entropies = native_model.analyze_tokens(val_tokens)
    forecast = [e < ref_limit for e in val_entropies]
    if not quiet:
        print()
        print("* TEST RESULTS *")
    metrics = util_fns.report_metrics(val_ground, forecast, quiet)
    return metrics


#### ************************************************************
####
#### Validation functions for discrimination on individual words.
####
#### ************************************************************

def validate_kfold_loanword_detection_dual_basis(
        tokens,
        borrowedscore,
        output_path,
        method='kni',
        smoothing=0.5,
        order=3,
        kfold=10
):

    ground = np.array(borrowedscore) < 0.5  # native versus loan

    kf = KFold(n_splits=kfold, shuffle=True)
    predictions = []
    for train_idx, val_idx in kf.split(tokens):
        train_tokens = np.array(tokens)[train_idx]
        val_tokens = np.array(tokens)[val_idx]
        train_ground = ground[train_idx]
        val_ground = ground[val_idx]


        results = validate_native_loan_dual_basis(
            train_tokens, train_ground, val_tokens, val_ground,
            method=method, smoothing=smoothing, order=order)

        predictions.append(results._asdict())

    # build report table as the one from pandas
    columns = ["Acc", "Maj_acc", "Prec", "Recall", "F1"]
    df = {"Mean": {}, "StDev": {}}
    for column in columns:
        vals = [predict[column] for predict in predictions]
        df["Mean"][column] = np.mean(vals)
        df["StDev"][column] = np.std(vals)

    print(f"{kfold}-fold validation.")
    print("Means", df["Mean"])
    print("StDevs", df["StDev"])

    return df


def validate_native_loan_dual_basis(
        train_tokens=None,
        train_ground=None,
        val_tokens=None,
        val_ground=None,
        method='kni',
        smoothing=0.5,
        order=3,
):

    native_model, loan_model = fit_native_loan_models(
        tokens=train_tokens, ground=train_ground, method=method,
        smoothing=smoothing, order=order, quiet=True,
    )
    metrics = evaluate_models_for_test(
        native_model, loan_model, val_tokens, val_ground, quiet=True,
    )
    return metrics



def validate_kfold_loanword_detection_native_basis(
        tokens,
        borrowedscore,
        output_path,
        method='kni',
        smoothing=0.5,
        order=3,
        p=0.995,
        kfold=10
):

    ground = np.array(borrowedscore) < 0.5  # native versus loan

    kf = KFold(n_splits=kfold, shuffle=True)
    predictions = []
    for train_idx, val_idx in kf.split(tokens):
        train_tokens = np.array(tokens)[train_idx]
        val_tokens = np.array(tokens)[val_idx]
        train_ground = ground[train_idx]
        val_ground = ground[val_idx]


        results = validate_native_basis(
            train_tokens, train_ground, val_tokens, val_ground,
            method=method, smoothing=smoothing, order=order)

        predictions.append(results._asdict())

    # build report table as the one from pandas
    columns = ["Acc", "Maj_acc", "Prec", "Recall", "F1"]
    df = {"Mean": {}, "StDev": {}}
    for column in columns:
        vals = [predict[column] for predict in predictions]
        df["Mean"][column] = np.mean(vals)
        df["StDev"][column] = np.std(vals)

    print(f"{kfold}-fold validation.")
    print("Means", df["Mean"])
    print("StDevs", df["StDev"])

    return df

def validate_native_basis(
            train_tokens,
            train_ground,
            val_tokens,
            val_ground,
            method='kni',
            smoothing=0.5,
            order=3,
            p=0.995
):


    native_model, ref_limit = fit_native_model(train_tokens, train_ground,
                            method=method, order=order, smoothing=smoothing,
                            p=p, quiet=True)

    # Evaluate on test set.
    val_entropies = native_model.analyze_tokens(val_tokens)
    forecast = [e < ref_limit for e in val_entropies]
    metrics = util_fns.report_metrics(val_ground, forecast, quiet=True)

    return metrics
