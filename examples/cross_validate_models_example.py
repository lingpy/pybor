#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:16:44 2020

@author: johnmiller
"""

import argparse
import math
from pathlib import Path
import csv
from tabulate import tabulate
import statistics as st

import pybor.config as config
import pybor.data as data
import pybor.evaluate as evaluate
import pybor.markov as markov
import pybor.neural as neural
import pybor.ngram as ngram
import pybor.svm as svm
import pybor.util as util


output_path = Path(config.BaseSettings().output_path).resolve()
logger = util.get_logger(__name__)

def get_user_fn(model_name, k_fold, writer, settings=None):

    def fn(language, form, table,
           model_name=model_name,
           k_fold=k_fold,
           writer=writer,
           settings=settings):

        # Allocate columns for k_fold
        p_ = []
        r_ = []
        f_ = []
        a_ = []

        for (i, train, test) in util.k_fold_samples(table, k=k_fold):

            if model_name == 'neuraldual':
                model = neural.NeuralDual(train, settings=settings)
                model.train()

            elif model_name == 'markovdual':
                model = markov.DualMarkov(train, settings=settings)

            elif model_name == 'ngram':
                model = ngram.NgramModel(train)
            elif model_name == 'bagofsounds':
                model = svm.BagOfSounds(train, kernel='linear')
            else:
                logger.warning(f'Unknown model_name {model_name} for language {language}.')
                return  # No valid model specified.

            # NgramModel and BagOfSounds don't like 3 column format for predict.
            predicted = model.predict_data([[row[0], row[1]] for row in test])
            p, r, f, a = evaluate.prf(test, predicted)
            p_.append(p)
            r_.append(r)
            f_.append(f)
            a_.append(a)


        writer.writerow([language, form, model_name, f'{k_fold:d}',
                        f'{st.mean(p_):.3f}', f'{st.mean(r_):.3f}',
                         f'{st.mean(f_):.3f}', f'{st.mean(a_):.3f}',
                        f'{st.stdev(p_):.3f}', f'{st.stdev(r_):.3f}',
                         f'{st.stdev(f_):.3f}', f'{st.stdev(a_):.3f}'])

    return fn

#  Still need to  accomodate this to k-fold statistics.
def print_summary(title, header, labels, summary):
    print(title)
    header = [''] + header
    summary[0] = [labels[0]] + summary[0]
    summary[1] = [labels[1]] + summary[1]
    summary[2] = [labels[2]] + summary[2]
    table = [header, summary[0], summary[1], summary[2]]
    print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.3f'))

def summarize_cross_validation(file_path, form, model_name, k_fold, series):
    with open(file_path.as_posix(), 'r', newline='') as fl:
        reader = csv.reader(fl)
        results = list(reader)

    header = ['prec', 'recall', 'f1', 'acc']
    results = results[1:]
    means = []
    stdevs = []
    measure_stdevs = []

    for col in range(4, 8):
        values = [float(row[col]) for row in results]
        means.append(st.mean(values))
        stdevs.append(st.stdev(values))
    for col in range(8, 12):
        values_sqr = [float(row[col])**2 for row in results]
        measure_stdevs.append(math.sqrt(st.mean(values_sqr)))

    title = (f'{k_fold}-fold cross-validation over language of borrowing prediction ' +
             f'for {form} form with {model_name} model {series}.')

    print_summary(title, header, labels=['Mean', 'StDev', 'Measure StDev'],
                  summary=[means, stdevs, measure_stdevs])



def cross_validate_model(languages, form, model_name, k_fold, series='', settings=None):

    filename = f'cv-{k_fold:d}-fold-{model_name}-{form}-{series}-prfa.csv'
    file_path = output_path / filename
    with open(file_path.as_posix(), 'w', newline='') as fl:
        writer = csv.writer(fl)
        writer.writerow(['language', 'form', 'model_name', 'k_fold',
                         'mean_prec', 'mean_recall', 'mean_f1', 'mean_acc',
                         'meas_stdev_prec', 'meas_stdev_recall', 'meas_stdev_f1', 'meas_stdev_acc'])

        fn = get_user_fn(model_name, k_fold, writer, settings)
        data.apply_function_by_language(languages, form, fn)

    summarize_cross_validation(file_path, form, model_name, k_fold, series)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=['ngram', 'bagofsounds', 'markovdual', 'neuraldual'],
        help="Model for the cross validation"
    )
    parser.add_argument(
        "--form",
        type=str,
        default="Tokens",
        help="Name of the column holding the forms for validation (default: \"Tokens\")"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="all",
        help='Language to use for example (default: \"all\")'
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
        help="K-fold value (default: 5)"

    )
    parser.add_argument(
        "--series",
        type=str,
        default="test",
        help="Series for the cross validation (default: \"test\")"
    )
    parser.add_argument(
        "--smoothing",
        default=0.3,
        type=float,
        help="Smoothing for Markov models (default: 0.3)"
    )
    parser.add_argument(
        "--embedding_len",
        default=16,
        type=int,
        help="Embedding length for Neural models (default: 16)"
    )
    parser.add_argument(
        "--val_split",
        default=0.1,
        type=float,
        help="Value split for Neural models (default: 0.1)"
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="Verbose operation for the methods that support it (default: False)"
    )
    args = parser.parse_args()

    # Build settings if needed
    settings = None
    if args.model == "neuraldual":
        settings =  config.RecurrentSettings(
                embedding_len=args.embedding_len, verbose=args.verbose, val_split=args.val_split)
    elif args.model == "markovdual":
        settings = config.MarkovSettings(smoothing=args.smoothing)

    # Run cross validation
    cross_validate_model(languages=args.languages,
    form="Tokens",
    model_name=args.model,
    k_fold=args.k_fold,
    series=args.series,
    settings=settings)