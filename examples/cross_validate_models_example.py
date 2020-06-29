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

def get_user_fn(model_name, mode, k_fold, holdout_n, max_iter, writer, settings=None):

    def fn(language, form, table,  # Supplied by 'apply method'.
           model_name=model_name,
           mode=mode,
           k_fold=k_fold,
           holdout_n=holdout_n,
           max_iter=max_iter,
           writer=writer,
           settings=settings):

        # Allocate columns for k_fold
        p_ = []
        r_ = []
        f_ = []
        a_ = []

        if mode == 'k_fold':
            gen = util.k_fold_samples(table, k=k_fold)
        else:
            gen = util.holdout_n_samples(table, n=holdout_n, max_iter=max_iter)

        for (i, train, test) in gen:

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

        logger.debug(f'{len(f_)} samples for {language}.')
        if len(f_) > 1:
            sd_p = st.stdev(p_)
            sd_r = st.stdev(r_)
            sd_f = st.stdev(f_)
            sd_a = st.stdev(a_)
        else:
            sd_p, sd_r, sd_f, sd_a = [0]*4
        writer.writerow([language,
                        f'{st.mean(p_):.3f}', f'{st.mean(r_):.3f}',
                        f'{st.mean(f_):.3f}', f'{st.mean(a_):.3f}',
                        f'{sd_p:.3f}', f'{sd_r:.3f}',
                        f'{sd_f:.3f}', f'{sd_a:.3f}'])

    return fn

def print_summary(title, header, labels, summary):
    print(title)
    header = [''] + header
    summary[0] = [labels[0]] + summary[0]
    summary[1] = [labels[1]] + summary[1]
    summary[2] = [labels[2]] + summary[2]
    table = [header, summary[0], summary[1], summary[2]]
    print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.3f'))

def summarize_cross_validation(file_path, form, model_name, mode, k_fold, holdout_n, series):
    with open(file_path.as_posix(), 'r', newline='') as fl:
        reader = csv.reader(fl)
        results = list(reader)

    header = ['prec', 'recall', 'f1', 'acc']
    results = results[1:]
    means = []
    stdevs = []
    measure_stdevs = []

    for col in range(1, 5):
        values = [float(row[col]) for row in results]
        means.append(st.mean(values))
        stdevs.append(st.stdev(values) if len(values)>1 else None)
    for col in range(5, 9):
        values_sqr = [float(row[col])**2 for row in results]
        measure_stdevs.append(math.sqrt(st.mean(values_sqr)))

    if mode == 'k_fold':
        title = f'{k_fold}-fold cross-validation'
    else:
        title = f'holdout-{holdout_n} cross-validation'

    title += f' - borrowing prediction by language for {form} - {model_name} - {series}.'


    print_summary(title, header, labels=['Mean', 'Between Language StDev', 'Within Language StDev'],
                  summary=[means, stdevs, measure_stdevs])



def cross_validate_model(languages, form, model_name, mode,
                         k_fold, holdout_n, max_iter, series='', settings=None):

    if mode == 'k_fold':
        filename = f'cv-{k_fold:d}-fold'
    else:
        filename = f'cv-holdout-{holdout_n:d}-{max_iter:d}-times'
        if holdout_n < 30:
            logger.info('Estimates of within language StDev inflated ' +
                        f'due to small holdout size {holdout_n}.')

    filename += f'-{model_name}-{form}-{series}-prfa.csv'

    file_path = output_path / filename
    with open(file_path.as_posix(), 'w', newline='') as fl:
        writer = csv.writer(fl)
        writer.writerow(['language',
                         'mean_prec', 'mean_recall',
                         'mean_f1', 'mean_acc',
                         'sample_stdev_prec', 'sample_stdev_recall',
                         'sample_stdev_f1', 'sample_stdev_acc'])

        fn = get_user_fn(model_name, mode, k_fold, holdout_n, max_iter, writer, settings)
        data.apply_function_by_language(languages, form, fn)

    summarize_cross_validation(file_path, form, model_name, mode, k_fold, holdout_n, series)

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
        "--mode",
        type=str,
        default='k_fold',
        choices=['k_fold', 'holdout_n'],
        help="whether to perform k-fold validation or holdout-n study (default: k_fold)"
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
        help="number of folds where each fold is 1-1/k training and 1/k testing (default: 5)"
    )
    parser.add_argument(
        "--holdout_n",
        type=int,
        default=5,
        help="number of samples held out for each train/test iteration (default: 5)"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1,
        help="maximum number of iterations of taking holdout samples (default: 1). " +
        "A -1 indicates until all data have been held out."
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
        "--val_split",
        default=0.1,
        type=float,
        help="Validation split for Neural models (default: 0.1)"
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
            verbose=args.verbose, val_split=args.val_split)
    elif args.model == "markovdual":
        settings = config.MarkovSettings(smoothing=args.smoothing)

    # Run cross validation
    cross_validate_model(languages=args.languages,
            form="Tokens",
            model_name=args.model,
            mode=args.mode,
            k_fold=args.k_fold,
            holdout_n=args.holdout_n,
            max_iter=args.max_iter,
            series=args.series,
            settings=settings)
