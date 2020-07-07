#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:03:57 2020

@author: johnmiller

Entropy distribution analysis examples.
Maybe precursor of something more general and applicable to all modules using predict protocol.
"""
import statistics
from pathlib import Path

from tabulate import tabulate

import pybor.wold as wold
from pybor.entropies import NeuralWordRecurrent
from pybor.neural import NeuralData
from pybor.plot import graph_word_distribution_entropies
import pybor.util as util

from pybor.dev.data import training1, testing1

import pybor.config as cfg

output_path = Path(cfg.BaseSettings().output_path).resolve()

def describe_entropies(data):
    n = len(data)
    avg = statistics.mean(data)
    stdev = statistics.stdev(data)
    min_ = min(data)
    max_ = max(data)
    return [n, avg, stdev, min_, max_]

def print_entropy_statistics(stats=None, title='', rownames=''):
    table = []
    table.append([title, 'n', 'mean', 'stdev', 'min', 'max'])
    for name, row in zip(rownames, stats):
        table.append([name]+row)
    print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.3f'))

def report_entropy_statistics(model=None, data=None):
        # Report descriptive statistics.
        stats = []
        fit_entropies = model.calculate_entropies(data.get_data_tokens_ids(data.fit))
        stats.append(describe_entropies(fit_entropies))
        if len(data.val) > 0:
            val_entropies = model.calculate_entropies(data.get_data_tokens_ids(data.val))
            stats.append(describe_entropies(val_entropies))
        if len(data.testing) > 0:
            test_entropies = model.calculate_entropies(data.get_data_tokens_ids(data.testing))
            stats.append(describe_entropies(test_entropies))
        return stats

def graph_entropies(tokens1=None, tokens2=None, data=None, model=None,
                    title='', label1='', label2='', filename=''):
        # Plot train and loan entropy distributions - all tokens basis
        tokens_ids1 = data.get_tokens_ids(tokens1)
        entropies1 = model.calculate_entropies(tokens_ids1)

        tokens_ids2 = data.get_tokens_ids(tokens2)
        entropies2 = model.calculate_entropies(tokens_ids2)
        # Could use mean + 6*stdev + 1 as upper limit for graphs.
        graph_path = output_path / filename
        graph_word_distribution_entropies(
                entropies1, entropies2, graph_path.as_posix(),
                title=title, label1=label1, label2=label2)


def analyze_neural_entropies_for_basis(
        train=None, test=None, train_x=None, test_x=None,
        language='', form='', basis='', model_type='recurrent'):

    # Get a vocabulary from all data.
    all_data = NeuralData(train, test)
    # Build the working dataset.
    data = NeuralData(train_x, test_x, vocab=all_data.vocab)

    model = NeuralWordRecurrent(
        data.vocab.size, language=language, basis=basis, series='demonstration')

    model.train(data.trainer, data.validator)
    model.evaluate_test(data.tester)

    # Report descriptive statistics.
    stats = report_entropy_statistics(model, data)
    title = f'{basis} basis'
    rownames = ['Fit', 'Validate', 'Test']
    print_entropy_statistics(stats, title, rownames)


    def get_prefix():
        return f"entropies.{model.construct_modelprefix()}-{form}-{model_type}-model"

    # Plot train, test entropy distributions
    train_tokens = [token for _, token, _ in train_x]
    test_tokens= [token for _, token, _ in test_x]

    filename = get_prefix()+'--train-test.pdf'
    title = f'{language} train and test entropies from {form} - {basis} basis'
    graph_entropies(train_tokens, test_tokens,
            data, model, title, 'Train', 'Test', filename)

    # Plot native, loan train entropy distributions - all tokens basis.
    train_native_tokens = [token for _, token, status in train if status == 0]
    train_loan_tokens = [token for _, token, status in train if status == 1]

    filename = get_prefix()+'--train-native-loan.pdf'
    title = f'{language} train - native and loan entropies from {form} - {basis} basis'
    graph_entropies(train_native_tokens, train_loan_tokens,
            data, model, title, 'Native', 'Loan', filename)

    # Plot native, loan test entropy distributions.
    test_native_tokens = [token for _, token, status in test if status == 0]
    test_loan_tokens = [token for _, token, status in test if status == 1]
    #print(f'test: native {len(test_native_tkns)}, loan {len(test_loan_tkns)}. ')

    filename = get_prefix()+'--test-native-loan.pdf'
    title = f'{language} test - native and loan entropies from {form} - {basis} basis'
    graph_entropies(test_native_tokens, test_loan_tokens,
            data, model, title, 'Native', 'Loan', filename)

def analyze_neural_entropies(language=None, table=None,
            form='', basis='', model_type='recurrent', test_split=None):

    # All data together.
    train, test = util.train_test_split(table, split=test_split)
    analyze_neural_entropies_train_test(language,
            train, test, form=form, basis=basis, model_type=model_type)


def analyze_neural_entropies_train_test(language =None,
            train=None, test=None, form='', basis='', model_type='recurrent'):
    # Function to perform graphical analysis of entropy distributions.
    # Possible languages are 'all', list of languages ([str]), or specific language (str).
    print(f'*** Graphical analysis of entropy distribution for {language}. ***')
    print(f'Basis is {basis}, neural model type is {model_type}.')

    if basis == 'all':
        analyze_neural_entropies_for_basis(train=train, test=test,
                train_x=train, test_x=test,
                language=language, form=form, basis=basis, model_type=model_type)

    elif basis == 'native':
        # Train and test native are based on train, test split of all data.
        # trainval_native = [[id_, token, status] for id_, token, status in trainval if status == 0]
        # test_native = [[id_, token, status] for id_, token, status in test if status == 0]
        train_native = [row for row in train if row[2]==0]
        test_native = [row for row in test if row[2]==0]
        analyze_neural_entropies_for_basis(train=train, test=test,
                train_x=train_native, test_x=test_native,
                language=language, form=form, basis=basis, model_type=model_type)

    elif basis == 'loan':
        # Train and test loan are based on train, test split of all data.
        # trainval_loan = [[id_, token, status] for id_, token, status in trainval if status == 1]
        # test_loan = [[id_, token, status] for id_, token, status in test if status == 1]
        train_loan = [row for row in train if row[2]==1]
        test_loan = [row for row in test if row[2]==1]
        analyze_neural_entropies_for_basis(train=train, test=test,
                train_x=train_loan, test_x=test_loan,
                language=language, form=form, basis=basis, model_type=model_type)


# Shell method to perform analysis.
def perform_analysis_by_language(languages=None, form='Tokens',
                basis='all', model_type='recurrent', test_split=None):

    wolddb = wold.get_wold_access()
    languages = wold.check_wold_languages(wolddb, languages)

    print(languages)
    for language in languages:
        table = wolddb.get_table(
                    language=language,
                    form=form,
                    classification='Borrowed'
                    )


        analyze_neural_entropies(
                    language=language,
                    table=table,
                    form=form,
                    basis=basis,
                    model_type=model_type,
                    test_split=test_split)

# Main
if __name__ == "__main__":
    languages = 'Hup'   # ['English']  # , 'Hup']
    perform_analysis_by_language(
                    languages=languages,
                    form='Tokens',
                    basis='loan',
                    model_type='recurrent',
                    test_split=0.15)

    # Use training1 and testing1.
    # analyze_neural_entropies_train_test(
    #                 language='German',
    #                 train=training1,
    #                 test=testing1,
    #                 form='Tokens',
    #                 basis='native',
    #                 model_type='recurrent')

