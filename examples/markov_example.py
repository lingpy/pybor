#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:54:13 2020

@author: johnmiller

markov_example.py
"""
import pickle

from pybor.data import LexibankDataset
from pybor.markov import DualMarkov, NativeMarkov
import pybor.evaluate as evaluate
import pybor.util as util

from pybor.dev.data import training1, testing1


def validate_loan_detection_dual_basis(train_data, test_data,
                                       model='kni', smoothing=0.5,
                                       order=3):


    dual_model = DualMarkov(train_data, model=model,
                            order=order, smoothing=smoothing)

    print("Evaluate train dataset.")
    predictions = dual_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    evaluate.print_evaluation(train_metrics)

    print("Evaluate test dataset.")
    predictions = dual_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    evaluate.print_evaluation(test_metrics)

    return dual_model, test_metrics


# =============================================================================
#
#  Validate loan detection - native model effectiveness.
#
# =============================================================================
def validate_loan_detection_native_basis(train_data, test_data,
                                         model='kni', smoothing=0.5,
                                         order=3, p=0.995):


    native_model = NativeMarkov(train_data, model=model,
                                order=order, smoothing=smoothing, p=p)

    print("Evaluate train dataset.")
    predictions = native_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    evaluate.print_evaluation(train_metrics)

    print("Evaluate test dataset.")
    predictions = native_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    evaluate.print_evaluation(test_metrics)

    return native_model, test_metrics


# Shell method to perform analysis.
def perform_analysis_by_language(languages=None, form='Tokens',
                                 basis='native', test_split=0.15):

    try:
        with open('wold.bin', 'rb') as f:
            lex = pickle.load(f)
    except:
        lex = LexibankDataset(
                'wold',
                transform={
                    "Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0}
                )
        with open('wold.bin', 'wb') as f:
            pickle.dump(lex, f)

    if languages == 'all':
        languages = [language["Name"] for language in lex.languages.values()]
    elif isinstance(languages, str):
        languages = [languages]
    elif not isinstance(languages, list):
        print("Language must be language name, list of languages, or keyword 'all'.")

    print(languages)
    for language in languages:
        table = lex.get_table(
                    language=language,
                    form=form,
                    classification='Loan'
                    )

        training, testing = util.train_test_split(
                            table, split=test_split)

        print("Language:", language)
        if basis == 'native':
            print("native basis")
            validate_loan_detection_native_basis(training, testing,
                                                 model='kni', smoothing=0.5,
                                                 order=3, p=0.995)
        else:  # 'dual'
            print("dual basis")
            validate_loan_detection_dual_basis(training, testing,
                                               model='kni', smoothing=0.5,
                                               order=3)


# Main
if __name__ == "__main__":
    languages = 'Hup'  # ['Hup' ,'English']  # all, individual-language

    perform_analysis_by_language(languages, form='Tokens',
                                  basis='native', test_split=0.15)

    perform_analysis_by_language(languages, form='Tokens',
                                  basis='dual', test_split=0.15)

    # Use training1 and testing1.
    # print("\nGerman - native basis")
    # validate_loan_detection_native_basis(training1, testing1, method='kni',
    #                     smoothing=0.5, order=3, p=0.995)
    # print("\nGerman - dual basis")
    # validate_loan_detection_dual_basis(training1, testing1,  method='kni',
    #                     smoothing=0.5, order=3)
