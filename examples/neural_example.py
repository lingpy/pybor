#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:44:50 2020

@author: johnmiller

Lexibank access borrowed from example by Tiago.
"""

import math
import random
import pickle
from pathlib import Path

from pybor.data import LexibankDataset
import pybor.evaluate as evaluate
from pybor.neural import NeuralNative, NeuralDual, NeuralData
import pybor.util as util
import pybor.config as cfg

from pybor.dev.data import training1, testing1

output_path = Path(cfg.BaseSettings().output_path).resolve()


def evaluate_neural_loanword_prediction(language='', table=None,
            detect_type='dual', model_type='recurrent', test_split=None, settings=None):

    train, test = util.train_test_split(table, split=test_split)
    val_split = (test_split if test_split is None or test_split >= 1.0
                            else test_split/(1-test_split))

    evaluate_neural_loanword_prediction_train_test(
                            language=language,
                            train=train,
                            test=test,
                            detect_type=detect_type,
                            model_type=model_type,
                            val_split=val_split,
                            settings=settings)

def evaluate_neural_loanword_prediction_train_test(language='',
            train=None, test=None, detect_type='dual',
            model_type='recurrent', val_split=None, settings=None):

    print(f'*** Evalution of prediction for {language}. ***')
    print(f'Detect type is {detect_type}, neural model type is {model_type}.')
    if detect_type == 'native':
        neural = NeuralNative(training=train,
                              testing=test,
                              language=language,
                              series='devel',
                              model_type=model_type,
                              val_split=val_split,
                              settings=settings)
    else:
        neural = NeuralDual(training=train,
                            testing=test,
                            language=language,
                            series='devel',
                            model_type=model_type,
                            val_split=val_split,
                            settings=settings)

    neural.train()

    print("Evaluate train dataset.")
    predictions = neural.predict_data(train)
    train_metrics = evaluate.evaluate_model(predictions, train)
    evaluate.print_evaluation(train_metrics)
    evaluate.false_positive(predictions, train)

    if test:
        print("Evaluate test dataset.")
        predictions = neural.predict_data(test)
        test_metrics = evaluate.evaluate_model(predictions, test)
        evaluate.print_evaluation(test_metrics)
        evaluate.false_positive(predictions, test)


def perform_detection_by_language(languages=None, form='FormChars',
                                  detect_type='native', model_type='recurrent',
                                  test_split=None, settings=None):

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

    print(f'Languages {languages}.')
    for language in languages:
        table = lex.get_table(
                    language=language,
                    form=form,
                    classification='Loan'
                    )

        evaluate_neural_loanword_prediction(language=language, table=table,
                                            detect_type=detect_type,
                                            model_type=model_type,
                                            test_split=test_split,
                                            settings=settings)


if __name__ == "__main__":
    languages = 'English'  # ['English', 'Hup', 'Imbabura Quechua']  # 'English'
    perform_detection_by_language(
                    languages=languages,
                    form='Tokens',
                    detect_type='dual',
                    model_type='attention',
                    test_split=0.10,
                    settings=None)


    # evaluate_neural_loanword_prediction_train_test('German',
    #                 training1, testing1, 'dual', 'recurrent')

    # evaluate_neural_loanword_prediction_train_test('German',
    #                 training1, detect_type='dual', model_type='recurrent')
