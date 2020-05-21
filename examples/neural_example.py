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
from pybor.neural import Neural
from pybor.data_tf import NeuralData
import pybor.neural_cfg as ncfg


output_path = Path(ncfg.system['output_path']).resolve()


def evaluate_neural_loanword_prediction(language='', table=None,
            detect_type='dual', model_type='recurrent'):
    print(f'*** Evalution of prediction goodness for {language}. ***')
    print(f'Detect type is {detect_type}, neural model type is {model_type}.')

    train, test = NeuralData.simple_train_test_split(table)

    neural = Neural(train_data=train, test_data=test, language=language,
                    series='devel', detect_type=detect_type, model_type=model_type)

    print("Evaluate train dataset.")
    predictions = neural.predict_data(train)
    train_metrics = evaluate.evaluate_model(predictions, train)
    evaluate.print_evaluation(train_metrics)
    evaluate.false_positive(predictions, train)

    print("Evaluate test dataset.")
    predictions = neural.predict_data(test)
    test_metrics = evaluate.evaluate_model(predictions, test)
    evaluate.print_evaluation(test_metrics)
    evaluate.false_positive(predictions, test)


def perform_detection_by_language(languages=None, form='FormChars',
            detect_type='native', model_type='recurrent'):

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
                                            detect_type=detect_type, model_type=model_type)


if __name__ == "__main__":
    languages = 'Hup'  # ['English', 'Hup', 'Imbabura Quechua']  # 'English'
    perform_detection_by_language(languages=languages, form='FormChars',
                    detect_type='dual', model_type='attention')


