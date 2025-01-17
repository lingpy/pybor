#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:44:50 2020

@author: johnmiller

Lexibank access borrowed from example by Tiago.
"""

from pathlib import Path

import pybor.wold as wold
import pybor.evaluate as evaluate
from pybor.neural import NeuralNative, NeuralDual, NeuralData
import pybor.util as util
import pybor.config as cfg

from pybor.dev.data import training1, testing1


def evaluate_neural_loanword_prediction(
    language="",
    table=None,
    detect_type="dual",
    model_type="recurrent",
    test_split=None,
    settings=None,
):

    train, test = util.train_test_split(table, split=test_split)
    val_split = (
        test_split
        if test_split is None or test_split >= 1.0
        else test_split / (1 - test_split)
    )

    evaluate_neural_loanword_prediction_train_test(
        language=language,
        train=train,
        test=test,
        detect_type=detect_type,
        model_type=model_type,
        val_split=val_split,
        settings=settings,
    )


def evaluate_neural_loanword_prediction_train_test(
    language="",
    train=None,
    test=None,
    detect_type="dual",
    model_type="recurrent",
    val_split=None,
    settings=None,
):

    print(f"*** Evalution of prediction for {language}. ***")
    print(f"Detect type is {detect_type}, neural model type is {model_type}.")
    if detect_type == "native":
        neural = NeuralNative(
            training=train,
            testing=test,
            language=language,
            series="devel",
            model_type=model_type,
            val_split=val_split,
            settings=settings,
        )
    else:
        neural = NeuralDual(
            training=train,
            testing=test,
            language=language,
            series="devel",
            model_type=model_type,
            val_split=val_split,
            settings=settings,
        )

    neural.train()

    print("Evaluate train dataset.")
    predictions = neural.predict_data(train)
    train_metrics = evaluate.evaluate_model(predictions, train)
    evaluate.false_positive(predictions, train)

    if test:
        print("Evaluate test dataset.")
        predictions = neural.predict_data(test)
        test_metrics = evaluate.evaluate_model(predictions, test)
        evaluate.false_positive(predictions, test)


def perform_detection_by_language(
    languages=None,
    form="FormChars",
    detect_type="native",
    model_type="recurrent",
    test_split=None,
    settings=None,
):

    wolddb = wold.get_wold_access()
    languages = wold.check_wold_languages(wolddb, languages)

    print(f"Languages {languages}.")
    for language in languages:
        table = wolddb.get_table(
            language=language, form=form, classification="Borrowed"
        )

        evaluate_neural_loanword_prediction(
            language=language,
            table=table,
            detect_type=detect_type,
            model_type=model_type,
            test_split=test_split,
            settings=settings,
        )


if __name__ == "__main__":
    languages = "English"  # ['English', 'Hup', 'Imbabura Quechua']  # 'English'
    perform_detection_by_language(
        languages=languages,
        form="Tokens",
        detect_type="dual",
        model_type="recurrent",
        test_split=0.10,
        settings=None,
    )

    # evaluate_neural_loanword_prediction_train_test('German',
    #                 training1, testing1, 'dual', 'recurrent')

    # evaluate_neural_loanword_prediction_train_test('German',
    #                 training1, detect_type='dual', model_type='recurrent')
