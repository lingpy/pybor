#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:16:27 2020

@author: johnmiller
"""

# Regression testing versus previous langauge tables and results.
from pathlib import Path
import csv

import pandas as pd

import pybor.evaluate as evaluate
from pybor.neural import NeuralNative, NeuralDual, NeuralData
import pybor.util as util
import pybor.config as cfg


output_path = Path(cfg.BaseSettings().output_path).resolve()


def evaluate_prediction(model=None, data=None, title=""):
    print(f"Evaluate {title} dataset.")
    predictions = model.predict_data(data)
    train_metrics = evaluate.evaluate_model(predictions, data)


def evaluate_borrowing_prediction_train_test(
    language=None,
    train=None,
    test=None,
    detect_type=None,
    model_type=None,
    val_split=None,
    settings=None,
):

    print(
        f"*** Evaluation for {language}, detect type is {detect_type}, neural model type is {model_type}"
    )

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

    evaluate_prediction(neural, train, "Training")
    evaluate_prediction(neural, test, "Testing")


# def make_orthographic_table(language):
#    tabledir = "tables/"
#    table = pd.read_csv(tabledir + language + ".tsv", sep="\t", index_col=0)
#    table["Borrowed"] = (table.BorrowedScore > 0.375).astype(
#        int
#    )  # Inverse of previous value
#    table["Segments"] = table["Segments"].str.split()
#    table = table[["Parameter_ID", "Segments", "Borrowed"]]
#
#    data = table.to_numpy().tolist()
#    return data
#
#
# def perform_prediction_by_language_from_table(
#    language=None, detect_type=None, model_type=None, test_split=None, settings=None
# ):
#
#    table = make_orthographic_table(language)
#
#    print(f"Table for {language}.")
#
#    train, test = util.train_test_split(table, split=test_split)
#
#    val_split = (
#        test_split
#        if test_split is None or test_split >= 1.0
#        else test_split / (1 - test_split)
#    )
#    evaluate_borrowing_prediction_train_test(
#        language=language,
#        train=train,
##        test=test,
#        detect_type=detect_type,
#        model_type=model_type,
#        val_split=val_split,
#        settings=settings,
#    )
#

# *******************************************************************************************
# Try with restore weights == true; if no go, try with false to match 'final.h5 from before.'
# Change entropies to patience=10.
settings = cfg.RecurrentSettings(
    rnn_cell_type="GRU",
    embedding_len=16,
    rnn_output_len=32,
    learning_rate=0.01,
    learning_rate_decay=0.95,
    merge_embedding_dropout=0.2,
    rnn_output_dropout=0.2,
    recurrent_l2=0.001,
    batch_size=50,
    epochs=40,
    restore_best_weights=False,
)

# perform_prediction_by_language_from_table(
#    language="English",
#    detect_type="dual",
#    model_type="recurrent",
#    test_split=0.15,
#    settings=settings,
# )
