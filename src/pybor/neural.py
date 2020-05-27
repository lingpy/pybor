#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:34:40 2020

@author: johnmiller

Detect loan words based on word entropies as calculated by a neural netword model.
Support for model trained on both native and borrowed words, or just native words
"""

#import tensorflow as tf
# import math
from pybor.neural_tf import NeuralWord
from pybor.data_tf import NeuralData
from pybor.util import find_ref_limit
import pybor.neural_cfg as ncfg

class Neural:
    """
    Construct and train neural model of native word entropy distribution,
    and used the distribution to predict native versus loan words.

    Train neural model on native word data and on loan word data if dual.
    Determine cutpoint to detect loan words if native, or entropy bias if dual
    Predict words as native or loan.
    """
    def __init__(self, train_data=None, test_data=None,
                 detect_type=None, model_type=None,
                 language=None, series=None):

        def get_param(variable, key):
            return variable if variable is not None else ncfg.neural[key]

        self.language = get_param(language, 'language')
        self.series = get_param(series, 'series')
        self.detect_type = get_param(detect_type, 'detect_type')
        self.model_type = get_param(model_type, 'model_type')
        self.vocab = None
        self.native_data = None
        self.loan_data = None
        self.native_model = None
        self.loan_model = None
        self.cut_point = None

        self.prepare_data(train_data, test_data)

        self.make_native_model()
        if self.detect_type == 'dual':
            self.make_loan_model()


    def prepare_data(self, train_data, test_data):

        # Row element 2 is native-loan status.
        native_train_data = [row for row in train_data if row[2] == 0]
        loan_train_data = [row for row in train_data if row[2] == 1]

        if test_data is not None:
            native_test_data = [row for row in test_data if row[2] == 0]
            loan_test_data = [row for row in test_data if row[2] == 1]
        else:
            native_test_data = None
            loan_test_data = None


        # Row element 1 is token list of str segments.
        all_tokens = [row[1] for row in train_data]
        if test_data is not None:
            all_tokens += [row[1] for row in test_data]

        self.vocab = NeuralData.build_vocab(all_tokens)

        # Use the dictionary already constructed.
        self.native_data = NeuralData(native_train_data, native_test_data, vocab=self.vocab)
        self.loan_data = NeuralData(loan_train_data, loan_test_data, vocab=self.vocab)


    def make_native_model(self):

        self.native_model = NeuralWord(
            vocab_len=len(self.vocab), model_type=self.model_type,
            language=self.language, basis='native', series=self.series)

        train_gen, val_gen, _ = self.native_data.make_generators()
        self.native_model.train(train_gen=train_gen, val_gen=val_gen)

    def make_loan_model(self):
        self.loan_model = NeuralWord(
            vocab_len=len(self.vocab), model_type=self.model_type,
            language=self.language, basis='loan', series=self.series)

        train_gen, val_gen, _ = self.loan_data.make_generators()
        self.loan_model.train(train_gen=train_gen, val_gen=val_gen)

    def calculate_ref_limit(self, entropies=None):
        return find_ref_limit(entropies=entropies, fraction=ncfg.neural['fraction'])

    def make_native_predictions(self, entropies, cut_point):
        return [int(entropy>cut_point) for entropy in entropies]

    def make_dual_predictions(self, native_entropies, loan_entropies):
        return [int(loanH<nativeH) for nativeH, loanH
                       in zip(native_entropies, loan_entropies)]

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # Entropy calculation and predictions depend on whether native or dual.

        tokens_ids = NeuralData._get_tokens_ids(tokens, self.vocab)
        native_entropies = self.native_model.calculate_entropies(tokens_ids)

        if self.detect_type == 'native':
            if self.cut_point is None:

                train_tokens_ids = self.native_data.get_chosen_tokens_ids('train')
                entropies = self.native_model.calculate_entropies(train_tokens_ids)
                self.cut_point = self.calculate_ref_limit(entropies=entropies)

            return self.make_native_predictions(native_entropies, self.cut_point)

        #  detect_type == 'dual'
        loan_entropies = self.loan_model.calculate_entropies(tokens_ids)
        return self.make_dual_predictions(native_entropies, loan_entropies)


    def predict_data(self, data):
        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        return self.predict_data([['', token]])[0][2]

