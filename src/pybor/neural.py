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
    """
    def __init__(self, train_data=None, test_data=None, language='', series='',
                 detect_type='dual', model_type='attention'):

        # Train neural model on native word data and on loan word data if dual.
        # Determine cutpoint to detect loan words if native, or entropy bias if dual
        # Predict words as native or loan.

        self.language = language
        self.series = series
        self.detect_type = detect_type
        self.model_type = model_type
        self.vocab = None
        self.native_data = None
        self.loan_data = None
        self.native_model = None
        self.loan_model = None

        self.params = ncfg.neural
        self.cut_point = None


        self.prepare_data(train_data, test_data)

        self.make_native_model()
        if self.detect_type == 'dual':
            self.make_loan_model()


    def prepare_data(self, train_data, test_data):

        native_train_data = [[id_, token, status]
                for id_, token, status in train_data if status == 0]
        loan_train_data = [[id_, token, status]
                for id_, token, status in train_data if status == 1]

        if test_data is not None:
            native_test_data = [[id_, token, status]
                    for id_, token, status in test_data if status == 0]
            loan_test_data = [[id_, token, status]
                    for id_, token, status in test_data if status == 1]
        else:
            native_test_data = None
            loan_test_data = None

        # Build the vocabulary from all tokens.
        all_tokens = [token for _, token, _ in train_data]
        if test_data is not None:
            all_tokens += [token for _, token, _ in test_data]

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
        return find_ref_limit(entropies=entropies, fraction=self.params['fraction'])

    def make_native_predictions(self, entropies, cut_point):
        return [int(entropy>cut_point) for entropy in entropies]

    def make_dual_predictions(self, native_entropies, loan_entropies):
        return [int(loanH<nativeH) for nativeH, loanH
                       in zip(native_entropies, loan_entropies)]

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # Entropy calculation and predictions depend on whether native or dual.

        tokens_ids = NeuralData.get_tokens_ids(tokens, self.vocab)
        native_entropies = self.native_model.calculate_entropies(tokens_ids)

        if self.detect_type == 'native':
            if self.cut_point is None:

                trainval_tokens_ids = (self.native_data.train_tokens_ids +
                                       self.native_data.val_tokens_ids)
                entropies = self.native_model.calculate_entropies(trainval_tokens_ids)
                self.cut_point = self.calculate_ref_limit( entropies=entropies)

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
        return self.predict_data([['', token]])



# =============================================================================
# Main for simple example
#
# =============================================================================

# if __name__ == "__main__":
#     # Test basic funtion.
#     import statistics
#     import pybor
#     from pybor.dev.data import testing, training
#     import pybor.evaluate as evaluate

#     training1 = training+testing[:3*len(testing)//4]
#     testing1 = testing[3*len(testing)//4:]
#     print('training len=',len(training1), 'testing len=', len(testing1))
#     neural = Neural( train_data=training1, test_data=testing1,
#                     language='German', series='devel', detect_type='native')
#     # neural.make_native_model()
#     neural.native_model.evaluate_test(neural.native_data.test_gen)
#     # neural.make_loan_model()
#     if neural.loan_model is not None:
#         neural.loan_model.evaluate_test(neural.loan_data.test_gen)

#     print('token entropies for testing')
#     # Need to offer servide method for conversion.
#     # And method from here to do the conversion and calculate the entropies.
#     tokens = [row[1] for row in testing1]
#     tokens_ids = neural.native_data.test_tokens_ids
#     #entropies = neural.native_model.calculate_entropies(tokens_ids)
#     #print('1x1 entropies:', statistics.mean(entropies))
#     entropies = neural.native_model.calculate_entropies(tokens_ids)
#     print('batch entropies:',statistics.mean(entropies))
#     print(testing1[5:19])
#     print('token =', testing1[6], 'result =', neural.predict(testing1[6][1]))
#     print('tokens =', testing1[5:19], 'results =', neural.predict_data(testing1[5:19]))

#     print("Evaluate train dataset.")
#     predictions = neural.predict_data(training1)
#     train_metrics = evaluate.evaluate_model(predictions, training1)
#     evaluate.print_evaluation(train_metrics)
#     evaluate.false_positive(predictions, training1)

#     print("Evaluate test dataset.")
#     predictions = neural.predict_data(testing1)
#     test_metrics = evaluate.evaluate_model(predictions, testing1)
#     evaluate.print_evaluation(test_metrics)
#     evaluate.false_positive(predictions, testing1)
