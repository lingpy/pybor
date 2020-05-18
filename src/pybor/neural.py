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
# from pybor.util import find_cut_point
from pybor.util import find_ref_limit

class Neural:
    """
    Construct and train neural model of native word entropy distribution,
    and used the distribution to predict native versus loan words.
    """
    def __init__(self, train_data=None, test_data=None, language='', series='',
                 detect_type='dual', model_type='attention', cell_type='LSTM',
                 embedding_len=32, rnn_output_len=32, frac=0.995,
                 make_models_separately=False):

        # Train neural model on native word dataa and of loan word data if dual.
        # Determine cutpoint to detect loan words if native, or entropy bias if dual
        # Predict words as native or loan.

        self.language = language
        self.series = series
        self.detect_type = detect_type
        self.model_type = model_type
        self.symbol_to_id = None
        self.vocab_len = None
        self.native_data = None
        self.loan_data = None
        #self.native_file_prefix = native_file_prefix
        #self.loan_file_prefix = loan_file_prefix
        self.native_model = None
        self.loan_model = None
        self.frac = frac
        self.cut_point = None

        self.prepare_data(train_data, test_data)

        # TODO: Add option to load existing models.

        # Leave now if making models separately.
        if make_models_separately: return

        self.make_model(basis='native', model_type=self.model_type,
                cell_type=cell_type, embedding_len=embedding_len,
                rnn_output_len=rnn_output_len)
        if self.detect_type == 'dual':
            self.make_model(basis= 'loan', model_type=self.model_type,
                cell_type=cell_type, embedding_len=embedding_len,
                rnn_output_len=rnn_output_len)


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
        self.symbol_to_id = NeuralData.get_symbol_dictionary(self.language, all_tokens)

        self.vocab_len = len(self.symbol_to_id)

        # Will use the dictionary already constructed.
        self.native_data = NeuralData(native_train_data,
                native_test_data, file_prefix=self.language)
        self.loan_data = NeuralData(loan_train_data,
                loan_test_data, file_prefix=self.language)


    def load_model(self, file_prefix=''):
        pass  # TODO: Add this capability.

    def make_model(self, basis='', model_type='attention', cell_type='LSTM',
                          embedding_len=32, rnn_output_len=32,
                          dropout_rate=0.2, l2_amt=0.001, epochs=40,
                          learning_rate=0.01, lr_decay=0.95):
        # Using dropout=0.2 default.  Consider use of 0.1.

        model = NeuralWord(vocab_len=self.vocab_len, model_type=model_type,
                                language=self.language, basis=basis, series=self.series,
                                cell_type=cell_type, embedding_len=embedding_len,
                                rnn_output_len=rnn_output_len,
                                dropout_rate=dropout_rate, l2_amt=l2_amt)

        if basis == 'native':
            self.native_model = model
            train_gen, val_gen, _ = self.native_data.make_generators()
        elif basis == 'loan':
            self.loan_model = model
            train_gen, val_gen, _ = self.loan_data.make_generators()

        model.train(train_gen=train_gen, val_gen=val_gen,
                epochs=epochs, learning_rate=learning_rate, lr_decay=lr_decay)

    # def calculate_cut_point(self, native_entropies, loan_entropies):
    #     return find_cut_point(native=native_entropies, loan=loan_entropies)

    def calculate_ref_limit(self, entropies=None, frac=0.99):
        return find_ref_limit(entropies=entropies, frac=frac)

    def make_native_predictions(self, entropies, cut_point):
        return [int(entropy>cut_point) for entropy in entropies]

    def make_dual_predictions(self, native_entropies, loan_entropies):
        return [int(loanS<nativeS) for nativeS, loanS
                       in zip(native_entropies, loan_entropies)]

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # Entropy calculation and predictions depend on whether native or dual.
        tokens_ids = NeuralData.get_tokens_ids(tokens, self.symbol_to_id)
        native_entropies = self.native_model.calculate_entropies(tokens_ids)

        if self.detect_type == 'native':
            if self.cut_point is None:
                # val_tokens_ids = self.native_data.data_repo.val_tokens_ids
                # val_native_entropies = self.native_model.calculate_entropies(val_tokens_ids)
                # val_tokens_ids = self.loan_data.data_repo.val_tokens_ids
                # val_loan_entropies = self.native_model.calculate_entropies(val_tokens_ids)
                # self.cut_point = self.calculate_cut_point(
                #        val_native_entropies, val_loan_entropies)

                # Jupyter implementation used train_val
                trainval_tokens_ids = self.native_data.data_repo.trainval_tokens_ids
                entropies = self.native_model.calculate_entropies(trainval_tokens_ids)
                self.cut_point = self.calculate_ref_limit( entropies=entropies, frac=self.frac)

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
#                     language='German', series='devel',
#                     detect_type='native', make_models_separately=True)
#     neural.make_model(basis='native', model_type='attention', epochs=10, lr_decay=0.975)
#     neural.native_model.evaluate_test(neural.native_data.test_gen)
#     neural.make_model(basis='loan', model_type='attention', epochs=40, lr_decay=0.95)
#     neural.loan_model.evaluate_test(neural.loan_data.test_gen)

#     print('token entropies for testing')
#     # Need to offer servide method for conversion.
#     # And method from here to do the conversion and calculate the entropies.
#     tokens = [row[1] for row in testing1]
#     tokens_ids = neural.native_data.data_repo.test_tokens_ids
#     entropies = neural.native_model.calculate_entropies(tokens_ids)
#     print('1x1 entropies:', statistics.mean(entropies))
#     entropies = neural.native_model._calc_tokens_ids_entropies(tokens_ids)
#     print('batch entropies:',statistics.mean(entropies))
    # print(testing1[5:19])
    # print('token =', testing1[6], 'result =', neural.predict(testing1[6][1]))
    # print('tokens =', testing1[5:19], 'results =', neural.predict_data(testing1[5:19]))

    # print("Evaluate train dataset.")
    # predictions = neural.predict_data(training1)
    # train_metrics = evaluate.evaluate_model(predictions, training1)
    # evaluate.print_evaluation(train_metrics)
    # evaluate.false_positive(predictions, training1)

    # print("Evaluate test dataset.")
    # predictions = neural.predict_data(testing1)
    # test_metrics = evaluate.evaluate_model(predictions, testing1)
    # evaluate.print_evaluation(test_metrics)
    # evaluate.false_positive(predictions, testing1)
