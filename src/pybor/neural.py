#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:34:40 2020

@author: johnmiller

Detect loan words based on word entropies as calculated by a neural netword model.
Support for model trained on both native and borrowed words, or just native words
"""

from pybor.util import find_ref_limit

import math
import random
from collections import Counter

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import attr
from pybor.config import *
from pybor.entropies import NeuralWord


@attr.s
class Vocab:
    data = attr.ib()
    start = attr.ib(default='<s>')
    end = attr.ib(default='</s>')
    null = attr.ib(default='<nul>')
    unknown = attr.ib(default='<unk>')

    def __attrs_post_init__(self):
        counter = 4
        self.vocab = {
                self.unknown: 1, self.start: 2, self.end: 3, self.null: 0,
                1: self.unknown, 2: self.start, 3: self.end, 0: self.null}
        for tokens in self.data:
            for token in tokens:
                try:
                    self.vocab[token]
                except:
                    self.vocab[token] = counter
                    self.vocab[counter] = token
                    counter += 1

    def __getitem__(self, item):
        return self.vocab[item]

    def get(self, item, alternative=None):
        return self.vocab.get(item, alternative)

    def __len__(self):
        return len(self.vocab)

    def translate(self, word):
        return [self.vocab[self.start]]+[self.vocab.get(x, self.unknown) for x
                in word]+[self.vocab[self.end]]


@attr.s
class NeuralData:
    """
    Parameters
    ----------
    train_data : [str, [str], int]
        Training data as rows of words. Each word consists of an identifier,
        token represented as list of character segments, and loan status
        (typically as native or loan 0/1).
    test_data : [str, [str], int]
        Same as train_data.
    vocab: {[str] : int}, optional
        Symbol to id translation dictionary. Calculated from tokens if not provided.
        If multiple data stores are defined then a common vocab should be used.
    val_split: float, optional
        Proportion of training data to use for validation.
        Uses value from neural_cfg.data if not overridden here.
    """
    training = attr.ib(default=[])
    testing = attr.ib(default=[])
    vocab = attr.ib(default=None)
    val_split = attr.ib(default=None)
    settings = attr.ib(default=BaseSettings())

    def __attrs_post_init__(self):
        self.val_split = self.val_split or self.settings.val_split
        self.all_data = self.training+self.testing
        random.shuffle(self.all_data)
        self.vocab = self.vocab or Vocab([x[1] for x in self.all_data])
        random.shuffle(self.testing)
        random.shuffle(self.training)
        s = math.ceil(len(self.training)*self.val_split)
        self.fit, self.val = self.training[:-s], self.training[-s:]
        self.label_counts = dict(Counter([x[2] for x in self.training]))

        if self.settings.verbose:
            print(f'Train length: {len(self.training)}, ',
                f'fit length: {len(self.fit)}',
                f'val length: {len(self.val) if self.val else 0}, ',
                f'test length: {len(self.testing) if self.testing else 0}.')

    def translate(self, sequences):
        """
        Translate a word to the internal numeric alphabet.
        """
        return [self.vocab.translate(t) for t in sequences]

    def get_batcher(self, data):
        return KerasBatchGenerator(
                self.translate([x[1] for x in data]),
                batch_size=self.settings.batch_size)
    @property
    def trainer(self):
        if hasattr(self, '_trainer'):
            return self._trainer
        self._trainer = self.get_batcher(self.fit)
        return self._trainer

    @property
    def validator(self):
        if hasattr(self, '_validator'):
            return self._validator
        self._validator = self.get_batcher(self.val)
        return self._validator

    @property
    def tester(self):
        if hasattr(self, '_tester'):
            return self._tester
        self._tester = self.get_batcher(self.testing)
        return self._tester


@attr.s
class KerasBatchGenerator(object):
    """
    Construct a generator for the neural network

    Parameters
    ----------
    data : [[int]]
        list of tokens. Tokens are represented by lists of ids.
        Each list of ids corresponds to a list of character string segments.
    batch_size : int, optional
        Batch size to use in neural network fit.
        If not given, the configuratiom batch size is used.
    """
    data = attr.ib(repr=False)
    batch_size = attr.ib(default=None)
    settings = attr.ib(default=BaseSettings(), repr=False)

    def __attrs_post_init__(self):
        self.batch_size = self.batch_size or self.settings.batch_size
        self.current_idx = 0
        self.data_len = len(self.data)

    def generate(self, sample=None):
        # Randomize order of words.
        data = [row for row in self.data]
        random.shuffle(data)
        count = 0
        while not sample or count < sample:
            count += 1
            x_lst = []
            y_lst = []
            for i in range(self.batch_size):
                ## Build 2-D list of lists of ids for each word.
                ## Apply Keras pad_sequences (post padding).
                ## Yield x and y numpy arrays.
                # Truncate last symbol because we don't have more y to predict.
                x_lst.append(data[self.current_idx][:-1])

                # Treat y as sparse.
                y_lst.append(data[self.current_idx][1:])

                self.current_idx += 1
                if self.current_idx >= self.data_len:
                    self.current_idx = 0

            x = pad_sequences(x_lst, padding='post')
            y = pad_sequences(y_lst, padding='post')
            yield x, y


@attr.s
class Neural:
    """
    Construct and train neural model of native word entropy distribution,
    and used the distribution to predict native versus loan words.

    Train neural model on native word data and on loan word data if dual.
    Determine cutpoint to detect loan words if native, or entropy bias if dual
    Predict words as native or loan.
    """
    training = attr.ib()
    testing = attr.ib(default=[])
    language = attr.ib(default='')
    series = attr.ib(default='series')
    detect_type = attr.ib(default=None)
    model_type = attr.ib(default='dual')
    settings = attr.ib(default=BaseSettings())

    def __attrs_post_init__(self):
        self.cut_point = None

        all_tokens = [row[1] for row in self.training]+[row[1] for row in
                self.testing]
        self.vocab = Vocab(data=all_tokens)
        self.native_data = NeuralData(
                training=[row for row in self.training if row[2] == 0],
                testing=[row for row in self.testing if row[2] == 0],
                vocab=self.vocab
                )
        self.loan_data = NeuralData(
                training=[row for row in self.training if row[2] == 1],
                testing=[row for row in self.testing if row[2] == 1],
                vocab=self.vocab)
        self.native_model = NeuralWord(
                vocab_len=len(self.vocab),
                model_type=self.model_type,
                language=self.language,
                basis='native',
                series=self.series)
        self.loan_model = NeuralWord(
            vocab_len=len(self.vocab), 
            model_type=self.model_type,
            language=self.language, 
            basis='loan', 
            series=self.series)


    def train(self, model='native'):
        if model == 'native':
            print('training native model')
            self.native_model.train(
                    train_gen=self.native_data.trainer,
                    val_gen=self.native_data.validator)
        elif model == 'loan':
            print('training loan model')
            self.loan_model.train(
                    train_gen=self.loan_data.trainer,
                    val_gen=self.loan_data.validator)

        #if self.detect_type == 'dual':
        #    self.make_loan_model()


    def calculate_ref_limit(self, entropies=None, fraction=None):
        return find_ref_limit(
                entropies=entropies, 
                fraction=fraction or NeuralSettings().fraction
                )

    def make_native_predictions(self, entropies, cut_point):
        return [int(entropy>cut_point) for entropy in entropies]

    def make_dual_predictions(self, native_entropies, loan_entropies):
        return [int(loanH<nativeH) for nativeH, loanH
                       in zip(native_entropies, loan_entropies)]

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # Entropy calculation and predictions depend on whether native or dual.

        tokens_ids = [self.vocab.translate(t) for t in tokens]
        native_entropies = self.native_model.calculate_entropies(tokens_ids)

        if self.detect_type == 'native':
            if self.cut_point is None:

                train_tokens_ids = [self.vocab.translate(t[1]) for t in
                        self.native_data.training]
                entropies = self.native_model.calculate_entropies(train_tokens_ids)
                self.cut_point = self.calculate_ref_limit(entropies=entropies)

            return self.make_native_predictions(native_entropies, self.cut_point)

        elif self.detect_type == 'dual':
            loan_entropies = self.loan_model.calculate_entropies(tokens_ids)
            return self.make_dual_predictions(native_entropies, loan_entropies)


    def predict_data(self, data):
        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        return self.predict_data([['', token]])[0][2]

