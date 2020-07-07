#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:34:40 2020

@author: johnmiller

Detect loan words based on word entropies as calculated by a neural netword model.
Support for model trained on both native and borrowed words, or just native words
"""

import abc
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import attr

from pybor.config import BaseSettings, NeuralSettings
from pybor.entropies import NeuralWordRecurrent

import pybor.util as util

output_path = Path(BaseSettings().output_path).resolve()

logger = util.get_logger(__name__)


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

        self.data = None  # Once used, no need to keep raw data.


    def __getitem__(self, item):
        return self.vocab[item]

    def get(self, item, alternative=None):
        return self.vocab.get(item, alternative)

    def __len__(self):
        return len(self.vocab)

    # Unkown value is 1 and used here.
    def translate(self, word):
#        return [self.vocab[self.start]]+[self.vocab.get(x, Vocab.UNK)
        return [self.vocab[self.start]]+[self.vocab.get(x, 1)
                                         for x in word]+[self.vocab[self.end]]
    @property
    def size(self):
        # Count only in 1 direction, not both forward and reverse translation.
        return len(self.vocab)//2

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
    settings = attr.ib(default=NeuralSettings())

    def __attrs_post_init__(self):
        self.all_data = self.training+self.testing
        random.shuffle(self.all_data)
        self.vocab = self.vocab or Vocab([x[1] for x in self.all_data])
        random.shuffle(self.testing)
        self.val_split = self.settings.val_split if self.val_split is None else self.val_split
        self.fit, self.val = util.train_test_split(self.training, self.val_split)
        self.label_counts = dict(Counter([x[2] for x in self.training]))

        if self.settings.verbose:
            logger.info(f'Train length: {len(self.training)}' +
                f', fit length: {len(self.fit)}' +
                f', val length: {len(self.val) if self.val else 0}' +
                f', test length: {len(self.testing) if self.testing else 0}.')

    def translate(self, sequences):
        """
        Translate words to the internal numeric alphabet.
        """
        return [self.vocab.translate(t) for t in sequences]

    def get_data_tokens(self, data):
        return [x[1] for x in data]

    def get_data_tokens_ids(self, data):
        return self.translate([x[1] for x in data])

    def get_tokens_ids(self, sequences):
        return self.translate(sequences)

    def get_data_states(self, data):
        return [x[2] for x in data]

    def get_batcher(self, data):
        # Return generator as None if no data.
        if data is None or len(data) == 0: return None
        return KerasBatchGenerator(
                self.translate([x[1] for x in data]),
                batch_size=self.settings.batch_size,
                vocab_size=self.vocab.size,
                settings=self.settings)
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
class KerasBatchGenerator:
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
    vocab_size : int
        Size of the vocabulary - number of string keys in the vocabulary.
    skip_step : int, optional
        Maximum leap in data sequence. Step chosen at random up to maximum.
    """
    data = attr.ib(repr=False)
    batch_size = attr.ib(default=None)
    vocab_size = attr.ib(default=None)
    skip_step = attr.ib(default=None)
    settings = attr.ib(default=NeuralSettings(), repr=False)

    def __attrs_post_init__(self):
        # In case settings object is not right type.
        if not isinstance(self.settings, BaseSettings):
            self.settings = BaseSettings()
        self.batch_size = self.batch_size or self.settings.batch_size
        self.skip_step = self.skip_step or self.settings.skip_step
        self.current_idx = 0
        self.data_len = len(self.data)
        # Test for unknowns.


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
                if self.current_idx >= self.data_len:
                    self.current_idx = 0
                ## Build 2-D list of lists of ids for each word.
                ## Apply Keras pad_sequences (post padding).
                ## Yield x and y numpy arrays.
                # Truncate last symbol because we don't have more y to predict.
                x_lst.append(data[self.current_idx][:-1])

                ## Treat y as sparse.
                #y_lst.append(data[self.current_idx][1:])
                temp_y = data[self.current_idx][1:]
                y_lst.append(to_categorical(temp_y, num_classes=self.vocab_size))

                self.current_idx += 1

            self.current_idx += random.randint(0, self.skip_step)
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
    __metaclass__ = abc.ABCMeta

    training = attr.ib()
    testing = attr.ib(default=[])
    language = attr.ib(default='')
    series = attr.ib(default='series')
    model_type = attr.ib(default='')
    val_split = attr.ib(default=None)


    def __attrs_post_init__(self):
        # In case settings object is not right type.
        self.language = self.language or self.settings.language
        self.series = self.series or self.settings.series
        self.model_type = self.model_type or self.settings.model_type
        all_tokens = [row[1] for row in self.training]
        all_tokens += [row[1] for row in self.testing] if self.testing else []
        self.vocab = Vocab(data=all_tokens)

        native_training = [row for row in self.training if row[2] == 0]
        loan_training = [row for row in self.training if row[2] == 1]
        # Oversample loan data.
        if self.settings.oversample and len(loan_training) < len(native_training):
            k=len(native_training)-len(loan_training)
            loan_training += random.choices(loan_training, k=k)

        self.native_data = NeuralData(
                training=native_training,
                testing=[row for row in self.testing if row[2] == 0] if self.testing else [],
                vocab=self.vocab,
                val_split=self.val_split,
                settings=self.settings
                )

        # Convenient to separate out loan data always.
        self.loan_data = NeuralData(
                training=loan_training,
                testing=[row for row in self.testing if row[2] == 1] if self.testing else [],
                vocab=self.vocab,
                val_split=self.val_split,
                settings=self.settings)

        self.native_model = NeuralWordRecurrent(
                vocab_len=self.vocab.size,
                language=self.language,
                basis='native',
                series=self.series,
                settings=self.settings)


    @abc.abstractmethod
    def train(self):
        """Train method implemented by native and dual subclasses"""
        return

    @abc.abstractmethod
    def predict_tokens(self, tokens):
        """Predict tokens method implemented by native and dual subclasses"""
        return


    def predict_data(self, data):
        if not data:
            logger.warn('No data to use in prediction.')
            return []

        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        return self.predict_data([['', token]])[0][2]


@attr.s
class NeuralNative(Neural):
    """
    Construct neural model for native words.
    Train neural model on native word data.
    Determine cutpoint to detect loan words.
    Predict words as native if less than empirical entropy cut-point and loan otherwise.
    """
    fraction = attr.ib(default=None)
    settings = attr.ib(default=NeuralSettings())

    def __attrs_post_init__(self):
        if not isinstance(self.settings, NeuralSettings):
            self.settings = NeuralSettings()
        super().__attrs_post_init__()
        self.fraction = self.fraction or self.settings.fraction
        self.cut_point = None



    # Train only native model if detect type is native.
    def train(self, epochs=None):
        logger.info('training native model')
        self.native_history = self.native_model.train(
                train_gen=self.native_data.trainer,
                val_gen=self.native_data.validator,
                epochs=epochs)

    def reset_cut_point(self, fraction=None):
        # Allows to set new fraction for prediction without having to start over.
        self.fraction = fraction if fraction is not None else self.settings.fraction
        self.calculate_cut_point()

    def calculate_cut_point(self):

        train_tokens_ids = self.native_data.get_data_tokens_ids(self.native_data.training)
        entropies = self.native_model.calculate_entropies(train_tokens_ids)
        self.cut_point = util.find_frac_cut_point(entropies=entropies, fraction=self.fraction)

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # If necessary form empirical distribution of training entropies and determine cutpoint.
        # All tokens with entropy > cut_point are loanwords.

        if self.cut_point is None:
            self.calculate_cut_point()

        tokens_ids = [self.vocab.translate(t) for t in tokens]
        native_entropies = self.native_model.calculate_entropies(tokens_ids)

        return [int(entropy>self.cut_point) for entropy in native_entropies]


@attr.s
class NeuralDual(Neural):
    """
    Construct neural models for native and loan words.
    Train corresponding neural models on native and loan word data.
    Determine entropy bias between native and loan entropy distributions.
    Predict words as native or loan based on whether native or loan calculates lesser entropy.
    """
    settings = attr.ib(default=NeuralSettings())

    def __attrs_post_init__(self):
        if not isinstance(self.settings, NeuralSettings):
            self.settings = NeuralSettings()
        super().__attrs_post_init__()
        self.cut_point = None


        self.loan_model = NeuralWordRecurrent(
                vocab_len=self.vocab.size,
                language=self.language,
                basis='loan',
                series=self.series,
                settings=self.settings)



    def train(self, epochs=None):
        logger.info('training native model')
        self.native_history = self.native_model.train(
                train_gen=self.native_data.trainer,
                val_gen=self.native_data.validator,
                epochs=epochs)
        logger.info('training loan model')
        self.loan_history = self.loan_model.train(
                train_gen=self.loan_data.trainer,
                val_gen=self.loan_data.validator,
                epochs=epochs)


    def calculate_delta_entropies(self, tokens_ids):
        native_entropies = self.native_model.calculate_entropies(tokens_ids)
        loan_entropies = self.loan_model.calculate_entropies(tokens_ids)
        deltas = [native - loan for native, loan in zip(native_entropies, loan_entropies)]

        return deltas

    def calculate_cut_point(self):
        if self.settings.prediction_policy == 'zero':
            self.cut_point = 0
        else:
            # Use training data for calculation as most likely to be representative.
            data = self.loan_data.training
            tokens_ids = self.loan_data.get_data_tokens_ids(data)
            loan_deltas = self.calculate_delta_entropies(tokens_ids)

            data = self.native_data.training
            tokens_ids = self.native_data.get_data_tokens_ids(data)
            native_deltas = self.calculate_delta_entropies(tokens_ids)

            if self.settings.prediction_policy == 'accuracy':
                self.cut_point = util.find_acc_cut_point_deltas(native=native_deltas,
                                                                loan=loan_deltas)
                logger.info(f'Accuracy optimized prediction: cut_point ={self.cut_point:.4f}.')
            else:  # fscore.
                self.cut_point = util.find_fscore_cut_point_deltas(native=native_deltas,
                                                                   loan=loan_deltas,
                                                                   beta=self.settings.fscore_beta)
                logger.info(f'F score optimized prediction: cut_point ={self.cut_point:.4f}.')

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # Entropy calculation and predictions depend on whether native or dual.
        if self.cut_point == None:
            self.calculate_cut_point()

        tokens_ids = [self.vocab.translate(t) for t in tokens]
        deltas = self.calculate_delta_entropies(tokens_ids)
        return [int(delta>self.cut_point) for delta in deltas]


