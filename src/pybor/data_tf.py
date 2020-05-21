#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:51:57 2020

@author: johnmiller

Process data in format expected by neural model - NeuralWord class.
Prepare generators for use by neural model fitting and test.
Prepare token data in format of lists of tokens with tokens
as lists of int ids of words from the segment vocabulary.
Stores segment vocabulary by language for reuse in stored neural models.
"""
import math
import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pybor.neural_cfg as ncfg


class NeuralData:

    def __init__(self, train_data=None, test_data=None, vocab=None, val_split=None):
        """
        Construct data store for neural net use in training and test.

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

        Returns
        -------
        NeuralData object reference.

        """
        self.vocab = vocab
        self.val_split = val_split if val_split is not None else ncfg.data['val_split']

        self.prepare_data(train_data, test_data)

        if ncfg.data['verbose'] > 0:
            print(f'Train length: {len(self.train_tokens)}, ',
                f'val length: {len(self.val_tokens) if self.val_data is not None else 0}, ',
                f'test length: {len(self.test_data) if self.test_data is not None else 0}.')


    @staticmethod
    def build_vocab(tokens=None):
        # Use complete dataset so that no undefined symbols.
        # Sorted for esthetics.
        symbols = set([symbol for token in tokens for symbol in token])
        # Add in unknown, start, stop, and nul symbols.
        symbols.add('<unk>')
        symbols.add("<s>")
        symbols.add("</s>")
        symbols = sorted(symbols)
        symbol_to_id = dict(zip(symbols, range(1, len(symbols)+1)))
        symbol_to_id["<nul>"] = 0 # zero padding indicator.
        return symbol_to_id

    @staticmethod
    def get_reverse_vocab(vocab):
        return dict(zip(vocab.values(), vocab.keys()))

    @staticmethod
    def get_tokens_ids(tokens=None, symbol_to_id=None):
        # Convert token list of segments to token list of int ids.
        unk_id = symbol_to_id.get('<unk>')
        tokens_ids = [[symbol_to_id.get(symbol, unk_id)
                for symbol in token]
                for token in tokens]
        # Add start and stop symbols.
        start_id = symbol_to_id["<s>"]
        stop_id = symbol_to_id["</s>"]
        tokens_ids = [[start_id]+token+[stop_id] for token in tokens_ids]
        return tokens_ids

    @staticmethod
    def get_ids_tokens(tokens_ids=None, id_to_symbol=None):
        #Convert lists of tokens_ids to lists of segments.
        tokens = [[id_to_symbol.get(id_, '<unk>')
                  for id_ in token_ids]
                  for token_ids in tokens_ids]
        # Don't bother to strip <s> and </s> symbols.
        return tokens

    @staticmethod
    def simple_train_test_split(table=None, test_split=None):
        table = random.sample(table, len(table))
        split = test_split if test_split is not None else ncfg.data['test_split']
        split = int(split) if split >= 1 else math.ceil(len(table)*split)

        train, test = table[:-split], table[-split:]
        print(f'table len={len(table)}, train len={len(train)}, test len={len(test)}.')
        return train, test

    def prepare_data(self, train_data=None, test_data=None):
        """
        Prepare data in format usable by neural network model.
        Neural network uses validation data split off from training data.

        Parameters
        ----------
        train_data : [str, [str], int]
            Formatted data structure of id, token, and loan status.
        test_data : [str, [str], int], optional
            Test data in same format as train_data. The default is None.

        Returns
        -------
        """

        trainval_data = train_data
        if test_data is not None:
            all_data = trainval_data + test_data
        else:
            all_data = trainval_data

        # Randomize order of data now to maintain correspondence with tokens.
        self.all_data = random.sample(all_data, len(all_data))
        self.all_tokens = [token for _, token, _ in self.all_data]

        if self.vocab == None:
            self.vocab = NeuralData.build_vocab(self.all_tokens)
        self.id_to_symbol = NeuralData.get_reverse_vocab(self.vocab)

        self.all_tokens_ids = NeuralData.get_tokens_ids(self.all_tokens, self.vocab)

        if test_data is not None:
            self.test_data = random.sample(test_data, len(test_data))
            self.test_tokens = [token for _, token, _ in self.test_data]
            self.test_tokens_ids = NeuralData.get_tokens_ids(self.test_tokens, self.vocab)
        else:
            self.test_data = None
            self.test_tokens = None
            self.test_tokens_ids = None

        # Split train into train and val datasets.
        # Reloading the data randomizes the train-val split.
        if  self.val_split > 0.0:
            self.train_data, self.val_data = NeuralData.simple_train_test_split(
                    table=trainval_data, test_split=self.val_split)
            self.train_tokens = [token for _, token, _ in self.train_data]
            self.train_tokens_ids = NeuralData.get_tokens_ids(self.train_tokens, self.vocab)
            self.val_tokens = [token for _, token, _ in self.val_data]
            self.val_tokens_ids = NeuralData.get_tokens_ids(self.val_tokens, self.vocab)

        else:
            self.train_data = random.sample(trainval_data, len(trainval_data))
            self.train_tokens = [token for _, token, _ in self.train_data]
            self.train_tokens_ids = NeuralData.get_tokens_ids(self.train_tokens, self.vocab)
            self.val_data = None
            self.val_tokens = None
            self.val_tokens_ids = None


    def make_generators(self):
        # Construct and return data generators for use with neural fit and test.
        self.train_gen = KerasBatchGenerator(
            self.train_tokens_ids, vocab_len=len(self.vocab))

        if self.val_tokens_ids is not None:
            self.val_gen = KerasBatchGenerator(
                self.val_tokens_ids, vocab_len=len(self.vocab))
        else:
            self.val_gen = None

        if self.test_tokens_ids is not None:
            self.test_gen = KerasBatchGenerator(
                self.test_tokens_ids, vocab_len=len(self.vocab))
        else:
            self.test_gen=None

        return self.train_gen, self.val_gen, self.test_gen


    def verify_token_encoding(self, tokens_ids):
        print('Symbol dictionary:', self.vocab)
        print('Reverse dictionary:', self.id_to_symbol)
        print(f'Sample train tokens ids: {tokens_ids}')
        sample_tokens = [" ".join([self.id_to_symbol.get(i, '<unk>')
                                   for i in token_ids]) for token_ids in tokens_ids]
        print(f'Sample train tokens: {sample_tokens}')



# Need to build batches with equal size words. Pad as necessary to accomplish this.
# Lengths of words between batches do not have to be equal.
# Return batches - both x and y.

class KerasBatchGenerator(object):


    def __init__(self, data=None, vocab_len=None, batch_size=None):
        """
        Construct a generator for the neural network

        Parameters
        ----------
        data : [[int]]
            list of tokens. Tokens are represented by lists of ids.
            Each list of ids corresponds to a list of character string segments.
        vocab_len : int
            Length of vocabulary of character string segments. The default is None.
        batch_size : int, optional
            Batch size to use in neural network fit.
            If not given, the configuratiom batch size is used.

        Returns
        -------
        None.

        """
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size if batch_size is not None else ncfg.data['batch_size']
        self.vocab_len = vocab_len
        # current_ids tracks the progress of each batch through the data set.
        # Once the data reaches the end of the data set it resets back to zero.
        self.current_idx = 0

    def generate(self):
        # Randomize order of words.
        data = random.sample(self.data, self.data_len)
        while True:
            x_lst = []
            y_lst = []
            for i in range(self.batch_size):
                ## Build 2-D list of lists of ids for each word.
                ## Apply Keras pad_sequences (post padding).
                ## Yield x and y numpy arrays.

                # Truncate last symbol because we don't have more y to predict.
                x_lst.append(data[self.current_idx][:-1])
                # Truncate first symbol because we predict y based on previous x
                temp_y = data[self.current_idx][1:]
                # Convert temp_y into a one hot representation
                y_lst.append(to_categorical(temp_y, num_classes=self.vocab_len))

                self.current_idx += 1
                if self.current_idx >= self.data_len:
                    self.current_idx = 0

            x = pad_sequences(x_lst, padding='post')
            y = pad_sequences(y_lst, padding='post')
            yield x, y


#--------------------------------------------------------

# if __name__ == "__main__":
#     import pybor
#     from pybor.dev.data import testing, training

#     nd = NeuralData(training, testing)
#     nd.verify_token_encoding(nd.test_tokens_ids[:5])
#     train_gen, val_gen, test_gen = nd.make_generators()
#     print(next(train_gen.generate()))
