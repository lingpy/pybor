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
from collections import Counter

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

        val_split = val_split if val_split is not None else ncfg.data['val_split']
        self.prepare_data(train_data, test_data, val_split=val_split)

        self.label_counts = dict(Counter(self.get_data_labels(self.train_data)))

        if ncfg.data['verbose'] > 0:
            print(f'Train length: {len(self.train_data)}, ',
                f'fit length: {len(self.fit_data)}',
                f'val length: {len(self.val_data) if self.val_data is not None else 0}, ',
                f'test length: {len(self.test_data) if self.test_data is not None else 0}.')


    def prepare_data(self, train_data=None, test_data=None, val_split=None):
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

        all_data = train_data if test_data is None else train_data + test_data

        # Randomize order of data now to maintain correspondence with tokens.
        self.all_data = random.sample(all_data, len(all_data))

        all_tokens = self.get_data_tokens(self.all_data)

        if self.vocab is None:
            self.vocab = NeuralData.build_vocab(all_tokens)
        self.reverse_vocab = NeuralData.get_reverse_vocab(self.vocab)

        if test_data is not None:
            self.test_data = random.sample(test_data, len(test_data))
        else:
            self.test_data = None

        self.train_data = random.sample(train_data, len(train_data))

        # Split train into fit and val datasets.
        if val_split > 0.0:
            self.fit_data, self.val_data = NeuralData.simple_train_test_split(
                    table=train_data, test_split=val_split)
        else:  # Fit data refers to train data.
            self.fit_data = self.train_data
            self.val_data = None


# =============================================================================
#
# Static service methods available internally or externally.
#
# =============================================================================
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
    def simple_train_test_split(table=None, test_split=None):
        table = random.sample(table, len(table))
        split = test_split if test_split is not None else ncfg.data['test_split']
        split = int(split) if split >= 1 else math.ceil(len(table)*split)

        train, test = table[:-split], table[-split:]
        print(f'table len={len(table)}, train len={len(train)}, test len={len(test)}.')
        return train, test

    @staticmethod
    def _get_tokens_ids(tokens=None, vocab=None):
        if tokens is None or len(tokens) == 0: return []
        # Convert token list of segments to token list of int ids.
        unk_id = vocab.get('<unk>')
        tokens_ids = [[vocab.get(symbol, unk_id)
                for symbol in token]
                for token in tokens]
        # Add start and stop symbols.
        start_id = vocab["<s>"]
        stop_id = vocab["</s>"]
        tokens_ids = [[start_id]+token+[stop_id] for token in tokens_ids]
        return tokens_ids

    @staticmethod
    def _get_ids_tokens(tokens_ids=None, reverse_vocab=None):
        if tokens_ids  is None or len(tokens_ids) == 0: return []
        #Convert lists of tokens_ids to lists of segments.
        tokens = [[reverse_vocab.get(id_, '<unk>')
                  for id_ in token_ids]
                  for token_ids in tokens_ids]
        # Don't bother to strip <s> and </s> symbols.
        return tokens



# =============================================================================
#
# Object access methods
#
# =============================================================================
    def get_tokens_ids(self, tokens=None):
        return NeuralData._get_tokens_ids(tokens=tokens, vocab=self.vocab)

    def get_ids_tokens(self, tokens_ids=None):
        return NeuralData._get_ids_tokens(
                tokens_ids=tokens_ids, reverse_vocab=self.reverse_vocab)

    def get_data_ids(self, data=None):
        if data is None or len(data) == 0: return []
        return [row[0] for row in data]

    def get_data_tokens(self, data=None):
        if data is None or len(data) == 0: return []
        return  [row[1] for row in data]

    def get_data_labels(self, data=None):
        if data is None or len(data) == 0: return []
        return [row[2] for row in data]

    def get_data_tokens_ids(self, data=None):
        if data is None or len(data) == 0: return []
        return self.get_tokens_ids(self.get_data_tokens(data))

    def get_data_for_gen(self, data=None):
        if data is None or len(data) == 0: return []
        return self.get_data_tokens_ids(data)

    def get_chosen_tokens_ids(self, choice=''):
        if choice == 'fit':
            data = self.fit_data
        elif choice == 'val':
            data = self.val_data
        elif choice == 'train':
            data = self.train_data
        elif choice == 'test':
            data = self.test_data
        else:  # Even if choice was other than 'all', return all_data
            data = self.all_data

        return self.get_data_tokens_ids(data)


# =============================================================================
#
# Construct generators for training and test.
#
# =============================================================================
    def make_generators(self):
        # Construct and return data generators for use with neural fit and test.
        vocab_len = len(self.vocab)
        batch_size = ncfg.data['batch_size']
        self.train_gen = KerasBatchGenerator(
            self.get_data_for_gen(self.fit_data),
            vocab_len=vocab_len,
            batch_size=batch_size)

        if self.val_data is not None:
            self.val_gen = KerasBatchGenerator(
                self.get_data_for_gen(self.val_data),
                vocab_len=vocab_len,
                batch_size=batch_size)
        else:
            self.val_gen = None

        if self.test_data is not None:
            self.test_gen = KerasBatchGenerator(
                self.get_data_for_gen(self.test_data),
                vocab_len=vocab_len,
                batch_size=batch_size)
        else:
            self.test_gen=None

        return self.train_gen, self.val_gen, self.test_gen




# =============================================================================
# Need to build batches with equal size words. Pad as necessary to accomplish this.
# Lengths of words between batches do not have to be equal.
# Return batches - both x and y.
#
# =============================================================================
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
        self.vocab_len = vocab_len
        self.batch_size = batch_size if batch_size is not None else ncfg.data['batch_size']
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

                # Treat y as sparse.
                y_lst.append(data[self.current_idx][1:])

                # Truncate first symbol because we predict y based on previous x
                #temp_y = data[self.current_idx][1:]
                # Convert temp_y into a one hot representation
                #y_lst.append(to_categorical(temp_y, num_classes=self.vocab_len))

                self.current_idx += 1
                if self.current_idx >= self.data_len:
                    self.current_idx = 0

            x = pad_sequences(x_lst, padding='post')
            y = pad_sequences(y_lst, padding='post')
            yield x, y


#--------------------------------------------------------

# if __name__ == "__main__":
#     from pybor.dev.data import testing1, training1

#     nd = NeuralData(training1, testing1)
#     train_gen, val_gen, test_gen = nd.make_generators()
#     x, y =  next(train_gen.generate())
#     print(x)
#     print(y)
