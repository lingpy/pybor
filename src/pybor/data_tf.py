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

import os
import csv
from collections import  namedtuple

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences



data_path = "neural-data/"

Data_repo = namedtuple("Data_repo",
                       ['train_data', 'test_data', 'all_ids', 'all_tokens',
                        'all_tokens_ids', 'all_status', 'trainval_ids',
                        'trainval_tokens', 'trainval_tokens_ids', 'trainval_status',
                        'train_ids', 'train_tokens', 'train_tokens_ids', 'train_status',
                        'val_ids', 'val_tokens', 'val_tokens_ids', 'val_status',
                        'test_ids', 'test_tokens', 'test_tokens_ids', 'test_status',
                        'symbol_to_id', 'id_to_symbol'])

class NeuralData:

    def __init__(self, train_data=None, test_data=None, file_prefix=None, val_split=0.15):
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

        Returns
        -------
        NeuralData object reference.

        """
        self.vocab = None
        self.vocab_len = None
        self.data_repo = None

        self.prepare_data(train_data, test_data,
                file_prefix=file_prefix, val_split=val_split)

        repo = self.data_repo
        print(f'Train length: {len(repo.train_tokens)}, ',
              f'val length: {len(repo.val_tokens)}, ',
              f'test length: {len(repo.test_tokens) if test_data is not None else 0}.')

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
    def get_symbol_dictionary(file_prefix=None, tokens=None, rebuild=False):
        # Get the dictionary if rebuild is False and the dictionary exists.
        # Otherwise build the dictionary from tokens and save the dictionary.
        if not rebuild:
            dict_name = file_prefix+"-dict.csv"
            dictionary_path = os.path.join(data_path, dict_name)
            if os.path.exists(dictionary_path):
                with open(dictionary_path, 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    symbol_to_id = dict(reader)
                    symbol_to_id = dict([s, int(i)] for s, i in symbol_to_id.items())
                    print(f'Loaded dictionary {dict_name} of {len(symbol_to_id)} entries.')
                return symbol_to_id

        # Either rebuild is true or dictionary does not exist.
        symbol_to_id = NeuralData.build_vocab(tokens)
        with open(dictionary_path, 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in symbol_to_id.items():
                writer.writerow([key, value])
            print(f'Created dictionary {dict_name} of {len(symbol_to_id)} entries.')
        return symbol_to_id

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

    def prepare_data(self, train_data=None, test_data=None, file_prefix=None, val_split=0.15):
        """
        Prepare data in format usable by neural network model.
        Neural network uses validation data split off from training data.

        Parameters
        ----------
        train_data : [str, [str], int]
            Formatted data structure of id, token, and loan status.
        test_data : [str, [str], int], optional
            Test data in same format as train_data. The default is None.
        file_prefix : str, optional
            Prefix of file name for storing segment vocabulary (symbol_to_id).
        val_split : float, optional
            Fraction of training data to be randomly assigned as validation data.
            The default is 0.15.

        Returns
        -------
        Data_repo
            Repository of prepared training and test data.
        """

        train_ids, train_tokens, train_status = zip(*train_data)
        all_ids = train_ids[:]
        all_tokens = train_tokens[:]
        all_status = train_status[:]
        if test_data is not None:
            test_ids, test_tokens, test_status = zip(*test_data)
            all_ids += test_ids
            all_tokens += test_tokens
            all_status += test_status

        # Either get existing dictionary or create from tokens.
        symbol_to_id = NeuralData.get_symbol_dictionary(file_prefix, all_tokens)
        vocab_len = len(symbol_to_id)
        id_to_symbol = dict(zip(symbol_to_id.values(), symbol_to_id.keys()))
        assert vocab_len == len(id_to_symbol)

        all_tokens_ids = NeuralData.get_tokens_ids(all_tokens, symbol_to_id)

        # Split train into train and val datasets.
        # Copy train to trainval before splitting into train and val datasets.
        trainval_tokens = train_tokens
        trainval_ids = train_ids
        trainval_status = train_status

        trainval_tokens_ids = NeuralData.get_tokens_ids(trainval_tokens, symbol_to_id)
        # Split into train and val
        # Reloading the data randomizes the train-val split.
        if val_split is not None and val_split > 0.0:
            (train_tokens, val_tokens, train_tokens_ids, val_tokens_ids,
             train_ids, val_ids, train_status, val_status) = train_test_split(
                                     trainval_tokens, trainval_tokens_ids,
                                     trainval_ids, trainval_status, test_size=val_split)

        if test_data is not None:
            # perpare test data.
            test_tokens_ids = NeuralData.get_tokens_ids(test_tokens, symbol_to_id)

        self.vocab = symbol_to_id
        self.vocab_len = len(symbol_to_id)
        self.data_repo = Data_repo(train_data=train_data, test_data=test_data,
                         all_ids=all_ids, all_tokens=all_tokens,
                         all_tokens_ids=all_tokens_ids, all_status=all_status,
                         trainval_ids=trainval_ids, trainval_tokens=trainval_tokens,
                         trainval_tokens_ids=trainval_tokens_ids, trainval_status=trainval_status,
                         train_ids=train_ids, train_tokens=train_tokens,
                         train_tokens_ids=train_tokens_ids, train_status=train_status,
                         val_ids=val_ids, val_tokens=val_tokens,
                         val_tokens_ids=val_tokens_ids, val_status=val_status,
                         test_ids=test_ids, test_tokens=test_tokens,
                         test_tokens_ids=test_tokens_ids, test_status=test_status,
                         symbol_to_id=symbol_to_id, id_to_symbol=id_to_symbol)


    def make_generators(self):
        # Construct and return data generators for use with neural fit and test.
        self.train_gen = KerasBatchGenerator(self.data_repo.train_tokens_ids,
                                        vocab_len=self.vocab_len)

        self.val_gen = KerasBatchGenerator(self.data_repo.val_tokens_ids,
                                      vocab_len=self.vocab_len)

        if self.data_repo.test_tokens_ids is not None:
            self.test_gen = KerasBatchGenerator(self.data_repo.test_tokens_ids,
                                           vocab_len=self.vocab_len)
        else:
            self.test_gen=None

        return self.train_gen, self.val_gen, self.test_gen


    def verify_token_encoding(self):
        print('Symbol dictionary:', self.data_repo.symbol_to_id)
        print(f'Sample train data: {self.data_repo.trainval_tokens_ids[7:9]}')
        sample_tokens = [" ".join([self.data_repo.id_to_symbol[i] for i in token])
                         for token in self.data_repo.trainval_tokens_ids[7:9]]
        print(f'Sample train tokens: {sample_tokens}')

    def prepare_native_loan_data(self, native_data=None,
                                 loan_data=None, file_prefix=None):
        # Possible use depending on rest of design.
        pass


# Need to build batches with equal size words. Pad as necessary to accomplish this.
# Lengths of words between batches do not have to be equal.
# Return batches - both x and y.
import random

class KerasBatchGenerator(object):


    def __init__(self, data=None, vocab_len=None, batch_size=32):
        self.data = data
        self.data_len = len(self.data)
        self.batch_size = batch_size
        self.vocab_len = vocab_len
        # current_ids tracks the progress of the batches sequentially through the
        # data set. Once the data reaches the end of the data set it resets
        # back to zero.
        self.current_idx = 0

    def generate(self):
        # Randomize order of words.
        #random.shuffle(self.data)
        # Make a copy of the data for local use.
        data_copy = random.sample(self.data, self.data_len)
        while True:
            x_lst = []
            y_lst = []
            for i in range(self.batch_size):
                ## Build 2-D list of lists of ids for each word.
                ## Truncate the lists to no more than num_steps.
                ## Apply Keras pad_sequences (post padding).
                ## Yield x and y numpy arrays.
                if self.current_idx >= self.data_len:
                    # reset the index back to the start of the data set
                    self.current_idx = 0

                # Truncate last symbol because we don't have more y to predict.
                x_lst.append(data_copy[self.current_idx][:-1])
                # Truncate first symbol because we predict y based on previous x
                temp_y = data_copy[self.current_idx][1:]
                # Convert temp_y into a one hot representation
                y_lst.append(to_categorical(temp_y, num_classes=self.vocab_len))
                self.current_idx += 1

            x = pad_sequences(x_lst, padding='post')
            y = pad_sequences(y_lst, padding='post')
            yield x, y

#--------------------------------------------------------

# if __name__ == "__main__":
#     import pybor
#     from pybor.dev.data import testing, training

#     nd = NeuralData(training, testing, 'German-dev-data')
#     nd.verify_token_encoding()
#     train_gen, val_gen, test_gen = nd.make_generators()
#     print(next(train_gen.generate()))
