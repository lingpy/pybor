#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:29:16 2020

@author: johnmiller

Neural configurations.
"""

recurrent = {
    # Model architecture parameters
    'embedding_len': 16,
    'rnn_output_len': 32,
    'rnn_cell_type': 'GRU',  # GRU, LSTM

    'print_summary': False,
    'plot_model': False,
    'plot_dpi': 400,
    # Model dropout and regulation parameters
    'embedding_dropout': 0.0,
    # 'rnn_kernel_l2': 0.0,
    'recurrent_l2': 0.001,
    'rnn_activity_l2': 0.0,
    # 'rnn_input_dropout': 0.0,
    'recurrent_dropout': 0.0,
    'rnn_output_dropout': 0.2,
    'merge_embedding_dropout': 0.2,

    # Model fitting parameters
    'epochs': 30,
    'learning_rate': 0.01,
    'lr_decay': 0.80,  # 0.8, 0.9, 0.95
    'neural_verbose': 1,
    'tf_verbose': 0,
    }

attention = {
    # Model architecture parameters
    'embedding_len': 32,
    'rnn_output_len': 32,
    'rnn_cell_type': 'LSTM',  # GRU, LSTM

    'print_summary': False,
    'plot_model': False,
    'plot_dpi': 400,
    # Model dropout and regulation parameters
    'embedding_dropout': 0.0,
    # 'rnn_kernel_l2': 0.0,  # Not implemented
    'recurrent_l2': 0.0,
    'rnn_activity_l2': 0.001,
    # 'rnn_input_dropout': 0.0,  # Not implemented
    'recurrent_dropout': 0.2,
    'rnn_output_dropout': 0.2,

    # Model fitting parameters
    'epochs': 50,
    'learning_rate': 0.01,
    'lr_decay': 0.95,  # 0.9, 0.95, 0.975
    'neural_verbose': 1,
    'tf_verbose': 0,
    }

data = {
    'batch_size': 32,
    'token_maxlen': 40,
    'val_split' : 0.15,
    'test_split' : 0.15,
    'basis': 'all',  # 'native', 'loan'
    'verbose': 1,
    }

neural = {
    'language': '',
    'series': '',
    'detect_type': 'dual',  # 'dual', 'native'.
    'model_type': 'recurrent',  #  'recurrent', 'attention'
    # Options for policy on entropy decision limit.
    # Fraction of train, max accuracy, max F1.
    # Fraction and F1 implemented.
    # 'native_policy': 'fraction',  # 'fraction', 'accuracy', 'F1' # Not implemented
    'fraction': 0.995,  # Fraction of train dataset below limit for native basis.
    }

system = {
    'output_path': './output'
    }

# 'form': 'tokens',  # 'Tokens', 'SCA', 'DOLGO', 'ASJP', 'FormChars'
