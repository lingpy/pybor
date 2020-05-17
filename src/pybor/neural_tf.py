#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:17:21 2020

@author: johnmiller

Neural net entropy estimation module to configure and fit a neural network model, and
estimate entropies given a token or list of tokens. Tokens are encoded as lists of
integer ids where each id corresponds to a symbol segment from a vocabulary.
"""

import math
import pickle
from pathlib import Path

import tensorflow as tf
#tf.autograph.set_verbosity(0, False)

from tensorflow.keras.models import Model
#from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import GRU, LSTM, AdditiveAttention
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# import numpy as np

# ** No saving of neural models or logs.
# model_path_best = "neural-models-best/"
# model_path_final = "neural-models-final/"
#log_path = "neural-logs/"  # Model summaries saved to log path.
output_path = Path('./output').resolve()


# @tf.autograph.experimental.do_not_convert
class NeuralWord:
    """
    Use lists of token id lists to calculate token entropies.
    Neural net model is configured and then trained from a list of tokens at
    time of construction. Entropy or entroy lists are calculated from the
    trained neural net model.
    """

    # @tf.autograph.experimental.do_not_convert
    def __init__(self, vocab_len=None, model_type='attention',
                 language='', basis='all', series='', name='',
                 cell_type='LSTM', embedding_len=32, rnn_output_len=32,
                 dropout_rate=0.1, l2_amt=0.001):
        """
        Neural net based model to calculate token entropy.
        Configure and complile the model and fit training data at construction.
        Calculate entropies on demand based on the fitted model.

        Parameters
        ----------
        vocab_len : int
            Length of the vocabulary. Note: All token segments are integer encoded.
        model_type : str ['recurrent', 'attention']
        language : str
            Language being modeled. Used in model naming for storage.
        basis : str ['all', 'native', 'loan', ...]
            Whether all tokens, just native, or just loan.
        series : str
            Study series to qualify model name.
        name : str
            Qualifying name given to this model.
        cell_type : str, optional
            Neural memory cell type for recurrent layer ['LSTM', 'GRU']
        embedding_len : int, optional
            Length of embedding layer. The default is 32.
        rnn_output_len : int, optional
            Length of recurrent output layer. The default is 32.
        dropout_rate : float, optional
            Fraction of cells that may be dropped out during training. The default is 0.1.
            Dropout >= 0.2 may result in poorer fit overall.
        l2_amt : float, optional
            Amount of l2 regulation applied to model during training. The default is 0.001.

        Returns
        -------
        NeuralWord object reference.
        """
        self.initialized = False

        if model_type not in ['recurrent', 'attention']:
            print(f'Invalid model {model_type}.')
            return
        if cell_type not in ['LSTM', 'GRU']:
            print('Invalid cell type {cell_type}')
            return
        if vocab_len is None:
            print(f'Require the vocabulary size to construct NeuralWord')
            return

        self.vocab_len = vocab_len
        self.model_type = model_type
        self.language = language
        self.basis = basis
        self.series = series
        self.name = name
        self.cell_type = cell_type
        self.embedding_len = embedding_len
        self.rnn_output_len = rnn_output_len
        self.dropout_rate = dropout_rate
        self.l2_amt = l2_amt
        self.model_name = None
        self.model = None

        self._construct_modelname(
                language=language, basis=basis, series=series, name=name)


        # Build the model.
        if self.model_type == 'recurrent':
            self._build_recurrent_model()
        else:  # 'attention'
            self._build_attention_model()

        self.initialized = True
        # Finally after all the other work, set to True.



    def calculate_entropy(self, token_ids):
        return self.calculate_entropies([token_ids])[0]


    # @tf.autograph.experimental.do_not_convert
    def calculate_entropies(self, tokens_ids):
        # Calculate entropy for a list of tokens.
        # Replicate the structure of data from batch generator
        # to improve performance and avoid advisory messages from tensorflow.
        # Use batch size of 32.

        batch_size = 32
        maxlen = 50
        idx = 0
        len_tokens = len(tokens_ids)
        entropies = []

        while idx < len_tokens:
            x_lst = []
            y_lst = []

            for i in range(batch_size):
                if idx >= len_tokens: break

                x_lst.append(tokens_ids[idx][:-1])
                y_lst.append(tokens_ids[idx][1:])
                idx += 1

            # Use maxlen to avoid tracing in tensorflow.
            x_tf = pad_sequences(x_lst, padding='post', maxlen=maxlen)
            probs_tf = self.model.predict(x_tf)
            #probs = np.squeeze(probs_tf)
            probs = probs_tf

            # Calculate entropy versus the actual token_ids.
            assert len(probs) == len(y_lst)
            for x_ids_probs, y_ids in zip(probs, y_lst):
                # Use len(y_ids) for the range since it retains token lengths.
                x_ids_lns = [math.log(x_ids_probs[i, y_ids[i]]) for i in range(len(y_ids))]
                entropy = -sum(x_ids_lns)/len(x_ids_lns)
                entropies.append(entropy)

        assert len(tokens_ids) == len(entropies)
        return entropies


    def _construct_modelname(self, language='', basis='A', series='', name=''):
        language_out = ''.join(language.split())
        model_name = f'{language_out}-{basis}-{series}-{name}-mdty{self.model_type}'
        model_name += f'-clty{self.cell_type}-embln{self.embedding_len}'
        model_name += f'-rnnln{self.rnn_output_len}-do{self.dropout_rate:1g}'
        model_name += f'-l2{self.l2_amt:.4g}'
        self.model_name = model_name

    # @tf.autograph.experimental.do_not_convert
    def _build_recurrent_model(self):

        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,))

        # Embedding of characters.
        # Mask zero works.
        embedding = Embedding(input_dim=self.vocab_len, output_dim=self.embedding_len,
                mask_zero=True, name='Segment_embedding')(inputs)

        if self.cell_type == 'LSTM':
            # Incorporate embeddings into hidden state and output state.
            rnn_output = LSTM(self.rnn_output_len, return_sequences=True,
                activity_regularizer=l2(self.l2_amt),
                recurrent_dropout=self.dropout_rate, name='LSTM_recurrent')(embedding)

        else:  # GRU
            rnn_output = GRU(self.rnn_output_len, return_sequences=True,
                activity_regularizer=l2(self.l2_amt),
                recurrent_dropout=self.dropout_rate, name='GRU_recurrent')(embedding)

        # Add in latest embedding per Bengio 2002.
        to_outputs = Concatenate(axis=-1, name='Merge_rnn_embedding')([rnn_output, embedding])
        if self.dropout_rate > 0.0:
            to_outputs = Dropout(self.dropout_rate, name='Dropout_to_outputs')(to_outputs)


        # Hidden state used to predict subsequent character.
        outputs = Dense(self.vocab_len, activation='softmax', name='Segment_output')(to_outputs)

        model = Model(inputs=[inputs], outputs=[outputs], name=self.model_name)
        self.model = model

        #print(model.summary())


    # @tf.autograph.experimental.do_not_convert
    def _build_attention_model(self):

        K.clear_session()

        inputs = Input(shape=(None,))

        embedding = Embedding(input_dim=self.vocab_len, output_dim=self.embedding_len,
                              mask_zero=True, name='Segment_embedding')(inputs)

        # Could add dropout and regulation to these.
        if self.cell_type == 'LSTM':
            rnn_forward, forward_hidden, _ = LSTM(self.rnn_output_len,
                                            return_sequences=True, return_state=True,
                                            activity_regularizer=l2(self.l2_amt),
                                            recurrent_dropout=self.dropout_rate,
                                            name='LSTM_attention')(embedding)
        else:
            rnn_forward, forward_hidden = GRU(self.rnn_output_len,
                                        return_sequences=True, return_state=True,
                                        activity_regularizer=l2(self.l2_amt),
                                        recurrent_dropout=self.dropout_rate,
                                        name='LSTM_attention')(embedding)

        if self.dropout_rate > 0.0:
            rnn_forward = Dropout(self.dropout_rate,
                                  name='Dropout_rnn_output')(rnn_forward)

        # hidden state output is for the entire word.  Not by character.
        forward_hidden = Reshape((-1, forward_hidden.shape[1]),
                               name='Reshape_hidden')(forward_hidden)

        context_vector = AdditiveAttention()([rnn_forward, forward_hidden])

        concatenate_layer = Concatenate(axis=2)([context_vector, rnn_forward])

        outputs = Dense(self.vocab_len, activation="softmax",
                        name='Segment_output')(concatenate_layer)

        model = Model(inputs=[inputs], outputs=[outputs], name=self.model_name)
        self.model = model
        self.print_model_summary()


    # @tf.autograph.experimental.do_not_convert
    def train(self, train_gen=None, val_gen=None,
              epochs=50, learning_rate=0.01, lr_decay=0.975):
        """
        Train the neural network using training and validation data in generators.

        Parameters
        ----------
        train_gen : Keras generator
            Training generator. The default is None.
        val_gen : Keras generator
            Validation generator. The default is None.
        epochs : int, optional
            Maximum number of epochs in training the model. The default is 50.
        learning_rate: float, optional
            Learning rate used in fitting neural model. The default is 0.01.
        lr_decay : TYPE, optional
            Multiplicative factor for learning rate each epoch. The default is 0.97.

        Returns
        -------
        None.

        """
        # Will invoke this after consruction of the model.
        # Too heavy weight to do in init of class.

        train_steps = train_gen.data_len//train_gen.batch_size
        lr_decay = (1.0/lr_decay-1.0)/train_steps
        optimizer = Adam(learning_rate=learning_rate, decay=lr_decay)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy', 'categorical_crossentropy'])

        # Filename without .h5 suffix, provides better handling of state.
        # Had trouble loading withoutwhen not using .h5 suffix, so still use it.
        # *** No saving of neural model, since we don't have way to load it still.
        # best_file_name = model_path_best+self.model_name+'-best.h5'
        # checkpointer = ModelCheckpoint(filepath=best_file_name,
        #                                verbose=1, save_best_only=True)

        earlystopper = EarlyStopping(monitor='val_loss', verbose=1,
                                     patience=5, restore_best_weights=True)

        val_steps=max(val_gen.data_len//val_gen.batch_size, 2)
        history = self.model.fit_generator(train_gen.generate(),
                        train_steps, epochs,
                        validation_data=val_gen.generate(),
                        validation_steps=val_steps,
                        callbacks=[earlystopper])
                        # callbacks=[checkpointer, earlystopper])
        # ** No saving of neural models until we have way to load them.
        # final_file_name = model_path_final+self.model_name+'-final.h5'
        # self.model.save(final_file_name)

        # Save history.
        # ** No saving of history log file until we have funtion to load them.
        # history_file_name = log_path+self.model_name+'-history.pickle'
        # with open(history_file_name, 'wb') as history_pi:
        #     pickle.dump(history.history, history_pi)
        # Load history with:
        # open(log_path+model_prefix+'-history.pickle', "rb") as history_pi:
        #    history_dict = pickle.load(history_pi)

    def evaluate_test(self, test_gen=None):
        # Evaluate by generator.
        score = self.model.evaluate_generator(test_gen.generate(), steps=50, verbose=False)

        print(f'Test dataset: loss={score[0]:.4f}, '+
              f'accuracy={score[1]:.4f}, '+
              f'crossentropy={score[2]:.4f}')
        return score

    def print_model_summary(self):
        # Prints model summary to log path.
        print_fn = output_path / (self.model_name+'.txt')
        with open(print_fn.as_posix(),'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def plot_model_summary(self, dpi=96):
        print_fn = output_path / (self.model_name+'.png')
        plot_model(self.model, print_fn.as_posix(), show_shapes=True,
                   show_layer_names=True, dpi=dpi)

#
if __name__ == "__main__":
    nw = NeuralWord(model_type='recurrent', cell_type='GRU', vocab_len=55, series='devel')
    nw.print_model_summary()
    nw.plot_model_summary()


