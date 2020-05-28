"""
Neural net entropy estimation module to configure and fit a neural network model

Notes
-----
Created on Tue May 12 13:17:21 2020

@author: johnmiller

Neural net entropy estimation module to configure and fit a neural network model, and
estimate entropies given a token or list of tokens. Tokens are encoded as lists of
integer ids where each id corresponds to a symbol segment from a vocabulary.
"""

import math
from pathlib import Path

import tensorflow as tf
#tf.autograph.set_verbosity(0, False)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import GRU, LSTM, AdditiveAttention
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping  # ModelCheckpoint,
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras import backend as K

import attr

from pybor.config import *

neus = NeuralSettings()
neud = BaseSettings()
attention = AttentionSettings()
recurrent = RecurrentSettings()
output_path = Path(neus.output_path).resolve()


class NeuralWord:
    """
    Use lists of token id lists to calculate token entropies.
    Neural net model is configured and then trained from a list of tokens at
    time of construction. Entropy or entroy lists are calculated from the
    trained neural net model.
    """

    @tf.autograph.experimental.do_not_convert
    def __init__(self, vocab_len=None, model_type='attention',
                 language='', basis='all', series='', name=''):
        """
        Neural net based model to calculate token entropy.
        Configure and complile the model and fit training data at construction.
        Calculate entropies on demand based on the fitted model.

        Parameters
        ----------
        vocab_len : int
            Length of the vocabulary. Note: All token segments are integer encoded.
        model_type : str ['recurrent', 'attention']
            References a recurrent model (Bengio, 2002) and a recurrent model with attention.
        language : str
            Language being modeled. Used in model naming for storage.
        basis : str ['all', 'native', 'loan', ...]
            Whether all tokens, just native, or just loan.
        series : str
            Study series to qualify model name.
        name : str
            Qualifying name given to this model.

        Returns
        -------
        NeuralWord object reference.

        Notes
        -----
        Based on recent research, dropout of 0.1 should be considered when model includes
        multiple levels of dropout.
        """

        if model_type not in ['recurrent', 'attention']:
            print(f'Invalid model type "{model_type}" set to "recurrent".')
            model_type = 'recurrent'
        if vocab_len is None or vocab_len <= 0:
            raise ValueError('Require a vocabulary size > 0 to construct NeuralWord')

        self.vocab_len = vocab_len
        self.model_type = model_type
        self.language = language
        self.basis = basis
        self.series = series
        self.name = name
        self.model = None

        # Build the model.
        if self.model_type == 'recurrent':
            self._build_recurrent_model()
        else:  # 'attention'
            self._build_attention_model()


    def calculate_entropy(self, token_ids):
        return self.calculate_entropies([token_ids])[0]


    def calculate_entropies(self, tokens_ids):
        # Calculate entropy for a list of tokens_ids
        # in format of int ids that correspond to str segments.

        # Get the probabilities for all str segment possibilities.
        maxlen = max([len(token_ids) for token_ids in tokens_ids])
        # Truncate right id for x and left id for y, so only 1 id extra.
        if maxlen > neud.token_maxlen+1:
            maxlen = neud.token_maxlen+1

        x_lst = []
        y_lst = []
        for token_ids in tokens_ids:
            x_lst.append(token_ids[:-1])
            y_lst.append(token_ids[1:])

        x_tf = pad_sequences(x_lst, padding='post', maxlen=maxlen)
        x_probs = self.model.predict(x_tf)

        # Compute cross-entropies

        entropies = []
        EPSILON = 1e-7
        for x_ids_probs, y_ids in zip(x_probs, y_lst):
            # Prevent overflow/underflow with clipping.
            x_ids_probs_ = tf.clip_by_value(x_ids_probs,  EPSILON, 1-EPSILON)
            x_ids_lns = [math.log(x_ids_probs_[i, y_ids[i]]) for i in range(len(y_ids))]
            entropy = -sum(x_ids_lns)/len(x_ids_lns)
            entropies.append(entropy)

        assert len(tokens_ids) == len(entropies)
        return entropies


    def param(self, key=None):
        return attention.__dict__[key] if self.model_type == 'attention' else recurrent.__dict__[key]

    def _construct_modelname(self):
        language_out = ''.join(self.language.split())
        model_name = f'{language_out}-' if language_out != '' else ''
        model_name += f'basis{self.basis}-' if self.basis != '' else ''
        model_name += f'{self.series}-' if self.series != '' else ''
        model_name += f'{self.name}-' if self.name != '' else ''
        model_name += f'model{self.model_type}-' if self.model_type != '' else ''
        embedding_len = self.param('embedding_len')
        rnn_output_len = self.param('rnn_output_len')
        rnn_cell_type = self.param('rnn_cell_type')
        model_name += f'-emblen{embedding_len}-rnnlen{rnn_output_len}-cel{rnn_cell_type}'
        return model_name

    @tf.autograph.experimental.do_not_convert
    def _build_recurrent_model(self):
        param = self.param
        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,))

        # Embedding of characters.
        # Mask zero works.
        embedding = Embedding(input_dim=self.vocab_len, output_dim=param('embedding_len'),
                              mask_zero=True, name='Segment_embedding')(inputs)

        if param('embedding_dropout') > 0.0:
            embedding = Dropout(param('embedding_dropout'),
                    name='Dropout_embedding')(embedding)

        if param('rnn_cell_type') == 'LSTM':
            # Incorporate embeddings into hidden state and output state.
            rnn_output = LSTM(param('rnn_output_len'),
                    return_sequences=True, return_state=False,
                    recurrent_regularizer=l2(param('recurrent_l2')),
                    activity_regularizer=l2(param('rnn_activity_l2')),
                    recurrent_dropout=param('recurrent_dropout'),
                    name='LSTM_recurrent')(embedding)

        else:  # GRU
            rnn_output = GRU(param('rnn_output_len'),
                    return_sequences=True, return_state=False,
                    recurrent_regularizer=l2(param('recurrent_l2')),
                    activity_regularizer=l2(param('rnn_activity_l2')),
                    recurrent_dropout=param('recurrent_dropout'),
                    name='GRU_recurrent')(embedding)

        if param('rnn_output_dropout') > 0.0:
            rnn_output = Dropout(param('rnn_output_dropout'),
                    name='Dropout_rnn_output')(rnn_output)

        if param('merge_embedding_dropout') > 0.0:
            embedding = Dropout(param('merge_embedding_dropout'),
                    name='Dropout_merge_embedding')(embedding)

        # Add in latest embedding per Bengio 2002.
        to_outputs = Concatenate(axis=-1,
                    name='Merge_rnn_embedding')([rnn_output, embedding])

        # Hidden state used to predict subsequent character.
        outputs = Dense(self.vocab_len, activation='softmax',
                    name='Segment_output')(to_outputs)

        model_name = self._construct_modelname()
        self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if param('print_summary') > 0:
            self.print_model_summary()
        if param('plot_model') > 0:
            self.plot_model_summary()

    @tf.autograph.experimental.do_not_convert
    def _build_attention_model(self):
        # K.clear_session()
        param = self.param
        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,))

        embedding = Embedding(input_dim=self.vocab_len, output_dim=param('embedding_len'),
                              mask_zero=True, name='Segment_embedding')(inputs)

        if param('embedding_dropout') > 0.0:
            embedding = Dropout(param('embedding_dropout'),
                    name='Dropout_embedding')(embedding)

        if param('rnn_cell_type') == 'LSTM':
            # Incorporate embeddings into hidden state and output state.
            rnn_output, rnn_hidden, _ = LSTM(param('rnn_output_len'),
                    return_sequences=True, return_state=True,
                    recurrent_regularizer=l2(param('recurrent_l2')),
                    activity_regularizer=l2(param('rnn_activity_l2')),
                    recurrent_dropout=param('recurrent_dropout'),
                    name='LSTM_recurrent')(embedding)

        else:  # GRU
            rnn_output, rnn_hidden = GRU(param('rnn_output_len'),
                    return_sequences=True, return_state=True,
                    recurrent_regularizer=l2(param('recurrent_l2')),
                    activity_regularizer=l2(param('rnn_activity_l2')),
                    recurrent_dropout=param('recurrent_dropout'),
                    name='GRU_recurrent')(embedding)

        if param('rnn_output_dropout') > 0.0:
            rnn_output = Dropout(param('rnn_output_dropout'),
                    name='Dropout_rnn_output')(rnn_output)

        # hidden state output is for the entire word.  Not by character.
        rnn_hidden = Reshape((-1, rnn_hidden.shape[1]),
                    name='Reshape_hidden')(rnn_hidden)
        context_vector = AdditiveAttention()([rnn_output, rnn_hidden])

        to_outputs = Concatenate(axis=2,
                    name='Merge_context_rnn_output')([context_vector, rnn_output])

        outputs = Dense(self.vocab_len, activation="softmax",
                    name='Segment_output')(to_outputs)

        model_name = self._construct_modelname()
        self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if param('print_summary') > 0:
            self.print_model_summary()
        if param('plot_model') > 0:
            self.plot_model_summary()


    @tf.autograph.experimental.do_not_convert
    def train(self, train_gen=None, val_gen=None):
        """
        Train the neural network using training and validation data in generators.

        Parameters
        ----------
        train_gen : Keras generator
            Training generator. The default is None.
        val_gen : Keras generator
            Validation generator. The default is None.

        Returns
        -------
        Tensorflow history.history.

        """
        # Will invoke this after consruction of the model.
        # Too heavy weight to do in init of class.
        param = self.param

        train_steps = train_gen.data_len//train_gen.batch_size
        lr_decay = (1.0/param('lr_decay')-1.0)/train_steps
        optimizer = Adam(learning_rate=param('learning_rate'), decay=lr_decay)

        # Use sparse encoding.  Tensorflow converts as needed.
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                metrics=['sparse_categorical_accuracy'])
        # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
        #               metrics=['accuracy', 'categorical_crossentropy'])

        # Filename without .h5 suffix, provides better handling of state.
        # Had trouble loading withoutwhen not using .h5 suffix, so still use it.
        # *** No saving of neural model, since we don't have way to load it still.
        # best_file_name = model_path_best+self.model_name+'-best.h5'
        # checkpointer = ModelCheckpoint(filepath=best_file_name,
        #                                verbose=1, save_best_only=True)

        earlystopper = EarlyStopping(monitor='val_loss',
                        verbose=param('tf_verbose'),
                        patience=5, restore_best_weights=True)

        val_steps=max(val_gen.data_len//val_gen.batch_size, 2)
        # Fit using generator - use fit directly.
        history = self.model.fit(train_gen.generate(),
                        steps_per_epoch=train_steps, epochs=param('epochs'),
                        validation_data=val_gen.generate(),
                        validation_steps=val_steps,
                        verbose=param('tf_verbose'),
                        callbacks=[earlystopper])

        # callbacks=[checkpointer, earlystopper])
        # ** No saving of neural models until we have way to load them.
        # final_file_name = model_path_final+self.model_name+'-final.h5'
        # self.model.save(final_file_name)

        if param('neural_verbose') > 0:
            self._show_quality_measures(history.history)

        # Save history.
        # ** No saving of history log file until we have function to load them.
        # history_file_name = log_path+self.model_name+'-history.pickle'
        # with open(history_file_name, 'wb') as history_pi:
        #     pickle.dump(history.history, history_pi)
        # Load history with:
        # open(log_path+model_prefix+'-history.pickle', "rb") as history_pi:
        #    history_dict = pickle.load(history_pi)
        # TODO: Optionally graph the quality measures from history.
        # This would serve in determing model parameters.
        # But should only be peperformed when requested.
        return history.history

    def _show_quality_measures(self, history):
        #measure_keys = history.keys()
        # val_loss is used to get the best fit with early stopping.
        val_loss = history['val_loss']
        best, best_val_loss = min(enumerate(val_loss), key=lambda v: v[1])
        print(f'Best epoch: {best} of {len(val_loss)}. Statistics from TensorFlow:')
        print(f"Train dataset: loss={history['loss'][best]:.4f}, " +
              f"accuracy={history['sparse_categorical_accuracy'][best]:.4f}")
              #f"crossentropy={history['sparse_categorical_crossentropy'][best]:.4f}")
        print(f"Validate dataset: loss={history['val_loss'][best]:.4f}, " +
              f"accuracy={history['val_sparse_categorical_accuracy'][best]:.4f}")


    @tf.autograph.experimental.do_not_convert
    def evaluate_test(self, test_gen=None):
        param = self.param
        # Evaluate using generator - use evaluate directly.

        test_steps=max(test_gen.data_len//test_gen.batch_size, 2)
        score = self.model.evaluate(test_gen.generate(),
                    steps=test_steps, verbose=param('tf_verbose'))

        print(f'Test dataset: loss={score[0]:.4f}, '+
              f'accuracy={score[1]:.4f}')
              #f'crossentropy={score[2]:.4f}')
        return score

    def print_model_summary(self):
        # Prints model summary to log path.
        print_fn = output_path / (self.model.name+'.txt')
        with open(print_fn.as_posix(),'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def plot_model_summary(self):
        print_fn = output_path / (self.model.name+'.png')
        plot_model(self.model, print_fn.as_posix(), show_shapes=True,
                   show_layer_names=True, dpi=self.param('plot_dpi'))




