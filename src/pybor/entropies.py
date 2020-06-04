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

import attr

import tensorflow as tf
#tf.autograph.set_verbosity(0, False)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import GRU, LSTM, AdditiveAttention, Attention
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping  # ModelCheckpoint,
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras import backend as K

import pybor.util as util
import pybor.config as cfg

output_path = Path(cfg.BaseSettings().output_path).resolve()
logger = util.get_logger(__name__)

@attr.s
class NeuralWord:
    """
    Use lists of token id lists to calculate token entropies.
    Neural net model is configured and then trained from a list of tokens at
    time of construction. Entropy or entroy lists are calculated from the
    trained neural net model.
    """

    vocab_len = attr.ib()
    language = attr.ib(default='')
    basis = attr.ib(default='all')
    series = attr.ib(default='')
    model = attr.ib(init=False)

    def __attrs_post_init__(self):
        if self.vocab_len is None or self.vocab_len <= 0:
            raise ValueError('Require a vocabulary size > 0 to construct neural entropy model.')

        """
        Neural net based model to calculate token entropy.
        Configure and complile the model and fit training data at construction.
        Calculate entropies on demand based on the fitted model.

        Parameters
        ----------
        vocab_len : int
            Length of the vocabulary. Note: All token segments are integer encoded.
        language : str
            Language being modeled. Used in model naming for storage.
        basis : str ['all', 'native', 'loan', ...]
            Whether all tokens, just native, or just loan.
        series : str
            Study series to qualify model name.

        Returns
        -------
        NeuralWord object reference.

        Notes
        -----
        Based on recent research, dropout of 0.1 should be considered when model includes
        multiple levels of dropout.
        """


    def calculate_entropy(self, token_ids):
        return self.calculate_entropies([token_ids])[0]


    def calculate_entropies(self, tokens_ids):
        # Calculate entropy for a list of tokens_ids
        # in format of int ids that correspond to str segments.

        # Get the probabilities for all str segment possibilities.
        maxlen = max([len(token_ids) for token_ids in tokens_ids])
        # Truncate right id for x and left id for y, so only 1 id extra.
        if maxlen > self.settings.token_maxlen+1:
            maxlen = self.settings.token_maxlen+1

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


    def train(self, train_gen=None, val_gen=None, epochs=None):
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
        if isinstance(self, NeuralWordRecurrent):
            logger.info('Training recurrent neural model.')
        elif isinstance(self, NeuralWordAttention):
            logger.info('Training attention neural model.')
        else:
            logger.warn("I don't know what of model I am training.")
        # Invoke this after consruction of the model.
        # Too heavy weight to do in init of class.

        train_steps = train_gen.data_len//train_gen.batch_size
        lr_decay = (1.0/self.settings.lr_decay-1.0)/train_steps
        optimizer = Adam(learning_rate=self.settings.learning_rate, decay=lr_decay)


        self.model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['sparse_categorical_accuracy'])

        epochs = epochs or self.settings.epochs

        earlystopper = EarlyStopping(monitor='val_loss',
                                     verbose=self.settings.tf_verbose,
                                     patience=epochs,
                                     restore_best_weights=self.settings.restore_best_weights)

        val_steps=max(val_gen.data_len//val_gen.batch_size, 2)
        # Fit using generator - use fit directly.
        history = self.model.fit(train_gen.generate(),
                                 steps_per_epoch=train_steps,
                                 epochs=epochs,
                                 validation_data=val_gen.generate(),
                                 validation_steps=val_steps,
                                 verbose=self.settings.tf_verbose,
                                 callbacks=[earlystopper])


        if self.settings.verbose > 0:
            self.show_quality_measures(history.history)


        return history.history

    def show_quality_measures(self, history):
        #measure_keys = history.keys()
        # val_loss is used to get the best fit with early stopping.
        val_loss = history['val_loss']
        best, best_val_loss = min(enumerate(val_loss), key=lambda v: v[1])
        logger.info(f'Best epoch: {best} of {len(val_loss)}. Statistics from TensorFlow:')
        logger.info(f"Train dataset: loss={history['loss'][best]:.4f}, " +
              f"accuracy={history['sparse_categorical_accuracy'][best]:.4f}")
        logger.info(f"Validate dataset: loss={history['val_loss'][best]:.4f}, " +
              f"accuracy={history['val_sparse_categorical_accuracy'][best]:.4f}")


    def evaluate_test(self, test_gen=None):
        # Evaluate using generator - use evaluate directly.

        test_steps=max(test_gen.data_len//test_gen.batch_size, 2)
        score = self.model.evaluate(test_gen.generate(),
                                    steps=test_steps,
                                    verbose=self.settings.tf_verbose)

        logger.info(f'Test dataset: loss={score[0]:.4f}, accuracy={score[1]:.4f}, ')
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
                   show_layer_names=True, dpi=self.settings.plot_dpi)

    def construct_modelprefix(self):
        language_out = ''.join(self.language.split())
        model_prefix = f'{language_out}-' if language_out != '' else ''
        model_prefix += f'{self.basis}-' if self.basis != '' else ''
        model_prefix += f'{self.series}-' if self.series != '' else ''
        return model_prefix

    def construct_modelsuffix(self):
        embedding_len = self.settings.embedding_len
        rnn_output_len = self.settings.rnn_output_len
        rnn_cell_type = self.settings.rnn_cell_type
        model_suffix = f'-emblen{embedding_len}-rnnlen{rnn_output_len}-celtyp{rnn_cell_type}'
        return model_suffix

    def construct_modelname(self, model_type):
        return self.construct_modelprefix()+model_type+self.construct_modelsuffix()


@attr.s
class NeuralWordRecurrent(NeuralWord):

    settings = attr.ib(default=cfg.RecurrentSettings())

    # Test whether this needs go here or if OK given en parent class.
    def __attrs_post_init__(self):
        if not isinstance(self.settings, cfg.RecurrentSettings):
            self.settings = cfg.RecurrentSettings()
        super().__attrs_post_init__()

        self.build_model()

    def build_model(self):
        params = self.settings  # Convenience

        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,))

        # Embedding of characters.
        # Mask zero works.
        embedding = Embedding(input_dim=self.vocab_len,
                              output_dim=params.embedding_len,
                              mask_zero=True,
                              name='Segment_embedding')(inputs)

        if params.embedding_dropout > 0.0:
            embedding = Dropout(params.embedding_dropout,
                                name='Dropout_embedding')(embedding)

        if params.rnn_levels == 1:
            if params.rnn_cell_type == 'LSTM':
                # Incorporate embeddings into hidden state and output state.
                rnn_output = LSTM(params.rnn_output_len,
                                  return_sequences=True, return_state=False,
                                  recurrent_regularizer=l2(params.recurrent_l2),
                                  activity_regularizer=l2(params.rnn_activity_l2),
                                  recurrent_dropout=params.recurrent_dropout,
                                  name='LSTM_recurrent')(embedding)

            else:  # GRU
                rnn_output = GRU(params.rnn_output_len,
                                 return_sequences=True, return_state=False,
                                 recurrent_regularizer=l2(params.recurrent_l2),
                                 activity_regularizer=l2(params.rnn_activity_l2),
                                 recurrent_dropout=params.recurrent_dropout,
                                 name='GRU_recurrent')(embedding)
        else:  # 2 levels
            if params.rnn_cell_type == 'LSTM':
                # Incorporate embeddings into hidden state and output state.
                rnn_output = LSTM(params.rnn_output_len,
                                  return_sequences=True, return_state=False,
                                  recurrent_regularizer=l2(params.recurrent_l2),
                                  activity_regularizer=l2(params.rnn_activity_l2),
                                  recurrent_dropout=params.recurrent_dropout,
                                  name='LSTM_recurrent_1')(embedding)
                rnn_output = LSTM(params.rnn_output_len,
                                  return_sequences=True, return_state=False,
                                  recurrent_regularizer=l2(params.recurrent_l2),
                                  activity_regularizer=l2(params.rnn_activity_l2),
                                  recurrent_dropout=params.recurrent_dropout,
                                  name='LSTM_recurrent_2')(rnn_output)

            else:  # GRU
                rnn_output = GRU(params.rnn_output_len,
                                 return_sequences=True, return_state=False,
                                 recurrent_regularizer=l2(params.recurrent_l2),
                                 activity_regularizer=l2(params.rnn_activity_l2),
                                 recurrent_dropout=params.recurrent_dropout,
                                 name='GRU_recurrent_1')(embedding)
                rnn_output = GRU(params.rnn_output_len,
                                 return_sequences=True, return_state=False,
                                 recurrent_regularizer=l2(params.recurrent_l2),
                                 activity_regularizer=l2(params.rnn_activity_l2),
                                 recurrent_dropout=params.recurrent_dropout,
                                 name='GRU_recurrent_2')(rnn_output)


        if params.rnn_output_dropout > 0.0:
            rnn_output = Dropout(params.rnn_output_dropout,
                                 name='Dropout_rnn_output')(rnn_output)

        if params.merge_embedding_dropout > 0.0:
            embedding = Dropout(params.merge_embedding_dropout,
                                name='Dropout_merge_embedding')(embedding)

        # Add in latest embedding per Bengio 2002.
        to_outputs = Concatenate(axis=-1, name='Merge_rnn_embedding')([rnn_output, embedding])

        # Hidden state used to predict subsequent character.
        outputs = Dense(self.vocab_len,
                        activation='softmax',
                        name='Segment_output')(to_outputs)

        model_name = self.construct_modelname('recurrent')
        self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if params.print_summary > 0:
            self.print_model_summary()
        if params.plot_model > 0:
            self.plot_model_summary()


@attr.s
class NeuralWordAttention(NeuralWord):

    settings = attr.ib(default=cfg.AttentionSettings())

    # Test whether this needs go here or if OK given en parent class.
    def __attrs_post_init__(self):
        if not isinstance(self.settings, cfg.AttentionSettings):
            self.settings = cfg.AttentionSettings()
        super().__attrs_post_init__()

        self.build_model()

    def build_model(self):
        params = self.settings  # Convenience

        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,))

        embedding = Embedding(input_dim=self.vocab_len,
                              output_dim=params.embedding_len,
                              mask_zero=True,
                              name='Segment_embedding')(inputs)

        if params.embedding_dropout > 0.0:
            embedding = Dropout(params.embedding_dropout,
                                name='Dropout_embedding')(embedding)

        if params.rnn_levels == 1:
            if params.rnn_cell_type == 'LSTM':
                # Incorporate embeddings into hidden state and output state.
                rnn_output, rnn_hidden, _ = LSTM(params.rnn_output_len,
                                                 return_sequences=True, return_state=True,
                                                 recurrent_regularizer=l2(params.recurrent_l2),
                                                 activity_regularizer=l2(params.rnn_activity_l2),
                                                 recurrent_dropout=params.recurrent_dropout,
                                                 name='LSTM_recurrent')(embedding)

            else:  # GRU
                rnn_output, rnn_hidden = GRU(params.rnn_output_len,
                                             return_sequences=True, return_state=True,
                                             recurrent_regularizer=l2(params.recurrent_l2),
                                             activity_regularizer=l2(params.rnn_activity_l2),
                                             recurrent_dropout=params.recurrent_dropout,
                                             name='GRU_recurrent')(embedding)
        else:  # 2 levels
            if params.rnn_cell_type == 'LSTM':
                # Incorporate embeddings into hidden state and output state.
                rnn_output, rnn_hidden, _ = LSTM(params.rnn_output_len,
                                                 return_sequences=True, return_state=True,
                                                 recurrent_regularizer=l2(params.recurrent_l2),
                                                 activity_regularizer=l2(params.rnn_activity_l2),
                                                 recurrent_dropout=params.recurrent_dropout,
                                                 name='LSTM_recurrent_1')(embedding)
                rnn_output, rnn_hidden, _ = LSTM(params.rnn_output_len,
                                                 return_sequences=True, return_state=True,
                                                 recurrent_regularizer=l2(params.recurrent_l2),
                                                 activity_regularizer=l2(params.rnn_activity_l2),
                                                 recurrent_dropout=params.recurrent_dropout,
                                                 name='LSTM_recurrent_2')(rnn_output)

            else:  # GRU
                rnn_output, rnn_hidden = GRU(params.rnn_output_len,
                                             return_sequences=True, return_state=True,
                                             recurrent_regularizer=l2(params.recurrent_l2),
                                             activity_regularizer=l2(params.rnn_activity_l2),
                                             recurrent_dropout=params.recurrent_dropout,
                                             name='GRU_recurrent_1')(embedding)
                rnn_output, rnn_hidden = GRU(params.rnn_output_len,
                                             return_sequences=True, return_state=True,
                                             recurrent_regularizer=l2(params.recurrent_l2),
                                             activity_regularizer=l2(params.rnn_activity_l2),
                                             recurrent_dropout=params.recurrent_dropout,
                                             name='GRU_recurrent_2')(rnn_output)


        if params.rnn_output_dropout > 0.0:
            rnn_output = Dropout(params.rnn_output_dropout,
                                 name='Dropout_rnn_output')(rnn_output)

        # hidden state output is for the entire word.  Not by character.
        rnn_hidden = Reshape((-1, rnn_hidden.shape[1]),
                             name='Reshape_hidden')(rnn_hidden)

        if params.attn_type == 'additive':
            context_vector = AdditiveAttention(causal=params.attn_causal,
                    dropout=params.attn_dropout)([rnn_output, rnn_hidden])
        else:  # dot-product
            context_vector = Attention(causal=params.attn_causal,
                    dropout=params.attn_dropout) ([rnn_output, rnn_hidden])

        to_outputs = Concatenate(axis=2,
                                 name='Merge_context_rnn_output')([context_vector, rnn_output])

        outputs = Dense(self.vocab_len, activation="softmax",
                        name='Segment_output')(to_outputs)

        model_name = self.construct_modelname('attention')
        self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if params.print_summary > 0:
            self.print_model_summary()
        if params.plot_model > 0:
            self.plot_model_summary()

