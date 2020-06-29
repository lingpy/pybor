"""
Settings for different models.
"""
import attr

@attr.s
class BaseSettings:
    val_split = attr.ib(default=0.0)
    test_split = attr.ib(default=0.15)
    detect_type = attr.ib(default='dual')

    verbose = attr.ib(default=1)
    print_summary = attr.ib(default=False, metadata={"deprecated": True})
    plot_model = attr.ib(default=False, metadata={"deprecated": True})
    plot_dpi = attr.ib(default=400, metadata={"deprecated": True})

    output_path = attr.ib(default='./output')

@attr.s
class NeuralSettings(BaseSettings):
    language = attr.ib(default='')
    series = attr.ib(default='')

    batch_size = attr.ib(default=32)
    skip_step = attr.ib(default=5)
    token_maxlen = attr.ib(default=30)
    model_type = attr.ib(default='recurrent')  # recurrent, attention
    fraction = attr.ib(default=0.995)  # For Native model.
    prediction_policy = attr.ib(default='zero')  # zero, accuracy, fscore
    fscore_beta = attr.ib(default=1.0)


@attr.s
class EntropiesSettings(NeuralSettings):
    # While not strictly a child of NeuralSettings, it seems more convenient.
    tf_verbose = attr.ib(default=0)
    basis = attr.ib(default='all')


@attr.s
class RecurrentSettings(EntropiesSettings):
    # Architecture parameters
    embedding_len = attr.ib(default=32)
    rnn_output_len = attr.ib(default=32)
    rnn_cell_type = attr.ib(default='GRU')  # GRU, LSTM
    rnn_levels = attr.ib(default=1)  # 1, 2

    # Dropout and regulation parameters
    embedding_dropout = attr.ib(default=0.0)
    recurrent_l2 = attr.ib(default=0.001)
    rnn_activity_l2 = attr.ib(default=0.0)
    recurrent_dropout = attr.ib(default=0.0)
    rnn_output_dropout = attr.ib(default=0.2)
    merge_embedding_dropout = attr.ib(default=0.2)

    # Model fitting parameters
    epochs = attr.ib(default=45)
    learning_rate = attr.ib(default=0.006)
    learning_rate_decay = attr.ib(default=0.95)  # Adjust for batch size, data len.
    restore_best_weights = attr.ib(default=True)

@attr.s
class AttentionSettings(EntropiesSettings):
    # Architecture parameters
    embedding_len = attr.ib(default=32)
    rnn_output_len = attr.ib(default=32)
    rnn_cell_type = attr.ib(default='LSTM')  # GRU, LSTM
    rnn_levels = attr.ib(default=1)  # 1, 2
    attention_type = attr.ib(default='additive')  # additive, dot-product
    attention_causal = attr.ib(default='False')

    # Dropout and regulation parameters
    embedding_dropout = attr.ib(default=0.0)
    recurrent_l2 = attr.ib(default=0.0)
    rnn_activity_l2 = attr.ib(default=0.001)
    recurrent_dropout = attr.ib(default=0.2)
    rnn_output_dropout = attr.ib(default=0.2)
    attention_dropout = attr.ib(default=0.0)

    # Model fitting parameters
    epochs = attr.ib(default=45)
    learning_rate = attr.ib(default=0.006)
    learning_rate_decay = attr.ib(default=0.95)  # 0.95
    restore_best_weights = attr.ib(default=True)


@attr.s
class MarkovSettings(BaseSettings):
    model = attr.ib(default='kni')
    order = attr.ib(default=3)
    p = attr.ib(default=0.995)
    smoothing = attr.ib(default=0.3)
    