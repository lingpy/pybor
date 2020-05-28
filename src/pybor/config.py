"""
Settings for different models.
"""
import attr

@attr.s
class BaseSettings:
    batch_size = attr.ib(default=32)
    token_maxlen = attr.ib(default=40)
    val_split = attr.ib(default=0.15)
    test_split = attr.ib(default=0.15)
    basis = attr.ib(default='all')
    verbose = attr.ib(default=1)

    embedding_len = attr.ib(default=16)
    rnn_output_len = attr.ib(default=32)
    rnn_cell_type = attr.ib(default='GRU')  # GRU, LSTM

    print_summary = attr.ib(default=False, metadata={"deprecated": True})
    plot_model = attr.ib(default=False, metadata={"deprecated": True})
    plot_dpi = attr.ib(default=400, metadata={"deprecated": True})

    # Model dropout and regulation parameters
    embedding_dropout = attr.ib(default=0.0)
    recurrent_l2 = attr.ib(default=0.001)
    rnn_activity_l2 = attr.ib(default=0.0)
    recurrent_dropout = attr.ib(default=0.0)
    rnn_output_dropout = attr.ib(default=0.2)
    merge_embedding_dropout = attr.ib(default=0.2)

    # Model fitting parameters
    epochs = attr.ib(30)
    learning_rate = attr.ib(0.003333)
    lr_decay = attr.ib(0.90)
    neural_verbose = attr.ib(1)
    tf_verbose = attr.ib(0)


@attr.s
class NeuralSettings:
    language = attr.ib(default='')
    series = attr.ib(default='')
    detect_type = attr.ib(default='dual')
    model_type = attr.ib(default='recurrent')
    fraction = attr.ib(default=0.995)
    output_path = './output'


@attr.s
class RecurrentSettings(BaseSettings):
    merge_embedding_dropout = attr.ib(default=0.2)


@attr.s
class AttentionSettings(BaseSettings):
    embedding_len = attr.ib(default=32)
    rnn_cell_type = attr.ib(default='LSTM')  # GRU, LSTM

    recurrent_l2 = attr.ib(default=0.0)
    rnn_activity_l2 = attr.ib(default=0.001)
    recurrent_dropout = attr.ib(default=0.2)

    # Model fitting parameters
    epochs = attr.ib(50)
    learning_rate = attr.ib(0.01)
    lr_decay = attr.ib(0.95)


