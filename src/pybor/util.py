"""
Utility functions for the package.
"""

# Import Python standard libraries
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import logging
import math
import random
import sys

# Build namespace
import pybor.config as cfg

output_path = Path(cfg.BaseSettings().output_path).resolve()


def test_data(*comps):
    return (
        Path(__file__).parent.parent.parent.joinpath("tests", "data", *comps).as_posix()
    )


def find_acc_cut_point(native, loan):
    """
    Calculate cut_point that will give highest accuracy for entropies.
    Use case is for entropies estimated from a single (native) entropy model.

    Parameters
    ----------
    native : [float]
        List of entropies from native distribution.
    loan : [float]
        List of entropies from loan distribution.

    Returns
    -------
    cut_point : float
        Critical value to test native versus loan word.
        Native if entropy <= cut_point else loan.

    Note
    ----
        Entropies are all non-negative and the algorithm used makes use
        of that fact.

    """
    abs_entropies = sorted([-s for s in native] + loan, key=abs)

    # All values > 0 are presumed correct at start.
    cut_point = 0
    correct = len(loan)
    correct_max = correct

    for value in abs_entropies:
        correct += 1 if value < 0 else -1
        if correct > correct_max:
            correct_max = correct
            cut_point = value

    cut_point = abs(cut_point)
    return cut_point


def find_acc_cut_point_deltas(native, loan):
    """
    Calculate cut_point that will give highest accuracy for entropy deltas.
    Use case is for differences between entropies calculated under competing models,
    where the native model typically estimates lower entropies for native words,
    and the loan model typically estimates lower entropies for loan words.

    Parameters
    ----------
    native : [float]
        Delta entropies for native words calculated under native and loan models.
        native = delta = entropy_{native_word|native_model} - entropy_{native_word|loan_model}
    loan : [float]
        Delta entropies for loan words calculated under native and loan models.
        loan = delta = entropy_{loan_word|native_model} - entropy_{loan_word|loan_model}

    Returns
    -------
    cut_point : float
        Critical value to test native versus loan word model deltas.
        Native if delta <= cut_pont else loan.
    """
    native = [(delta, 0) for delta in native]
    loan = [(delta, 1) for delta in loan]
    data = sorted(native + loan, key=lambda row: row[0])
    # All loan deltas > -inf so correct. No native < -inf so incorrect.
    cut_point = float("-inf")
    correct = len(loan)
    correct_max = correct

    for value, source in data:
        # Count native correct if cut_point were set here.
        # Count loan wrong if cut_point were set here.
        correct += 1 if source == 0 else -1
        if correct > correct_max:
            correct_max = correct
            cut_point = value

    return cut_point


def calculate_fscore(n_true_pos, n_true, n_pos, beta=1):
    if n_true_pos == 0:
        return 0.0
    if n_true == 0:
        return 0.0
    if n_pos == 0:
        return 0.0
    prec = n_true_pos / n_pos
    recall = n_true_pos / n_true
    return (1 + beta ** 2) * (prec * recall) / (beta ** 2 * prec + recall)


def find_fscore_cut_point_deltas(native, loan, beta=1.0):
    """
    Calculate cut_point that will give highest F score for entropy deltas.

    Parameters
    ----------
    native : [float]
        Delta entropies for native words calculated under native and loan models.
        native = delta = entropy_{native_word|native_model} - entropy_{native_word|loan_model}
    loan : [float]
        Delta entropies for loan words calculated under native and loan models.
        loan = delta = entropy_{loan_word|native_model} - entropy_{loan_word|loan_model}
    beta : float, optional.
        beta used to customize F score.  Default is 1.0.
    Returns
    -------
    cut_point : float
        Critical value to test native versus loan word model deltas.
        Native if delta <= cut_pont else loan.

    Notes
    -----
    native = [-1.5, -1.0, -0.75, -0.5, 0.0, 1.0]
    loan = [-0.65, 0.5, 1.5, 2.0]

    """
    native = [(delta, 0) for delta in native]
    loan = [(delta, 1) for delta in loan]
    data = sorted(native + loan, key=lambda row: row[0])
    # All loan deltas > -inf so true and pos.
    # All loan true
    # All delta > -inf so pos
    cut_point = float("-inf")
    n_true_pos = len(loan)
    n_true = len(loan)
    n_pos = len(data)
    f = calculate_fscore(n_true_pos, n_true, n_pos, beta)
    # print(f'n_true_pos={n_true_pos}, n_pos={n_pos}, f={f:.2f},
    #   val=-inf, f_max={f:.2f}, cut_point=-inf.')
    f_max = f

    for value, source in data:
        # Count true_pos and pos if cut_point were set here.
        n_pos -= 1  # reduce n_pos for each value below this point.
        n_true_pos -= 1 if source == 1 else 0
        f = calculate_fscore(n_true_pos, n_true, n_pos, beta)
        if f > f_max:
            f_max = f
            cut_point = value
        # print(f'n_true_pos={n_true_pos}, n_pos={n_pos}, f={f:.2f}, val={value},
        #   f_max={f_max:.2f}, cut_point={cut_point}.')

    return cut_point


def find_frac_cut_point(entropies, fraction=0.995):
    """
    Calculate a cut-off point from fraction of entropy distribution.
    Use case is with native entropy distribution to qualify on true negatives,
    under the assumption that loan entropies will be generally greater than native,
    and so result in low false negatives and high true positives.
    """
    # Entropies are not power law distributed, but neither are they Gaussian.
    # Better to use fraction of distribution rather than Gaussian z value
    # as cut-point for discriminating between native and loan.
    entropies = sorted(entropies)
    idx = min((len(entropies) - 1) * fraction, len(entropies) - 1)
    return (entropies[math.floor(idx)] + entropies[math.ceil(idx)]) / 2


def train_test_split(table, split=0):
    random.shuffle(table)
    if split == 0:
        return table, []
    split = int(split) if split >= 1 else math.ceil(len(table) * split)
    return table[:-split], table[-split:]


# =============================================================================
# Logger standard routines
# Adapted from: https://gist.github.com/nguyenkims/e92df0f8bd49973f0c94bddf36ed7fd0
# =============================================================================

FILE_FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
CONSOLE_FORMATTER = logging.Formatter("%(name)s — %(levelname)s — %(message)s")
LOG_FILE = output_path / "pybor.log"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CONSOLE_FORMATTER)
    console_handler.setLevel(logging.INFO)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE.as_posix(), when="midnight")
    file_handler.setFormatter(FILE_FORMATTER)
    file_handler.setLevel(logging.DEBUG)
    return file_handler


def get_logger(logger_name):
    _logger = logging.getLogger(logger_name)
    if not _logger.handlers:
        _logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
        _logger.addHandler(get_console_handler())
        #    _logger.addHandler(get_file_handler())
        _logger.propagate = False

    return _logger


logger = get_logger(__name__)

# =============================================================================
# Generator for use in k-fold cross-validation
# =============================================================================


def k_fold_samples(table, k):
    size = len(table)
    fraction = 1 / k
    table = random.sample(table, size)

    i = 0
    while i < k:
        test_begin = math.ceil(i * size * fraction)
        test_end = math.ceil((i + 1) * size * fraction)
        test = table[test_begin:test_end]
        train = table[:test_begin] + table[test_end:]
        yield i, train, test
        i += 1


# =============================================================================
# Generator for use in holdout-n cross-validation
# =============================================================================
def holdout_n_samples(table, n, max_iter):
    size = len(table)
    table = random.sample(table, size)
    abs_max_iter = int(math.ceil(size / n))
    max_iter = abs_max_iter if max_iter == -1 or max_iter > abs_max_iter else max_iter
    if max_iter == abs_max_iter:
        logger.debug("Using max number iterations %i.", max_iter)
    i = 0
    while i < max_iter:
        test_begin = math.ceil(i * n)
        test_end = math.ceil((i + 1) * n)
        test = table[test_begin:test_end]
        train = table[:test_begin] + table[test_end:]
        yield i, train, test
        i += 1
