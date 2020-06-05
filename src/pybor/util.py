"""
Utility functions for the package.
"""
import math
import random
from pathlib import Path

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import pybor.config as cfg

output_path = Path(cfg.BaseSettings().output_path).resolve()

def test_data(*comps):
    return Path(__file__).parent.parent.parent.joinpath( 'tests', 'data',
                                                        *comps).as_posix()


def find_cut_point(native, loan):
    """
    Find cut_point that will give highest accuracy.

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
        Native if entropy < cut_point else loan.

    """
    total = sorted([-s for s in native] + loan, key=abs)

    # All values are > 0 are correct.
    cut_point = 0
    correct = len(loan)
    correct_max = correct

    for t in total:
        correct += (1 if t < 0 else -1)
        if correct > correct_max:
            correct_max = correct
            cut_point = t

    cut_point = abs(cut_point)
    return cut_point



def find_ref_limit(entropies=None, fraction=0.995):
    """
    Calculate a cut-off point for entropies from the data.
    """
    # Entropies are not power law distributed, but neither are they Gaussian.
    # Better to use fraction of distribution rather than Gaussian z value
    # as cut-point for discriminating between native and loan.
    entropies = sorted(entropies)
    idx = min((len(entropies)-1)*fraction, len(entropies)-1)
    return (entropies[math.floor(idx)]+entropies[math.ceil(idx)])/2


def train_test_split(table, split=None):
    random.shuffle(table)
    split = int(split) if split >= 1 else math.ceil(len(table)*split)
    return table[:-split], table[-split:]


# =============================================================================
# Logger standard routines
# Adapted from: https://gist.github.com/nguyenkims/e92df0f8bd49973f0c94bddf36ed7fd0
# =============================================================================

FILE_FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
CONSOLE_FORMATTER = logging.Formatter("%(name)s — %(levelname)s — %(message)s")
LOG_FILE = output_path / "pybor.log"

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CONSOLE_FORMATTER)
    console_handler.setLevel(logging.INFO)
    return console_handler

def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE.as_posix(), when='midnight')
    file_handler.setFormatter(FILE_FORMATTER)
    file_handler.setLevel(logging.DEBUG)
    return file_handler

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG) # better to have too much log than not enough
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False

    return logger

