import math
import statistics
import numpy as np
#from collections import namedtuple

from nltk.util import ngrams
import nltk.lm as lm
import nltk.lm.preprocessing as pp
from sklearn.model_selection import train_test_split
#import sklearn.metrics as metrics

#from tabulate import tabulate

import pybor.evaluate as evaluate

# =============================================================================
#
# Markov model for operationg on tokens.
# Used by all detection and analysis modules.
# Will update to not need splitter, when format of input tokens better defined.
#
# =============================================================================

# Temporay fix for dataset input. Puts in standard form and removes blanks.
def _splitter(item):
    if isinstance(item, (list, tuple)):
        return [ch for ch in item if ch != ' ']
    elif " " in item:
        return [ch for ch in item.split(" ") if ch != ' ']
    else:
        return [ch for ch in item]

class MarkovWord:
    def __init__(self, tokens, model="kni", order=2, smoothing=0.5):
        # smoothing is either discount for KNI or gamma for Lidstone.
        # default discount in KNI is 0.1 in NLTK while default gamma in Lidstone is 0.2
        self._tokens = tokens
        self._order = order

        # If tokens is a string with spaces, we assume they are separators
        # and split accordingly; otherwise, we take each character of the
        # string as a token. If it is a list or tuple, we don't touch it.
        words = [_splitter(token) for token in self._tokens]

        # NLTK recommended padding explicitly for training.
        train, vocab = pp.padded_everygram_pipeline(self._order, words)

        # Define the vocabulary.  Allow for out of vocabulary, thus cutoff=2.
        vocab = lm.Vocabulary(vocab, unk_cutoff=2)

        # Define the language model.
        self._lm = None
        if model == "kni":
            self._lm = lm.KneserNeyInterpolated(
                order, vocabulary=vocab, discount=smoothing
            )  # Default nltk discount iis 0.1; we use 0.5
        elif model == "wbi":
            self._lm = lm.WittenBellInterpolated(order, vocabulary=vocab)
        elif model == "lp":  # Laplace
            self._lm = lm.Laplace(order=order, vocabulary=vocab)
        elif model == "ls":  # Lidstone
            self._lm = lm.Lidstone(
                order=order, gamma=smoothing, vocabulary=vocab)
        else:  # MLE
            self._lm = lm.MLE(order, vocabulary=vocab)

        self._lm.fit(train)  #


    def calculate_entropies(self, tokens):
        words = [_splitter(token) for token in tokens]
        ### verified by experiment - use order for left and order 2 for right.
        padded_words = [
            list(
                pp.pad_both_ends(
                    pp.pad_both_ends(word, pad_right=False, n=self._order),
                    pad_left=False, n=2)
            )
            for word in words
        ]

        # Convert lists of padded words to lists of ngrams.
        ngrams_lst = [list(ngrams(seq, self._order)) for seq in padded_words]
        entropies = [self._lm.entropy(segs) for segs in ngrams_lst]
        return entropies


    # Function to calculate per sound average entropy of single token.
    def calculate_entropy(self, token):
        return self.calculate_entropies([token])[0]


#### ************************************************
####
#### Functions for discrimination on individual words.
####
#### ************************************************

# =============================================================================
#
# Dual model approach - native and loan models compete for prediction.
#
# =============================================================================

class DualMarkov:
    def __init__(self, tokens, ground, method='kni', order=3, smoothing=0.5):

        nativetokens = [token for token, select in zip(tokens, ground) if select]
        self.nativemodel = MarkovWord(
            nativetokens, model=method, order=order, smoothing=smoothing)

        loantokens = [token for token, select in zip(tokens, ground) if not select]
        self.loanmodel = MarkovWord(
            loantokens, model=method, order=order, smoothing=smoothing)

    def predict_data(self, tokens):
        nativeentropies = self.nativemodel.calculate_entropies(tokens)
        loanentropies = self.loanmodel.calculate_entropies(tokens)
        predictions = np.less(nativeentropies, loanentropies)
        return predictions.astype(int)

    def predict(self, token):
        return self.predict_data([token])[0]


# =============================================================================
#
#  For compatibility with prevous analysis module.
#  But also because we still need to provide this functionality
#  for the purpose of demonstrating overall effectiveness.
#
# =============================================================================
def detect_native_loan_dual_basis(tokens=None, borrowedscore=None, method='kni',
                                  smoothing=0.5, order=3, trainfrac=0.8):

    ground = np.array(borrowedscore) < 0.5  # native versus loan
    train_idx, val_idx = train_test_split(range(len(tokens)), test_size=1-trainfrac)
    train_tokens = np.array(tokens)[train_idx]
    train_ground = np.array(ground)[train_idx]

    dual_model = DualMarkov(train_tokens, train_ground, method=method,
                            order=order, smoothing=smoothing)

    print("\nEvaluate train dataset.")
    train_metrics = evaluate.evaluate_model(dual_model, train_tokens, train_ground)
    evaluate.print_evaluation(train_metrics)

    print("\nEvaluate test dataset.")
    val_tokens = np.array(tokens)[val_idx]
    val_ground = np.array(ground)[val_idx]
    val_metrics = evaluate.evaluate_model(dual_model, val_tokens, val_ground)
    evaluate.print_evaluation(val_metrics)

    return dual_model, val_metrics


# =============================================================================
#
# Native model approach - entropies are tested versus a numerical limit.
#
# =============================================================================

class NativeMarkov:
    def __init__(self, tokens, ground, method='kni', order=3, smoothing=0.5, p=0.995):

        nativetokens = [token for token, select in zip(tokens, ground) if select]
        self.nativemodel = MarkovWord(
            nativetokens, model=method, order=order, smoothing=smoothing)
        nativeentropies = self.nativemodel.calculate_entropies(nativetokens)
        self.ref_limit = calculate_empirical_ref_limit(nativeentropies, frac=p)

    def predict_data(self, tokens):
        entropies = self.nativemodel.calculate_entropies(tokens)
        predictions = np.less(entropies, self.ref_limit)
        return predictions.astype(int)

    def predict(self, token):
        return self.predict_data([token])[0]


def calculate_empirical_ref_limit(entropies, frac=0.995):
    # Entropies are not power law distributed, but neither are they Gaussian.
    # Better to use fraction of distribution to use as cut-point for
    # discriminating between native and loan.
    entropies = np.sort(entropies)
    idx = min((len(entropies)-1)*frac, len(entropies)-1)
    return (entropies[math.floor(idx)]+entropies[math.ceil(idx)])/2


def detect_native_loan_native_basis(tokens=None, borrowedscore=None, method='kni',
                                    smoothing=0.5, order=3, p=0.995, trainfrac=0.8):

    ground = np.array(borrowedscore) < 0.5  # native versus loan
    train_idx, val_idx = train_test_split(range(len(tokens)), test_size=1-trainfrac)
    train_tokens = np.array(tokens)[train_idx]
    train_ground = np.array(ground)[train_idx]

    native_model = NativeMarkov(train_tokens, train_ground, method=method,
                                order=order, smoothing=smoothing, p=p)

    print("\nEvaluate train dataset.")
    train_metrics = evaluate.evaluate_model(native_model, train_tokens, train_ground)
    evaluate.print_evaluation(train_metrics)

    print("\nEvaluate test dataset.")
    val_tokens = np.array(tokens)[val_idx]
    val_ground = np.array(ground)[val_idx]
    val_metrics = evaluate.evaluate_model(native_model, val_tokens, val_ground)
    evaluate.print_evaluation(val_metrics)

    return native_model, val_metrics

    # avg = statistics.mean(entropies)
    # stdev = statistics.stdev(entropies)
    # print(f'Native avg={avg:.3f}, stdev={stdev:.3f}')
    # print(f'fraction {frac:.3f}, ref limit={ref_limit:.3f}')

