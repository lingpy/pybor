"""
Bag of sound approaches for borrowing detection.
"""

from collections import defaultdict
import numpy as np

from lingpy.sequence import NgramModel as LP_Ngram


class NgramModel(object):
    """
    Use an ngram classifier to try to find borrowed words.
    """

    def __init__(
        self, data, order=3, smooth="wittenbell", use_length=True, **kw
    ):

        self.data = data
        self.use_length = use_length

        # Collect sequences by their class
        self.sequences = defaultdict(list)
        for idx, sequence, seq_class in data:
            self.sequences[seq_class].append(sequence)

        # Build forward (left-to-right) and backward equivalent (right-to-left)
        # estimators for each class
        self.f_estimator = {}
        self.b_estimator = {}
        for seq_class, sequences in self.sequences.items():
            self.f_estimator[seq_class] = LP_Ngram(order, sequences=sequences)
            self.b_estimator[seq_class] = LP_Ngram(
                pre_order=0, post_order=order, sequences=sequences
            )

            self.f_estimator[seq_class].train(smooth)
            self.b_estimator[seq_class].train(smooth)

    def predict(self, word):
        """
        Predict if a word is borrowed or not.
        """

        scores = []
        for seq_class in self.sequences:
            f = self.f_estimator[seq_class].score(
                word, use_length=self.use_length
            )
            b = self.b_estimator[seq_class].score(
                word, use_length=self.use_length
            )

            # NOTE: here it is only using the forward estimation, but we
            # could combine them, for example just doing `(f+b)/2`
            scores.append([seq_class, f])

        # sort scores and return the seq_class
        scores = sorted(scores, key=lambda s: -s[1])

        return np.array([scores[0][0]])

    def predict_data(self, data):
        """
        Predict for a range of words.
        """
        out = []
        for idx, word in data:
            out += [[idx, word, self.predict(word)]]

        return out
