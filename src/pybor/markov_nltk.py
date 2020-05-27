"""
Markov word model (based on nltk) used for borrowing detection.

Markov word model is used to calculate word entropies
which are used to detect borrowed words.

Notes
-----
Uses nltk (natural language toolkit) language model:
    https://www.nltk.org/api/nltk.lm.html

Book citation:
    Bird, Steven, Edward Loper and Ewan Klein (2009).
    Natural Language Processing with Python.  O'Reilly Media Inc.
"""
from nltk.util import ngrams
import nltk.lm as lm
import nltk.lm.preprocessing as pp

class MarkovWord:
    """
    Use 2nd or 3rd order ngrams (1st or 2nd order dependency) to calculate word entropies.
    Markov model is trained from list of tokens at time of construction.
    Entropy or entropy lists are calculated from the trained Markov model.
    """
    def __init__(self, tokens, model="kni", order=3, smoothing=0.5):
        """
        Estimate model probabilities from inial input.

        Parameters
        ----------
        tokens : [[str]]
            list of tokens, each token a list of character segments
            tokens/words used to fit the Markov model.
        model : str, optional
            Markov word model option from nltk. The default is "kni".
            kni = Kneser Ney interpolated,
            wbi = Witten Bell interpolated
            lp = Laplace,
            ls = Lidstone,
            ml = Maximum likelihood without smoothing.
        order : int, optional
            ngram order of model which counts symbol emitted and number
            of conditioning symbols. The default is 3.
        smoothing : float, optional
            Smoothing quantity for use by 'kni' or 'ls'. The default is 0.5.

        Returns
        -------
        MarkovWord object reference.

        """
        # smoothing is either discount for KNI or gamma for Lidstone.
        # default discount would be 0.1 for kni and
        # default gamma would be 0.2 for Lidstone in nltk.
        self._tokens = tokens
        self._order = order

        # NLTK recommended padding explicitly for training.
        train, vocab = pp.padded_everygram_pipeline(self._order, tokens)

        # Define the vocabulary.  Allow for out of vocabulary, thus cutoff=2.
        vocab = lm.Vocabulary(vocab, unk_cutoff=2)

        # Define and then train the language model.
        self._lm = None
        if model == "kni":
            if smoothing is None or smoothing <= 0.0 or smoothing >=1.0:
                # Default nltk discount is 0.1
                self._lm = lm.KneserNeyInterpolated(order, vocabulary=vocab)
            else:
                # We default to 0.5 smoothing.
                self._lm = lm.KneserNeyInterpolated(order, vocabulary=vocab,
                                                    discount=smoothing)
        elif model == "wbi":
            self._lm = lm.WittenBellInterpolated(order, vocabulary=vocab)
        elif model == "lp":  # Laplace
            self._lm = lm.Laplace(order=order, vocabulary=vocab)
        elif model == "ls":  # Lidstone
            if smoothing is None or smoothing <=0.0 or smoothing >= 1.0:
                self._lm = lm.Lidstone(order=order, gamma=0.1, vocabulary=vocab)
            else:
                self._lm = lm.Lidstone(order=order, gamma=smoothing, vocabulary=vocab)
        else:  # MLE
            self._lm = lm.MLE(order, vocabulary=vocab)

        self._lm.fit(train)


    def calculate_entropies(self, tokens):
        """
        Calculate entropies for list of tokens.

        Parameters
        ----------
        tokens : [[str]]
            list of tokens, each token a list of characters.

        Returns
        -------
        entropies : [float]
            calclated token entropies.

        """
        # Verified by experiment - use order for left and order 2 for right.
        # Use of order 2 for right stops calculation of first pad symbol.
        # This is in order to be consistent with neural imlementation.
        padded_tokens = [
            list(pp.pad_both_ends(
                    pp.pad_both_ends(token, pad_right=False, n=self._order),
                    pad_left=False, n=2)
            )
            for token in tokens
        ]

        # Convert lists of padded tokens to lists of ngrams.
        ngrams_lst = [list(ngrams(seq, self._order)) for seq in padded_tokens]
        entropies = [self._lm.entropy(segs) for segs in ngrams_lst]
        return entropies


    def calculate_entropy(self, token):
        """
        Calculate per sound average entropy of single token.

        Parameters
        ----------
        token : [str]
            Token as list of character segments.

        Returns
        -------
        float
            calcalated token entropy.

        """
        return self.calculate_entropies([token])[0]


