"""
Detect loan words based on word entropies as calculated by Markov word models.

Notes
-----
Support for model trained on both native and borrowed words (dual model),
and for model trained on just native words (native model).
"""

# Import Python standard libraries
import math
from pathlib import Path

# Import 3rd-party libraries
from nltk.util import ngrams
import attr
import nltk.lm as lm
import nltk.lm.preprocessing as pp

# Build namespace
import pybor.config as config
import pybor.util as util

output_path = Path(config.BaseSettings().output_path).resolve()
logger = util.get_logger(__name__)


@attr.s
class Markov:
    """
    Base class.
    """

    data = attr.ib(default=[], repr=False)
    model = attr.ib(default=None)
    order = attr.ib(default=None)
    smoothing = attr.ib(default=None)
    settings = attr.ib(default=config.MarkovSettings())

    def __attrs_post_init__(self):
        if not isinstance(self.settings, config.MarkovSettings):
            self.settings = config.MarkovSettings()
        self.model = self.settings.model if self.model is None else self.model
        self.order = self.settings.order if self.order is None else self.order
        self.smoothing = (
            self.settings.smoothing if self.smoothing is None else self.smoothing
        )


@attr.s
class MarkovWord(Markov):
    """
    Use 2nd or 3rd order ngrams (1st or 2nd order dependency) to calculate word entropies.

    Notes
    -----
    Markov model is trained from list of tokens at time of construction.
    Entropy or entropy lists are calculated from the trained Markov model.

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

    """

    tokens = attr.ib(default=[], repr=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        # NLTK recommended padding explicitly for training.
        train, vocab = pp.padded_everygram_pipeline(self.order, self.tokens)

        # Define the vocabulary.  Allow for out of vocabulary, thus cutoff=2.
        vocab = lm.Vocabulary(vocab, unk_cutoff=2)

        # Define and then train the language model.
        options = {
            "kni": (lm.KneserNeyInterpolated, {"discount": self.smoothing}),
            "wbi": (lm.WittenBellInterpolated, {}),
            "lp": (lm.Laplace, {}),
            "ls": (lm.Lidstone, {"gamma": self.smoothing}),
            "mle": (lm.MLE, {}),
        }

        self.lm = options[self.model][0](
            order=self.order, vocabulary=vocab, **options[self.model][1]
        )
        self.lm.fit(train)

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
            list(
                pp.pad_both_ends(
                    pp.pad_both_ends(token, pad_right=False, n=self.order),
                    pad_left=False,
                    n=2,
                )
            )
            for token in tokens
        ]

        # Convert lists of padded tokens to lists of ngrams.
        ngrams_lst = [list(ngrams(seq, self.order)) for seq in padded_tokens]
        entropies = [self.lm.entropy(segs) for segs in ngrams_lst]
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


@attr.s
class DualMarkov(Markov):
    """
    Construct Markov models of native and loan distributions.

    Notes
    -----
    Fit dual Markov models - one to native data, another to loan data.

    Parameters
    ----------
    data : [[str, [str], int]]
        List of language tokens in format:
            identifier,
            token as list of character segments,
            binary (0, 1) indicator of borrowed word status.
    model : str, optional
        Model type from MarkovWord class. The default is 'kni'.
    order : int, optional
        Order from MarkovWord class. The default is 3.
    smoothing : float, optional
        smoothing from MarkovWord class. The default is 0.5.
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        nativetokens = [token for _, token, status in self.data if status == 0]
        self.native = MarkovWord(
            tokens=nativetokens,
            data=self.data,
            model=self.model,
            order=self.order,
            smoothing=self.smoothing,
        )

        loantokens = [token for _, token, status in self.data if status == 1]
        self.loan = MarkovWord(
            tokens=loantokens,
            data=self.data,
            model=self.model,
            order=self.order,
            smoothing=self.smoothing,
        )

    def predict_tokens(self, tokens):
        """
        Predict loan word statuses for a list of tokens.

        Parameters
        ----------
        tokens : [[str]]
            List of tokens as list of character segments.

        Returns
        -------
        predictions : int (binary 0,1)
            List of loan word predictions corresponding to tokens.

        """
        nativeentropies = self.native.calculate_entropies(tokens)
        loanentropies = self.loan.calculate_entropies(tokens)
        predictions = [
            int(loanS < nativeS)
            for nativeS, loanS in zip(nativeentropies, loanentropies)
        ]
        return predictions

    def predict_data(self, data):
        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        # Insert '' for identifer before invoking predict_data
        return self.predict_data([["", token]])[0]


@attr.s
class NativeMarkov(Markov):
    """
    Unsupervised language model approach.
    """

    p = attr.ib(default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.p = self.settings.p if self.p is None else self.p

        nativetokens = [token for _, token, status in self.data if not status]
        self.nativemodel = MarkovWord(
            tokens=nativetokens,
            data=self.data,
            model=self.model,
            order=self.order,
            smoothing=self.smoothing,
        )
        nativeentropies = self.nativemodel.calculate_entropies(nativetokens)
        entropies = sorted(nativeentropies)
        idx = min((len(entropies) - 1) * self.p, len(entropies) - 1)
        self.ref_limit = (entropies[math.floor(idx)] + entropies[math.ceil(idx)]) / 2

    def predict_tokens(self, tokens):
        entropies = self.nativemodel.calculate_entropies(tokens)
        predictions = [int(tokenS > self.ref_limit) for tokenS in entropies]
        return predictions

    def predict_data(self, data):
        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        # Insert '' for identifer before invoking predict_data
        return self.predict_data([["", token]])[0]
