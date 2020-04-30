from mobor.data import Wordlist

# from lingpy.sequence import ngrams
from collections import OrderedDict
import nltk.lm.preprocessing as pp
from nltk.lm import Vocabulary
import nltk.lm as lm
from nltk.util import ngrams
import math

# class Markov(ngrams.NgramModel):
# Drop inheritance from NgramModel for now.
# Add my Markov class as easy to add interface to NLTK3.5 for now.
class Markov:
    def __init__(
        self,
        wordlist,
        language,
        fields,
        dtypes=None,
        pre_order=0,
        post_order=2,
    ):
        if dtypes and len(dtypes) != len(fields):
            raise ValueError("dtypes needs to be same length as fields")

        self.wordlist = wordlist
        self.now = {
            "language": language,
            "fields": fields,
            "dtypes": dtypes or [str for str in fields],
        }
        self.now["table"] = self.wordlist.get_language(language, fields, dtypes)
        self.now["dicts"] = [
            OrderedDict(zip(["id"] + fields, row)) for row in self.now["table"]
        ]
        # ngrams.NgramModel.__init__(self, pre_order, post_order)
        self.pre_order = pre_order
        self.post_order = post_order

    # Add training sequences.
    def add_sequences(self, sequences):
        self.now["sequences"] = sequences
        # self.sequences = sequences
        # self.add_sequences(sequences)

    # def train(self, normalize='laplace', bins=None, **kwargs):
    #    self.ngrams.train(normalize, bins=bins, **kwargs)
    # Construct model, prepare training data, build vocabulary.
    # Train according to model parameters.
    def train(self, normalize="kneserney", smoothing=None):
        order = self.post_order + 1  # NLTK uses order of the entire ngram.
        # tokens = [t for t in self._sequences]
        tokens = [t.split() for t in self.now["sequences"]]
        tokens, vocab = pp.padded_everygram_pipeline(order, tokens)
        # Define the vocabulary.  Allow for out of vocabulary, thus cutoff=2.
        vocab = Vocabulary(vocab, unk_cutoff=2)
        # With NLTK, constructor needs to know normalize method.
        # So ngrammodel gets created here and trained here as well.
        self._lm = None
        if normalize == "kneserney":
            self._lm = lm.KneserNeyInterpolated(
                order, vocabulary=vocab, discount=smoothing
            )
        elif normalize == "wittenbell":
            self._lm = lm.WittenBellInterpolated(order, vocabulary=vocab)
        elif normalize == "laplace":  # Laplace
            self._lm = lm.Laplace(order=order, vocabulary=vocab)
        elif normalize == "lidstone":  # Lidstone
            self._lm = lm.Lidstone(
                order=order, gamma=smoothing, vocabulary=vocab
            )
        else:  # MLE
            self._lm = lm.MLE(order, vocabulary=vocab)

        self._lm.fit(tokens)  #

    # def score(self, sequence, use_length=True):
    #    return self.ngrams.score(sequence, use_length=True)

    # def model_entropy(self):
    #   return self.ngrams.model_entropy()

    def entropy(self, sequence, base=2):
        # return self.ngrams.entropy(sequence, base)
        return self.calculate_entropies([sequence], base=base)[0]

    def perplexity(self, sequence):
        # return self.ngrams.perplexity(sequence)
        # Default for entropies is base 2.
        return math.pow(2, self.entropy(sequence))

    # Calculate training entropies.
    def entropies(self, base=2):
        # return [self.entropy(seq) for seq in self.now['sequences']]
        self.entropies = self.calculate_entropies(
            self.now["sequences"], base=base
        )
        return self.entropies

    def calculate_entropies(self, sequences, base=2):
        order = self.post_order + 1
        tokens = [t.split() for t in sequences]
        ### verified by experiment - use order for left and 2 for right.
        tokens = [
            list(
                pp.pad_both_ends(
                    pp.pad_both_ends(t, pad_right=False, n=order),
                    pad_left=False,
                    n=2,
                )
            )
            for t in tokens
        ]

        # Convert lists of padded tokens to lists of segment ngrams.
        ngrams_lst = [list(ngrams(segments, order)) for segments in tokens]
        entropies = [self._lm.entropy(ngrams) for ngrams in ngrams_lst]
        if base != 2:
            factor = math.log(2, base)
            entropies = [factor * e for e in entropies]
        return entropies
