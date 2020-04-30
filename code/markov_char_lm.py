import nltk.lm.preprocessing as pp
import nltk.lm as lm
from nltk.util import ngrams
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import pad_both_ends
import numpy as np


class MarkovCharLM():
    # tokens and names are pandas series.
    # maybe later version would permit Python list or Numpy array?
    def __init__(self, tokens, names=None, model='KNI', order=2, smoothing=0.5):
        # smoothing is either discount for KNI or gamma for Lidstone.
        # default discount in KNI is 0.1 in NLTK while default gamma in Lidstone is 0.2
        self._tokens = tokens
        self._names = names
        self._order = order
        self._entropies = None
        self._ref_limit = None

        tokens = [t for t in self._tokens]

        words = [word.split() for word in tokens] #
        # Must we use padding on both ends?
        train, vocab = pp.padded_everygram_pipeline(self._order, words) #
        # Define the vocabulary.  Need to allow for out of vocabulary, thus cutoff=2.
        vocab = Vocabulary(vocab, unk_cutoff=2)

        # Define the language model.
        self._lm = None
        if model=='KNI':
            self._lm = lm.KneserNeyInterpolated(order, vocabulary=vocab, discount=smoothing) # default discount = 0.1
        elif model=='WBI':
            self._lm = lm.WittenBellInterpolated(order, vocabulary=vocab)
        elif model=='LP': # Laplace
            self._lm = lm.Laplace(order=order, vocabulary=vocab)
        elif model=='LS': # Lidstone
            self._lm = lm.Lidstone(order=order, gamma=smoothing, vocabulary=vocab)
        else: # MLE
            self._lm = lm.MLE(order, vocabulary=vocab)

        self._lm.fit(train) #
        #print('model', model, self._lm)

    # General function for analysis of training tokens.
    def analyze_training(self):
        self._entropies = self.analyze_tokens(self._tokens)
        return self._entropies

    def validate(self, tokens, names=None):
        self._valtokens = tokens
        self._valnames = names

        self._valentropies = self.analyze_tokens(self._valtokens)
        return self._valentropies

    def analyze_tokens(self, tokens):
        words = [word.split() for word in tokens]
        ### verified by experiment - use order for left and order 2 for right.
        padded_words = [list(pad_both_ends(pad_both_ends(word, pad_right=False, n=self._order), pad_left=False, n=2)) for word in words]

        #Convert lists of padded words to lists of sound ngrams.
        sound_ngrams_lst = [list(ngrams(sounds, self._order)) for sounds in padded_words]
        return [self._lm.entropy(sound_ngrams) for sound_ngrams in sound_ngrams_lst]


    # Function to calculate per sound average entropy of single token.
    def calculate_token_entropy(self, token):
        return self.analyze_tokens([token])

    def get_concepts_for_training_token(self, token):
        return self._names[self._tokens==token].tolist()


    def get_concepts_for_val_token(self, token):
        return self._valnames[self._valtokens==token].tolist()

