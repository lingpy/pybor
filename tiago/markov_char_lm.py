import nltk
import nltk.lm.preprocessing as pp
import nltk.lm as lm
import nltk.util as ut
from nltk.util import ngrams
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import pad_both_ends


class MarkovCharLM:
    # tokens and names are pandas series.
    # maybe later version would permit Python list or Numpy array?
    def __init__(self, tokens, names=None, model="KNI", order=2, smoothing=0.5):
        # smoothing is either discount for KNI or gamma for Lidstone.
        # default discount in KNI is 0.1 in NLTK while default gamma in Lidstone is 0.2
        self._tokens = tokens
        self._names = names
        self._order = order
        self._entropies = None
        self._ref_limit = None

        tokens = [t for t in self._tokens]

        words = [word.split() for word in tokens]  #
        # Must we use padding on both ends?
        train, vocab = pp.padded_everygram_pipeline(self._order, words)  #
        # Define the vocabulary.  Need to allow for out of vocabulary, thus cutoff=2.
        vocab = Vocabulary(vocab, unk_cutoff=2)

        # Define the language model.
        self._lm = None
        if model == "KNI":
            self._lm = lm.KneserNeyInterpolated(
                order, vocabulary=vocab, discount=smoothing
            )  # default discount = 0.1
        elif model == "WBI":
            self._lm = lm.WittenBellInterpolated(order, vocabulary=vocab)
        elif model == "LP":  # Laplace
            self._lm = lm.Laplace(order=order, vocabulary=vocab)
        elif model == "LS":  # Lidstone
            self._lm = lm.Lidstone(
                order=order, gamma=smoothing, vocabulary=vocab
            )
        else:  # MLE
            self._lm = lm.MLE(order, vocabulary=vocab)

        self._lm.fit(train)  #
        # print('model', model, self._lm)

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
        padded_words = [
            list(
                pad_both_ends(
                    pad_both_ends(word, pad_right=False, n=self._order),
                    pad_left=False,
                    n=2,
                )
            )
            for word in words
        ]

        # Convert lists of padded words to lists of sound ngrams.
        sound_ngrams_lst = [
            list(ngrams(sounds, self._order)) for sounds in padded_words
        ]
        return [
            self._lm.entropy(sound_ngrams) for sound_ngrams in sound_ngrams_lst
        ]

    # Allow to reset order; but necessarily retrain language model.
    def set_order(self, order):
        self._order = order
        self.train_lm()

    # Function to calculate per sound average entropy of single token.
    def calculate_token_entropy(self, token):
        return self.analyze_valtokens([token])

    # Training functions - reporting
    def get_training_types_gtlimit(self, limit):
        tokens = self._tokens[np.asarray(self._entropies) > limit]
        return list(set(tokens))

    def print_extreme_training_types(self, limit):
        print("\nentropy>" + str(limit), self.get_training_types_gtlimit(limit))

    # Function to retrieve per sound average entropy of token.
    def get_training_token_entropy(self, token):
        flags = self._tokens == token
        return np.array(self._entropies)[flags]

    def get_concepts_for_training_token(self, token):
        return self._names[self._tokens == token].tolist()

    def get_concepts_for_training_tokens_gtlimit(self, limit):
        return {
            tk: [
                concepts
                for concepts in self.get_concepts_for_training_token(tk)
            ]
            for tk in self.get_training_types_gtlimit(limit)
        }

    def print_extreme_training_concepts(self, limit):
        print(
            "\nentropy>" + str(limit),
            self.get_concepts_for_training_tokens_gtlimit(limit),
        )

    # Validation functions - reporting
    def get_val_types_gtlimit(self, limit):
        tokens = self._valtokens[np.asarray(self._valentropies) > limit]
        return list(set(tokens))

    def print_extreme_val_types(self, limit):
        print("\nentropy>" + str(limit), self.get_val_types_gtlimit(limit))

    # Function to retrieve per sound average entropy of token.
    def get_val_token_entropy(self, token):
        flags = self._valtokens == token
        return np.array(self._valentropies)[flags]

    def get_concepts_for_val_token(self, token):
        return self._valnames[self._valtokens == token].tolist()

    def get_concepts_for_val_tokens_gtlimit(self, limit):
        return {
            tk: [concepts for concepts in self.get_concepts_for_val_token(tk)]
            for tk in self.get_val_types_gtlimit(limit)
        }

    def print_extreme_val_concepts(self, limit):
        print(
            "\nentropy>" + str(limit),
            self.get_concepts_for_val_tokens_gtlimit(limit),
        )
