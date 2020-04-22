from mobor.data import Wordlist
from lingpy.sequence import ngrams
from collections import OrderedDict

class Markov(ngrams.NgramModel):
    
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
            raise ValueError('dtypes needs to be same length as fields')

        self.wordlist = wordlist
        self.now = {
                'language': language,
                'fields': fields,
                'dtypes': dtypes or [str for str in fields],
                }
        self.now['table'] = self.wordlist.get_language(language, fields, dtypes)
        self.now['dicts'] = [OrderedDict(zip(['id']+fields, row)) for row in
                self.now['table']]
        ngrams.NgramModel.__init__(self, pre_order, post_order)

    #def add_sequences(self, sequences):
    #    self.now['sequences'] = sequences
    #    self.add_sequences(sequences)

    #def train(self, normalize='laplace', bins=None, **kwargs):
    #    self.ngrams.train(normalize, bins=bins, **kwargs)

    #def score(self, sequence, use_length=True):
    #    return self.ngrams.score(sequence, use_length=True)

    #def model_entropy(self):
    #    return self.ngrams.model_entropy()

    #def entropy(self, sequence, base=2.0):
    #    return self.ngrams.entropy(sequence, base)

    #def perplexity(self, sequence):
    #    return self.ngrams.perplexity(sequence)

    #def entropies(self):
    #    return [self.entropy(seq) for seq in self.now['sequences']]


