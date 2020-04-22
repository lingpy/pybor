import tabulate
from mobor.data import Wordlist
from mobor.markov import Markov
from mobor.plot import plot_word_distributions

wl = Wordlist.from_lexibank(
        'wold',
        fields=['loan', 'borrowed'],
        fieldfunctions={
            "borrowed": lambda x: (int(x[0])*-1+5)/4
            })

print('loaded markov')
wl.add_soundclass('sca', clts=False)
print('added sound classes')

mk = Markov(
        wl, 
        'English', 
        ['concept', 'form', 'tokens', 'sca', 'borrowed', 'loan'],
        dtypes = [str, list, list, list, float, bool]
        )
mk.add_sequences(
        [row['tokens'] for row in mk.now['dicts']])
mk.train(normalize='laplace')

# retrieve distribution for borrowed words
borrowed, unborrowed = [], []
for row in mk.now['dicts']:
    if row['loan']:
        borrowed += [mk.entropy(row['tokens'])]
    else:
        unborrowed += [mk.entropy(row['tokens'])]

# plot the distribution
plot_word_distributions(borrowed, unborrowed, 'test.pdf')







