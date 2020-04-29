"""
Plot the entropy of words and contrast them.
"""
from mobor.data import Wordlist
from mobor.markov import Markov
from mobor.plot import plot_word_distributions

def register(parser):
    parser.add_argument(
            '--language',
            default='English',
            help='Select your language',
            type=str
        )
    parser.add_argument(
            '--sequence',
            default='formchars',
            help='select the sequence type you want',
            type=str)

    parser.add_argument(
            '--file',
            default='english-test.pdf',
            help='select the filename for plotting',
            type=str)

def run(args):
    wl = Wordlist.from_lexibank(
            'wold',
            #fields=['loan', 'borrowed'],
            fields=['borrowed'],
            fieldfunctions={
                "borrowed": lambda x: (int(x[0])*-1+5)/4
                })

    args.log.info('loaded markov')
    #wl.add_soundclass('sca', clts=False)
    #args.log.info('added sound classes')

    mk = Markov(
            wl,
            args.language,
            ['concept', 'form', 'formchars', 'tokens', 'sca', 'borrowed'],
            dtypes = [str, str, str, str, str, float],
            #['concept', 'form', 'tokens', 'sca', 'borrowed', 'loan'],
            #dtypes = [str, list, list, list, float, bool],
            post_order=2,
            pre_order=0
            )
    mk.add_sequences(
            [row[args.sequence] for row in mk.now['dicts']])
    mk.train(normalize='kneserney')

    # retrieve distribution for borrowed words
    loan, native = [], []
    for row in mk.now['dicts']:
        if row['borrowed']>=0.5:
            loan += [mk.entropy(row[args.sequence])]
        else:
            native += [mk.entropy(row[args.sequence])]

    #borrowed_avg = sum(borrowed)/len(borrowed)
    #unborrowed_avg = sum(borrowed)/len(borrowed)

    # plot the distribution
    plot_word_distributions(native, loan, args.file,
        graphlimit=max([max(loan), max(native)])+1)



