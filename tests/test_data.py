from mobor.data import Wordlist
from lingpy import log
import tabulate


wl = Wordlist.from_lexibank(
        'wold', 
        fields=['loan', 'borrowed'],
        fieldfunctions={
            "borrowed": lambda x: (int(x[0])*-1+5)/4
            }
        )
log.debug(
        f'loaded wordlist with {wl.height} concepts and {wl.width} doculects'
        )

wl.add_soundclass('sca', clts=False)

# select one language
table = wl.get_language(
        'English',
        [
            'concept', 
            'form', 
            'tokens', 
            'sca', 
            'borrowed',
            'loan'],
        dtypes = [str, str, str, lambda x: ' '.join(x), 
            lambda x: '{0:.2f}'.format(x), str]
        )
print(tabulate.tabulate(table[:20], headers=['id', 'concept', 'form', 'tokens', 'sca', 'borrowed',
    'loan'], tablefmt='pipe'))


