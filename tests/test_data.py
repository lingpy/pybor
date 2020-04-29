#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('src')

from mobor.data import Wordlist
from lingpy import log
import tabulate


wl = Wordlist.from_lexibank(
        'wold',
        fields=['borrowed'],
        fieldfunctions={
            "borrowed": lambda x: (int(x[0])*-1+5)/4
            }
        )
log.debug(
        f'loaded wordlist with {wl.height} concepts and {wl.width} doculects'
        )

# Select one language.
table = wl.get_language(
        'English',
        [
            'concept',
            'form',
            'formchars',
            'tokens',
            'sca',
            'borrowed'],
        dtypes = [str, str, str, str, str,
            lambda x: '{0:.2f}'.format(x)]
        )
print(tabulate.tabulate(table[:20], headers=['id', 'concept', 'form',
                'formchars', 'tokens', 'sca', 'borrowed'], tablefmt='pipe'))
