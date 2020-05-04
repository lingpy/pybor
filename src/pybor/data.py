"""
Wrapper for handling data in lexibank packages.
"""
from importlib import import_module

import lingpy
from pyclts import CLTS

class Wordlist(lingpy.Wordlist):
    
    @classmethod
    def from_lexibank(
            self,
            package,
            fields=None,
            fieldmapper=None,
            fieldfunctions=None,
            columns=(
                "concept_name",
                "concept_concepticon_id",
                "language_id",
                "language_glottocode",
                "value",
                "form",
                "segments"
                ),
            namespace=(
                ("concept_name", "concept"),
                ("concept_concepticon_id", "concepticon"),
                ("language_id", "doculect"),
                ("language_glottocode", "glottocode"),
                ("value", "value"),
                ("form", "form"),
                ("segments", "tokens")
                )
            ):
        if fieldmapper:
            for field, target in fieldmapper.items():
                namespace += ((field, target))
        if fields:
            for field in fields:
                columns += tuple([field])
        module = import_module('lexibank_'+package)
        lingpy.log.debug('imported module lexibank_'+package)
        wordlist = self.from_cldf(
                module.Dataset().cldf_dir.joinpath(
                    'cldf-metadata.json'),
                columns=columns,
                namespace=namespace)
        lingpy.log.info(
            f'loaded wordlist {wordlist.height} concepts and {wordlist.width} languages'
            )
        if fieldfunctions:
            for field, function in fieldfunctions.items():
                for idx in wordlist:
                    wordlist[idx, field] = function(
                            wordlist[idx, field])
        return wordlist

    def add_soundclass(self, model, clts=True):

        if clts:
            clts = CLTS()
            soundclass = lambda x: clts.soundclass(model)(str(x))
        else:
            soundclass = lambda x: lingpy.tokens2class(x, model)
            
        self.add_entries(
                str(model),
                'tokens',
                soundclass)

    def get_language(self,
            language,
            fields,
            dtypes=None):
        if not dtypes:
            dtypes = [str for field in fields]
        idxs = self.get_list(
                col=language,
                flat=True)
        out = []
        for idx in idxs:
            row = [idx]
            for field, dtype in zip(fields, dtypes):
                row += [dtype(self[idx, field])]
            
            out += [row]
        return out

