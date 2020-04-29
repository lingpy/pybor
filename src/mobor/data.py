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

        # Use join to put as space delimited string not individual symbols.
        if clts:
            clts = CLTS()
            soundclass = lambda x: ' '.join(clts.soundclass(model)(str(x)))
        else:
            soundclass = lambda x: ' '.join(lingpy.tokens2class(x, model))
        self.add_entries(
                str(model),
                'tokens',
                soundclass)

    def add_formchars(self):
    # Convert form to space delimited form without garbage symbols.
        formchars = lambda x: segmentchars(cleanform(x))
        self.add_entries(
                'formchars',
                'form',
                formchars)


    def get_language(self,
            language,
            fields,
            dtypes=None):
        if not dtypes:
            dtypes = [str for field in fields]
        # Add sound class.
        if 'tokens' in fields:
        	self.add_soundclass(model='sca')
        if 'form' in fields:
        	self.add_formchars()

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


# Global functions instead of within class.
def cleanform(word):
# Clean word forms.
# Does not solve problem of variants which would add words to table.
	import re
	pnre = re.compile('\(\d+\)')
	space = re.compile('\s+')
	question = re.compile('[?¿]')
	affix = re.compile('(^-)|(-$)')
	optional = re.compile('_?\(-?\w+-?\)_?')

	try:
		word = word.strip().lower()
		word = pnre.sub('', word).strip()
		# previously used # for internal space.
		word = space.sub('_', word)
		word = question.sub('', word)
		word = affix.sub('', word)
		word = optional.sub('', word)
		return word
	except ValueError as error:
		print('Value Error', error)
		print(f'*{word}*')
		return word
	except Exception as exc:
		print('Exception', exc)
		print(f'*{word}*')
		return word

def segmentchars(form):
	try:
		return ' '.join(list(form))
	except ValueError as error:
		print('Value Error', error)
		print(f'*{form}*')
		return form
	except Exception as exc:
		print('Exception', exc)
		print(f'*{form}*')
		return form
