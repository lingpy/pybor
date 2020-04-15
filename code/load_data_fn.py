from lexibank_wold import Dataset as DS
from tabulate import tabulate
import lingpy

def load_wordlist():
    # Build absolute path to metadata
    metadata = DS().cldf_dir.joinpath("cldf-metadata.json")

    # Load a wordlist from the metadata
    wl = lingpy.Wordlist.from_cldf(
        metadata,
        columns=(
            "parameter_id",
            "language_id",
            "form",
            "segments",
            "loan",
            "concept",
            "borrowed"
        ),
    )
    return wl

import re

def stripalternative(segments):
	# Match on any sequence of characters. 
	# Partial seqeuence terminated by / is ignored and only last sequence is used.
	# Some error proofing. Guards against no-match or white space match.
	sr = re.compile(r"^(.+/)*([^/]+)$")

	return [['' if (mat := sr.match(seg.strip())) is None 
		else mat.group(2) for seg in segs] for segs in segments]

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

def get_language_data(wordlist=None, language=None):
    concept = wordlist.get_list(language=language, entry="parameter_id",  flat=True)
    form = wordlist.get_list(language=language, entry="form",  flat=True)
    loan = wordlist.get_list(language=language, entry="loan",  flat=True)
    borrowed = wordlist.get_list(language=language, entry="borrowed",  flat=True)
    borrowedscore = [(int(score[0])*-1+5)/4 for score in borrowed]
    form = [cleanform(form) for form in form]
    form_chars = [segmentchars(form) for form in form]
    segments = stripalternative(wordlist.get_list(language=language, entry="tokens",  flat=True))
    segments = [' '.join(list(word)) for word in segments]
    langdata = list(zip(concept, form, form_chars, segments, loan, borrowedscore))
    print(f'Language {language} from wordlist has {len(langdata)} concepts/forms')
    
    return langdata


import csv

def write_language_table(languagetable=None, language=None):
    # Write language table.
    tabledir = 'tables/'
    hdr = ('concept', 'form', 'formchars', 'segments', 'loan', 'borrowedscore')
    
    with open(tabledir+language+'.tsv', 'w', newline='\n') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(hdr)
        tsv_output.writerows(languagetable)
    print(f'Language {language} written to table has {len(languagetable)} concepts/forms')

# Function to read data and write individual language tables.
def get_write_language(wordlist=None, language=None):
    languagedata = get_language_data(wordlist=wordlist, language=language)
    write_language_table(languagetable=languagedata, language=language)

