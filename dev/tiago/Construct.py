import utils

# List of languages to process.
study_languages = [
    "English",
    "Hup",
    "ImbaburaQuechua",
    "Mapudungun",
    "Qeqchi",
    "Wichi",
]

wl = utils.load_wordlist()
for l in study_languages:
    utils.get_write_language(wordlist=wl, language=l)
