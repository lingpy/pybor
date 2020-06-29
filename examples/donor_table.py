import pybor
import pybor.data as data

language_ = "all"  # or language name
form = "Tokens"  # one in ["Tokens", "FormChars", "ASJP", "DOLGO", "SCA"],


lex = data.get_lexibank_access()
languages = data.check_languages_with_lexibank(lex, language_)

for language in languages:
    table = lex.get_donor_table(language=language, form=form, classification="Loan")

    for row in table:
        print(row)
