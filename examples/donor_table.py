import pybor
import pybor.wold as wold

language_ = "all"  # or language name
form = "Tokens"  # one in ["Tokens", "FormChars", "ASJP", "DOLGO", "SCA"],


wolddb = wold.get_wold_access()
languages = wold.check_wold_languages(wolddb, language_)

for language in languages:
    table = wolddb.get_donor_table(language=language, form=form, classification="Loan")

    for row in table:
        print(row)
