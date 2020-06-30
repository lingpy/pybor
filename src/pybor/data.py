"""
Wrapper for handling data in lexibank packages.
"""
from importlib import import_module
from pyclts import CLTS
from csvw.dsv import UnicodeDictReader
from tqdm import tqdm


class LexibankDataset(object):
    def __init__(self, package, transform=None):
        """
        Load the data of a lexibank dataset.
        """
        clts = CLTS()
        modify = {
            "Tokens": lambda x, y, z: [
                str(clts.bipa[token]) for token in x["Segments"].split() if token != "+"
            ],
            "Language": lambda x, y, z: y[x["Language_ID"]]["Name"],
            "Glottocode": lambda x, y, z: y[x["Language_ID"]]["Glottocode"],
            "Concept": lambda x, y, z: z[x["Parameter_ID"]]["Name"],
            "Concepticon_ID": lambda x, y, z: z[x["Parameter_ID"]]["Concepticon_ID"],
            "Concepticon_GLOSS": lambda x, y, z: z[x["Parameter_ID"]][
                "Concepticon_Gloss"
            ],
            "FormChars": lambda x, y, z: list(x["Form"]),
            "ASJP": lambda x, y, z: clts.soundclass("asjp")(x["Segments"]),
            "DOLGO": lambda x, y, z: clts.soundclass("dolgo")(x["Segments"]),
            "SCA": lambda x, y, z: clts.soundclass("sca")(x["Segments"]),
        }
        transform = transform or {}
        modify.update(transform)
        module = import_module("lexibank_" + package)
        self.ds = module.Dataset()
        self.forms = []
        self.concepts = {}
        with UnicodeDictReader(self.ds.cldf_dir.joinpath("parameters.csv")) as reader:
            for row in reader:
                self.concepts[row["ID"]] = row
        self.languages = {}
        with UnicodeDictReader(self.ds.cldf_dir.joinpath("languages.csv")) as reader:
            for row in reader:
                self.languages[row["ID"]] = row

        with UnicodeDictReader(self.ds.cldf_dir.joinpath("forms.csv")) as reader:
            for row in tqdm(reader, desc="loading data"):
                for key, fun in modify.items():
                    row[key] = fun(row, self.languages, self.concepts)
                self.forms.append(row)

    def get_table(self, language=None, form="Form", classification="Borrowed"):
        out = []
        for row in self.forms:
            if not language or row["Language"] == language:
                out.append([row["ID"], row[form], row[classification]])
        return out

    def get_donor_table(self, language=None, form="Form", classification="Borrowed"):
        out = []
        for row in self.forms:
            if not language or row["Language"] == language:
                out.append([row["ID"], row[form], row[classification],
                    row["donor_language"],
                    row["donor_description"],
                    row["donor_value"],
                    row["age_description"],
                    row["age_start_year"],
                    row["age_end_year"],])

        return out


# =============================================================================
# Apply user function to tables from WOLD.
# =============================================================================
import pickle
import pybor.util as util

logger = util.get_logger(__name__)


def apply_function_by_language(languages, form=None, function=None):
    """

    Parameters
    ----------
    languages : str or [str] or 'all'
        Language name, list of language names, or 'all' for all languages.
    form : str, optional
        Form designation from ['Tokens', 'FormChars', 'ASJP', 'DOLGO', 'SCA'].
        The internal default is 'Tokens'.
    function : function
        User provided funtion applied to each languge table.
        Initial arguments from function are language (str), form (str),
        and table ([str, [str], str]) corresponding to [id, form, loan_flag].

    Returns
    -------
    None.

    Notes
    -----

    Example:

import csv

def get_user_fn(detect_type, model_type, settings, writer):

    def fn_example(language, form, table,
                   detect_type=detect_type,
                   model_type=model_type,
                   settings=settings,
                   writer=writer):

        logger.info(f'language={language}, form={form},
                    detect type={detect_type}, model type={model_type}.')
        logger.info(f'table[:3]: {table[:3]}')
        logger.info(f'settings.embedding_len={settings.embedding_len}.')
        logger.info('Appropriate work by function.')

        writer.writerow([language, form, table[0][0], table[0][1]])

    return fn_example

# Try the function.
# Output path comes from package environment.
filename = 'test_flout.csv'
file_path = output_path / filename
with open(file_path.as_posix(), 'w', newline='') as fl:
    writer = csv.writer(fl)

    settings = config.RecurrentSettings(embedding_len=32)
    fn = get_user_fn('dual', 'recurrent', settings, writer)
    apply_function_by_language(languages=['English', 'Hup'], form='FormChars', function=fn)

    """
    logger.debug(f"Apply function to languages {languages} and form {form}.")

    lex = get_lexibank_access()
    languages = check_languages_with_lexibank(lex, languages)

    for language in languages:
        table = lex.get_table(language=language, form=form, classification="Borrowed")

        function(language, form, table)


# =============================================================================
# Get tables from WOLD via generator.
# =============================================================================


def language_table_gen(languages="all", form="Tokens"):

    logger.debug(f"Generator for {languages} languages.")

    lex = get_lexibank_access()
    languages = check_languages_with_lexibank(lex, languages)

    for language in languages:
        table = lex.get_table(language=language, form=form, classification="Borrowed")

        yield language, table


# =============================================================================
# Language table access functions
# =============================================================================
def get_lexibank_access():
    def to_score(x):
        num = float(x.lstrip()[0])
        return (5-num)/4

    try:
        with open("wold.bin", "rb") as f:
            lex = pickle.load(f)
    except:
        lex = LexibankDataset(
            "wold",
            transform={
                "Borrowed": lambda x, y, z:
                    1 if to_score(x["Borrowed"]) >= 0.9 else 0
            },
        )

        with open("wold.bin", "wb") as f:
            pickle.dump(lex, f)

    return lex


def check_languages_with_lexibank(lexibank, languages="all"):

    all_languages = [language["Name"] for language in lexibank.languages.values()]
    if languages == "all":
        return all_languages

    if isinstance(languages, str):
        languages = [languages]

    if isinstance(languages, list):
        for language in languages:
            if language not in all_languages:
                raise ValueError(f"Language {language} not in Lexibank.")

    return languages  # Checked as valid.

    logger.warning(
        "Language must be language name, list of languages, or keyword 'all'."
    )
    raise ValueError(f"Language list required, instead received {languages}.")
