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


# =============================================================================
# Everything else moved to wold.py
# =============================================================================
