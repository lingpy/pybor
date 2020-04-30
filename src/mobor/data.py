"""
Wrapper for handling data in lexibank packages.
"""

# Import Python standard libraries
import csv
import datetime
from importlib import import_module
from pathlib import Path

# Import MPI-SHH libraries
import lingpy
from pyclts import CLTS

# TODO: Load with pycldf and not lingpy?
# TODO: use a different logger (either standard or clldutils)


class Wordlist(lingpy.Wordlist):
    """
    Class for loading data from a Lexibank dataset as a LingPy wordlist.
    """

    @classmethod
    def from_lexibank(
        cls,
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
            "segments",
        ),
        namespace=(
            ("concept_name", "concept"),
            ("concept_concepticon_id", "concepticon"),
            ("language_id", "doculect"),
            ("language_glottocode", "glottocode"),
            ("value", "value"),
            ("form", "form"),
            ("segments", "tokens"),
        ),
    ):
        # Build/extend the list of columns to read
        if fieldmapper:
            for field, target in fieldmapper.items():
                namespace += (field, target)
        if fields:
            for field in fields:
                columns += tuple([field])

        # Load data by importing the module
        module = import_module("lexibank_" + package)
        lingpy.log.debug("imported module lexibank_" + package)
        wordlist = cls.from_cldf(
            module.Dataset().cldf_dir.joinpath("cldf-metadata.json"),
            columns=columns,
            namespace=namespace,
        )
        lingpy.log.info(
            f"loaded wordlist {wordlist.height} concepts and {wordlist.width} languages"
        )

        # Apply the field functions if provided
        if fieldfunctions:
            for field, function in fieldfunctions.items():
                for idx in wordlist:
                    wordlist[idx, field] = function(wordlist[idx, field])

        return wordlist

    def add_soundclass(cls, model, clts=True):
        # Use join to put as space delimited string not individual symbols.
        if clts:
            clts = CLTS()
            soundclass = lambda x: " ".join(clts.soundclass(model)(str(x)))
        else:
            soundclass = lambda x: " ".join(lingpy.tokens2class(x, model))

        cls.add_entries(str(model), "tokens", soundclass)

    def add_formchars(cls):
        # Convert form to space delimited form without garbage symbols.
        formchars = lambda x: segmentchars(cleanform(x))

        cls.add_entries("formchars", "form", formchars)

    def get_language(cls, language, fields, dtypes=None):
        """
        Extract information for a given language.
        """

        # Default types to string
        if not dtypes:
            dtypes = [str for field in fields]

        subset = [
            {
                field: dtype(cls[idx, field])
                for field, dtype in zip(fields, dtypes)
            }
            for idx in cls
            if cls[idx, "doculect"] == language
        ]

        return subset


# Global functions instead of within class.
def cleanform(word):
    # Clean word forms.
    # Does not solve problem of variants which would add words to table.
    import re

    pnre = re.compile(r"\(\d+\)")
    space = re.compile(r"\s+")
    question = re.compile(r"[?¿]")
    affix = re.compile(r"(^-)|(-$)")
    optional = re.compile(r"_?\(-?\w+-?\)_?")

    try:
        word = word.strip().lower()
        word = pnre.sub("", word).strip()
        # previously used # for internal space.
        word = space.sub("_", word)
        word = question.sub("", word)
        word = affix.sub("", word)
        word = optional.sub("", word)
        return word
    except ValueError as error:
        print("Value Error", error)
        print(f"*{word}*")
        return word
    except Exception as exc:
        print("Exception", exc)
        print(f"*{word}*")
        return word


def segmentchars(form):
    try:
        return " ".join(list(form))
    except ValueError as error:
        print("Value Error", error)
        print(f"*{form}*")
        return form
    except Exception as exc:
        print("Exception", exc)
        print(f"*{form}*")
        return form


# Standard function for loading data, designed for Lexibank WOLD mostly
def load_data(dataset):
    # Load data
    wl = Wordlist.from_lexibank(
        dataset,
        fields=["borrowed"],
        fieldfunctions={"borrowed": lambda x: (int(x[0]) * -1 + 5) / 4},
    )

    wl.add_soundclass("sca", clts=False)
    wl.add_formchars()

    return wl


# quick function for updating results
# TODO: code properly, maybe in sqlite
def update_results(parameters, results, filename):
    # Extract fields from the new entry being passed
    parameter_fields = list(parameters.keys())
    result_fields = list(results.keys()) + ["timestamp"]

    # Make sure all values are strings (as read from disk)
    parameters = {key: str(value) for key, value in parameters.items()}
    results = {key: str(value) for key, value in results.items()}

    # Read results in disk, if available
    if not Path(filename).exists():
        entries = {}
    else:
        with open(filename) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            entries = {}
            for entry in reader:
                entry_parameters = {
                    field: entry.get(field, None) for field in parameter_fields
                }
                entry_results = {
                    field: entry.get(field, None) for field in result_fields
                }

                entry_key = tuple(
                    sorted(entry_parameters.items(), key=lambda i: i[0])
                )
                entries[entry_key] = entry_results

    # Add timestamp to `results`
    results["timestamp"] = str(datetime.datetime.now())

    # Update `entries` with the new item
    new_entry_key = tuple(sorted(parameters.items(), key=lambda i: i[0]))
    entries[new_entry_key] = results

    # Write results to disk
    with open(filename, "w") as tsvfile:
        writer = csv.DictWriter(
            tsvfile, delimiter="\t", fieldnames=parameter_fields + result_fields
        )
        writer.writeheader()

        for entry_key, entry_results in entries.items():
            # Make a copy of the results, making sure we don't change in place
            row = entry_results.copy()

            # Update row with info in the key and write
            for item in entry_key:
                row[item[0]] = item[1]
            writer.writerow(row)
