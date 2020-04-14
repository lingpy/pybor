from lexibank_wold import Dataset as DS

import lingpy
from tabulate import tabulate

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
        "borrowed",
        "concept",
    ),
)


# Print wordlist columns
print(wl.columns)

# Print wordlist header
print(tabulate([wl[idx] for idx in range(1, 10)]))
