# Load Python standard libraries
from pathlib import Path
from statistics import mean
import argparse
import random

# Load Pybor
from pybor.dev.data import training, testing
from pybor.evaluate import prf
from pybor.markov import DualMarkov
from pybor.neural import NeuralDual
from pybor.ngram import NgramModel
from pybor.svm import BagOfSounds
import pybor.util as util
import pybor.wold as wold

def bigrams(sequence):
    return list(zip(["^"] + sequence[:-1], sequence[1:] + ["$"]))


def trigrams(sequence):
    return list(
        zip(
            ["^", "^"] + sequence[:-1],
            ["^"] + sequence + ["$"],
            sequence[1:] + ["$", "$"],
        )
    )


def run_experiment(
        model_name, language_, form, brate, order, test_split,
        verbose, output):

    # output buffer
    buffer = ["Language,Precision,Recall,Fs,Accuracy"]

    # Collect all native words from German word table in order
    # to seed native German words as fake in other language tables.
    fakes = []
    for a, b, c in training + testing:
        if c != 1:
            fakes += [[a, b, 1]]

    table = []
    stats = []

    wolddb = wold.get_wold_access()
    languages = wold.check_wold_languages(wolddb, language_)

    for language in languages:
        # Set training and test lists
        # train, test = [], []

        # Get language table, delete loan words, seed fakes, split into train and test.
        table = wolddb.get_table(
            language=language, form=form, classification="Borrowed"
        )
        table = [row for row in table if row[2] != 1]
        # How many fakes? Want 1/brate borrowed words in resulting table.
        # So we add 1/(brate-1) fraction of words.
        add_len = int(round(len(table) / (brate - 1)))
        table += random.sample(fakes, add_len)
        train, test = util.train_test_split(table, test_split)
        train_add_len = sum([row[2] for row in train])
        test_add_len = sum([row[2] for row in test])
        # Seed native German words into training and test

        if verbose:
            logger.info(
                f"{language} language, {form} form, table len {len(table)}, "
                + f"table borrowed {add_len}, borrow rate {int(round(len(table)/add_len))}."
            )
            logger.info(
                f"train len {len(train)}, train borrowed {train_add_len}, "
                + f"test len {len(test)}, test borrowed {test_add_len}."
            )

        if model_name == "bagofsounds":
            # Building bigram and trigram test sets
            train2, test2 = (
                [[a, bigrams(b), c] for a, b, c in train],
                [[a, bigrams(b), c] for a, b, c in test],
            )
            train3, test3 = (
                [[a, trigrams(b), c] for a, b, c in train],
                [[a, trigrams(b), c] for a, b, c in test],
            )

            # Train the bag of words according to the requested order
            if order == "monogram":
                bag = BagOfSounds(train)
                guess = bag.predict_data([[a, b] for a, b, c in test])
            elif order == "bigram":
                bag = BagOfSounds(train2)
                guess = bag.predict_data([[a, b] for a, b, c in test2])
            elif order == "trigram":
                bag = BagOfSounds(train3)
                guess = bag.predict_data([[a, b] for a, b, c in test3])
        else:
            if model_name == "ngram":
                ngrams = NgramModel(train)
                guess = ngrams.predict_data(test)
            elif model_name == "markovdual":
                markov = DualMarkov(train)
                guess = markov.predict_data(test)
            else:  # Neural
                neural = NeuralDual(train)
                neural.train()
                guess = neural.predict_data(test)

        # Collect performance statistics
        p, r, f, a = prf(test, guess)
        stats += [[p, r, f, a]]
        buffer.append(
            "{4},{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(p, r, f, a, language)
        )

    # Add totals
    totals = "{4},{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(
        mean([line[0] for line in stats]),
        mean([line[1] for line in stats]),
        mean([line[2] for line in stats]),
        mean([line[3] for line in stats]),
        "TOTAL/MEAN",
    )
    buffer.append(totals)
    print(totals)

    # Write results to disk
    output_path = Path(output).joinpath(
            f"fakeborrowing_{model_name}_{language_}_{form}_{brate:.1f}br.csv"
            ).as_posix()

    with open(output_path, "w") as handler:
        for row in buffer:
            handler.write(row)
            handler.write("\n")


if __name__ == "__main__":
    logger = util.get_logger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        choices=["ngram", "bagofsounds", "markovdual", "neuraldual"],
        help="Model for the fake borrowing experiment",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        type=str,
        default="all",
        help='Languages to use for example (default: "all")',
    )
    parser.add_argument(
        "--form",
        type=str,
        default="Tokens",
        choices=["Tokens", "FormChars", "ASJP", "DOLGO", "SCA"],
        help="Form to take from language table.",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["monogram", "bigram", "trigram"],
        default="monogram",
        help='Ngram order for experiment (default: "monogram")',
    )
    parser.add_argument(
        "--brate",
        type=int,
        default=10,
        help="Set the borrowing rate (default: 10, for 1 in 10)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.2,
        help="Set the test split proportion (default: 0.2)",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Verbose reporting (default: False)"
    )
    parser.add_argument(
        "--output",
        default="output",
        help="output")
    args = parser.parse_args()
    languages = "all" if args.languages[0] == "all" else args.languages

    run_experiment(
        args.model,
        languages,
        args.form,
        args.brate,
        args.order,
        args.split,
        args.verbose,
        args.output
    )
