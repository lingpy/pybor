#!/usr/bin/env python3

# Load Python standard libraries
from statistics import mean
from sys import argv
from pathlib import Path
import argparse
import pickle
import random

# Load Pybor
from pybor.dev.data import training, testing
from pybor.svm import *
from pybor.data import LexibankDataset
from pybor.evaluate import prf
from pybor.ngram import NgramModel
from pybor.markov import DualMarkov


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


def run_experiment(model, brate, order):
    # output buffer
    buffer = ["Language,Precision,Recall,Fs,Accuracy"]

    # load binary wold
    try:
        with open("wold.bin", "rb") as f:
            lex = pickle.load(f)
    except:
        lex = LexibankDataset(
            "wold",
            transform={
                "Loan": lambda x, y, z: 1
                if x["Borrowed"] != "" and float(x["Borrowed_score"]) >= 0.9
                else 0
            },
        )
        with open("wold.bin", "wb") as f:
            pickle.dump(lex, f)

    # Collect all native words
    fakes = []
    for a, b, c in training + testing:
        if c != 1:
            fakes += [[a, b, 1]]

    table = []
    stats = []
    for language in lex.languages.values():
        # Set training and test list
        train, test = [], []

        # Get language table and shuffle it
        table = lex.get_table(
            language=language["Name"], form="FormChars", classification="Loan"
        )
        random.shuffle(table)

        # Pop native words in training and test
        fakeidx = list(range(len(fakes)))
        for i in range(len(table) // 2):
            if random.randint(0, brate) == 0:
                fake = fakes[fakeidx.pop(random.randint(0, len(fakeidx) - 1))]
                train += [fake]
            else:
                train += [[table[i][0], table[i][1], 0]]
        for i in range(len(table) // 2, len(table)):
            if random.randint(0, brate) == 0:
                fake = fakes[fakeidx.pop(random.randint(0, len(fakeidx) - 1))]
                test += [fake]
            else:
                test += [[table[i][0], table[i][1], 0]]

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
            bag = model(train)
            guess = bag.predict_data([[a, b] for a, b, c in test])
        elif order == "bigram":
            bag = model(train2)
            guess = bag.predict_data([[a, b] for a, b, c in test2])
        elif order == "trigram":
            bag = model(train3)
            guess = bag.predict_data([[a, b] for a, b, c in test3])

        # Collect performance statistics
        p, r, f, a = prf(test, guess)
        stats += [[p, r, f, a]]
        buffer.append(
            "{4},{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(p, r, f, a, language["Name"])
        )

    # Add totals
    buffer.append(
        "{4},{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(
            mean([line[0] for line in stats]),
            mean([line[1] for line in stats]),
            mean([line[2] for line in stats]),
            mean([line[3] for line in stats]),
            "TOTAL/MEAN",
        )
    )

    # Write results to disk
    output_path = Path(__file__).parent.parent.absolute()
    output_path = output_path / "output" / "fakeborrowing.csv"
    with open(output_path.as_posix(), "w") as handler:
        for row in buffer:
            handler.write(row)
            handler.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=["ngram", "bagofsounds", "markovdual"],
        help="Model for the fake borrowing experiment",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["monogram", "bigram", "trigram"],
        default="monogram",
        help='Ngram order for experiment (default: "monogram")',
    )
    parser.add_argument(
        "--brate", type=int, default=19, help="Set the borrowing rate (default: 20)"
    )
    args = parser.parse_args()

    # select the correct model
    if args.model == "ngram":
        model = NgramModel
    elif args.model == "markovdual":
        model = DualMarkov
    else:
        model = BagOfSounds

    # We increate borrowing rate by one, as in original code by @lingulist
    run_experiment(model, args.brate + 1, args.order)
