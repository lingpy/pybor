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
from pybor.svm import BagOfSounds
from pybor.data import LexibankDataset
from pybor.evaluate import prf
from pybor.ngram import NgramModel
from pybor.markov import DualMarkov
from pybor.neural import NeuralDual
import pybor.util as util

logger = util.get_logger(__name__)

def get_wold_data():
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

    return lex

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

def run_experiment(model_name, brate, order, test_split):

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


    lex = get_wold_data()

    for language in lex.languages.values():
        # Set training and test lists
        train, test = [], []

        # Get language table and break into train and test.
        table = lex.get_table(
            language=language["Name"], form="FormChars", classification="Loan"
        )
        train_table, test_table = util.train_test_split(table, test_split)

        # Seed native German words into training and test
        fakeidx = list(range(len(fakes)))
        for i in range(len(train_table)):
            if random.randint(0, brate) == 0:
                fake = fakes[fakeidx.pop(random.randint(0, len(fakeidx) - 1))]
                train += [fake]
            else:
                # Treat all words from table as native. ???
                train += [[train_table[i][0], train_table[i][1], 0]]

        for i in range(len(test_table)):
            if random.randint(0, brate) == 0:
                fake = fakes[fakeidx.pop(random.randint(0, len(fakeidx) - 1))]
                test += [fake]
            else:
                test += [[test_table[i][0], test_table[i][1], 0]]


        if model_name == 'bagofsounds':
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
            "{4},{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(p, r, f, a, language["Name"])
        )


    # Add totals
    totals = "{4},{0:.2f},{1:.2f},{2:.2f},{3:.2f}".format(
            mean([line[0] for line in stats]),
            mean([line[1] for line in stats]),
            mean([line[2] for line in stats]),
            mean([line[3] for line in stats]),
            "TOTAL/MEAN")
    buffer.append(totals)
    print(totals)


    # Write results to disk
    output_path = Path(__file__).parent.parent.absolute()
    output_path = output_path / "output" / f"fakeborrowing_{model_name}_{brate:.2f}br.csv"
    with open(output_path.as_posix(), "w") as handler:
        for row in buffer:
            handler.write(row)
            handler.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=["ngram", "bagofsounds", "markovdual", "neuraldual"],
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
    parser.add_argument(
        "--split", type=float, default=0.1, help="Set the test split proportion (default: 0.1)"
    )
    args = parser.parse_args()


    # We increment borrowing rate by one, as in original code by @lingulist
    run_experiment(args.model, args.brate + 1, args.order, args.split)
