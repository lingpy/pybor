# Import Python standard libraries
import pickle

# Import 3rd-party libraries
from tabulate import tabulate
import pyclts

# Build namespace
import pybor.wold as wold

# Build CLTS object
# TODO: use default and accept user path
clts = pyclts.CLTS("/home/tresoldi/.config/cldf/clts")


def get_bigrams(sequence):
    return list(zip(["^"] + sequence[:-1], sequence[1:] + ["$"]))


def get_trigrams(sequence):
    return list(
        zip(
            ["^", "^"] + sequence[:-1],
            ["^"] + sequence + ["$"],
            sequence[1:] + ["$", "$"],
        )
    )


def main():

    lex = wold.get_wold_access()

    mytable = []
    for language in lex.languages.values():
        table = lex.get_table(
            language=language["Name"], form="Segments", classification="Borrowed"
        )
        # count borrowings
        b = len([x for x in table if x[-1] == 1])
        n = len([x for x in table if x[-1] == 0])

        # count sounds
        bs, ns = set(), set()
        for row in table:
            for sound in row[1].split():
                sound = str(clts.bipa[sound])
                if sound != "+":
                    if row[-1] == 1:
                        bs.add(sound)
                    else:
                        ns.add(sound)

        # collect trigrams
        b_trigram, n_trigram = set(), set()
        for row in table:
            seq = [str(clts.bipa[sound]) for sound in row[1].split() if sound != "+"]
            trigrams = set(get_trigrams(seq))
            if row[-1] == 1:
                b_trigram |= trigrams
            else:
                n_trigram |= trigrams

        # build results
        mytable += [
            [
                language["Name"],
                b / len(table),
                n / len(table),
                len(bs.difference(ns)) / len(bs.union(ns)),
                len(ns.difference(bs)) / len(bs.union(ns)),
                len(b_trigram.difference(n_trigram)) / len(b_trigram.union(n_trigram)),
                len(n_trigram.difference(b_trigram)) / len(b_trigram.union(n_trigram)),
            ]
        ]

    print(
        tabulate(
            sorted(mytable),
            headers=["Language", "BorrW", "NatW", "UBorrS", "UNatS", "UBorr3", "UNat3"],
            floatfmt=".2f",
        )
    )


if __name__ == "__main__":
    main()
