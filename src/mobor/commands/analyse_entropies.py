"""
Build results of a distribution analysis.
"""

# Import Python standard libraries
from pathlib import Path

import mobor.data
import mobor.plot
import mobor.stats
import mobor.analyse_entropies_ngram

# TODO: add function for collect list of installed lexibank datasets


def register(parser):
    parser.add_argument(
        "--language",
        default="English",
        help='sets the ID of the language ("doculect") to be analyzed',
        type=str,
    )

    parser.add_argument(
        "--sequence",
        default="formchars",
        help="sets the column for the analysis: formchars, tokens, sca",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        default="wold",
        help="sets the Lexibank dataset for analysis (must have been "
        "installed beforehand)",
        type=str,
    )
    # TODO: should be allow for loan word basis as well?
    parser.add_argument(
        "--basis",
        default="all",
        help="whether to use 'all', 'native' only or 'loan' only as "
        "training set",
        type=str,
    )

    parser.add_argument(
        "--graphlimit",
        default=None,
        help="upper limit to set on entropy distribution graph ",
        type=float,
    )

    parser.add_argument(
        "-n",
        default=1000,
        help="sets the number of iterations (the larger, the better "
        "and the slower)",
        type=int,
    )

    parser.add_argument(
        "--order", default=3, help="sets the ngram order", type=int
    )

    parser.add_argument(
        "--test",
        choices=["t", "ks", "md"],
        default="ks",
        type=str,
        help="sets the type of test (`t` for 2-sample student, `ks` for "
        "two sample Kolmogorov-Schmirnoff, `md` for mean difference)",
    )

    parser.add_argument(
        "--method",
        choices=["kni", "wbi", "lp", "ls", "mle"],
        default="kni",
        type=str,
        help="sets the smoothing method to use ("
        "`kni` is for interpolated Kneser-Ney, "
        "`wbi` is for interpolated Witten-Bell, "
        "`lp` is for Laplace, "
        "`ls` is for Lidstone, "
        "`mle` is for Maximum-Likelihood Estimation)",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help='set the smoothing method value ("gamma")',
    )

    parser.add_argument(
        "-o",
        "--output",
        help="sets the output directory (default to `output/` in "
        "main directory)",
        type=str,
    )

    parser.add_argument(
        "-k",
        "--kfold",
        default=1,
        help="sets the number of splits between train and validation datasets",
        type=int,
    )

def run(args):

    # print('** new route **')
    # Build output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path(__file__).parent.parent.parent.parent / "output"
    print(output_path)

    # Load data
    wordlist = mobor.data.load_data(args.dataset)

    # Subset data and select only borrowed items (suited for WOLD)
    # TODO: replace hardcoded selector, allow no selector
    # could include criterion as argument to permit other than 0.375
    # could allow for lambda function as well
    subset = wordlist.get_language(
        args.language, [args.sequence, "borrowed"], dtypes=[list, float]
    )
    tokens = [row[args.sequence] for row in subset]
    selector = [row["borrowed"] < 0.5 for row in subset]

    if args.kfold <=1:
        if args.basis == 'all':
            # Run analysis
            # TODO: decide on allowing not `logebase` from command line
            logebase = True
            mobor.analyse_entropies_ngram.analyze_word_distributions(
                tokens,
                selector,
                output_path,
                sequence=args.sequence,
                dataset=args.dataset,
                language=args.language,
                method=args.method,
                smoothing=args.smoothing,
                order=args.order,
                graphlimit=args.graphlimit,
                test=args.test,
                n=args.n,
                logebase=logebase,
            )
        else:
            # Run analysis
            # TODO: decide on whether to allow for loan word basis as well.
            logebase = True
            mobor.analyse_entropies_ngram.analyze_word_distributions_native_basis(
                tokens,
                selector,
                output_path,
                sequence=args.sequence,
                dataset=args.dataset,
                language=args.language,
                method=args.method,
                smoothing=args.smoothing,
                order=args.order,
                graphlimit=args.graphlimit,
                test=args.test,
                n=args.n,
                logebase=logebase,
            )
    else:  # >1  so kfold.
        print("kfold validation on entropy distributions not imlemented.")
