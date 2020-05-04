"""
Build results of a distribution analysis.
"""

# Import Python standard libraries
from pathlib import Path

import mobor.data
import mobor.detect_borrowing_ngram

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
        help="sets the column for the analysis",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        default="wold",
        help="sets the Lexibank dataset for analysis (must have been "
        "installed beforehand)",
        type=str,
    )

    parser.add_argument(
        "--basis",
        default="all",
        help="whether to use 'native' only or 'native-loan' as "
        "training set",
        type=str,
    )


    parser.add_argument(
        "--order", default=3, help="sets the ngram order", type=int
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
        "--trainfrac",
        type=float,
        default=0.8,
        help='set the training fraction',
    )

    parser.add_argument(
        "-o",
        "--output",
        help="sets the output directory (default to `output/` in "
        "main directory)",
        type=str,
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
    borrowedscore = [row["borrowed"] for row in subset]

    if args.basis == 'native':
        # Run analysis
        mobor.detect_borrowing_ngram.detect_native_loan_dual_basis(
            tokens,
            borrowedscore,
            output_path,
            method=args.method,
            smoothing=args.smoothing,
            order=args.order,
            trainfrac=args.trainfrac
        )
