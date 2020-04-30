"""
Build results of a distribution analysis.
"""

# Import Python standard libraries
import math
from pathlib import Path

import mobor.data
import mobor.plot
import mobor.stats
from mobor.markov import MarkovCharLM

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


# TODO add field to output name
def run(args):

    # Build output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path(__file__).parent.parent.parent.parent / "output"
    print(output_path)

    # Load data
    wl = mobor.data.load_data(args.dataset)

    # Subset data and select only borrowed items (suited for WOLD)
    # TODO: replace hardcoded selector, allow no selector
    subset = wl.get_language(
        args.language, [args.sequence, "borrowed"], dtypes=[list, float]
    )
    tokens = [row[args.sequence] for row in subset]
    selector = [row["borrowed"] < 0.375 for row in subset]

    # Run analysis
    # TODO: decide on allowing not `logebase` from command line
    logebase = True
    mlm = analyze_word_distributions(
        tokens,
        selector,
        output_path,
        sequence=args.sequence,
        dataset=args.dataset,
        language=args.language,
        method=args.method,
        smoothing=args.smoothing,
        order=args.order,
        test=args.test,
        n=args.n,
        logebase=logebase,
    )


def analyze_word_distributions(
    tokens,
    selector,
    output_path,
    sequence="",
    dataset="",
    language="unknown",
    order=3,
    method="kni",
    smoothing=0.5,
    test="ks",
    n=1000,
    logebase=True,
):

    # tokens - in space segmented form.
    # selector - which tokens to use for indicator of likely native tokens.
    # figuredir - directory to put .pdf of histogram.
    # language - name of language for identification in figures and reports.
    # model - model estimation method - default is KNI.
    # order - model order - default is 2.
    # smoothing - Kneser Ney smoothing - default is 0.5 appropriate for this study.
    # test - test statistic for training versus val difference.
    # n - number of iterations of randomization test.

    # Build the Markov model
    # TODO: decide on `model` <-> `method` terminology
    mlm = MarkovCharLM(tokens, model=method, order=order, smoothing=smoothing)

    # Compute entropies, using logebase if requested
    entropies = mlm.analyze_training()
    if logebase:
        log2ofe = math.log2(math.e)
        entropies = [entropy / log2ofe for entropy in entropies]

    # Split native and loan entropies based on selector
    native_entropies = [
        entropy
        for entropy, select in zip(entropies, selector)
        if select == True
    ]
    loan_entropies = [
        entropy
        for entropy, select in zip(entropies, selector)
        if select == False
    ]

    # Perform randomization tests, plot distribution and write data
    (
        stat_ref,
        prob,
        plot_stats,
    ) = mobor.stats.calculate_randomization_test_between_distributions(
        entropies, selector, test, n
    )

    print(f"prob ({test} stat >= {stat_ref[0]:.5f}) = {prob:.5f}")

    filename = f"distribution.{language}-{sequence}-{order}-{method}-{smoothing}-{test}-{n}.pdf"
    dist_plot = output_path / filename
    mobor.plot.draw_dist(plot_stats, dist_plot.as_posix(), title=f"test {test}")

    # Plot entropies
    filename = (
        f"entropies.{language}-{sequence}-{order}-{method}-{smoothing}.pdf"
    )
    entropies_plot = output_path / filename
    mobor.plot.graph_word_distribution_entropies(
        native_entropies,
        loan_entropies,
        entropies_plot.as_posix(),
        language=language,
        title=f"{language} native and loan entropy distribution - undifferentiated fit",
        graphlimit=5,
    )

    # Update general results in disk
    result_file = output_path / "analysis_distribution.tsv"
    parameters = {
        "language": language,
        "sequence": sequence,
        "dataset": dataset,
        "order": order,
        "method": method,
        "smoothing": smoothing,
        "test": test,
        "n": n,
        "logebase": logebase,
    }
    results = {
        "stat_ref": "%.5f" % stat_ref[0],
        "prob": "%.5f" % prob,
        "dist_file": dist_plot.name,
        "entropies_plot": entropies_plot.name,
    }
    mobor.data.update_results(parameters, results, result_file.as_posix())
