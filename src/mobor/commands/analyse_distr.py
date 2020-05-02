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


def run(args):

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
    # TODO: retain current analysis useing true for native and false for loan
    # TODO: could include criterion as argument to permit other than 0.375
    # TODO: could allow for lambda as well
    subset = wordlist.get_language(
        args.language, [args.sequence, "borrowed"], dtypes=[list, float]
    )
    tokens = [row[args.sequence] for row in subset]
    selector = [row["borrowed"] < 0.375 for row in subset]

    if args.basis == 'all':
        # Run analysis
        # TODO: decide on allowing not `logebase` from command line
        logebase = True
        analyze_word_distributions(
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
        analyze_word_distributions_native_basis(
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



def analyze_word_distributions(
    tokens,
    selector,
    output_path="",
    sequence="formchars",
    dataset="",
    language="unknown",
    method="kni",
    smoothing=0.5,
    order=3,
    graphlimit=None,
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
        if select
    ]
    loan_entropies = [
        entropy
        for entropy, select in zip(entropies, selector)
        if not select
    ]

    # Perform randomization tests, plot distribution and write data
    (
        stat_ref,
        prob,
        plot_stats,
    ) = mobor.stats.calculate_randomization_test_between_distributions(
        entropies, selector, test, n
    )

    print(f"prob ({test} stat >= {stat_ref:.5f}) = {prob:.5f}")

    filename = f"distribution.{language}-{sequence}-{order}-{method}-{smoothing}-{test}-{n}.pdf"
    dist_plot = output_path / filename
    mobor.plot.draw_dist(plot_stats, dist_plot.as_posix(),
                title=f"{language}-{sequence}-test {test}-{'native basis'}")

    # Plot entropies
    filename = (
        f"entropies.{language}-{sequence}-{order}-{method}-{smoothing}.pdf"
    )

    if graphlimit==None: gl=max([max(loan_entropies), max(native_entropies)])+1
    else: gl=graphlimit

    entropies_plot = output_path / filename
    mobor.plot.graph_word_distribution_entropies(
        native_entropies,
        loan_entropies,
        entropies_plot.as_posix(),
        title=f"{language} native and loan entropy distribution - undifferentiated fit",
        graphlimit=gl,
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
        "basis": "all",
    }
    results = {
        "stat_ref": "%.5f" % stat_ref,
        "prob": "%.5f" % prob,
        "dist_file": dist_plot.name,
        "entropies_plot": entropies_plot.name,
    }
    mobor.data.update_results(parameters, results, result_file.as_posix())


def analyze_word_distributions_native_basis(
    tokens,
    selector,
    output_path="",
    sequence="formchars",
    dataset="",
    language="unknown",
    method="kni",
    smoothing=0.5,
    order=3,
    graphlimit=None,
    test="ks",
    n=200,  # low # repetitions, but each one is expensive.
    logebase=True,
):

    # tokens - in space segmented form.
    # selector - which tokens to use for indicator of likely native tokens.
    # output_path - directory to put images.
    # sequence - sequence analyzed (form, segments, sound classes).
    # dataset - only wold supported for now.
    # language - name of language for identification in figures and reports.
    # method - model estimation method - default is kni.
    # order - model order - default is 3 grams which gives 2nd order dependency.
    # smoothing - Kneser Ney default of 0.5 is appropriate for this study.
    # test - test statistic for training versus val difference.
    # n - number of iterations of randomization test.
    # logebase - natural log basis (true) or log 2 basis (false).

    # Build the Markov model
    # TODO: decide on `model` <-> `method` terminology
    native_tokens = [
        token for token, select in zip(tokens, selector) if select == True
    ]
    loan_tokens = [
        token for token, select in zip(tokens, selector) if select == False
    ]

    mlm = MarkovCharLM(
        native_tokens, model=method, order=order, smoothing=smoothing
    )
    native_entropies = mlm.analyze_training()
    loan_entropies = mlm.analyze_tokens(loan_tokens)

    if logebase:
        log2ofe = math.log2(math.e)
        native_entropies = [entropy / log2ofe for entropy in native_entropies]
        loan_entropies = [entropy / log2ofe for entropy in loan_entropies]

    # Plot distribution, perform randomization tests, and write data
    # Plot entropies
    filename = (
        f"entropies.{language}-{sequence}-{order}-{method}-{smoothing}-{'native'}.pdf"
    )

    if graphlimit==None: gl= max([max(loan_entropies), max(native_entropies)])+1
    else: gl=graphlimit

    entropies_plot = output_path / filename
    mobor.plot.graph_word_distribution_entropies(
        native_entropies,
        loan_entropies,
        entropies_plot.as_posix(),
        title=f"{language} native and loan entropy distribution - native basis fit",
        graphlimit=gl,
    )


    # Perform randomization tests
    (
        stat_ref,
        prob,
        plot_stats,
    ) = mobor.stats.calculate_differentiated_randomization_test_between_distributions(
            tokens=tokens,
            selector=selector,
            order=order,
            method=method,
            smoothing=smoothing,
            test=test,
            n=n,
            )

    print(f"prob ({test} stat >= {stat_ref:.5f}) = {prob:.5f}")

    filename = f"distribution.{language}-{sequence}-{order}-{method}-{smoothing}-{test}-{n}-{'native'}.pdf"
    dist_plot = output_path / filename
    mobor.plot.draw_dist(plot_stats, dist_plot.as_posix(),
                title=f"{language}-{sequence}-test {test}-{'native basis'}")

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
        "basis": "native",
    }
    results = {
        "stat_ref": "%.5f" % stat_ref,
        "prob": "%.5f" % prob,
        "dist_file": dist_plot.name,
        "entropies_plot": entropies_plot.name,
    }
    mobor.data.update_results(parameters, results, result_file.as_posix())
