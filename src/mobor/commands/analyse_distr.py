"""
Build results of a distribution analysis.
"""

# Import Python standard libraries
import math
from pathlib import Path

import scipy
import numpy as np
import matplotlib.pyplot as plt

import mobor.data
from mobor.markov import Markov
from mobor.markov import MarkovCharLM
import mobor.plot

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
        help="sets the Lexibank dataset for analysis (must have been installed beforehand)",
        type=str,
    )

    parser.add_argument(
        "-n",
        default=1000,
        help="sets the number of iterations (the larger, the better and the slower)",
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
        help="sets the type of test (`t` for 2-sample student, `ks` for two sample Kolmogorov-Schmirnoff, `md` for mean difference)",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="sets the output directory (default to `output/` in main directory)",
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
    # TODO: allow to decide smoothing method
    logebase = True
    mlm = analyze_word_distributions(
        tokens,
        selector,
        output_path,
        language=args.language,
        model="KNI",
        order=args.order,
        test=args.test,
        n=args.n,
        logebase=logebase,
    )


def analyze_word_distributions(
    tokens,
    selector,
    output_path,
    language="unknown",
    model="KNI",
    order=3,
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
    mlm = MarkovCharLM(tokens, model=model, order=order, smoothing=smoothing)

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

    mobor.plot.graph_word_distribution_entropies(
        native_entropies,
        loan_entropies,
        output_path,
        language=language,
        title=language
        + " native and loan entropy distribution - undifferentiated fit",
        graphlimit=5,
        figurequal="all-basis-native-loan-entropies",
    )

    # Perform randomization tests.
    # Efficient since just permutation of selector for constructing alternate test results.
    stat_ref, prob, plot_stats = calculate_randomization_test_between_distributions(
        entropies, selector, test, n)
    print(f"prob ({test} stat >= {stat_ref[0]:.5f}) = {prob:.5f}")
    mobor.plot.draw_dist(plot_stats, output_path, title=f"test {test}")

    return mlm


Statistic_Name = {
    "t": "Two Sample Student's t -- Unequal Variances",
    "ks": "Two Sample Kolmogorov Schmirnoff",
    "md": "Mean Difference Between Samples",
}


#### statistics
def calculate_test_statistic_between_distributions(x, y, test="ks"):
    # Returns test statistic.
    if test == "t":
        statistic = scipy.stats.ttest_ind(
            x, y, equal_var=False, nan_policy="propagate"
        )
    elif test == "ks":  # test == 'ks'
        statistic = scipy.stats.ks_2samp(x, y)
    elif test == "md":
        statistic = MeanDif(
            dif=np.mean(x) - np.mean(y), x=np.mean(x), y=np.mean(y),
        )
    else:
        statistic = None
    return statistic


def calculate_randomization_test_between_distributions(
    values, selector, test, n
):
    # Calculate test statistic value.
    # Repeatedly permute selector and calculate randomized test statistic.
    # Plot empirical distribution of test statistic.
    # Report empirical probability of test result.

    x = [value for value, select in zip(values, selector) if select == True]
    y = [value for value, select in zip(values, selector) if select == False]

    stat_ref = calculate_test_statistic_between_distributions(x, y, test=test)
    # print(f'{test} statistic =', stat_ref[0])

    stats = [0] * n
    for i in range(n):
        selector = np.random.permutation(selector)
        x = [value for value, select in zip(values, selector) if select == True]
        y = [
            value for value, select in zip(values, selector) if select == False
        ]

        test_stat = calculate_test_statistic_between_distributions(
            x, y, test=test
        )
        stats[i] = test_stat[0]

    count = sum([val < stat_ref[0] for val in stats])
    prob = (count + 0.5) / (len(stats) + 1)

    return stat_ref, 1 - prob, stats
