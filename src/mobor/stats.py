import scipy
import numpy as np
import collections

from mobor.markov import MarkovCharLM

Statistic_Name = {
    "t": "Two Sample Student's t -- Unequal Variances",
    "ks": "Two Sample Kolmogorov Schmirnoff",
    "md": "Mean Difference Between Samples",
}

MeanDif = collections.namedtuple("MeanDif", ["dif", "x", "y"])


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

    x = [value for value, select in zip(values, selector) if select]
    y = [value for value, select in zip(values, selector) if not select]

    stat_ref = calculate_test_statistic_between_distributions(x, y, test=test)[0]

    stats = [0] * n
    for i in range(n):
        selector = np.random.permutation(selector)
        x = [value for value, select in zip(values, selector) if select]
        y = [
            value for value, select in zip(values, selector) if not select
        ]

        test_stat = calculate_test_statistic_between_distributions(
            x, y, test=test
        )
        stats[i] = test_stat[0]

    count = sum([val < stat_ref for val in stats])
    prob = (count + 0.5) / (len(stats) + 1)

    return stat_ref, 1 - prob, stats



## Randomization test between distributions with separate estimator per randomization.
# Returns new statistical estimate for each randomization.
def calculate_differentiated_randomization_test_between_distributions(
    tokens, selector, order=3, method="kni", smoothing=0.5, test="ks", n=200
):
    # ts: t, ks, md
    # Initial selector value defines the reference condition.
    teststat = analyze_language_tokens_test(
        tokens=tokens,
        selector=selector,
        method=method,
        order=order,
        smoothing=smoothing,
        test=test,
    )
    stat_ref = teststat[0]

    stats = [0] * n
    for i in range(n):
        selector = np.random.permutation(selector)
        teststat = analyze_language_tokens_test(
            tokens=tokens,
            selector=selector,
            method=method,
            order=order,
            smoothing=smoothing,
            test=test,
        )
        stats[i] = teststat[0]

    count = sum([val < stat_ref for val in stats])
    prob = (count + 0.5) / (len(stats) + 1)

    return stat_ref, 1 - prob, stats


def analyze_language_tokens_test(
    tokens, selector, method, order, smoothing, test
):
    # Without selector, there is no test.
    if selector is None:
        return None

    # tokens - in space segmented form.
    # selector - which tokens to use for indicator of native tokens.
    # method - model estimation method - default is kni.
    # order - model ngram order.
    # test - statistic: ks=Kolmogorov-Smirnov, t=Student's t, md=Mean difference

    # Selector splits corpus into train and test.
    train_tokens = []
    val_tokens = []
    for entry, split in zip(tokens, selector):
        if split:
            train_tokens.append(entry)
        else:
            val_tokens.append(entry)

    # Construct and fit language model.
    mlm = MarkovCharLM(
        train_tokens, model=method, order=order, smoothing=smoothing
    )
    train_entropies = mlm.analyze_training()

    # Validate language model.
    val_entropies = mlm.analyze_tokens(val_tokens)

    # Compute test statistic.
    stat = calculate_test_statistic_between_distributions(
        train_entropies, val_entropies, test=test
    )
    return stat
