import scipy
import numpy as np

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
