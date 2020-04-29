import random

import numpy as np

import math
import statistics
from collections import namedtuple


# Utility functions for loan word discrimination

# TODO: allow to set seed?
def train_test_split(table, trainfrac=0.80):

    train = []
    test = []
    for entry in table:
        if random.random() < trainfrac:
            train.append(entry)
        else:
            test.append(entry)

    #    sz = len(table.index)
    #    trainCnt = math.ceil(sz*trainfrac)
    #    selector = pd.Series([True]*trainCnt + [False]*(sz-trainCnt))
    #    selector = np.random.permutation(selector)
    #    train = table[selector==True]
    #    test = table[selector==False]
    print(f"num train={len(train):5d}, num test={len(test):5d}")

    return train, test


# Measure classification performance.

# 4.3 Calculate evaluation metrics.
Test_prediction = namedtuple(
    "Binary_prediction", ["Acc", "Maj_acc", "Prec", "Recall", "F1"]
)


def report_metrics(ground, forecast):
    import sklearn.metrics as metrics

    # ground and entropies are on token basis.
    assert len(ground) == len(
        forecast
    ), f"ground size ({len(ground)}) not equal forecast size ({len(forecast)})"
    # forecast is versus the upper ref_limit.  Most should be < the limit and so native words.
    # forecast = [e < self._ref_limit for e in self._entropies]
    # print('Test for entropy <', self._ref_limit)
    prec_recall_f1 = metrics.precision_recall_fscore_support(
        ground, forecast, average="binary"
    )[:-1]
    print("precision, recall, F1 =", prec_recall_f1)
    acc = metrics.accuracy_score(ground, forecast)
    print("n =", len(ground), " accuracy =", acc)
    print(
        "confusion matrix: tn, fp, fn, tp",
        metrics.confusion_matrix(ground, forecast).ravel(),
    )
    maxpredict = max(sum(ground), len(ground) - sum(ground))
    maj_acc = maxpredict / len(ground)
    print("Predict majority: accuracy=", maj_acc)

    return Test_prediction(
        Acc=acc,
        Maj_acc=maj_acc,
        Prec=prec_recall_f1[0],
        Recall=prec_recall_f1[1],
        F1=prec_recall_f1[2],
    )


###
def calculate_ref_limit(entropies, z=2.58):
    avg = statistics.mean(entropies)
    stdev = statistics.stdev(entropies)
    ref_limit = avg + z * stdev
    print(
        f"Native avg={avg:.3f}, stdev={stdev:.3f}, z={z:.3f}, ref limit={ref_limit:.3f}"
    )
    return ref_limit


def calculate_empirical_ref_limit(entropies, frac=0.99):
    # Entropies are not power law distributed, but neither are they Gaussian.
    # Better to use fraction of distribution to use as cut-point for discriminating between native and loan.
    idx = min((len(entropies) - 1) * frac, len(entropies) - 1)
    entropies = np.sort(entropies)
    ref_limit = (entropies[math.floor(idx)] + entropies[math.ceil(idx)]) / 2
    avg = statistics.mean(entropies)
    stdev = statistics.stdev(entropies)
    print(f"Native avg={avg:.3f}, stdev={stdev:.3f}")
    print(f"fraction {frac:.3f}, idx {idx:.2f}, ref limit={ref_limit:.3f}")
    return ref_limit


# Utility functions for graphing distributions_from_metadata(import numpy as np
import matplotlib.pyplot as plt
import statistics

# Draw histogram of log probabilities from language model.
def draw_lnprob_dist(lnprobs, order=None, file=None):
    cnt = f"{len(lnprobs):6d}"
    avg = f"{statistics.mean(lnprobs):9.4f}"
    std = f"{statistics.stdev(lnprobs):9.4f}"
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure(figsize=(8, 5))

    n, bins, patches = plt.hist(
        x=lnprobs, bins="auto", color="#0504aa", alpha=0.75, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel(
        "Log2 probability" + (" -- Markov order " + str(order) if order else "")
    )
    plt.ylabel("Frequency of form")
    plt.title(
        "Log2 probability "
        + r"$(n="
        + cnt
        + ", \mu="
        + avg
        + ", \sigma="
        + std
        + ")$"
    )
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if file:
        plt.savefig(file + str(order) + ".pdf", dpi=600)
    plt.show()
    plt.close()


# Draw histogram of statistic.
# Used in randomization tests.


def draw_dist(x, title="Distribution of Statistic"):
    cnt = f"{len(x):6d}"
    avg = f"{statistics.mean(x):9.4f}"
    std = f"{statistics.stdev(x):9.4f}"
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure(figsize=(8, 5))

    n, bins, patches = plt.hist(
        x=x, bins="auto", color="#0504aa", alpha=0.75, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Statistic")
    plt.ylabel("Frequency")
    plt.title(
        title + r" $(n=" + cnt + ", \mu=" + avg + ", \sigma=" + std + ")$"
    )
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.close()
