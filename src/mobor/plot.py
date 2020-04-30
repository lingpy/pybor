from matplotlib import pyplot as plt
import numpy as np
import statistics


def plot_word_distributions(
    native,
    loan,
    filename,
    title="[untitled]",
    graphlimit=5,
    figsize=(8, 5),
    alpha_native=0.65,
    alpha_loan=0.65,
    label_native="native entropies",
    label_loan="loan entropies",
    color_native="blue",
    color_loan="red",
    xlabel="Entropies",
    ylabel="Frequencies",
):

    # Calculate basis for labels.
    native_cnt = f"{len(native):6d}"
    native_avg = f"{statistics.mean(native):9.4f}"
    native_std = f"{statistics.stdev(native):9.4f}"
    loan_cnt = f"{len(loan):6d}"
    loan_avg = f"{statistics.mean(loan):9.4f}"
    loan_std = f"{statistics.stdev(loan):9.4f}"

    # Set frame horizontal for this measure.
    bins = np.linspace(1, graphlimit, 60)

    plt.figure(figsize=figsize)
    plt.hist(
        native,
        bins,
        alpha=alpha_native,
        label=label_native
        + r"$(n="
        + native_cnt
        + ", \mu="
        + native_avg
        + ", \sigma="
        + native_std
        + ")$",
        color=color_native,
    )
    plt.hist(
        loan,
        bins,
        alpha=alpha_loan,
        label=label_loan
        + r"$(n="
        + loan_cnt
        + ", \mu="
        + loan_avg
        + ", \sigma="
        + loan_std
        + ")$",
        color=color_loan,
    )

    plt.grid(axis="y", alpha=0.8)
    plt.legend(loc="upper right")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(filename)
