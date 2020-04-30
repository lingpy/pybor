from matplotlib import pyplot as plt
import numpy as np
import statistics


def graph_word_distribution_entropies(
    native_entropies,
    loan_entropies,
    output_path,
    language="unknown",
    title="title",
    graphlimit=5,
    figurequal="",
):
    # entropies.
    # selector - which tokens to use for native versus loan.
    # figuredir - directory to put .pdf of histogram.
    # language - name of language for identification in figures and reports.
    # title - title for graph.
    # graphlimit - upper graph limit for histogram bins.

    # selector = selector.tolist() # just in case it is Pandas series.
    # Divide into native and loan entropies.
    # native_entropies = [entropy for entropy, select in zip(entropies, selector) if select==True]
    # loan_entropies = [entropy for entropy, select in zip(entropies, selector) if select==False]

    native_cnt = f"{len(native_entropies):6d}"
    native_avg = f"{np.mean(native_entropies):9.4f}"
    native_std = f"{np.std(native_entropies):9.4f}"
    loan_cnt = f"{len(loan_entropies):6d}"
    loan_avg = f"{np.mean(loan_entropies):9.4f}"
    loan_std = f"{np.std(loan_entropies):9.4f}"

    # Set frame horizontal for this measure.
    bins = np.linspace(1, graphlimit, 60)
    plt.figure(figsize=(8, 5))
    plt.hist(
        native_entropies,
        bins,
        alpha=0.65,
        label="native entropies"
        + r"$(n="
        + native_cnt
        + ", \mu="
        + native_avg
        + ", \sigma="
        + native_std
        + ")$",
        color="blue",
    )
    plt.hist(
        loan_entropies,
        bins,
        alpha=0.65,
        label="loan entropies"
        + r"$(n="
        + loan_cnt
        + ", \mu="
        + loan_avg
        + ", \sigma="
        + loan_std
        + ")$",
        color="red",
    )
    plt.grid(axis="y", alpha=0.8)
    plt.legend(loc="upper right")

    plt.xlabel("Entropies")
    plt.ylabel("Frequency")
    plt.title(title)

    # Build file output and write
    output_file = language + figurequal + ".pdf"
    output_filepath = output_path / output_file
    plt.savefig(output_filepath.as_posix(), dpi=600)

    plt.close()


def draw_dist(x, output_path, title="Distribution of Statistic"):
    cnt = f"{len(x):6d}"
    avg = f"{np.mean(x):9.4f}"
    std = f"{np.std(x):9.4f}"
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

    # Build file output and write
    output_file = title + ".pdf"
    output_filepath = output_path / output_file
    plt.savefig(output_filepath.as_posix(), dpi=600)

    plt.close()
