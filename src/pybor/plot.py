from matplotlib import pyplot as plt
import statistics
import numpy as np

def graph_word_distribution_entropies(
    entropies1=None,
    entropies2=None,
    output_path=None,
    title='',
    label1='',
    label2='',
    graph_limit=None,
):
    # entropies1.
    # entropies2.
    # language - name of language for identification in figures and reports.
    # title - title for graph.
    # graphlimit - upper graph limit for histogram bins.


    cnt1 = f"{len(entropies1):6d}"
    avg1 = f"{statistics.mean(entropies1):6.3f}"
    std1 = f"{statistics.stdev(entropies1):6.3f}"
    cnt2 = f"{len(entropies2):6d}"
    avg2 = f"{statistics.mean(entropies2):6.3f}"
    std2 = f"{statistics.stdev(entropies2):6.3f}"

    limit = graph_limit if graph_limit is not None else (
            max(max(entropies1[:-2]), max(entropies2[:-2])))

    # Set frame horizontal for this measure.
    bins = np.linspace(0, limit, 60)
    plt.figure(figsize=(8, 5))
    plt.hist(
        entropies1,
        bins,
        alpha=0.65,
        label=label1
        + r"$(n="
        + cnt1
        + ", \mu="
        + avg1
        + ", \sigma="
        + std1
        + ")$",
        color="blue",
    )
    plt.hist(
        entropies2,
        bins,
        alpha=0.65,
        label=label2
        + r"$(n="
        + cnt2
        + ", \mu="
        + avg2
        + ", \sigma="
        + std2
        + ")$",
        color="red",
    )
    plt.grid(axis="y", alpha=0.8)
    plt.legend(loc="upper right")

    plt.xlabel("Entropies")
    plt.ylabel("Frequency")
    plt.title(title)

    # Build file output and write
    plt.savefig(output_path, dpi=600)

    plt.close()


# def draw_dist(x, output_path, title="Distribution of Statistic"):
#     cnt = f"{len(x):6d}"
#     avg = f"{np.mean(x):9.4f}"
#     std = f"{np.std(x):9.4f}"
#     # An "interface" to matplotlib.axes.Axes.hist() method
#     plt.figure(figsize=(8, 5))

#     n, bins, patches = plt.hist(
#         x=x, bins="auto", color="#0504aa", alpha=0.75, rwidth=0.85
#     )
#     plt.grid(axis="y", alpha=0.75)
#     plt.xlabel("Statistic")
#     plt.ylabel("Frequency")
#     plt.title(
#         title + r" $(n=" + cnt + ", \mu=" + avg + ", \sigma=" + std + ")$"
#     )
#     maxfreq = n.max()
#     # Set a clean upper y-axis limit.
#     plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

#     # Build file output and write
#     plt.savefig(output_path, dpi=600)

#     plt.close()
