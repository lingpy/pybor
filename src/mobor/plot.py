from matplotlib import pyplot as plt
import numpy as np

def plot_word_distributions(
        native, 
        borrowed, 
        filename,
        title='title', 
        graphlimit=5, 
        figsize=(8, 5),
        alpha_native=0.65,
        alpha_borrowed=0.65,
        label_native='native entropies',
        label_borrowed='borrowed entropies',
        color_native='blue',
        color_borrowed='red',
        xlabel='Entropies',
        ylabel='Frequencies'
        ):

    # Set frame horizontal for this measure.
    bins = np.linspace(1, graphlimit, 60)

    plt.figure(figsize=figsize)
    plt.hist(
            native, 
            bins, 
            alpha=alpha_native,
            label=label_native,
            color=color_native
            )
    plt.hist(
            borrowed, 
            bins, 
            alpha=alpha_borrowed,
            label=label_borrowed,
            color=color_borrowed
            )

    plt.grid(axis='y', alpha=0.8)
    plt.legend(loc='upper right')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(filename)


