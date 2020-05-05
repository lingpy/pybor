#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:05:45 2020

@author: johnmiller
"""

# Import Python standard libraries
import math
import numpy as np
import scipy

import mobor.data
import mobor.plot
import mobor.stats
from mobor.markov import MarkovCharLM

# TODO: add function for collect list of installed lexibank datasets


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

    #print('** new route **')
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

    graphlimit = graphlimit or max([max(loan_entropies), max(native_entropies)])+1
    entropies_plot = output_path / filename
    mobor.plot.graph_word_distribution_entropies(
        native_entropies,
        loan_entropies,
        entropies_plot.as_posix(),
        title=f"{language} native and loan entropy distribution - undifferentiated fit",
        graphlimit=graphlimit,
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

    graphlimit = graphlimit or max([max(loan_entropies), max(native_entropies)])+1
    entropies_plot = output_path / filename
    mobor.plot.graph_word_distribution_entropies(
        native_entropies,
        loan_entropies,
        entropies_plot.as_posix(),
        title=f"{language} native and loan entropy distribution - native basis fit",
        graphlimit=graphlimit,
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

#### ******************************************
####
#### Validation functions for entropy analysis.
####
#### ******************************************

def validate_kfold_entropy_distributions_all_basis(
    tokens,
    output_path="",
    sequence="formchars",
    dataset="",
    language="unknown",
    method="kni",
    smoothing=0.5,
    order=3,
    logebase=True,
    kfold=10
):

    results = kfold_entropy(tokens, method=method, smoothing=smoothing,
                           order=order, kfold=kfold, logebase=logebase)
    print_kfold_entropy(results)


def validate_kfold_entropy_distributions_native_basis(
    tokens,
    selector,
    output_path="",
    sequence="formchars",
    dataset="",
    language="unknown",
    method="kni",
    smoothing=0.5,
    order=3,
    logebase=True,
    kfold=10
):

    tokens = [token for token, select in zip(tokens, selector) if select]
    results = kfold_entropy(tokens, method=method, smoothing=smoothing,
                           order=order, kfold=kfold, logebase=logebase)
    print_kfold_entropy(results)


def print_kfold_entropy(results):
    sample = results[0]
    summary = results[1]
    print(
        f"Sample={sample[0]}, k-fold={sample[1]}, val={sample[2]}, model={sample[3]}, order={sample[4]}, smoothing={sample[5]}."
    )

    print(f"Statistic: Train mean Train stdev    Val mean  Val stdev")
    print(
        f"Mean        {summary[0, 0]:9.3f}   {summary[0, 1]:9.3f}   {summary[0, 2]:9.3f}  {summary[0, 3]:9.3f}"
    )
    print(
        f"StdDev      {summary[1, 0]:9.4f}   {summary[1, 1]:9.4f}   {summary[1, 2]:9.4f}  {summary[1, 3]:9.4f}"
    )
    print(
        f"StdErr      {summary[2, 0]:9.5f}   {summary[2, 1]:9.5f}   {summary[2, 2]:9.5f}  {summary[2, 3]:9.5f}"
    )


def kfold_entropy(tokens,
                  method="kni",
                  smoothing=0.5,
                  order=3,
                  kfold=10,
                  logebase=True):

    # TODO Use scipy kfsplit instead of homegrown solution.
    sz = len(tokens)
    frac = 1 / kfold
    tokens = np.random.permutation(tokens)

    entropy_stats = []
    for i in range(kfold):
        # Process k_fold train - val sets.
        val_begin = math.ceil(i * sz * frac)
        val_end = math.ceil((i + 1) * sz * frac)
        val_tokens = tokens[val_begin:val_end]
        train_tokens = np.concatenate(
            (tokens[:val_begin], tokens[val_end:])
        )

        # Train on Markov model.
        # Calculate entropy distributions for train and val.
        entropies = entropy_train_val(
            train_tokens, val_tokens, method, smoothing,  order, logebase,
        )

        entropy_stats.append(entropies)

    entropy_stats = np.array(entropy_stats)

    summary = np.zeros((3, 4))
    for j in range(4):
        stat_values = entropy_stats[:, j]
        # Calculate mean, standard deviation, and standard error of mean
        summary[0, j] = np.mean(stat_values)
        summary[1, j] = np.std(stat_values)
        summary[2, j] = scipy.stats.sem(stat_values)

    return [
        [sz, kfold, math.ceil(sz * frac), method, order, smoothing],
        summary,
    ]


def entropy_train_val(train_tokens=None, val_tokens=None, method="kni",
                      smoothing=0.5, order=3, logebase=True
):

    # train_tokens - in space segmented form.
    # val_tokens - in space segmented form.
    # order - model order as used in NLTK - default is 3.
    # default model is Kneser Ney.
    # Kneser Ney smoothing for entropy model.

    mlm = MarkovCharLM(
        train_tokens, model=method, order=order, smoothing=smoothing
    )
    train_entropies = mlm.analyze_tokens(train_tokens)
    # Validate language model. Errors if not sufficient data for vocabulary.
    val_entropies = mlm.analyze_tokens(val_tokens)
    if logebase:
        log2ofe = math.log2(math.e)
        train_entropies = [entropy / log2ofe for entropy in train_entropies]
        val_entropies = [entropy / log2ofe for entropy in val_entropies]

    return [
        np.mean(train_entropies),
        np.std(train_entropies),
        np.mean(val_entropies),
        np.std(val_entropies),
    ]
