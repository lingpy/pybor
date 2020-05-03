#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:36:58 2020

@author: johnmiller

Demonstrate access to function methods within commands.
"""

from pathlib import Path
import mobor.data
import mobor.commands.analyse_distr as analyse_distr

output_path = Path('.') / "output"
print(output_path)

# Load data
wordlist = mobor.data.load_data("wold")
subset = wordlist.get_language(
    "English", ["formchars", "borrowed"], dtypes=[list, float]
)
tokens = [row["formchars"] for row in subset]
selector = [row["borrowed"] < 0.375 for row in subset]

logebase = True
analyse_distr.analyze_word_distributions(
    tokens,
    selector,
    output_path,
    sequence="formchars",
    dataset="wold",
    language="English",
    method="kni",
    smoothing=0.5,
    order=3,
    graphlimit=4,
    test="ks",
    n=200,
    logebase=logebase,
    )
