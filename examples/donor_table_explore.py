#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:52:13 2020

@author: johnmiller
"""
from sys import argv
import argparse
import random
from collections import Counter
import csv
import pybor.wold as wold

def get_donor(table, donor_num=1):
    if donor_num <= 0: return ''
    # Count and order the language donors.
    donors = Counter([row[3] for row in table if row[2]==1]).most_common()
    if not donors or len(donors) == 0: return ''
    # Drop '' and 'Unidentified'.
    donors = [donor for donor in donors if donor[0] != '' and donor[0] !='Unidentified']
    if len(donors) < donor_num: return ''
    donors = donors[:4]
    print(donors)

    return donors.pop(donor_num-1)[0]


# Use this function to protype get_lead_donor_table
def get_native_donor_table(language, form='Tokens'):

    print(f'***Donors for language {language} ***')
    table = wolddb.get_donor_table(language=language, form=form, classification="Borrowed")

    donor = get_donor(table, 1)
    print(f'Lead donor for {language} is {donor}.')

    # Now select the rows containing this donor.
    donor_table = [[row[0], row[1], row[2]] for row in table
                        if row[2] == 1 and donor in row[3]]
    # Get this just to count
    native_donor_ids = [row[0] for row in table
                          if row[2] == 0 and donor in row[3]]

    native_table = [[row[0], row[1], row[2]] for row in table if row[2] == 0]
    print(f'Original table size {len(table)}, ' +
          f'Number borrowed from {donor} is {len(donor_table)}, ' +
          f'and native from {donor} is {len(native_donor_ids)}, ' +
          f'and all native is {len(native_table)}.')

    return random.sample(native_table + donor_table, len(native_table)+len(donor_table))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #languages = ["English", "Swahili", "Hup", "Oroqen", "Imbabura Quechua"]  # 'all' or language name
    #form = "Tokens"  # one in ["Tokens", "FormChars", "ASJP", "DOLGO", "SCA"],

    parser.add_argument(
        "--languages",
        nargs='*',
        type=str,
        default='all',
        help="'all' or language_names",
    )

    parser.add_argument(
        "--form",
        type=str,
        default='Tokens',
        choices=["Tokens", "FormChars", "ASJP", "DOLGO", "SCA"],
        help="Form to take from language table.",
    )
    args = parser.parse_args()
    languages = 'all' if args.languages[0] == 'all' else args.languages

    wolddb = wold.get_wold_access()
    languages = wold.check_wold_languages(wolddb, languages)

    for language in languages:
        table = get_native_donor_table(language, args.form)

        filename = "discovery/"+language+"_native_donor_table"+".csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(table)
