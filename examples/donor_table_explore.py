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
    donors = donors[:6]
    #print(donors)

    return donors.pop(donor_num-1)[0]


# Use this function to protype get_lead_donor_table
def report_native_donors(language, form='Tokens'):

    print(f'***Donors for language {language} ***')
    table = wolddb.get_donor_table(language=language, form=form, classification="Borrowed")

    donor = get_donor(table, 1)
    print(f'Lead donor for {language} is {donor}.')

    # Count the rows containing this donor.
    donor_len = len([row[0] for row in table if row[2] == 1 and donor in row[3]])

    native_len =len([row[0] for row in table if row[2] == 0])
    borrowed_len = len(table) - native_len
    donor_frac = donor_len/borrowed_len
    print(f'Language {language}, ' +
          f'Donor language {donor}, ' +
          f'Table len {len(table)}, ' +
          f'Borrowed len {borrowed_len}, ' +
          f'Native len {native_len}, ' +
          f'Lead donor len {donor_len}, ' +
          f'Fraction lead donor {donor_frac:.3f}')

    return [language, donor, len(table), borrowed_len,
            native_len, donor_len, f'{donor_frac:.4f}']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #languages = ["English", "Swahili", "Hup", "Oroqen"]  # 'all' or language name
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

    parser.add_argument(
        "--min_borrowed",
        type=int,
        default=0,
        help="Minimum borrowed quantity.",
        )

    args = parser.parse_args()
    languages = 'all' if args.languages[0] == 'all' else args.languages

    wolddb = wold.get_wold_access()
    languages = wold.check_wold_languages(wolddb, languages)

    min_borrowed = args.min_borrowed
    with open('output/language_donors_mb'+str(min_borrowed)+'.csv', "w", newline="") as fl:
        writer = csv.writer(fl)
        writer.writerow(
            ["language", "donor", "table_len", "borrowed_len",
             "native_len", "donor_len", "donor_frac"])
        for language in languages:
            language_donor_rpt = report_native_donors(language, args.form)
            if language_donor_rpt[3] >= min_borrowed:
                writer.writerow(language_donor_rpt)