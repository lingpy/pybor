#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Markov module of detection of loan words.
"""

import statistics

from pybor.markov import DualMarkov, validate_loan_detection_dual_basis
from pybor.markov import NativeMarkov, validate_loan_detection_native_basis
from pybor.dev.data import testing, training
import pybor.evaluate as evaluate
from pybor.data import LexibankDataset


def test_DualMarkov_predict_data():
    print('\nTest dual Markov predict data')
    dm = DualMarkov(training)

    print('Training set predictions')
    predictions = dm.predict_data(training)
    tp, tn, fp, fn = evaluate.false_positive(predictions, training, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')

    results =  evaluate.evaluate_model(predictions, training)
    evaluate.print_evaluation(results)

    print('Testing set predictions')
    predictions = dm.predict_data(testing)
    tp, tn, fp, fn = evaluate.false_positive(predictions, testing, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')
    results =  evaluate.evaluate_model(predictions, testing)
    evaluate.print_evaluation(results)

def test_NativeMarkov_predict_data():
    print('\nTest native Markov predict data')
    nm = NativeMarkov(training)

    print('Training set predictions')
    predictions = nm.predict_data(training)
    tp, tn, fp, fn = evaluate.false_positive(predictions, training, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')

    results =  evaluate.evaluate_model(predictions, training)
    evaluate.print_evaluation(results)

    print('Testing set predictions')
    predictions = nm.predict_data(testing)
    tp, tn, fp, fn = evaluate.false_positive(predictions, testing, True)
    #print(f'tp, tn, fp, fn = {(tp, tn, fp, fn)}')
    results =  evaluate.evaluate_model(predictions, testing)
    evaluate.print_evaluation(results)


def test_validate_loan_detection_dual_basis():
    print('\nTest validate loan detection dual basis')
    validate_loan_detection_dual_basis(training)

def test_validate_loan_detection_native_basis():
    print('\nTest validate loan detection native basis')
    validate_loan_detection_native_basis(training)

def test_validate_loan_detection_dual_basis_lexibank():
    print('\nTest validate loan detection native basis')
    ds = LexibankDataset('wold', transform=
        {"Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0})
    table = ds.get_table(
        language='English', form='FormChars', classification='Loan')
    assert table
    print(f'table size={len(table)}')

    validate_loan_detection_dual_basis(table)



def get_tokens(data):
    return [token for _, token, _ in data]

if __name__ == "__main__":
    test_DualMarkov_predict_data()
    test_NativeMarkov_predict_data()
    test_validate_loan_detection_dual_basis()
    test_validate_loan_detection_native_basis()
    #test_validate_loan_detection_dual_basis_lexibank()