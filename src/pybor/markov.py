#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect loan words based on word entropies as calculated by Markov word model.
Support for model trained on both native and borrowed words (dual model),
and for model trained on just native words (native model).
"""
import math
from sklearn.model_selection import train_test_split

import pybor.markov_nltk as mk
import pybor.evaluate as evaluate

# =============================================================================
#
# Dual model approach - native and loan models compete for prediction.
#
# =============================================================================

class DualMarkov:
    def __init__(self, data, method='kni', order=3, smoothing=0.5):
        """
        Fit dual Markov models - one to native data, another to loan data.

        Parameters
        ----------
        data : [[str, [str], int]]
            List of language tokens in format:
                identifier,
                token as list of character segments,
                binary (0, 1) indicator of borrowed word status.
        method : str, optional
            Model type from MarkovWord class. The default is 'kni'.
        order : int, optional
            Order from MarkovWord class. The default is 3.
        smoothing : float, optional
            smoothing from MarkovWord class. The default is 0.5.

        Returns
        -------
        DualMarkov object reference.

        """

        nativetokens = [token for _, token, status in data if status]
        self.nativemodel = mk.MarkovWord(
            nativetokens, model=method, order=order, smoothing=smoothing)

        loantokens = [token for _, token, status in data if status]
        self.loanmodel = mk.MarkovWord(
            loantokens, model=method, order=order, smoothing=smoothing)

    def predict_tokens(self, tokens):
        """
        Predict loan word statuses for a list of tokens.

        Parameters
        ----------
        tokens : [[str]]
            List of tokens as list of character segments.

        Returns
        -------
        predictions : int (binary 0,1)
            List of loan word predictions corresponding to tokens.

        """
        nativeentropies = self.nativemodel.calculate_entropies(tokens)
        loanentropies = self.loanmodel.calculate_entropies(tokens)
        predictions = [int(loanS>nativeS) for nativeS, loanS
                       in zip(nativeentropies, loanentropies)]
        return predictions

    def predict_data(self, data):
        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        return self.predict_data([token])[0]


# =============================================================================
#
#  Validate loan detection - dual model effectiveness.
#
# =============================================================================

def validate_loan_detection_dual_basis(data, method='kni', smoothing=0.5,
                                       order=3, trainfrac=0.8):

    train_data, test_data = train_test_split(data, test_size=1-trainfrac)

    dual_model = DualMarkov(train_data, method=method,
                            order=order, smoothing=smoothing)

    print("Evaluate train dataset.")
    predictions = dual_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    evaluate.print_evaluation(train_metrics)

    print("Evaluate test dataset.")
    predictions = dual_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    evaluate.print_evaluation(test_metrics)

    return dual_model, test_metrics


# =============================================================================
#
# Native model approach - entropies are tested versus a numerical limit.
#
# =============================================================================

class NativeMarkov:
    def __init__(self, data, method='kni', order=3, smoothing=0.5, p=0.995):

        nativetokens = [token for _, token, status in data if not status]
        self.nativemodel = mk.MarkovWord(
            nativetokens, model=method, order=order, smoothing=smoothing)
        nativeentropies = self.nativemodel.calculate_entropies(nativetokens)
        self.ref_limit = calculate_empirical_ref_limit(nativeentropies, frac=p)

    def predict_tokens(self, tokens):
        entropies = self.nativemodel.calculate_entropies(tokens)
        predictions = [int(tokenS>self.ref_limit) for tokenS in entropies]
        return predictions

    def predict_data(self, data):
        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        return self.predict_data([token])[0]


def calculate_empirical_ref_limit(entropies, frac=0.995):
    # Entropies are not power law distributed, but neither are they Gaussian.
    # Better to use fraction of distribution rather than Gaussian z value
    # as cut-point for discriminating between native and loan.
    entropies = sorted(entropies)
    idx = min((len(entropies)-1)*frac, len(entropies)-1)
    return (entropies[math.floor(idx)]+entropies[math.ceil(idx)])/2


# =============================================================================
#
#  Validate loan detection - native model effectiveness.
#
# =============================================================================
def validate_loan_detection_native_basis(data, method='kni', smoothing=0.5,
                                    order=3, p=0.995, trainfrac=0.8):

    train_data, test_data = train_test_split(data, test_size=1-trainfrac)

    native_model = NativeMarkov(train_data, method=method,
                                order=order, smoothing=smoothing, p=p)

    print("Evaluate train dataset.")
    predictions = native_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    evaluate.print_evaluation(train_metrics)

    print("Evaluate test dataset.")
    predictions = native_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    evaluate.print_evaluation(test_metrics)

    return native_model, test_metrics


