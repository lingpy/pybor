#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:07:30 2020

@author: johnmiller

Explore how we might pass an uninitialized function to a shell function.
In particular how to use the data routine with any function.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:03:57 2020

@author: johnmiller

Entropy distribution analysis examples.
Maybe precursor of something more general and applicable to all modules using predict protocol.
"""
import math
import random
import statistics
import pickle
from pathlib import Path

from tabulate import tabulate

from pybor.data import LexibankDataset
from pybor.neural_tf import NeuralWord
from pybor.data_tf import NeuralData
from pybor.plot import graph_word_distribution_entropies

output_path = Path('./output').resolve()
outlier = 2.5

def describe_entropies(data):
    n = len(data)
    avg = statistics.mean(data)
    stdev = statistics.stdev(data)
    min_ = min(data)
    max_ = max(data)
    return [n, avg, stdev, min_, max_]

def print_entropy_statistics(stats=None, title='', rownames=''):
    table = []
    table.append([title, 'n', 'mean', 'stdev', 'min', 'max'])
    for name, row in zip(rownames, stats):
        table.append([name]+row)
    print(tabulate(table, tablefmt='pipe', headers='firstrow', floatfmt='.3f'))

def report_entropy_statistics(model=None, repo=None):
    # Report descriptive statistics.
    stats = []
    trainH = model.calculate_entropies(repo.train_tokens_ids)
    stats.append(describe_entropies(trainH))
    valH = model.calculate_entropies(repo.val_tokens_ids)
    stats.append(describe_entropies(valH))
    trainvalH = model.calculate_entropies(repo.trainval_tokens_ids)
    stats.append(describe_entropies(trainvalH))
    testH = model.calculate_entropies(repo.test_tokens_ids)
    stats.append(describe_entropies(testH))
    return stats

def calculate_entropies_str(model=None, tokens=None, vocab=None):
    tokens_ids = NeuralData.get_tokens_ids(tokens, vocab)
    entropies =  model.calculate_entropies(tokens_ids)
    extremes = [(token, entropy) for token, entropy in zip(tokens, entropies) if entropy>outlier]
    print('extremes:', extremes)
    return entropies

def report_entropy_statistics_str(model=None, repo=None):
    # Report descriptive statistics.
    stats = []
    trainH = calculate_entropies_str(model, repo.train_tokens, repo.symbol_to_id)
    stats.append(describe_entropies(trainH))
    valH = calculate_entropies_str(model, repo.val_tokens, repo.symbol_to_id)
    stats.append(describe_entropies(valH))
    trainvalH = calculate_entropies_str(model, repo.trainval_tokens, repo.symbol_to_id)
    stats.append(describe_entropies(trainvalH))
    testH = calculate_entropies_str(model, repo.test_tokens, repo.symbol_to_id)
    stats.append(describe_entropies(testH))
    return stats

def graph_entropies(tokens1=None, tokens2=None, vocab=None, model=None,
                    title='', label1='', label2='', filename=''):
    # Plot train and loan entropy distributions - all tokens basis
    tokens_ids1 = NeuralData.get_tokens_ids(tokens1, vocab)
    entropies1 = model.calculate_entropies(tokens_ids1)

    tokens_ids2 = NeuralData.get_tokens_ids(tokens2, vocab)
    entropies2 = model.calculate_entropies(tokens_ids2)
    # Could use mean + 6*stdev + 1 as upper limit for graphs.
    graph_path = output_path / filename
    graph_word_distribution_entropies(
            entropies1, entropies2, graph_path.as_posix(),
            title=title, label1=label1, label2=label2)


def simple_train_test_split(table=None, test_split=0.15):
    table = random.sample(table, len(table))
    test_split = int(test_split) if test_split >= 1 else math.ceil(len(table)*test_split)
    train, test = table[:-test_split], table[-test_split:]
    print(f'table len={len(table)}, train len={len(train)}, test len={len(test)}.')
    return train, test

def print_tokens_from_repo(repo=None):
    # First try with test.
    test_tokens_ids = repo.test_tokens_ids
    tokens = NeuralData.get_ids_tokens(test_tokens_ids, repo.id_to_symbol)
    print('Test tokens:', tokens)

def describe_neural_entropies_for_basis(
        trainval=None, test=None, trainval_x=None, test_x=None,
        model_type='', language='', basis='', prefix=''):

    data = NeuralData(trainval_x, test_x, language)
    train_gen, val_gen, test_gen = data.make_generators()

    model = NeuralWord(data.vocab_len, model_type=model_type, language=language, basis=basis)
    model.train(train_gen, val_gen, epochs=50)
    model.evaluate_test(test_gen)

    # Report descriptive statistics.
    # stats = report_entropy_statistics(model, data.data_repo)
    # title = f'{basis} basis'
    # rownames = ['Train', 'Validate', 'Train-validate', 'Test']
    # print_entropy_statistics(stats, title, rownames)

    stats = report_entropy_statistics_str(model, data.data_repo)
    title = f'{basis} basis'
    rownames = ['Train', 'Validate', 'Train-validate', 'Test']
    print_entropy_statistics(stats, title, rownames)

    # Plot train, test entropy distributions
    train_tokens = [token for _, token, _ in trainval_x]
    test_tokens= [token for _, token, _ in test_x]

    filename = prefix+'train-test.pdf'
    title = f'{language} train and test entropies - {basis} basis'
    graph_entropies(train_tokens, test_tokens,
            data.vocab, model, title, 'Train', 'Test', filename)

    # Reverse report the tokens of validation and test.
    # No problem.
    # print_tokens_from_repo(data.data_repo)


def describe_neural_entropies(language=None, table=None, form='',
        basis='', model_type='', test_split=0.15):

    # Function to perform graphical analysis of entropy distributions.
    # Possible languages are 'all', list of languages ([str]), or specific language (str).
    print(f'*** Graphical analysis of entropy distribution for {language}. ***')
    print(f'Basis is {basis}, neural model type is {model_type}.')

    def get_prefix():
        return f"entropies.{language}-{form}-{basis}-basis-{model_type}-model--"

    # Build the vocabulary.
    vocab = NeuralData.get_symbol_dictionary(language, [token for _, token, _ in table])

    # All data together.
    trainval, test = simple_train_test_split(table, test_split=0.15)

    if basis == 'all':
        prefix = get_prefix()
        describe_neural_entropies_for_basis(trainval=trainval, test=test,
                trainval_x=trainval, test_x=test,
                model_type=model_type, language=language, basis=basis, prefix=prefix)


    elif basis == 'native':
        # Train and test native are based on train, test split of all data.
        trainval_native = [[id_, token, status] for id_, token, status in trainval if status == 0]
        test_native = [[id_, token, status] for id_, token, status in test if status == 0]
        prefix = get_prefix()
        describe_neural_entropies_for_basis(trainval=trainval, test=test,
                trainval_x=trainval_native, test_x=test_native,
                model_type=model_type, language=language, basis=basis, prefix=prefix)

    elif basis == 'loan':
        # Train and test loan are based on train, test split of all data.
        trainval_loan = [[id_, token, status] for id_, token, status in trainval if status == 1]
        test_loan = [[id_, token, status] for id_, token, status in test if status == 1]
        prefix = get_prefix()
        describe_neural_entropies_for_basis(trainval=trainval, test=test,
                trainval_x=trainval_loan, test_x=test_loan,
                model_type=model_type, language=language, basis=basis, prefix=prefix)





def perform_function_by_language(languages=None, form='', basis='', model_type='', function=None):


    try:
        with open('wold.bin', 'rb') as f:
            lex = pickle.load(f)
    except:
        lex = LexibankDataset(
                'wold',
                transform={
                    "Loan": lambda x, y, z: 1 if x['Borrowed'].startswith('1') else 0}
                )
        with open('wold.bin', 'wb') as f:
            pickle.dump(lex, f)

    if languages == 'all':
        languages = [language["Name"] for language in lex.languages.values()]
    elif isinstance(languages, str):
        languages = [languages]
    elif not isinstance(languages, list):
        print("Language must be language name, list of languages, or keyword 'all'.")

    print(f'Languages {languages}.')
    for language in languages:
        table = lex.get_table(
                    language=language,
                    form=form,
                    classification='Loan'
                    )

        fn(language, table, form, basis, model_type)

if __name__ == "__main__":

    languages = 'English'  # ['English', 'Hup', 'Imbabura Quechua']
    form = 'FormChars'
    basis = 'all'
    model_type = 'attention'
    fn = describe_neural_entropies

    perform_function_by_language(languages=languages, form=form,
            basis=basis, model_type=model_type, function=fn)
