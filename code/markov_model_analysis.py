import numpy as np
import pandas as pd
import math
import statistics
import sys
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy import stats
from sklearn.model_selection import KFold

from markov_char_lm import MarkovCharLM
from utility_functions import *


MeanDif = namedtuple('MeanDif', ['dif', 'x', 'y'])

Statistic_Name = {'t': "Two Sample Student's t -- Unequal Variances" ,
            'ks': "Two Sample Kolmogorov Schmirnoff",
            'md': "Mean Difference Between Samples"}

def calculate_test_statistic_between_distributions(x, y, test='ks'):
    # Returns test statistic.
    if test == 't':
        statistic = stats.ttest_ind(x, y, equal_var=False, nan_policy='propagate')
    elif test == 'ks': # test == 'ks'
        statistic = stats.ks_2samp(x, y)
    elif test == 'md':
        statistic = MeanDif(dif=statistics.mean(x)-statistics.mean(y), 
        					x=statistics.mean(x), y=statistics.mean(y))
    else:
        statistic = None
    return statistic

def calculate_randomization_test_between_distributions(values, selector, test='ks', n=1000):
    # Calculate test statistic value.
    # Repeatedly permute selector and calculate randomized test statistic.
    # Plot empirical distribution of test statistic.
    # Report empirical probability of test result.
    
    x = [value for value, select in zip(values, selector) if select==True]
    y = [value for value, select in zip(values, selector) if select==False]

    stat_ref = calculate_test_statistic_between_distributions(x, y, test=test)
    #print(f'{test} statistic =', stat_ref[0])
    
    stats = [0]*n
    for i in range(n):
        selector = np.random.permutation(selector)
        x = [value for value, select in zip(values, selector) if select==True]
        y = [value for value, select in zip(values, selector) if select==False]
        
        test_stat = calculate_test_statistic_between_distributions(x, y, test=test)
        stats[i] = test_stat[0]
   
    count = sum([val < stat_ref[0] for val in stats])
    prob = (count+0.5)/(len(stats)+1)
    print(f'prob ({test} stat >= {stat_ref[0]:.5f}) = {1-prob:.5f}')    
    draw_dist(stats, title=Statistic_Name[test])
    
    return stat_ref, 1-prob

#### ************************************************
####
#### Functions for assessing entropy distributions.
####
#### ************************************************


##

def graph_word_distribution_entropies(native_entropies=None, loan_entropies=None, figuredir=None,
	language='unknown', title='title', graphlimit=5, figurequal=''):
    # entropies.
    # selector - which tokens to use for native versus loan.
    # figuredir - directory to put .pdf of histogram.
    # language - name of language for identification in figures and reports.
    # title - title for graph.
    # graphlimit - upper graph limit for histogram bins.
    
    #selector = selector.tolist() # just in case it is Pandas series.
    # Divide into native and loan entropies.
    #native_entropies = [entropy for entropy, select in zip(entropies, selector) if select==True]
    #loan_entropies = [entropy for entropy, select in zip(entropies, selector) if select==False]
 
    native_cnt = f'{len(native_entropies):6d}'
    native_avg = f'{statistics.mean(native_entropies):9.4f}'
    native_std = f'{statistics.stdev(native_entropies):9.4f}'
    loan_cnt = f'{len(loan_entropies):6d}'
    loan_avg = f'{statistics.mean(loan_entropies):9.4f}'
    loan_std = f'{statistics.stdev(loan_entropies):9.4f}'

    # Set frame horizontal for this measure.
    bins = np.linspace(1, graphlimit, 60)
    plt.figure(figsize=(8, 5))
    plt.hist(native_entropies, bins, alpha=0.65, 
             label='native entropies'+r'$(n='+native_cnt+', \mu='+
             native_avg+', \sigma='+native_std+')$', color="blue")
    plt.hist(loan_entropies, bins, alpha=0.65, 
             label='loan entropies'+r'$(n='+loan_cnt+', \mu='+loan_avg+
             ', \sigma='+loan_std+')$', color="red")
    plt.grid(axis='y', alpha=0.8)
    plt.legend(loc='upper right')

    plt.xlabel('Entropies')
    plt.ylabel('Frequency')
    plt.title(title)

    if figuredir is not None:
        file = figuredir+language+figurequal
        plt.savefig(file+'.pdf', dpi=600)

    plt.show()
    plt.close()


def analyze_word_distributions(tokens=None, selector=None, figuredir=None,
	language='unknown', model='KNI', order=3, smoothing=0.5, test='ks', n=1000, logebase=True):

    # tokens - in space segmented form.
    # selector - which tokens to use for indicator of likely native tokens.
    # figuredir - directory to put .pdf of histogram.
    # language - name of language for identification in figures and reports.
    # model - model estimation method - default is KNI.
    # order - model order - default is 2.
    # smoothing - Kneser Ney smoothing - default is 0.5 appropriate for this study.
    # test - test statistic for training versus val difference. 
    # n - number of iterations of randomization test.
    
    print(f'Language={language}')

    train_tokens = tokens        
    mlm = MarkovCharLM(train_tokens, model=model, order=order, smoothing=smoothing)
    entropies = mlm.analyze_training()

    if logebase:
        log2ofe = math.log2(math.e)
        entropies = [entropy/log2ofe for entropy in entropies]
        
    native_entropies = [entropy for entropy, select in zip(entropies, selector) if select==True]
    loan_entropies = [entropy for entropy, select in zip(entropies, selector) if select==False]
    graph_word_distribution_entropies(native_entropies, loan_entropies, figuredir=figuredir,
		language=language, title=language+' native and loan entropy distribution - undifferentiated fit', 
		graphlimit=5, figurequal='all-basis-native-loan-entropies')
	

    # Perform randomization tests.
    # Efficient since just permutation of selector for constructing alternate test results.
    calculate_randomization_test_between_distributions(entropies, selector=selector, test=test, n=n)
    
    return mlm


def analyze_language_word_distributions(language=None, form='formchars', test='ks', n=1000, logebase=True):
    tabledir = 'tables/'
    table = pd.read_csv(tabledir+language+'.tsv', delimiter='\t', encoding='utf-8')

    selector = table.borrowedscore < 0.375
    #selector = ~table.loan
    #mlm = analyze_word_distributions(tokens=table.Segments, selector=selector, figuredir='figures/',
    mlm = analyze_word_distributions(tokens=table[form], selector=selector, 
    	figuredir='paper-figures-mm/', language=language, model='KNI', order=3, test=test, n=n, 
    	logebase=logebase)

####
####

## General function to do analysis of a language with selector for separate train and test.

def analyze_word_distributions_native_basis(tokens=None, selector=None, figuredir=None,
	language='unknown', model='KNI', order=3, smoothing=0.5, logebase=True):
	
    train_tokens = [token for token, select in zip(tokens, selector) if select==True]
    val_tokens = [token for token, select in zip(tokens, selector) if select==False]
    
    mlm = MarkovCharLM(train_tokens, model=model, order=order, smoothing=smoothing)
    native_entropies = mlm.analyze_training()
    loan_entropies = mlm.validate(val_tokens)
    
    if logebase:
        log2ofe = math.log2(math.e)
        native_entropies = [entropy/log2ofe for entropy in native_entropies]
        loan_entropies = [entropy/log2ofe for entropy in loan_entropies]

    graph_word_distribution_entropies(native_entropies, loan_entropies, figuredir=figuredir,
        language=language, title=language+' native and loan entropy distribution - undifferentiated fit', 
        graphlimit=7, figurequal='-native-basis-native-loan-entropies')



# Calculate and return test statistic on the difference between train and test distributions.

def analyze_language_tokens_test(tokens=None, selector=None, model='KNI', order=3, smoothing=0.5, test='ks'):
    # Without selector, there is no test.
    if selector is None: return None
    
    # tokens - in space segmented form.
    # selector - which tokens to use for indicator of likely native tokens.
    # language - name of language for identification in figures and reports.
    # model - model estimation method - default is KNI.
    # order - model order - default is 3.
    # test - default test is ks=Kolmogorov-Smirnov
    
    # Selector splits corpus into train and test.
    split = selector
    train_tokens = tokens[split==True]
    val_tokens = tokens[split!=True]
    # Construct and fit language model.        
    mlm = MarkovCharLM(train_tokens, model=model, order=order, smoothing=smoothing)
    train_entropies = mlm.analyze_training()
        
    # Validate language model.
    val_entropies = mlm.validate(val_tokens)  

    # Compute test statistic.
    stat = calculate_test_statistic_between_distributions(train_entropies, val_entropies, test=test)
    return stat

####


## Randomization test between distributions with separate estimator per randomization.
# Returns new statistical estimate for each randomization.
def differentiated_randomization_test(tokens, selector, n=100, order=3, model='KNI', smoothing=0.5, test='t'):
    # ts: t, ks, md 
    # Initial selector value defines the reference condition.
    teststat = analyze_language_tokens_test(tokens=tokens, selector=selector, model=model, order=order, smoothing=smoothing, test=test)
    stat_ref = teststat[0]
    #print('ref stat =', stat_ref)
    
    stats = [0]*n
    for i in range(n):
        selector = np.random.permutation(selector)
        teststat = analyze_language_tokens_test(tokens=tokens, selector=selector, model=model, order=order, smoothing=smoothing, test=test)
        stats[i] = teststat[0]
        
    count = sum([val < stat_ref for val in stats])
    prob = (count+0.5)/(len(stats)+1)
    
    print('prob (stat >=',stat_ref, ') =', 1-prob)    
    draw_dist(stats)
    
    return stat_ref, 1-prob


def analyze_language_word_distributions_native_basis(language=None, form='formchars', test='ks', n=1000, logebase=True):

    tabledir = 'tables/'
    table = pd.read_csv(tabledir+language+'.tsv', delimiter='\t', encoding='utf-8')

    selector = table.borrowedscore < 0.375
    analyze_word_distributions_native_basis(tokens=table[form], selector=selector,
    	figuredir='paper-figures-mm/', language=language, model='KNI', order=3, logebase=logebase)
    	
    differentiated_randomization_test(tokens=table[form], selector=selector, 
    	n=n, order=3, model='KNI', smoothing=0.5, test='ks')
    	
####
####

#### ************************************************
####
#### Functions for discrimination on individual words.
####
#### ************************************************

## Dual model approach - native and loan

def fit_native_loan_models(tokens=None, ground=None, model='KNI', order=3, smoothing=0.5):
    if ground is None: return None  # Must have ground to fit.
    
    tokens_native = [token for token, select in zip(tokens, ground) if select==True]
    nativemodel = MarkovCharLM(tokens_native, model=model, order=order, smoothing=smoothing)
    nativeentropies = nativemodel.analyze_tokens(tokens)

    tokens_loan = [token for token, select in zip(tokens, ground) if select==False]
    loanmodel = MarkovCharLM(tokens_loan, model=model, order=order, smoothing=smoothing)
    loanentropies = loanmodel.analyze_tokens(tokens)

    forecast = np.less(nativeentropies, loanentropies)
    print()
    print("* TRAIN RESULTS *")
    report_metrics(ground, forecast)
    
    return nativemodel, loanmodel

def evaluate_models_for_test(nativemodel=None, loanmodel=None, val_tokens=None, valground=None):

    # Calculate entropies for test set.
    nativeentropies = nativemodel.analyze_tokens(val_tokens)
    loanentropies = loanmodel.analyze_tokens(val_tokens)
    forecast = np.less(nativeentropies, loanentropies)
    print()
    print("* TEST RESULTS *")  
    metrics = report_metrics(valground, forecast)
    return metrics

def analyze_native_loan_dual_basis(train=None, val=None, form='formchars', smoothing=0.5):
    
    trainground = train.borrowedscore < 0.375 # native
    valground = val.borrowedscore < 0.375 # native
    nativemodel, loanmodel = fit_native_loan_models(tokens=train[form], ground=trainground, smoothing=smoothing)
    metrics = evaluate_models_for_test(nativemodel, loanmodel, val[form], valground)
    return metrics
    
def language_word_discrimination_dual_basis(language=None, form='formchars', smoothing=0.5, trainfrac=0.8):
    tabledir = 'tables/'
    print(f'\n{language}')
    table = pd.read_csv(tabledir+language+'.tsv', delimiter='\t', encoding='utf-8')
    
    train, val = train_test_split(table, trainfrac=trainfrac)
    analyze_native_loan_dual_basis(train, val, form=form, smoothing=smoothing)


## Native only model approach.

def fit_native_model(tokens=None, ground=None, model='KNI', order=3, smoothing=0.5, p=0.995):
    if ground is None: return None  # Must have ground to fit.
    
    tokens_native = [token for token, select in zip(tokens, ground) if select==True]
    nativemodel = MarkovCharLM(tokens_native, model=model, order=order, smoothing=smoothing)
    # Calculate empirical distribution limit of native only entropies.
    nativeentropies = nativemodel.analyze_tokens(tokens_native)
    ref_limit = calculate_empirical_ref_limit(nativeentropies, frac=p)
    # Then test  versus all entropies.
    trainentropies = nativemodel.analyze_tokens(tokens)
    forecast = [e < ref_limit for e in trainentropies]
    print()
    print('* TRAIN RESULTS *')
    report_metrics(ground, forecast)
   
    return nativemodel, ref_limit


def analyze_native_loan_native_basis(train=None, val=None, form='formchars', smoothing=0.5, p=0.995):
    # table with forms and ground.
    
    # Ground variable for native words of train and test sets.
    trainground = train.borrowedscore < 0.375
    valground = val.borrowedscore < 0.375
    nativemodel, ref_limit = fit_native_model(train[form], trainground, smoothing=smoothing, p=p)
    # Evaluate on test set.
    valentropies = nativemodel.analyze_tokens(val[form])
    forecast = [e < ref_limit for e in valentropies]
    print()
    print('* TEST RESULTS *')
    metrics = report_metrics(valground, forecast)
    return metrics


# Overall script for individual language test - native word basis 
def language_word_discrimination_native_basis(language=None, form='formchars', smoothing=0.5, trainfrac=0.8, p=0.995):
    tabledir = 'tables/'
    print(f'\n{language}')
    table = pd.read_csv(tabledir+language+'.tsv', delimiter='\t', encoding='utf-8')
    
    train, val = train_test_split(table, trainfrac=trainfrac)
    analyze_native_loan_native_basis(train, val, form=form, smoothing=smoothing, p=p)


#### *************************************************
####
#### n-fold analysis functions for word discrimination
####
#### *************************************************

## n-fold analysis of native based discrimination between native and loan.

# Function to iterate on train-test split and estimate test measurement.
# Rewrite complete_native_analysis to take multiple samples.
# Overall script for individual language test - native word basis 
def k_fold_language_word_discrimination_native_basis(language, form='formchars', smoothing=0.5, k_fold=10, p=.995):
    # language = language archive name, e.g., English
    # smoothing = Kneser Ney smoothing for Markov model
    # k_fold = number of times validation study will be be repeated
    # p = cumulative proportion of empirical probability distribution to use as cutoff.
    
    print() 
    print(language)
    tabledir = 'tables/'
    table = pd.read_csv(tabledir+language+'.tsv', delimiter='\t', encoding='utf-8')
    
    # Iterate k_fold times over train, test splits.
    # More accepted option is to break into k_fold train - test virtual splits.
    
    #test KFold - to replace train_test_split
    kf = KFold(n_splits=k_fold, shuffle=True)
    #trainfrac = 1.0-1/k_fold
    predictions = []
    # Replace the for loop and the subsequent function call with kf.split(table) call!
    #for _ in range(k_fold):
    for train_index, val_index in kf.split(table):
        #train, val = train_test_split(table, trainfrac=trainfrac)
        train, val = table.iloc[train_index], table.iloc[val_index]
        results = analyze_native_loan_native_basis(train, val, form=form, smoothing=smoothing, p=p)
        predictions.append(results)
    df = pd.DataFrame.from_records(predictions, columns=Test_prediction._fields)
    #columns=['Acc', 'Maj_acc', 'Prec', 'Recall', 'F1'])
    
    means = df.mean()
    stds = df.std()
    print()
    print(pd.DataFrame([means, stds], index=['Mean', 'StDev']))
    
    return df

## n-fold analysis of dual native-loan based discrimination between native and loan.

def k_fold_language_word_discrimination_dual_basis(language, form='formchars', smoothing=0.5, k_fold=10):
    # language = language archive name, e.g., English
    print()
    print(language)
    tabledir = 'tables/'
    table = pd.read_csv(tabledir+language+'.tsv', delimiter='\t', encoding='utf-8')
    
    #test KFold - to replace train_test_split
    kf = KFold(n_splits=k_fold, shuffle=True)
    # Iterate k_fold times over train, test splits.
    #trainfrac = 1.0-1/k_fold
    predictions = []
    #for _ in range(k_fold):
    for train_index, val_index in kf.split(table):
        #train, val = train_test_split(table, trainfrac=trainfrac)   
        train, val = table.iloc[train_index], table.iloc[val_index]
        results = analyze_native_loan_dual_basis(train, val, form=form, smoothing=smoothing)
        predictions.append(results)

    df = pd.DataFrame.from_records(predictions, columns=Test_prediction._fields)
        #columns=['Acc', 'Maj_acc', 'Prec', 'Recall', 'F1'])

    means = df.mean()
    stds = df.std()
    print()
    print(pd.DataFrame([means, stds], index=['Mean', 'StDev']))
    
    return df

#### **********************************************************
####
#### n-fold analysis functions for train and validation entropy
####
#### **********************************************************

# k-fold entropy for table of segments. 

import scipy as scipy

def entropy_train_val(train_tokens=None, val_tokens=None, order=3, model='KNI', smoothing=0.5):
    # train_tokens - in space segmented form.
    # val_tokens - in space segmented form.
    # order - model order as used in NLTK - default is 3.
    # default model is Kneser Ney.
    # Kneser Ney smoothing for entropy model.   
    
    mlm = MarkovCharLM(train_tokens, model=model, order=order, smoothing=smoothing)
    train_entropies = mlm.analyze_tokens(train_tokens)
    # Validate language model. Errors if not sufficient data for vocabulary.
    val_entropies = mlm.analyze_tokens(val_tokens)
    return [np.mean(train_entropies), np.std(train_entropies), np.mean(val_entropies), np.std(val_entropies)]

def k_fold_entropy(segments, k_fold=5, order=3, model='KNI', smoothing=0.5):
    sz = len(segments)
    frac = 1/k_fold
    segments = np.random.permutation(segments)
    
    entropy_stats = []
    for i in range(k_fold):
        # Process k_fold train vs val sets.
        val_begin = math.ceil(i*sz*frac)
        val_end = math.ceil((i+1)*sz*frac)
        val_segments = segments[val_begin:val_end]
        train_segments = np.concatenate((segments[:val_begin], segments[val_end:]))        
        
        # Train on Markov model. 
        # Calculate entropy distributions for train and val.
        entropies = entropy_train_val(train_segments, val_segments, order, model, smoothing)
        entropy_stats.append(entropies)

    entropy_stats = np.array(entropy_stats)
    
    summary = np.zeros((3, 4))
    for j in range(4):
        stat_values = entropy_stats[:, j]
        # Calculate mean, standard deviation, and standard error of mean
        summary[0, j] = np.mean(stat_values)
        summary[1, j] = np.std(stat_values)
        summary[2, j] = scipy.stats.sem(stat_values)
        
    return [[sz, k_fold, math.ceil(sz*frac), model, order, smoothing], summary]


###

# Display results of k-fold crossvalidation.
def print_k_fold_entropy(results):
    sample = results[0]
    summary = results[1]
    print(f'Sample={sample[0]}, k-fold={sample[1]}, val={sample[2]}, model={sample[3]}, order={sample[4]}, smoothing={sample[5]}.')
    
    print(f'Statistic: Train mean Train stdev    Val mean  Val stdev')
    print(f'Mean        {summary[0, 0]:9.3f}   {summary[0, 1]:9.3f}   {summary[0, 2]:9.3f}  {summary[0, 3]:9.3f}')
    print(f'StdDev      {summary[1, 0]:9.4f}   {summary[1, 1]:9.4f}   {summary[1, 2]:9.4f}  {summary[1, 3]:9.4f}')
    print(f'StdErr      {summary[2, 0]:9.5f}   {summary[2, 1]:9.5f}   {summary[2, 2]:9.5f}  {summary[2, 3]:9.5f}')

###

# Execute k-fold entropy for given language, sample selection criterion, order, model, smoothing
# Enter smoothing as scalar or as list of smoothing values
#import inspect as inspect

def k_fold_entropy_for_language(language=None, form='formchars', selector=None, k_fold=5, order=3, model='KNI', smoothing=[0.5]):
    
    tabledir='tables/'
    table = pd.read_csv(tabledir+language+'.tsv', delimiter='\t', encoding='utf-8')
    print()
    print(f'{k_fold}-fold entropy for {language}.')
    
    if selector is None:
        # Get the column containing segments for analysis.
        # Use entire column of segments in analysis.
        segments = table[form]
    else:
        #print(f'Selector={inspect.getsource(selector)}')
        print('Subset selected: See sample and val sizes.')
        selection = selector(table)
        # Get the column containing segments for analysis.
        # Get the subset of segments for analysis based on the selector.
        segments = table[form]
        segments = segments[selection]

    # Handle scalar or list for smoothing.
    smoothing = np.asarray([smoothing]) if np.isscalar(smoothing) else np.asarray(smoothing)
    for sm in smoothing:
        summary = k_fold_entropy(segments, k_fold=k_fold, order=order, model=model, smoothing=sm)
        print_k_fold_entropy(summary)
