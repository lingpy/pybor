# Commands for constructing tables for the study "Using lexical language models to detect borrowings in monolingual wordlists"

## Requirements for installation

```
$ git clone https://github.com/lingpy/pybor.git
$ pip install -e pybor
$ git clone https://github.com/lexibank/wold.git
$ pip install -e wold
```

## 1 Detection of artificial borrowings

Having installed the two code packages, the easiest way to carry out the initial experiment on artificially seeded borrowings is to use our Makefile. Having `cd`ed into the folder `examples`, just type:

```
$ make fake
```

If you want to carry out the experiments in isolation, just `cd` again into the `examples` folder and type:

```
$ python fakeborrowings.py bagofsounds --brate 5 --output "../output"
$ python fakeborrowings.py bagofsounds --brate 10 --output "../output"
$ python fakeborrowings.py bagofsounds --brate 20 --output "../output"
$ python fakeborrowings.py markovdual --brate 5 --output "../output"
$ python fakeborrowings.py markovdual --brate 10 --output "../output"
$ python fakeborrowings.py markovdual --brate 20 --output "../output"
$ python fakeborrowings.py neuraldual --brate 5 --output "../output"
$ python fakeborrowings.py neuraldual --brate 10 --output "../output"
$ python fakeborrowings.py neuraldual --brate 20 --output "../output"
```

# Cross-validation of borrowing detection on real language data:

## Bag of Sounds

`% examples/cross_validate_models_example.py bagofsounds --k_fold 10 --series "10-fold-CV" --min_borrowed 0`

## Markov model


% examples/cross_validate_models_example.py markovdual --k_fold 10 --series "10-fold-CV" --min_borrowed 0



## Neural network

`% examples/cross_validate_models_example.py neuraldual --k_fold 10 --val_split 0.0 --series "10-fold-CV" --min_borrowed 0`

# Determining factors that influence borrowing detection

The analysis was performed in Minitab and based on phonology summary statistics versus cross-validation of borrowing detection on real languages languages.

*** Tiago ***  Are we including your python script to obtain this file of summary statistics?


# Detecting borrowings when there is dominant donor in intensive contact situations

Cross-validation of borrowing detection on real language data results are stratified by minimum borrowing and dominant donor. These commands report minimum borrowing and percent borrowing by primary donor -- whose results were used to stratify the cross-validation results. 

`monolingual-borrowing-detection% examples/donor_table_explore.py --min_borrowed 300`

`monolingual-borrowing-detection% examples/donor_table_explore.py --min_borrowed 200`

`monolingual-borrowing-detection% examples/donor_table_explore.py --min_borrowed 100`

# Comparing entropy distributions to investigate borrowing prediction performance

This analysis is performed in a Jupyter notebook.  It uses modules from PyBor.

notebooks/entropy_distributions.ipynb
