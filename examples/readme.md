Commands for constructing tables for in the paper "Using lexical language models to detect borrowings in monolingual wordlists".

# Detection of artificially borrowings:

## Bag of Sounds

NOTE: —brate is the sampling interval.  Its inverse is the percent seeded borrowing.

`monolingual-borrowing-detection % examples/fakeborrowings.py bagofsounds --brate 5`

`monolingual-borrowing-detection % examples/fakeborrowings.py bagofsounds --brate 10`

`monolingual-borrowing-detection % examples/fakeborrowings.py bagofsounds --brate 20`


## Markov model;

`monolingual-borrowing-detection % examples/fakeborrowings.py markovdual --brate 5`

`monolingual-borrowing-detection % examples/fakeborrowings.py markovdual --brate 10`

`monolingual-borrowing-detection % examples/fakeborrowings.py markovdual --brate 20`


## Neural network

`monolingual-borrowing-detection % examples/fakeborrowings.py neuraldual --brate 5`

`monolingual-borrowing-detection % examples/fakeborrowings.py neuraldual --brate 10`

`monolingual-borrowing-detection % examples/fakeborrowings.py neuraldual --brate 20`


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

This analysis was performed in a Jupyter notebook.  

*** The notebook would be called experimental at best.  
I would need to reduce to something pretty basic to make useful.  
Maybe worth the effort as it is useful to show the distributions. ***