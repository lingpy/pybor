from utility_functions import *
from markov_char_lm import MarkovCharLM
from markov_model_analysis import analyze_language_word_distributions
from markov_model_analysis import (
    analyze_language_word_distributions_native_basis,
)
from markov_model_analysis import language_word_discrimination_dual_basis
from markov_model_analysis import language_word_discrimination_native_basis
from markov_model_analysis import language_word_discrimination_dual_basis
from markov_model_analysis import language_word_discrimination_native_basis
from markov_model_analysis import (
    k_fold_language_word_discrimination_native_basis,
)
from markov_model_analysis import k_fold_language_word_discrimination_dual_basis
from markov_model_analysis import k_fold_entropy_for_language

# List of languages to process.
# TODO: share with Construct, or read from tables
study_languages = [
    "English",
    "Hup",
    "ImbaburaQuechua",
    "Mapudungun",
    "Qeqchi",
    "Wichi",
]


# Markov model entropies calculated from undifferentiated (entire) word table¶
print("============ 1")
analyze_language_word_distributions('English', form='formchars', test='ks', n=100)
analyze_language_word_distributions('English', form='segments', test='ks', n=100)
analyze_language_word_distributions('English', form='scas', test='ks', n=100)

# Constructs native and loan Markov models for each randomization¶
print("============ 2")
analyze_language_word_distributions_native_basis('English', test='ks', n=100)
analyze_language_word_distributions_native_basis('English', form='segments', test='ks', n=100)
analyze_language_word_distributions_native_basis('English', form='scas', test='ks', n=100)

# Word discrimination - based on native versus loan entropy models
print("============ 3")
language_word_discrimination_dual_basis("English", smoothing=0.5)
language_word_discrimination_dual_basis("English", form='segments', smoothing=0.5)
language_word_discrimination_dual_basis("English", form='scas', smoothing=0.5)

# Word discrimination - based on just native known¶
print("============ 4")
language_word_discrimination_native_basis('English', smoothing=0.5, p=.995)
language_word_discrimination_native_basis('English', form='segments', smoothing=0.5, p=.995)
language_word_discrimination_native_basis('English', form='scas', smoothing=0.5, p=.995)

# Test k-fold native basis word discrimination.
# No interest in return table of k-fold trials.
print("============ 5")
_ = k_fold_language_word_discrimination_native_basis('English', form='formchars', smoothing=0.5, k_fold=10, p=.995)
_ = k_fold_language_word_discrimination_native_basis('English', form='segments', smoothing=0.5, k_fold=10, p=.995)
_ = k_fold_language_word_discrimination_native_basis('English', form='scas', smoothing=0.5, k_fold=10, p=.995)

# Word discrimination with native and loan known
# Test k_fold dual basis word discrimination.
print("============ 6")
_ = k_fold_language_word_discrimination_dual_basis('English', form='formchars', smoothing=0.5, k_fold=10)
_ = k_fold_language_word_discrimination_dual_basis('English', form='segments', smoothing=0.5, k_fold=10)
_ = k_fold_language_word_discrimination_dual_basis('English', form='scas', smoothing=0.5, k_fold=10)

# Overfitting of Markov model
print("============ 7")
k_fold_entropy_for_language("English", k_fold=5, smoothing=[0.5])
k_fold_entropy_for_language(
    "English",
    k_fold=10,
    smoothing=0.5,
    selector=lambda e: e["borrowedscore"] < 0.375,
)
k_fold_entropy_for_language(
    "English", form="segments", k_fold=5, smoothing=[0.5]
)
k_fold_entropy_for_language(
    "English",
    form="segments",
    k_fold=10,
    smoothing=0.5,
    selector=lambda e: e["borrowedscore"] < 0.375,
)
k_fold_entropy_for_language("English", form="scas", k_fold=5, smoothing=[0.5])
k_fold_entropy_for_language(
    "English",
    form="scas",
    k_fold=10,
    smoothing=0.5,
    selector=lambda e: e["borrowedscore"] < 0.375,
)
