#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:27:02 2020

@author: johnmiller

Test data_tf.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest testys/test_data_tf.py -—cov=pybor.data_tf

Or to save as html report:
$ pytest tests/test_data_tf.py -—cov=pybor.data_tf --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_data_tf.py --cov=pybor.data_tf --cov-report term-missing

"""
from pybor.data_tf import NeuralData
from pybor.dev.data import testing1, training1

# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================

def test_instantiation():
    data = NeuralData(training1, testing1)
    assert 179 == len(data.test_data)
    # Test default 0.15 test split.
    assert 152 == len(data.val_data)
    assert 861 == len(data.train_data)
    # Verify related vector lengths.
    assert len(data.train_data) == len(data.train_tokens)
    assert len(data.train_data) == len(data.train_tokens_ids)
    assert len(data.val_data) == len(data.val_tokens)
    assert len(data.val_data) == len(data.val_tokens_ids)
    assert len(data.test_data) == len(data.test_tokens)
    assert len(data.test_data) == len(data.test_tokens_ids)

    print('Review vocab to be sure that segments are correctly represented.')
    print('Vocab', data.vocab)

    # Without test dataset.
    data = NeuralData(training1)
    assert data.test_data is None
    assert 152 == len(data.val_data)
    assert 861 == len(data.train_data)
    assert len(data.train_data) == len(data.train_tokens)
    assert len(data.train_data) == len(data.train_tokens_ids)
    assert len(data.val_data) == len(data.val_tokens)
    assert len(data.val_data) == len(data.val_tokens_ids)
    assert data.test_tokens is None
    assert data.test_tokens_ids is None

    data = NeuralData(training1, testing1, val_split=0.0)
    assert data.val_data is None
    assert len(data.train_data) == len(training1)


def test_encoding():
    tokens_ref = [
        ['a', 'ŋ', 's', 't'],
        ['m', 'i', 't', 'l', 'ai', 't'],
        ['ʃ', 't', 'øː', 'n', 'ə', 'n']]

    vocab = NeuralData.build_vocab(tokens_ref)
    # Should be:
    ref_vocab = {'</s>': 1, '<s>': 2, '<unk>': 3, 'a': 4, 'ai': 5, 'i': 6,
                 'l': 7, 'm': 8, 'n': 9, 's': 10, 't': 11, 'øː': 12, 'ŋ': 13,
                 'ə': 14, 'ʃ':15, '<nul>': 0}
    print('vocabulary: ', vocab)
    assert len(vocab) == len(ref_vocab)
    assert vocab == ref_vocab

    # Check conversion.
    tokens_ids_ref = [
        [4, 7, 13, 8],
        [2, 10, 12, 14, 1, 0, 0]]
    # Conversion wraps id list with start and stop symbols.
    tokens_ids_ref_out = [
        [2, 4, 7, 13, 8, 1],
        [2, 2, 10, 12, 14, 1, 0, 0, 1]]

    tokens_ref = [
        ['a', 'l', 'ŋ', 'm'],
        ['<s>', 's', 'øː', 'ə', '</s>', '<nul>', '<nul>']]

    tokens_ids = NeuralData.get_tokens_ids(tokens_ref, vocab)
    assert tokens_ids == tokens_ids_ref_out, ("tokens_ids %r != expected %r"
                                              %(tokens_ids, tokens_ids_ref_out))
    tokens = NeuralData.get_ids_tokens(tokens_ids_ref,
                    NeuralData.get_reverse_vocab(vocab))
    assert tokens == tokens_ref, "tokens %r != expected %r" %(tokens, tokens_ref)

    # Try out of vocabulary segments and ids.
    tokens_ids_ref = [
        [4, 7, 23, 8],
        [2, 17, 10]]
    tokens_ids_ref_out = [
        [2, 4, 7, 3, 8, 1],
        [2, 2, 3, 10, 1]]

    tokens_ref = [
        ['a', 'l', '<unk>', 'm'],
        ['<s>', '<unk>', 's']]
    tokens_ids = NeuralData.get_tokens_ids(tokens_ref, vocab)
    assert tokens_ids == tokens_ids_ref_out, ("tokens_ids %r != expected %r"
                                              %(tokens_ids, tokens_ids_ref_out))
    tokens = NeuralData.get_ids_tokens(tokens_ids_ref,
                    NeuralData.get_reverse_vocab(vocab))
    assert tokens == tokens_ref, "tokens %r != expected %r" %(tokens, tokens_ref)


    # Set vocabulary with instantiation.
    data = NeuralData(training1, testing1, vocab=vocab)
    assert data.vocab == vocab

    data.verify_token_encoding(tokens_ids_ref)
    # Should printout short dictionary from ref_vocab and reverse.
    # should also print: [['a', 'l', '<unk>', 'm'],
    #                    [''<s>', '<unk>', 's']]

def test_make_generator():
    data = NeuralData(training1, testing1)
    train_gen, val_gen, test_gen = data.make_generators()

    assert train_gen.vocab_len == 57  # Num segments in training1+testing1
    assert train_gen.batch_size == 32  # default from config-tf
    assert train_gen.data_len == 861  # from training1+testing1

    gen = train_gen.generate()
    x, y = next(gen)
    assert len(x) == train_gen.batch_size
    assert len(y) == train_gen.batch_size
    assert len(y[0][0]) == train_gen.vocab_len

    gen = val_gen.generate()
    x, y = next(gen)
    assert len(x) == val_gen.batch_size
    assert len(y) == val_gen.batch_size
    assert len(y[0][0]) == val_gen.vocab_len

    gen = test_gen.generate()
    x, y = next(gen)
    assert len(x) == test_gen.batch_size
    assert len(y) == test_gen.batch_size
    assert len(y[0][0]) == test_gen.vocab_len
    #print(x)
    #print(y)

    data = NeuralData(training1, val_split=0.0)
    train_gen, val_gen, test_gen = data.make_generators()
    assert val_gen is None
    assert test_gen is None
    assert train_gen.current_idx == 0
    assert train_gen.data_len == len(training1), ('train_gen.data_len %r != training1 %r'
                                                  % (train_gen.data_len, len(training1)))

    gen = train_gen.generate()
    _ = next(gen)
    assert train_gen.current_idx == train_gen.batch_size
    idx = train_gen.current_idx
    cnt = 1
    while train_gen.current_idx >= idx:
        _ = next(gen)
        cnt += 1
    assert cnt == len(train_gen.data)//train_gen.batch_size + 1, 'cnt was %r' % cnt
# =============================================================================
# Helper functions for testing
#
# =============================================================================



# =============================================================================
# Execute tests as standalone file
# =============================================================================

# if __name__ == "__main__":
#      test_instantiation()
#      test_encoding()
#      test_make_generator()
