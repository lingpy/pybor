#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test data.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest tests/test_data.py -—cov=pybor.data

Or to save as html report:
$ pytest tests/test_data.py -—cov=pybor.data --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_data.py --cov=pybor.data --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_data.py::test_gettable --cov=pybor.data --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

"""

# from pybor.data import LexibankDataset
# from pybor.dev.data import training
# import pybor.data as data
import pybor.util as util

logger = util.get_logger(__name__)


# def test_get_table():
#     lex = data.get_lexibank_access()
#     assert lex
#     table = lex.get_table(language='English', form='FormChars', classification='Borrowed')
#     assert table

#     print(f'table size={len(table)}')
#     assert len(table) == 1516  # As of June 18, 2020
#     assert len(table[0]) == 3

#     print('English table sample:')
#     print(table[10:25])
#     num_loan = sum([row[2] for row in table])
#     print(f'Sum loan={num_loan}')
#     assert num_loan > 0
#     print('Compare devel table for format:')
#     print(training[10:25])

# def test_language_table_gen():
#     gen = data.language_table_gen('English')
#     language, table = next(gen)
#     assert language == 'English'
#     assert len(table) == 1516  # As of June 18, 2020
#     assert len(table[0]) == 3
#     print(table[:4])

#     gen = data.language_table_gen(['English', 'Hup'])
#     language, table = next(gen)
#     assert language == 'English'
#     assert len(table) == 1516  # As of June 18, 2020
#     assert len(table[0]) == 3
#     print(table[:2])
#     for language, table in gen:
#         assert language == 'Hup'
#         assert len(table) == 1179
#         assert len(table[0]) == 3

#     gen = data.language_table_gen()
#     count = 0
#     for language, table in gen:
#         print(language, table[0])
#         count += 1
#     assert count == 41

# def test_language_table_gen_donor():
#     gen = data.language_table_gen('English', donor_num=1)
#     language, table = next(gen)
#     assert language == 'English'
#     assert len(table) == 1304  # As of June 18, 2020
#     assert len(table[0]) == 3
#     print(table[:4])


# def test_access_functions():
#     lex = data.get_lexibank_access()
#     language='Hup'
#     languages = data.check_languages_with_lexibank(lex, language)
#     print(languages)
#     assert languages[0] == language


# def test_get_donor_table():
#     lex = data.get_lexibank_access()
#     table = data.get_native_donor_table(lex, 'English', form='Tokens',
#                                         classification='Borrowed', donor_num=0)
#     assert len(table) == 1516
#     assert len(table[0]) == 3

#     table = data.get_native_donor_table(lex, 'English', form='Tokens',
#                                         classification='Borrowed', donor_num=1)
#     assert len(table) == 1304
#     assert len(table[0]) == 3

# def test_apply_function():

#     donor_num = 0

#     def get_user_fn(detect_type, model_type):

#         def fn_example(language, form, table,
#                        detect_type=detect_type,
#                        model_type=model_type):
#             assert language in ['English', 'Hup']
#             if donor_num==0:
#                 if language == 'English':
#                     assert len(table) == 1516
#                     assert len(table[0]) == 3
#                 else:
#                     assert len(table) == 1179
#                     assert len(table[0]) == 3
#             elif donor_num==1:
#                 if language == 'English':
#                     assert len(table) == 1304
#                     assert len(table[0]) == 3
#                 else:
#                     assert len(table) == 1144
#                     assert len(table[0]) == 3
#             elif donor_num==2:
#                 if language == 'English':
#                     assert len(table) == 1059
#                     assert len(table[0]) == 3
#                 else:
#                     assert len(table) == 1095
#                     assert len(table[0]) == 3


#             assert detect_type == 'dual'
#             assert model_type == 'recurrent'

#             logger.info(f'language={language}, form={form}, ' +
#                         f'detect type={detect_type}, model type={model_type}.')
#             logger.info(f'table[:2] = {table[:2]}')


#         return fn_example

#     fn = get_user_fn('dual', 'recurrent')
#     data.apply_function_by_language(languages=['English', 'Hup'], form='FormChars', function=fn)

#     donor_num = 1
#     fn = get_user_fn('dual', 'recurrent')
#     data.apply_function_by_language(languages=['English', 'Hup'],
#                                     form='FormChars', function=fn, donor_num=donor_num)

#     donor_num = 2
#     fn = get_user_fn('dual', 'recurrent')
#     data.apply_function_by_language(languages=['English', 'Hup'],
#                                     form='FormChars', function=fn,
#                                     donor_num=donor_num, min_borrowed=40)
#     fn = get_user_fn('dual', 'recurrent')
#     data.apply_function_by_language(languages=['English', 'Hup'],
#                                     form='FormChars', function=fn, donor_num=donor_num)


# if __name__ == "__main__":
#     # test_get_table()
#     # test_language_table_gen()
#     # test_access_functions()
#     # test_get_donor_table()
#     # test_language_table_gen_donor()
#     test_apply_function()