#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:00:36 2020

@author: johnmiller
"""
import math
import random
import pybor
from pybor.dev.data import training, testing

data = testing+training
print(f'Orginal data: training {len(training)}, testing {len(testing)}, total {len(data)}.')
test_split = 0.15
print(f'Current setting for split {test_split:.3f}.')
print('Enter new value for split', end=':')
split_str = input()
test_split = float(split_str) if split_str != '' else test_split
print(f'New split setting {test_split:.3f}')

data = random.sample(data, len(data))
test_len = math.ceil(test_split*len(data))

training1 = data[:-test_len]
testing1 = data[-test_len:]
print(f'New data: training {len(training1)}, testing {len(testing1)}.')
print('New training example: ', training1[:5])
print('New testing example:', testing1[:5])

training1 = sorted(training1, key=lambda word: word[0])
testing1 = sorted(testing1, key=lambda word: word[0])

print(f'Sorted data: training {len(training1)}, testing {len(testing1)},'+
      f'total {len(training1)+len(testing1)}.')
print('Sorted training example:', training1[:5])
print('Sorted testing example:', testing1[:5])

with open('test_split.txt', 'w') as fl:
    fl.write('\n\n')
    fl.write(f'# Test split of {test_split:.2f}.')
    fl.write('\n\ntraining1 = [')
    for word in training1:
        fl.write('\n')
        fl.write(f'    {word},')
    fl.write('\n    ]\n')
    fl.write('\ntesting1 = [')
    for word in testing1:
        fl.write('\n')
        fl.write(f'    {word},')
    fl.write('\n    ]\n\n')

