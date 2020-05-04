# SVM Code Example

The SVM is trained by all sounds in the sample. To run the test code, just type:

```shell
$ python svm_example.py
```

The output is given in total numbers of false positives, false negatives as found in the classification.

The results are then shown in a table as follows:

|       |   Borrowing |   No Borrowing |   Score |
|:------|------------:|---------------:|--------:|
| True  |        6.00 |         524.00 |    0.88 |
| False |        2.00 |          69.00 |    0.12 |
| Score |        0.01 |           0.99 |         | 

As can be seen, the method is not performing well, since it fails to identify the majority of the borrowings, which are classified as not being borrowed, so the method has a high rate of false negatives, only masked since the dataset does not have so many borrowings in the sample.
