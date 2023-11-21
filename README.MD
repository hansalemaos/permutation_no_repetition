# Permutation without repetition - written in Cython - for Numpy 

## pip install permutation-no-repetition

### Tested against Windows / Python 3.11 / Anaconda


### This module requires Cython and a C/C++ Compiler to be installed.

```python

	
This module provides a Cython-based utility for generating permutations (without repetitions!) of arrays, supporting various data types and dimensions.

Usage:
from permutation_no_repetition import cython_permutations
import numpy as np

inputarray1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
i1 = cython_permutations(inputarray1, start=0, end=-1)
print(i1)

inputarray2 = np.array(
    [
        [1, 2, 3, 4],
        [3, 3, 3, 6],
        [2, 0, 0, 2],
        [2, 0, 0, 2],
        [8, 2, 8, 2],
        [4, 5, 4, 5],
        [3, 3, 3, 6],
        [4, 5, 4, 5],
        [0, 9, 8, 7],
        [1, 2, 3, 4],
    ]
)
i2 = cython_permutations(inputarray2, start=0, end=-1)
print(i2)

inputarray3 = np.array(
    [
        100,
        1,
        2,
        2,
        3,
        4,
        115,
        5,
        6,
        7,
    ],
    dtype=np.float32,
)
i3 = cython_permutations(inputarray3, start=0, end=-1)
print(i3)

int_array_7 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
test_int_array_7 = cython_permutations(int_array_7, start=0, end=-1)
print(test_int_array_7)
int_array_8 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
test_int_array_8 = cython_permutations(int_array_8, start=0, end=-1)
print(test_int_array_8)
int_array_9 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
test_int_array_9 = cython_permutations(int_array_9, start=0, end=-1)
print(test_int_array_9)
int_array_10 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
test_int_array_10 = cython_permutations(int_array_10, start=0, end=-1)
print(test_int_array_10)
float_array_7 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7], dtype=np.float32)
test_float_array_7 = cython_permutations(float_array_7, start=0, end=-1)
print(test_float_array_7)
float_array_8 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8], dtype=np.float64)
test_float_array_8 = cython_permutations(float_array_8, start=0, end=-1)
print(test_float_array_8)
complex_array_7 = np.array(
    [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 9 + 10j, 11 + 12j, 13 + 14j], dtype=np.complex64
)
test_complex_array_7 = cython_permutations(complex_array_7, start=0, end=-1)
print(test_complex_array_7)
complex_array_8 = np.array(
    [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 9 + 10j, 11 + 12j, 13 + 14j, 15 + 16j],
    dtype=np.complex128,
)
test_complex_array_8 = cython_permutations(complex_array_8, start=0, end=-1)
print(test_complex_array_8)
bool_array_7 = np.array([True, False, True, False, True, False, True], dtype=np.bool_)
test_bool_array_7 = cython_permutations(bool_array_7, start=0, end=-1)
print(test_bool_array_7)
str_array_7 = np.array(
    ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"], dtype=np.str_
)
test_str_array_7 = cython_permutations(str_array_7, start=0, end=-1)
print(test_str_array_7)
unicode_array_7 = np.array(["α", "β", "γ", "δ", "ε", "ζ", "η"], dtype=np.unicode_)
test_unicode_array_7 = cython_permutations(unicode_array_7, start=0, end=-1)
print(test_unicode_array_7)
bytes_array_7 = np.array([b"a", b"b", b"c", b"d", b"e", b"f", b"g"], dtype="S1")
test_bytes_array_7 = cython_permutations(bytes_array_7, start=0, end=-1)
print(test_bytes_array_7)
bytes_array_8 = np.array([b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"], dtype="S1")
test_bytes_array_8 = cython_permutations(bytes_array_8, start=0, end=-1)
print(test_bytes_array_8)

```