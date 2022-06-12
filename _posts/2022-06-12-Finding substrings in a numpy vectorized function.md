---
tags:
    - scrollto=7tzc6rcmwhi2
mathjax: true
comments: true
title:  Finding substrings in a numpy vectorized function
header:
  teaser: 
---

# Hide 10 needles in the haystack

20220612174519

---


This document explores how a vectorized numpy solution for finding all occurrences of a sublist (needle) in a bigger list (haystack) works. 

The solution (and question) comes form this stack overflow page called [Python/NumPy first occurrence of subarray](https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray) and is referenced also in the [From Python to Numpy ebook](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)

```python
import numpy as np

haystack = np.random.randint(1000, size=(int(1e6),))
needle = np.random.randint(1000, size=(100,))
place = np.random.randint(int(1e6 - 100 + 1), size=10)
for idx in place:
    haystack[idx:idx+100] = needle
```

## Solution

```python
def find_subsequence(seq, subseq):
    target = np.dot(subseq, subseq)
    candidates = np.where(np.correlate(seq,
                                       subseq, mode='valid') == target)[0]
    # some of the candidates entries may be false positives, double check
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask]

>>> find_subsequence(haystack, needle)

array([ 25719, 149766, 279629, 372581, 373305, 535210, 573245, 806295,
       838102, 954196])
```

```python
>>> np.all(np.sort(place) == find_subsequence(haystack, needle))
True
```

```python
>>> %timeit find_subsequence(haystack, needle)
1 loop, best of 5: 113 ms per loop
```

## Analisys

```
target = np.dot(needle, needle)
>>> target
31037624
```

`np.correlate` computes the `dot` product between needle and every sliding window of the same size in haystack. This means that where this function returns values equal to the `target` variable, you probably have a needle.

```python
>>> np.correlate(haystack, needle, mode='valid')
array([26408171, 23900354, 25843323, ..., 22086714, 23579285, 22736943])
```

Naturally, the same `dot-product` may be the result of multiple possible pairs of vectors, so the result of `np.correlate` where it is equal with the `target` indicates only of *possible* places where the needle exists. 

```python
candidates = np.where(np.correlate(haystack, needle, mode='valid') == target)[0]
>>> candidates.shape, candidates

((10,), array([ 25719, 149766, 279629, 372581, 373305, 535210, 573245, 806295,
        838102, 954196]))
```

```python
check = candidates[:, np.newaxis] + np.arange(len(needle))
>>> check.shape, check[:4, :10]

((10, 100),
 array([[ 25719,  25720,  25721,  25722,  25723,  25724,  25725,  25726,
          25727,  25728],
        [149766, 149767, 149768, 149769, 149770, 149771, 149772, 149773,
         149774, 149775],
        [279629, 279630, 279631, 279632, 279633, 279634, 279635, 279636,
         279637, 279638],
        [372581, 372582, 372583, 372584, 372585, 372586, 372587, 372588,
         372589, 372590]]))
```

`check` builds a list of consecutive indices for each candidate equal in length to the length of the needle and starting from the `candidates` indices returned by the `np.correlate` function, which is used as a mask for elements that should be equal to the actual needle values. 

```python
mask = np.all((np.take(haystack, check) == needle), axis=-1)
>>> mask

array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True])
```

Mask makes the direct `1-to-1` check that the candidates actually are the start of the needles, by comparing the check values with the needle itself.

## Use cases
This vectorisation could be useful for searching (finding) an embedding or a hash in a large list of hashes, where the hashes are larger than 64 bits (so they can't fit in a single float value). If this isn't the case (if the hash is smaller than 64 bits) you don't need the `np.correlate`. Just call `np.where` and be done.

I wonder if this is (in practice) faster than using a plain dictionary which has a `O(1)` lookup. I guess the dictionary may be potentially faster but sure is a lot more memory hungry! At the same time, if you need to optimise for memory maybe you shouldn't use `Python` to begin with..

## Maybe a faster implementation via `scipy.correlate`

Numpy notes in [their docs for correlate](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html):

```
numpy.correlate may perform slowly in large arrays (i.e. n = 1e5) because it does not use the FFT to compute the convolution; in that case, scipy.signal.correlate might be preferable.
```

I've modified the function bellow to use the `FFT` method from `scipy` to see how much of an improvement we get by using that.

```python
from scipy.signal import correlate

def find_subsequence_scipy(seq, subseq):
    target = np.dot(subseq, subseq)
    candidates = np.where(correlate(seq, subseq, mode='valid', method='fft') == target)[0]
    # some of the candidates entries may be false positives, double check
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask]

>>> find_subsequence_scipy(haystack, needle), np.all(np.sort(place) == find_subsequence_scipy(haystack, needle))

(array([ 25719, 149766, 279629, 372581, 373305, 535210, 573245, 806295,
        838102, 954196]), True)

```

```python
>>> %timeit find_subsequence(haystack, needle)

1 loop, best of 5: 112 ms per loop
```

```python
>>> %timeit find_subsequence_scipy(haystack, needle)

10 loops, best of 5: 89.9 ms per loop
```

We get a 20% speedup (also by adding a new dependency on `scipy`)!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fHcmH1iR0aWJi2LPE36ZMrQROtY8L8Zr[#scrollto=7tzc6rcmwhi2](/tags/#scrollto=7tzc6rcmwhi2))
