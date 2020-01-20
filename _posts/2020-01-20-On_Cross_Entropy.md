---
categories: 
    - tutorial
tags:
    - fundamental
    - loss
---

Cross entropy can be used to define a loss function in machine learning and is usualy used when training a classification problem. 

> In information theory, the cross entropy between two probability distributions $p$ and $q$ over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set if a coding scheme used for the set is optimized for an estimated probability distribution $q$, rather than the true distribution $p$. ([source](https://en.wikipedia.org/wiki/Cross_entropy))

This post tries to implement it in pure python to better understand it's inner workings and then compare it to other popular implementations for cross-validation.

# Our implementation


```
import numpy as np
import tensorflow as tf
import torch 
from matplotlib import pyplot as plt
```


The crossentropy function is defined as:

$$Loss = -\sum_{i}{target_i * \log(prediction_i)}$$ 

This seems simple enough so let's implement this!


```python
def categorical_crossentropy(y_true, y_pred):
    # - SUM(target * log(pred))
    return -np.sum(y_true * np.log(y_pred))

categorical_crossentropy([0, 1], [0.5, 0.5])
```




    0.6931471805599453




```python
categorical_crossentropy([0, 1], [0.5, 0.5])
```




    0.6931471805599453



I don't trust my code so I need to certify that my implementation is working correctly by comparing it to known and proven implementations. 

The first one that comes to mind is the `sklearn` one. It is not (confusingly) called `crossentropy` but goes by its other name: `log_loss`


```python
from sklearn.metrics import log_loss

log_loss([0, 1], [0.5, 0.5])
```




    0.6931471805599453



Ok! The results matched on both (and also match my analitical computation). Time for a few more tests to make sure we're not missing something with this happy flow.


```python
def certify():
    tests = [
            [[0, 0, 1], [0.3, 0.7, 0.0]],
            [[0, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]],
            [[1, 0], [0.4, 0.6]],
    ]

    for [y_true, y_pred] in tests:
        my_xent = categorical_crossentropy(y_true, y_pred)
        xent = log_loss(y_true, y_pred)
        assert my_xent == xent, f"{y_true}\t{y_pred}\n{my_xent} != {xent}" 

certify()
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:3: RuntimeWarning: divide by zero encountered in log
      This is separate from the ipykernel package so we can avoid doing imports until



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-5-f1d99854990a> in <module>()
         11         assert my_xent == xent, f"{y_true}\t{y_pred}\n{my_xent} != {xent}"
         12 
    ---> 13 certify()
    

    <ipython-input-5-f1d99854990a> in certify()
          9         my_xent = categorical_crossentropy(y_true, y_pred)
         10         xent = log_loss(y_true, y_pred)
    ---> 11         assert my_xent == xent, f"{y_true}\t{y_pred}\n{my_xent} != {xent}"
         12 
         13 certify()


    AssertionError: [0, 0, 1]	[0.3, 0.7, 0.0]
    inf != 12.033141381058451


Hmm.. it crashes on the first example..


```python
categorical_crossentropy([0, 0, 1], [0.3, 0.7, 0.0]), log_loss([0, 0, 1], [0.3, 0.7, 0.0])
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:3: RuntimeWarning: divide by zero encountered in log
      This is separate from the ipykernel package so we can avoid doing imports until





    (inf, 12.033141381058451)



The problem is of course in our implementation. We have a `0.0` value (the third in the `y_pred`) on which we are applying the `log`. You may remember that the log function is undefined on `0.0`. The `sklearn` implementation actually clips the end of the provided `y_pred` so it will never be `0.0` or `1.0`. 

*Offtopic*: `log(1.0)` is actually 0, it is defined, and I'm unsure why they clip the top as well. I assume is related either to the `vanishing gradient problem` or to the idea that a prediction is never actually 100% certain of a result (?).

The clipping is performed, employing a sufficiently small `epsilon` value (`sklearn` defaults to `1e-15`), as:
    
    y_pred = max(eps, min((1-eps), y_pred))

We can use the above or make use of `np.clip` which will implement the exact formula above, but faster (they claim).


```python
def _clip_for_log(y_pred, eps=1e-15): 
    # y_pred = np.maximum(eps, np.minimum((1-eps), y_pred)) # equivalent
    y_pred = np.clip(y_pred, eps, 1-eps)
    return y_pred

_clip_for_log(1), _clip_for_log(np.array([1, 1, 0, 1, 0, 0.5, 0.4, 0.3]))
```




    (0.999999999999999,
     array([1.e+00, 1.e+00, 1.e-15, 1.e+00, 1.e-15, 5.e-01, 4.e-01, 3.e-01]))



The improved `crossentropy` function is now:


```python
def categorical_crossentropy(y_true, y_pred):
    y_pred = _clip_for_log(y_pred)
    return -np.sum(y_true * np.log(y_pred))

categorical_crossentropy([0, 0, 1], [0.3, 0.7, 0.0])
```




    34.538776394910684




```python
certify()
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-13-e627f102689a> in <module>()
    ----> 1 certify()
    

    <ipython-input-7-b39bd15b1efe> in certify()
          7 
          8     for [y_true, y_pred] in tests:
    ----> 9         assert categorical_crossentropy(y_true, y_pred) == log_loss(y_true, y_pred), f"{y_true}\t{y_pred}"
         10 
         11 certify()


    AssertionError: [0, 0, 1]	[0.3, 0.7, 0.0]


Trying to run the test again shows that (even if the code doesn't crashes anymore) we are getting different results:


```python
categorical_crossentropy([0, 0, 1], [0.3, 0.7, 0.0]), log_loss([0, 0, 1], [0.3, 0.7, 0.0])
```




    (34.538776394910684, 12.033141381058451)



What happens is that, in reality, calling 

    log_loss([0, 0, 1], [0.3, 0.7, 0.0])

is interpreted as 

```
log_loss([
    0, 
    0, 
    1
], 
[
    0.3,
    0.7,
    0.0
])
``` 

where each list is a batch of predictions. So the log_loss is actually used as a `binary_crossentropy` on each pair of (target, prediction) and the results (equal to the number of values in the lists) is averaged togheter. 

Explicitly, we have:


```python
(log_loss([0], [0.3], labels=[0, 1]) +
 log_loss([0], [0.7], labels=[0, 1]) + 
 log_loss([1], [0.0], labels=[0, 1])) / 3
```




    12.033141381058451



This means that we need to make the `sklearn` `log_loss` think that we're not having batches but a single prediction to evaluate (so instead of shape `(3,)` we need a `(1, 3)`).


```python
categorical_crossentropy([0, 0, 1], [0.3, 0.7, 0.0]), log_loss([[0, 0, 1]], [[0.3, 0.7, 0.0]])
```




    (34.538776394910684, 34.538776394910684)




```python
def certify():
    tests = [
            [[0, 0, 1], [0.3, 0.7, 0.0]],
            [[0, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]],
            [[1, 0], [0.4, 0.6]],
    ]

    for [y_true, y_pred] in tests:
        my_xent = categorical_crossentropy(y_true, y_pred)
        xent = log_loss([y_true], [y_pred])
        assert my_xent == xent, f"{y_true}\t{y_pred}\n{my_xent} != {xent}" 
    print("Success, results are equal!")
    
certify()
```

    Success, results are equal!


Does this mean that our implementation does not work on batches?


```python
categorical_crossentropy([[0, 0, 1], [0, 1, 0]], [[0.3, 0.7, 0.0], [0.5, 0.2, 0.3]]), log_loss([[0, 0, 1], [0, 1, 0]], [[0.3, 0.7, 0.0], [0.5, 0.2, 0.3]])
```




    (36.14821430734479, 18.074107153672394)



The results of our computatin and `sklearn`'s `log_loss` with batches is different..


```python
categorical_crossentropy([[0, 1, 0]], [[0.5, 0.2, 0.3]]) + categorical_crossentropy([0, 0, 1], [0.3, 0.7, 0.0])
```




    36.14821430734479



It works but not correctly. Our implementation does a `sum` over all errors in a batch but we need to return a mean, so we need to divide it by the number of examples in the batch (the batch_size). As such, the new implementation is:


```python
def ensure_ndarray(value):
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return value

def categorical_crossentropy(y_true, y_pred):
    """
    Implements the crossentropy function:
    Loss = - SUM(target * log(pred))
    """

    y_true = ensure_ndarray(y_true)
    y_pred = ensure_ndarray(y_pred)

    # dimensions must match
    assert y_true.shape == y_pred.shape

    y_pred = _clip_for_log(y_pred)
    batch_size = y_true.shape[0]
    
    return -np.sum(y_true * np.log(y_pred)) / batch_size

categorical_crossentropy([[0, 0, 1], [0, 1, 0]], [[0.3, 0.7, 0.0], [0.5, 0.2, 0.3]]), log_loss([[0, 0, 1], [0, 1, 0]], [[0.3, 0.7, 0.0], [0.5, 0.2, 0.3]])
```




    (18.074107153672394, 18.074107153672394)




```python
certify()
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-14-e627f102689a> in <module>()
    ----> 1 certify()
    

    <ipython-input-12-f328bbe97973> in certify()
          9         my_xent = categorical_crossentropy(y_true, y_pred)
         10         xent = log_loss([y_true], [y_pred])
    ---> 11         assert my_xent == xent, f"{y_true}\t{y_pred}\n{my_xent} != {xent}"
         12     print("Success, results are equal!")
         13 


    AssertionError: [0, 0, 1]	[0.3, 0.7, 0.0]
    11.512925464970229 != 34.538776394910684


Hmm... We're back to square one. The first example doesn't fit anymore because we are using a single list (and not batches). Our current implementation assumes we have a single prediction to make, but computes the batch size on the first dimension, which is 3 (but that's actually the number of classes in our single one-hot-encoded vector). 

We need to compute the `batch_size` a little more carefully (considering we have a batch computation if the inputs have at least 2 dimensions, else if only a single dimensions is used, the inputs are a single prediction)


```python
def ensure_ndarray(value):
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return value

def categorical_crossentropy(y_true, y_pred):
    """
    Implements the crossentropy function:
    Loss = - SUM(target * log(pred))
    """

    y_true = ensure_ndarray(y_true)
    y_pred = ensure_ndarray(y_pred)

    # dimensions must match
    assert y_true.shape == y_pred.shape

    y_pred = _clip_for_log(y_pred)
    batch_size = y_true.shape[0] if len(y_true.shape) > 1 else 1
    
    return -np.sum(y_true * np.log(y_pred)) / batch_size

certify()
```

    Success, results are equal!


Success!

# Sklearn's vs Ours discussion

Now, there's an interesting discussion about our above heuristic:
* form an interface perspective, the implementation acts is two ways, given the shape of the inputs:
    * in batch mode (dims >=2, first dim is the batch one) 
    * in single prediction mode (dims == 1, we only have a single prediction to evaluate)

Since we have a hybrid behavior, we may try to standardize a part of it:
* consider that we **always** have batches.

This makes the behavior on the:
```categorical_crossentropy([0, 0, 1], [0.3, 0.7, 0.0])``` be interpreted as a batch of `[(0, 0.3), (0, 0.7), (1, 0.0)]` examples (3 in this case) and the result be a mean of these. 

This is actually the behavior of the `sklearn` implementation. **It always assumes you send in batches**. It may make sense, since the vast majority of the time you want to use this function is in a stochastic gradient descent (batch based) training loop.

Unfortunately this still adds some uncertainties (or heterogenous behaviour) as the pairs above `[(0, 0.3), (0, 0.7), (1, 0.0)]` cannot be plainly computed anymore with the initial formula:
$$Loss = -\sum_{i}{target_i*\log(prediction_i)}$$ since this formulation is valid for a one-hot-encoded `target` variable where there is exactly one value of `1`. In the pair `(0, 0.3)` there is no 1 value in the target, so using this formula yields the result `0` (and it always is `0` for targets equal to `0`). This basically leads to the `Loss` value only represent the errors of the `positive` (`target == 1`) samples in the batch, because these are the only ones in which (the single) product is not `0`.

## Binary Crossentropy

The `sklearn` implementation solves this case by assuming that if your input dimension is 1 (you have a list of scalar values) the values will not be computed on the **categorical crossentropy** function but a simplified version of it where the pairs `(0, 0.3)` have following internal representation:
* `0 is translated to [1, 0] <=> [1 - 0, 0]` 
* `0.3 is translated to [0.7, 0.3] <=> [1 - 0.3, 0.3]` 

This redefinition can be translated as:
`the correct output is label 0 but the prediction for label 1 is 0.3. So basically I want to say that I predict 0 because I predict a really low label 1 value.`. 

Generically, the pairs $(label, pred)$ where $label \in \{0,1\}$ and $pred \in (0, 1]$ are equivaleted to `target = [1 - label, label]` and `prediction = [1 - pred, pred]`. Now we can compute the regular crossentropy formula:
$$ Loss = - \sum_{i}{target_i} *\log(prediction_i) $$
$$ Loss = - [( 1 - label) * \log(1 - pred) + label * \log(pred)]$$ 

The last formulation is called **binary crossentropy**. 

So in essence, `sklearn.log_loss` chooses to assume that we allways have batches in the input, and when in doubt (single dimension inputs), doesn't compute the `categorical crossentropy` but the `binary crossentropy`. 

For my taste and implementation I'm going to assume that we always compute the `categorical crossentropy` and relax the batching assumption as the function is called `categroical..`. I always do `categorical`.

# Keras / Tensorflow crossentropy


```python
from tensorflow.keras.metrics import categorical_crossentropy as keras_cat_xent
keras_cat_xent([0, 1], [0.5, 0.5]).numpy()
```




    0.6931472




```python
targets = [[0, 0, 1], [0, 1, 0]]
predics = [[0.3, 0.7, 0.0], [0.5, 0.2, 0.3]]
categorical_crossentropy(targets, predics), keras_cat_xent(targets, predics).numpy()
```




    (18.074107153672394, array([16.118095,  1.609438], dtype=float32))



Ok, using the tensorflow / keras version leads to the following 3 questions:
* why do we get 2 values?
* what do these values mean?
* how can make our numbers match?

The answers to the first and second question are somewhat obvious: we have as many results as samples in the batch, ant they are the results of the per-sample categorical crossentropy function. 

This is because the `K.categorical_crossentropy` function also has a `axis=-1` parameter which instructs on which dimensions to do the reduction. Since we're asking for a reduction on **only** the last dimension (the dimension of one-hot-encoded values) we are left with the dimension 0 elements (the batch size).

We can demonstrate this by showing that calling the `K.categorical_crossentropy` function individually for each sample in a batch with size 1 will lead the the same 2 values as above.


```python
keras_cat_xent([[0, 0, 1]], [[0.3, 0.7, 0.0]]), keras_cat_xent([[0, 1, 0]], [[0.5, 0.2, 0.3]])
```




    (<tf.Tensor: shape=(1,), dtype=float32, numpy=array([16.118095], dtype=float32)>,
     <tf.Tensor: shape=(1,), dtype=float32, numpy=array([1.609438], dtype=float32)>)



Now for the last question ("how can we make our previous numbers match" / "why don't they match?"). 

Recall that on the first sample, our function returned `34.538`, the same for `log_loss` whereas the keras version returned `16.118`.. 


```python
targets = [[0, 0, 1]]
predics = [[0.3, 0.7, 0.0]]
categorical_crossentropy(targets, predics), log_loss(targets, predics), keras_cat_xent(targets, predics).numpy().sum()
```




    (34.538776394910684, 34.538776394910684, 16.118095)



After reading the source code of the keras implementation and couldn't find any difference with our implementation, I decided to recompute by hand their answer, when I noticed something strange.


```python
TF_EPSILON = 1e-7
SK_EPSILON = 1e-15

- ((1 * np.log(TF_EPSILON))), - ((1 * np.log(SK_EPSILON)))
```




    (16.11809565095832, 34.538776394910684)



Tensorflow / Keras uses a different epsilon! 

We were previously using `1e-15` but they choose `1e-7`. It is a bit surprinsing that the resulting errors are that large, while the change between them is rather small. 

Sure, mathematically it make sense that the log of a `10^8` smaller value should result in a bigger error, but from an API point of view, predicting either `1/10^7` or `1/10^15` while the correct answer is `1`, should give pretty close errors. These two predictions are after all synonimus to `pure wrong`. 


Now, for the question of "why we have 2 values instead of a single one per on a batch_size == 2?" the answer is both suprising and confusing..

* [`tensorflow.keras.metrics.categorical_crossentropy`](https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/keras/backend.py#L4443-L4504) is a `function` and only computes the per-sample result (without doing a mean over the results).

* [`tensorflow.keras.metrics.CategoricalCrossentropy`](https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/keras/metrics.py#L2807-L2818) is a `class` (and `layer`) that computes the per-batch (with the mean, because it subclasses [`tensorflow.keras.metrics.MeanMetricWrapper`](https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/metrics.py#L547)). This is actually the class instanced when using a `Model` if you specify `loss='categorical_crossentropy'`.  

* even more surprising, the official [`keras.io`](www.keras.io) implementation of the [`keras.losses.CategoricalCrossentropy`](https://github.com/keras-team/keras/blob/master/keras/losses.py) is returning the per-batch result but defaults to reducing it by doing the `losses_utils.Reduction.SUM_OVER_BATCH_SIZE`, so not a mean, but a **sum**. 

* Should I tell you that there's also a [`keras.**metrics**.CategoricalCrossentropy`](https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/metrics.py#L857) **`class`** that mirros the Tensorflow implementation (it does the mean)? Presumably you'd use the first version for the `loss=` part and the second one in the `metrics=` part..

* and there is also the [`keras.losses.categorical_crossentropy`](https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/backend/tensorflow_backend.py#L3360) `function` which redirects to the Tensorflow `function` (the first point) which returns the per-sample crossentropy.. 


In all, using the vanila `keras.io` you can get (depending on what you use):
* per-batch with SUM reduction
* per-batch with MEAN reduction
* per-sample 

On the TensorFlow `keras` port you can get:
* per-batch with MEAN reduction
* per-sample

What about the batching dilema (the one where `sklearn` and ours diverged?)


```python
keras_cat_xent([0, 0, 1], [0.3, 0.7, 0.0])
```




    16.118095



Well, `keras` chooses to **always** do the crossentropy, like we did. 

One last note about Tensorflow / Keras. The `categroical_crossentropy` has also a parameter `from_logits=False` that can interpret the values of the predictions as logits, meaning that you can use it for multi-class predictions:

    lables = [1, 0, 0, 1]

where the network is expected to produce results for mutiple classes at the same time. This is interpreted as if each value of the lable represents a `binary_crossentropy` evaluation.

Setting `from_logits=True` redirects you to using the [tensorflow.nn.softmax_cross_entropy_with_logits_v2](https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/nn_ops.py#L3108) function. This function has a few caveats to understand:

> **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.

This means that even I've stated you could have `lables=[1, 0, 0, 1]` this function actually requires that you send it something like `lables = [0.5, 0, 0, 0.5]` for it to be a valid probability distribution. You can convert `[1, 0, 0, 1]` to `[0.5, 0, 0, 0.5]` by passing it through a `softmax` or through a simples scalling method:

    def scale(values):
        return values / np.sum(values)

> **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

This means that while we are **required to scale** the `lables` we are **required NOT to scale** the `logits` (i.e. predictions) 

# PyTorch crossentropy


[NLLLoss](https://pytorch.org/docs/master/nn.html#nllloss) is the negative log likelihood implementation:
* uses the format `(y_pred, y_true)` instead of the common `(y_true, y_pred)` found in `sklearn`, `keras`, `tensorflow`
* `y_pred` is expected to have log values (i.e. `y_pred == log(orig_y_pred)`
* `y_true` should contain **class indexes** (i.e. ordinal values not one-hot-encoded values). This is equivalent to the `sparse_categorical_crossentropy` class of modules in `keras` and TensorFlow
* expects certain types: 
    * `torch.Long` for the `y_true`  
    * `torch.Float` for the `y_pred`


**Obervation**: Because the function **requires** the `y_pred` values to be in `log` format that means that is up to the called to do the `clipping` with whatever values he wishes to use.


```python
def f(values):
    return torch.tensor(values).float()

def l(values):
    return torch.tensor(values).long()

from torch import nn
nn.NLLLoss()(f([np.log([0.5, 0.5])]), l([1]))
```




    tensor(0.6931)



So let's respect these **documented** assumptions and try to check that we can correctly match the results of the `sklearn.metrics.log_loss` and `torch.nn.NLLLoss`


```python
orig_targets = [[0, 0, 1], [0, 1, 0]]
orig_predics = [[0.3, 0.7, 0.0], [0.5, 0.2, 0.3]]

targets = np.argmax(orig_targets, axis=-1)
predics = np.log(np.clip(orig_predics, SK_EPSILON, 1-SK_EPSILON)) # same clipping type
```


```
nn.NLLLoss()(f(predics), l(targets)).numpy(), log_loss(orig_targets, orig_predics)
```




    (array(18.074108, dtype=float32), 18.074107153672394)



Notice that by default, calling `.float()` on a PyTorch tensor yields a `float32` values wich leads to a reduction in precision of the results.

Let's try to make the tensor a `float64` value and notice what happens 


```python
def f(values):
    return torch.tensor(values).type(torch.DoubleTensor)

def l(values):
    return torch.tensor(values).long()

orig_targets = [[0, 0, 1], [0, 1, 0]]
orig_predics = [[0.3, 0.7, 0.0], [0.5, 0.2, 0.3]]

targets = np.argmax(orig_targets, axis=-1)
predics = np.log(np.clip(orig_predics, SK_EPSILON, 1-SK_EPSILON))   # same clipping type
nn.NLLLoss()(f(predics), l(targets)).numpy(), log_loss(orig_targets, orig_predics)
```




    (array(18.07410715), 18.074107153672394)



We get more decimal points but the results are still a bit off compared to the `sklearn` implementation. I'm not sure why that is.


There is also the [CrossEntropyLoss](https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss) layer, which is read it correctly only does a `softmax` on the predictions, before computing the `log_loss`. 

This means that either we need to invert the softmax before calling it, or we apply the softmax on the `sklearn` one if we wish to compare the results.

It's easyer to do the second option.


```python
nn.CrossEntropyLoss()(f(orig_predics), l(targets)), log_loss(orig_targets, softmax(orig_predics))
```




    (tensor(1.3566, dtype=torch.float64), 1.3565655522346258)



It worked!!

# Conclusions

* I wasn't expected things to be so nuanced when I started writing this!
* `keras` in bit of a mess. There are multiple confusing ways to compute the crossentropy.
* small details (the epsilon) matter
* if not carefull we may sometime get to see the results of a `binary_crossentropy` rather than a `categorical_crossentropy`
* PyTorch makes you to explicitly do stuff (like the applying the `log`, the `clipping` or the `softmax`) in order to make you aware of the subtle details that if made implicit (like `keras` and `sklearn` supperbly do) might make you shoot yourself in the foot (without even noticing it)

My **main** takeawys are these:
* implement everything yourself (or read the sourcecode). I'm affraid of how many details I've missed until now in other more convoluted (get it?!) layers / concepts.
* [details matter](https://www.curs-ml.com/)

A realy nice article about the cross-entropy loss can also be found [here](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

