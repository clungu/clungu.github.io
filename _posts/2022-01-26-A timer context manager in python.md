---
tags:
    - python
    - engineering
mathjax: true
comments: true
title:  A timer context manager in python
header:
  teaser: 
---

[#engineering](/tags/#engineering) [#python](/tags/#python)

20220125205230

---


When writing code you may want to quickly do a performance analysis on your algorithms. The most likely tool that you need is obviously an profiler (perf_counter ?!) but if that is too much of a hassle to set up, the quickest way is to write your own time measuring tool.

The easiest way of doing it is:

```python
import time

actual_time = time.perf_counter()
comput_time = time.process_time()

result = f(*args, **kw)

actual_time = time.perf_counter() - actual_time
comput_time = time.process_time() - comput_time
```

## Using a function decorator
For a better looking code you may wish to use a decorator if all you need is to print the execution times on `stdout`

```python
import time
from functools import wraps

def timing(f):
    """
    Decorator to be used to measure the time it takes for a single call
    """

    @wraps(f)
    def wrap(*args, **kw):
        actual_time = time.perf_counter()
        comput_time = time.process_time()

        result = f(*args, **kw)

        actual_time = time.perf_counter() - actual_time
        comput_time = time.process_time() - comput_time

        print(
            '[%2.4fs|%2.4fs] func:%r args:[%r, %r]' % (
                actual_time,
                comput_time,
                f.__name__,
                args if len(args) < 1000 else "<something large>",
                kw
            )
        )
        return result

    return wrap
```

Example:
```python
import time

@timing
def my_heavy_function(a, b):
    time.sleep(1)
    
my_heavy_function(1, "2")

>     [1.0011s|0.0014s] func:'my_heavy_function' args:[(1, '2'), {}]
```

## Using a context manager

But if you want to have access to the `actual_time` and `comput_time` you may want to use [context manager](https://www.python.org/dev/peps/pep-0343/) (inspired by [this thread](https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time)):

```python
import time

class timer(object):
    """
    A small timer context manager
    """
    def __enter__(self):
        self.actual_time = time.perf_counter()
        self.comput_time = time.process_time()
        return self

    def __exit__(self, type, value, traceback):
        self.actual_time = time.perf_counter() - self.actual_time
        self.comput_time = time.process_time() - self.comput_time
```
 
 Example
```python
with timer() as t:
    time.sleep(4)
    
print(t.actual_time, t.comput_time)

> 4.004134978000366 0.0028222399996593595
```