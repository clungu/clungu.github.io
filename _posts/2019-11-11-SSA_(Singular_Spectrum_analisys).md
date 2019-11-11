---
mathjax: true
---

# Problem description

We have some data. Our goal for today would be to find the cyclical pattern that the data presents. 

To give one use case for this, let's say we work for an online store company. We're planning to increase our sales by promoting (through advertising) some products, discounts or special offers. The end goal would be to have people that see ads, buy these stuff. Someone seeing an ad, then making a purchase is called a `conversion`.

Unfortunately, the number of `conversions` happen irregularly (we're not seeing each hour roughly the same amount of conversions). At the same time though, the `conversions` we're seeing have a clear pattern, one of which we've identified by looking at the charts is the day/night pattern. This makes sense: people usually buy during the day and not that often at night. 

Obiously day / night is not the only pattern, there are potentially more, which look when summed up like complex looking shapes.


Our problem is identifying all these patterns, for:
* tailoring our marketing campaing by spending more on the intervals with high conversion rate, and stop the campaing on the slughish intervals.
* understand the main `conversion` driving factors for our clients (holidays, vacation periods, etc..?)
* understand our clients. We've assumed day / night is a pattern but the cycles for days don't line up with our local time. This may be because:
    * our customers come from a different part of the world 
    * it's not a day / night but a after-hours / morning-hours one
    * others..

I hope I've convinced you that exactly knowing the caracteristics of the patterns in our data is of great help.

## Generate data



Normally, any data that we work with has 3 main components:
* **Trend** (generically going up, or generically going down). The trend is an overall shape that the data seems to follow. It may be a straight line, or a parabola, or some other shape, whose domain spans all the data we have.
* **Seasonality** (the actual pattern we are looking for). This is a periodical function, that spans multiple cycles in our dataset (like the day / night cycle, winter / summer, full moon / no moon, fiscal Q1 / Q2 / Q3 / Q4, etc..)
* **Noise** - Some random stuff that we want to ignore, but which is already contained in the data. 

So we assume, of the bat, that most data has the following structure:

$$ f(t) = trend(t) + seasonality(t) + noise(t) $$

We're initerested in **seasonality** but finding that implies actually finding the other two components (because the trend is usually a low polinomyal regression over $trend(t) + noise(t)$). So we will be exploring (one way) of decomposing our data into the three parts above.

## Definining the main components of the data

In our case, we will be generating our toy data using the following definitions:

$$ trend(t) = .001 * t ^ 2$$ 

$$seasonality(t) = sin(\frac{\pi}{4}) + 5 * cos(\frac{\pi}{2})$$

$$ noise(t) = Random(0, 5) $$

$$ f(t) = 0.001*t^2 + sing(\frac{\pi}{4}) + 5 * cos(\frac{\pi}{2}) + Random(0, 5) $$ 

The `trend` is the overall pattern of the data. The trend is not about repeating patterns but about the generic direction.


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
```


```python
def trend(t):
    """
    Function that for a certain t returns a quadratic function applied on that point. 
    """
    return 0.001 * t ** 2

T = list(range(-100, 100))
Trend = [trend(t) for t in T]
plt.plot(Trend)
```




    [<matplotlib.lines.Line2D at 0x7ffb36807f98>]




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_11_1.png)



```python
def apply(func, T):
    """
    Mapping function. Applies to all the values in T the function func and returns a list with the results.
    """
    return [func(t) for t in T]
```

Then we add two periodical function to the mix (essentialy these are what we're after)


```python
def period_1(t, period = 20):
    return math.sin(math.pi * t / period)

plt.plot(apply(period_1, T))
```




    [<matplotlib.lines.Line2D at 0x7ffb3472a9b0>]




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_14_1.png)



```python
def period_2(t, period = 40):
    return math.cos(math.pi * t / period)

plt.plot(apply(period_2, T))
```




    [<matplotlib.lines.Line2D at 0x7ffb346a2198>]




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_15_1.png)


And add in some noise to make the problem hard.


```python
def noise(t):
    return np.random.randint(0, 3)

plt.plot(apply(noise, T))
```




    [<matplotlib.lines.Line2D at 0x7ffb346104a8>]




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_17_1.png)


Our final function is then:

$$ f(t) = trend(t) + period_1(t) + period_2(t) + noise(t) = 0.001*t^2 + sing(\frac{\pi}{4}) + 5 * cos(\frac{\pi}{2}) + Random(0, 5) $$ 


```python
def f(t):
    return trend(t) + period_1(t) + period_2(t) + noise(t)

f(10)
```




    1.8071067811865476




```python
plt.figure(figsize=(8, 6))
plt.plot(apply(trend, T), alpha=0.4)
plt.plot(apply(period_1, T), alpha=0.4)
plt.plot(apply(period_2, T), alpha=0.4)
plt.plot(apply(noise, T), alpha=0.4)
plt.plot(apply(f, T))
plt.legend(["trend", "period_1", "period_2", "noise", "function"], loc='upper right')
plt.title("All the components of the function")
```




    Text(0.5, 1.0, 'All the components of the function')




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_20_1.png)



```python
F = np.array(apply(f, T))
F.shape
```




    (200,)



Decompose that!!

# Idea of the approach

We've said that we assume, most data to have the following structure:

$$ f(t) = trend(t) + seasonality(t) + noise(t) $$

**Some notes**:
* Even thought trend is seen in this formulation as something different from **seasionality**, it can be (and usually is, given data that spans enough time), the trend ca also be a **seasionality** component, althoug one with a very long periodicity.
* Even though **seasonality** is seen as a single factor, it can actually be a composition of **multiple** periodic functions, with different frequency and amplitudes. For example in case of some spending data, we can have day/night patterns, Q1 / Q4 patterns, holiday and year-on-year patterns **at the same time**. Some of these components will have a longer cycle, some smaller. They all constitute **seasonality** factors because we should have (by definition) multiple cycles (they manifest repeatably) in our data. 
* Even though noise(t) is seen in this formulation as something different from **seasionality**, like a random process, is can also be some **seasionality** component, although a very short one (with minute / second long frequency). Or even in the case where noise is trully random, we can simulate it using some really short frequency **seasionality** components added togheter.

Having said that, we will try to do the following:
* model $f(t)$ as a sum of **periodic** functions (components)
* find (heuristical) ways to group these components into the 3 categories (trend, seasonality, noise)
    * low periodicity means trend, medium periodicity means seasionality, others are noise
    * for example, one heuristic would be to group as trend all the components that have a period greather than the timespan of the data we have.

So 
$$f(t) = \sum_{period=long}{component_i(t)} + \sum_{period=medium}{component_j(t)} + \sum_{period=short}{component_k(t)}$$

# Choosing an approach

We will be using here the [**Singular Spectrum Analisys (SSA)**](https://www.crcpress.com/Analysis-of-Time-Series-Structure-SSA-and-Related-Techniques/Golyandina-Nekrutkin-Zhigljavsky/p/book/9781584881940) which is based on Singular Value Decomposition (SVD).

* the shape above is really similar to a Fourier Transform decomposition into basic trigonometric functions. Although we won't be using it, a Fourier decomposition can be used to replicate this. It's called [**Fourier Analisys (FA)**](https://en.wikipedia.org/wiki/Fourier_analysis).


The reason to use SSA over FA is
>due to the fact that a single pair of data-adaptive SSA eigenmodes often will capture better the basic periodicity of an oscillatory mode than methods with fixed basis functions, such as the sines and cosines used in the Fourier transform.  
>([quote](https://en.wikipedia.org/wiki/Singular_spectrum_analysis))

This roughly sais that **often** SSA works better because it has a more complex modeling technique (based on eigenvalues) compared to Fourier Analisys (which is trignometrical functions based).

It's possible that the practical reason is that FA will yield to multiple basic components compared to SSA and thus:
* grouping them into trend, seasionality and noise will be harder 
* interpreting the FA may be more difficult

# SSA (Singular Spectrum analisys) EDA

## [Main algorithm description of SSA](https://en.wikipedia.org/wiki/Singular_spectrum_analysis)


You will find bellow a lot of math notation but we'll going to do this step by step in code after the next section. I'm including this here just to have an outline of what we're going to do next. It's ok if you only get 30-40% of what this means. 

Hopefully, after we go through the actual implementation, comming back to this will gruadually start to make more sense.

### Step 1: Embedding

Form the *trajectory matrix* of the series $\mathbf X$ , which is the $L * K$

$$\displaystyle \mathbf {X} =[X_{1}:\ldots :X_{K}]=(x_{ij})_{i,j=1}^{L,K}={\begin{bmatrix}x_{1}&x_{2}&x_{3}&\ldots &x_{K}\\x_{2}&x_{3}&x_{4}&\ldots &x_{K+1}\\x_{3}&x_{4}&x_{5}&\ldots &x_{K+2}\\\vdots &\vdots &\vdots &\ddots &\vdots \\x_{L}&x_{L+1}&x_{L+2}&\ldots &x_{N}\\\end{bmatrix}}$$


where $$X_{i}=(x_{i},\ldots ,x_{i+L-1})^{\mathrm  {T}}\;\quad (1\leq i\leq K)$$ are lagged vectors of size $L$. 

The matrix $\mathbf {X}$  is a *Hankel matrix* which means that $\mathbf {X}$  has equal elements $x_{ij}$ on the anti-diagonals $i+j=\,{\rm {const}}$.

###  Step 2: Singular Value Decomposition (SVD)

Perform the singular value decomposition (SVD) of the trajectory matrix $\mathbf {X}$. Set ${\mathbf  {S}}={\mathbf  {X}}{\mathbf  {X}}^{\mathrm  {T}}$ and denote by:
* $\lambda _{1},\ldots ,\lambda _{L}$ the eigenvalues of $\mathbf {S}$  taken in the decreasing order of magnitude ($\lambda _{1}\geq \ldots \geq \lambda _{L}\geq 0$)
* $U_{1},\ldots ,U_{L}$ the orthonormal system of the eigenvectors of the matrix $\mathbf {S}$  corresponding to these eigenvalues.
* $V_{i}={\mathbf  {X}}^{\mathrm  {T}}U_{i}/{\sqrt  {\lambda _{i}}} (i=1,\ldots ,d)$, where $d={\mathop  {\mathrm  {rank}}}({\mathbf  {X}})=\max\{i,\ {\mbox{such that}}\ \lambda _{i}>0\}$ (note that $d=L$ for a typical real-life series)

In this notation, the SVD of the trajectory matrix $\mathbf {X}$  can be written as

$${\mathbf  {X}}={\mathbf  {X}}_{1}+\ldots +{\mathbf  {X}}_{d}$$,
where

$${\mathbf  {X}}_{i}={\sqrt  {\lambda _{i}}}U_{i}V_{i}^{\mathrm  {T}}$$
are matrices having rank 1; these are called elementary matrices. 

The collection ($\sqrt  {\lambda_{i}},U_{i},V_{i}$) will be called the $i$th eigentriple (abbreviated as ET) of the SVD. Vectors $U_{i}$ are the left singular vectors of the matrix $\mathbf {X}$ , numbers ${\sqrt  {\lambda_{i}}}$ are the singular values and provide the singular spectrum of $\mathbf {X}$ ; this gives the name to SSA. Vectors ${\sqrt  {\lambda_{i}}}V_{i}={\mathbf  {X}}^{\mathrm  {T}}U_{i}$ are called vectors of principal components (PCs).

### Step 3: Eigentriple grouping

Partition the set of indices $\{1,\ldots ,d\}$ into $m$ disjoint subsets $I_{1},\ldots ,I_{m}$.

Let $I=\{i_{1},\ldots ,i_{p}\}$. Then the resultant matrix $${\mathbf {X}}_{I}$$ corresponding to the group $I$ is defined as $${\mathbf  {X}}_{I}={\mathbf  {X}}_{i_{1}}+\ldots +{\mathbf  {X}}_{i_{p}}$$. The resultant matrices are computed for the groups $I=I_{1},\ldots ,I_{m}$ and the grouped SVD expansion of $\mathbf {X}$  can now be written as

$${\mathbf  {X}}={\mathbf  {X}}_{I_{1}}+\ldots +{\mathbf  {X}}_{I_{m}}$$

### Step 4: Diagonal averaging

Each matrix $${\mathbf  {X}}_{I_{j}}$$ of the grouped decomposition is hankelized and then the obtained Hankel matrix is transformed into a new series of length $N$ using the one-to-one correspondence between Hankel matrices and time series. Diagonal averaging applied to a resultant matrix $${\mathbf  {X}}_{I_{k}}$$ produces a reconstructed series 

$$\widetilde {\mathbb  {X}}^{(k)}=(\widetilde {x}_{1}^{(k)},\ldots ,\widetilde {x}_{N}^{(k)})$$ 

In this way, the initial series $x_1,\ldots,x_N$ is decomposed into a sum of $${\displaystyle m}m$$ reconstructed subseries:

$$x_{n}=\sum \limits _{k=1}^{m}\widetilde {x}_{n}^{(k)}\ \ (n=1,2,\ldots ,N)$$
This decomposition is the main result of the SSA algorithm. The decomposition is meaningful if each reconstructed subseries could be classified as a part of either trend or some periodic component or noise.

## Actual implementation

### Step 1: Embedding

* compute the **trajectory matrix** (autocorrelation matrix). The matrix is column wise a stack of windows of size L over all the data. So:
    * column 1 is $f_0 .. f_{L-1}$
    * column 2 is $f_1 .. f_L$
    * column 3 is $f_2 .. f_{L+1}$
    * ..

$$\mathbf{X} = \begin{bmatrix}
f_0 & f_1 & f_2 & f_3 &\ldots & f_{N-L} \\ 
f_1 & f_2 & f_3 & f_4 &\ldots & f_{N-L+1} \\
f_2 & f_3 & f_4 & f_5 &\ldots & f_{N-L+2} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
f_{L-1} & f_{L} & f_{L+1} & f_{L+2} & \ldots & f_{N-1} \\ 
\end{bmatrix}$$

* this is also called a **Hankel** matrix


```python
L = 70 # window size
N = len(T) # number of observations
```


```python
X = np.array([F[offset:(offset+L)] for offset in range(0, N - L)]).T
plt.matshow(X[:30])
plt.title("First few examples of values in X");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_41_0.png)


An alternative way of composing the above is as follows (using `np.column_stack`)


```python
X_ = np.column_stack([F[offset:(offset+L)] for offset in range(0, N - L)])
np.allclose(X, X_)
```




    True



### Step 2: Applying SVD

The main idea of this step is to have find $i$ elementarry matrices that exhibit the following property:

$${\mathbf  {X}}={\mathbf  {X}}_{1}+\ldots +{\mathbf  {X}}_{d}$$,

#### The wikipedia way

On the Step 2 we are required to compute the $X*X^T$. 

**Note**: On Python 3 we can use the `@` operator for doing a matrix multiplication. This should be equivalent to doing `np.matmul`


```python
np.allclose(X @ X.T, np.matmul(X, X.T))
```




    True



We're going to print the $X*X^T$ matrix to have a visual idea of what that looks like.


```python
plt.matshow(X @ X.T)
```




    <matplotlib.image.AxesImage at 0x7ffb32cea208>




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_50_1.png)



```python
U, S, _V_t = np.linalg.svd(X @ X.T)
_V = _V_t.T

U.shape, S.shape, _V.shape
```




    ((70, 70), (70,), (70, 70))



Now S consists of L `eigenvalues` $\lambda _{1},\ldots ,\lambda _{L}$ of $\mathbf {S}$  taken in the decreasing order of magnitude ($\lambda _{1}\geq \ldots \geq \lambda _{L}\geq 0$)


```python
plt.plot(S[:10])
```




    [<matplotlib.lines.Line2D at 0x7ffb32c2aef0>]




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_53_1.png)


If we follow the algorithm, we should be constructing $V$ as follows:
* $$V_{i}={\mathbf  {X}}^{\mathrm  {T}}U_{i}/{\sqrt  {\lambda_{i}}} (i=1,\ldots ,d)$$, where $$d={\mathop  {\mathrm  {rank}}}({\mathbf  {X}})=\max\{i,\ {\mbox{such that}}\ \lambda_{i}>0\}$$ (note that $d=L$ for a typical real-life series)

The $rank$ of the matrix is the cardinal of the last index $i$, where number $S_i > 0$. If the rank is lower than the dimentions of $U$ this means that the matrix can be decomposed without loss into fewer dimensions. In most real world examples we're not that lucky, and $rank == U.shape[0]$


```python
rank = np.where(S > 0)[0][-1] + 1
rank
```




    70



Now let's reconstruct the V matrix as in the wiki algorithm.


```python
V = np.zeros(shape=(130, 130))
for column in range(rank):
    V[:, column] = (X.T@U[:, column]) / np.sqrt(S[column])
plt.matshow(V)
```




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_58_1.png)


Numpy decomposes our $\Sigma$ value into a 1D array of singular values, but mathematically, $\Sigma$ should have the shape `(U.shape[-1], V.shape[0])` for the computation to be well defined.

As such we will recompose the propper matrix $\Sigma$ using the function bellow.


```python
def inflate_s(U, S, V):
    """
    Function that takes in the S value returned by 
    np.linalg.svd decomposition and reconstructs the mathematical
    diagonal matrix that it represents. 
    """
    _S = np.zeros(shape=(U.shape[-1], V.T.shape[0]))
    _S[:U.shape[1],:U.shape[1]] = np.diag(S)
    return _S

_S = inflate_s(U, S, V)
plt.matshow(_S)
```




![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_60_1.png)


While $\Sigma$ looks empty, if you look closely you can barely see on the (0, 0), (1, 1) and (2, 2) - the first diagonal- some non zero values.  So it's working properly.

Now let's see what the actual $U * \Sigma * V^T$ looks like. It will obviously not be the same matrix reconstruction as $X$ since we'r using $U$ and $\Sigma$ from the decomposition of the matrix $X * X_T$, and V is a derivative of these two ($U$ and $\Sigma$)


```python
plt.matshow(U @ _S @ V.T)
plt.title(r"Pseudo decomposition of $X * X^T$")
plt.matshow(X)
plt.title(r"Original matrix X")
```



![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_63_1.png)



![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_63_2.png)


Now let's compute the elementary matrices 
$${\mathbf  {X}}_{i}={\sqrt  {\lambda_{i}}}U_{i}V_{i}^{\mathrm  {T}}$$


```python
_X = np.zeros((rank, X.shape[0], X.shape[1]))
for i in range(rank):
    _X[i] = (np.sqrt(S[i])*U[:, [i]])@V[:, [i]].T
```

Notice that in this formulation, we are doing a regular matrix multiplication between `U[:, i]` and `V[:, i].T` but this is equivalent on doing an `outer product` between `U[:, i]` and `V[:, i]`.

In the check bellow we are using `[i]` indexing on columns so we end up with matrices rather than vectors that we can then transpose via `.T` otherwise transposing a vector in numpy doesn't have any effect (so not our intended 1 column and many rows shape for V). This is rather an API quirk.


```python
np.allclose(
    np.outer(U[:, i], V[:, i]),
    U[:, [i]] @ V[:, [i]].T
)
```




    True




```python
def is_square(integer):
    """
    This function is harder to write (correctly) than I have previously imagined.
    
    # https://stackoverflow.com/questions/2489435/check-if-a-number-is-a-perfect-square
    """
    root = math.sqrt(integer)
    return integer == int(root + 0.5) ** 2

def prepare_axis(ax, i):
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel(f"{i}")
    ax.set_yticks([])
    ax.set_yticklabels([])
    return ax

def show_elementary_matrices(X_elem, rank, max_matices_to_show = 9):
    figure_max = 15
    dims_to_show = min((figure_max, max_matices_to_show, rank))

    axes_needed = dims_to_show + 1
    n_rows = int(np.sqrt(axes_needed)) if is_square(axes_needed) else int(np.ceil(np.sqrt(axes_needed)))
    n_cols = int(np.ceil(np.sqrt(axes_needed)))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axs = [prepare_axis(ax, i) for i, ax in enumerate(axs.flatten())];

    for i, s_i in enumerate(S[:dims_to_show]):
        axs[i].matshow(X_elem[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].set_ylabel(f"{i}")
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])

    X_rec = np.sum(X_elem, axis=0)
    axs[-1].matshow(X_rec)
    axs[-1].get_xaxis().set_visible(False)
    axs[-1].set_ylabel("$X_{rec}$")
    axs[-1].set_yticks([])
    axs[-1].set_yticklabels([''])
    fig.suptitle("Diagonalized main elementary matrices of X")
```


```python
show_elementary_matrices(X_elem=_X, rank=rank, max_matices_to_show=15)
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_69_0.png)


We're expecting that the elementarry matrices in _X will all sum up to X.


```python
np.allclose(np.sum(_X, axis=0), X, atol=1e-10)
```




    True



So the difference between them is not significant! We're all set.


```python
def plot_original_vs_rec(X, X_elem):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 2.5))
    ax1.matshow(X)
    ax1.set_title("X")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    X_rec = np.sum(X_elem, axis=0)
    ax2.matshow(X_rec)
    ax2.set_title("X (reconstructed)")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)


    ax3.matshow(np.abs(X - X_rec))
    ax3.set_title("abs(X - X_rec)")
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)


    ax4.matshow(np.round(np.sqrt(np.power(X - X_rec, 2)), 3))
    ax4.set_title("round(MSE(X - X_rec), 3)")
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    fig.suptitle("Original matrix X vs. reconstruction")
    fig.tight_layout()

plot_original_vs_rec(X, _X)
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_73_0.png)


#### The kaggle way

There is also another way of decomposing $X$ into i elementarry matrices $X_i$ that can be summed up to $X$, presented [on this kaggle kernel](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition#3.-Time-Series-Component-Separation-and-Grouping).

This basically avoids doing the $X * X^T$ decomposition through SVD in order to get to a series of elementary matrices $X_i$ and in turn opt for directly decomposing $X$ into $U$, $\Sigma$ and $V$ which will be used to construct the $X_i$ elementarry matrices.  

This has one big advantage in that the result $X$ is closer to the actual values of $X_i$ (fewer steps) and on the whole:
* reduce the number of mistakes that can be done
* reduce the possiblity of floating point operations underflows 
* is more explicit and easyer to understand
* eliminates a lot of mathemathical machinery that is not really obvoius as to why they are needed (remember the convoluted way in which we've build $V$ using the wikipedia way).

Remember that this step is only about finding suitable $X_i$ such as:

$${\mathbf  {X}}={\mathbf  {X}}_{1}+\ldots +{\mathbf  {X}}_{d}$$

We first deconstruct X in $U * \Sigma * V.T$ through SVD.


```python
U, S, V_T = np.linalg.svd(X)
V = V_T.T
U.shape, S.shape, V.shape
```




    ((70, 70), (70,), (130, 130))



Now our elementary matrices are, 
$$X_i = \sigma_i U_i V_i^{\text{T}}$$

and,

$$X = \sum_{i=0}^{d-1}X_i$$

* $\sigma_i$ is the i-th singular value of $\Sigma$. This means `S[i]` in numpy notation (because numpy's SVD returns a vector for S, with the non zero values of the diagonal - remember that S is a diagnoal matrix). 
* $U_i$ is the column i of U
* $V^T_i$ is the column i of $V^T$
* the product between $U_i$ and $V^T_i$ is an [outer product](https://en.wikipedia.org/wiki/Outer_product)
* d is the rank of the matrix $\mathbf{X}$, and is the maximum value of $i$ such that $\sigma_i > 0$. 

(**Note**: for noisy real-world time series data, the S matrix is likely to have  rank=𝐿  dimensions.)

In order to replicate the formula, we will have to construct multiple separable matices $X_i$ that we will then need to sum up. We will have as many such separable matrices as we have singular non-zero values in $S$ (aka rank d).

You can compute the rank using `np.linalg.matrix_rank(X)` but that internally computes the SVD decomposition all over again and as we've seen above we can compute the rank using plain Python with the same result.


```python
d = np.where(S > 0)[0][-1] + 1
assert d == np.linalg.matrix_rank(X)
```


```python
X_ = np.array([S[i] * np.outer(U[:, i], V[:, i]) for i in range(d)])
X_.shape
```




    (70, 70, 130)




```python
show_elementary_matrices(X_elem=X_, rank=d, max_matices_to_show=15)
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_84_0.png)


Let's show in more detail the result of the reconstructions.


```python
plot_original_vs_rec(X, X_)
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_86_0.png)


We observe that in actual fact, there are some errors in the reconstruction, but the total MSE over all pixels is null. This is given by the fact, that the errors are subunitary, on the order of 1e-10. Raising by the power 2 makes these number even furter small. 

We could have illustrated this by rounding as well (like we did bellow). Notice that we now have only 0s.


```python
np.round((X - X_rec)[:10, :], 2)
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])



#### Comparing the two elementary matrices sets

So by now we have to ways of decomposing X into a sum of elemenatary matrices and I'm wondering weather they result in the same values. 


```python
np.allclose(_X, X_)
```




    True



It seems so! To be honest, it's not obvious to me why the two are equivalent, but I'll just take this conclusion based on this experimental evidence.

So whichever route you choose (wikipedia or kaggle) both will yield the same decomposition of X.

#### Understanting each elementary matrices contribution

We expect that the values in S be sorted descending (the first component being the dominant one) 


```python
plt.bar(np.arange(d),S[:d])
```



![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_95_1.png)



```python
S[:d]
```




    array([296.38093887, 169.08832106,  57.86482217,  46.40141   ,
            31.90894601,  15.34198554,  14.86843714,  14.78694165,
            14.5247375 ,  14.37088651,  13.58001463,  13.43212073,
            12.48453521,  12.38210474,  12.23913403,  12.19359197,
            12.06379809,  11.84667796,  10.87244838,  10.54201025,
            10.4285727 ,  10.25184033,  10.1185066 ,   9.87839322,
             9.66298264,   9.62024667,   9.52666637,   9.34031991,
             9.08076815,   8.98204041,   8.88240115,   8.79184386,
             8.69677049,   8.6275053 ,   8.57455309,   8.32935018,
             8.3016696 ,   8.19605236,   8.16795458,   8.0082251 ,
             7.87639454,   7.84820178,   7.78861214,   7.76982839,
             7.52879927,   7.29338437,   7.21568859,   7.10035755,
             6.96965055,   6.87384928,   6.65348868,   6.59733372,
             6.47399648,   6.32487688,   6.20118721,   5.96992107,
             5.836588  ,   5.73879453,   5.71891451,   5.63213598,
             5.4930785 ,   5.3414355 ,   5.30227011,   4.99551775,
             4.58242342,   4.4556975 ,   3.91478919,   3.90050813,
             3.44812975,   3.36613955])



Math shows that 
$$\lvert\lvert \mathbf{X} \rvert\rvert_{\text{F}}^2 = \sum_{i=0}^{d-1} \sigma_i^2$$
i.e. the squared Frobenius norm of the trajectory matrix is equal to the sum of the squared singular values. 


```python
np.allclose(np.linalg.norm(X) ** 2, np.sum(S[:d] ** 2))
```




    True



This suggests that we can take the ratio $\sigma_i^2 / \lvert\lvert \mathbf{X} \rvert\rvert_{\text{F}}^2$ as a measure of the contribution that the elementary matrix $\mathbf{X}_i$ makes in the expansion of the trajectory matrix $X$.

If we're only insterested on the order of the components and their magnitude we could get away without dividing on the norm, as it is the same computation on all the sigular values.


```python
s_sumsq = (S**2).sum()
fig, ax = plt.subplots(1, 2, figsize=(14,5))
number_of_components_to_show = 20

ax[0].plot(S**2 / s_sumsq * 100, lw=2.5)
ax[0].set_xlim(0,number_of_components_to_show + 1)
ax[0].set_title("Relative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
ax[0].set_xlabel("$i$")
ax[0].set_ylabel("Contribution (%)")
ax[1].plot((S**2).cumsum() / s_sumsq * 100, lw=2.5)
ax[1].set_xlim(0,number_of_components_to_show + 1)
ax[1].set_title("Cumulative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
ax[1].set_xlabel("$i$")
ax[1].set_ylabel("Contribution (%)");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_100_0.png)



```python
def cummulative_contributions(S):
    return (S**2).cumsum() / (S**2).sum()

def relevant_elements(S, variance = 0.99):
    """
    Returns the elementary elements X_i needed to ensure that we retain the given degree of variance.
    S is the singular value matrix returned by the svd procedure while decomposing X.
    """
    cumm_contributions = cummulative_contributions(S)
    last_contributing_element = np.where(cumm_contributions > variance)[0][0] + 1
    return np.arange(last_contributing_element)

relevants = relevant_elements(S, variance=0.99)
cummulative_contributions(S)[relevants]
```




    array([0.68453083, 0.90733315, 0.93342601, 0.95020458, 0.95813904,
           0.95997327, 0.96169603, 0.96339995, 0.96504398, 0.96665336,
           0.96809048, 0.96949647, 0.97071108, 0.97190584, 0.97307317,
           0.97423183, 0.97536595, 0.97645962, 0.97738081, 0.97824685,
           0.97909436, 0.97991338, 0.98071124, 0.98147168, 0.98219931,
           0.98292053, 0.98362778, 0.98430764, 0.98495023, 0.98557893,
           0.98619376, 0.98679611, 0.98738551, 0.98796556, 0.9885385 ,
           0.98907915, 0.98961621, 0.9901397 ])



#### Partially reconstructing X, by only using the top 4 components

We see that only the first 4 components (maybe even 3) contribute some meaningfull values to the reconstruction of $X$. The other singular values in S (and thus respective elementary matrices $X_i$ could be ignored).

Let's see what happens if we remove them and only reconstruct X from the first few elementary matrices.


```python
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 6))

def no_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

ax1, ax3 = no_axis(ax1), no_axis(ax3)

compomnents_to_use = 4
ax1.matshow(np.sum(X_[:compomnents_to_use], axis=0))
ax1.set_title(f"Partial reconstruction using only first {compomnents_to_use} components")

ax3.matshow(X)
ax3.set_title("Original matrix")

fig.tight_layout();
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_104_0.png)


Let's also show an animation of that looks like for more components.


```python
from matplotlib import animation, rc
from IPython.display import HTML

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 5))
ax1, ax3 = no_axis(ax1), no_axis(ax3)


plot = ax1.matshow(np.sum(X_[:d], axis=0))
ax1.set_title(f"Partial reconstruction using only first {1} components")

ax3.matshow(X)
ax3.set_title("Original matrix")

fig.tight_layout()

def animate(i, plot):
    ax1.set_title(f"Recon. with {i} comp.")
    plot.set_data(np.sum(X_[:i], axis=0))
    return [plot]

anim = animation.FuncAnimation(fig, animate, fargs=(plot,), frames=min(100, d), interval=200, blit=False)
display(HTML(anim.to_html5_video()))
plt.close()
```


<video width="720" height="360" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAJfsm1kYXQAAAKtBgX//6ncRem9
5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTUyIHIyODU0IGU5YTU5MDMgLSBILjI2NC9NUEVHLTQg
QVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDE3IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcv
eDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9
MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVm
PTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6
b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9NiBsb29r
YWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFj
ZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJh
bWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdl
aWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49NSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNo
PTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFw
bWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAf2ZliIQA
Ev/+963fgU3AQO1rulc4tMurlDQ9UfaUpni2SAAAAwAAAwAAAwB6mPL+rE35KLD0AAADAECACpAX
+mv43JPHwAQ2MsCFFzLLqVD3BXbwR4rQKlJCaR0r+iOYOv1z0+5/43HWm6I9ZdtvNGaNRVk9bR+0
pYpY3LjYvi1FG9d+yUkbKAYkZl/mxWN6RIaU4ZHzT7FZmYJdsD3uJM+5Wq2ICujAeS6b8QU+tmQJ
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
">
  Your browser does not support the video tag.
</video>


### Step4: Diagonal averaging

The wikipedia article and the kaggle kernel diverge somewhat at this point. There are 3 steps that still need to be done:
1. Group the elementary matrices into disjoint groups, summing the groups into a new set of (still) elementary matrices 

$$\begin{align*}
\mathbf{X} &  = \sum_{k \in \mathcal{S}}\mathbf{X}_k + \sum_{l \in \mathcal{T}}\mathbf{X}_l + \ldots \\
             &  = \sum_j \mathbf{X}^{(j)}
\end{align*}$$

2. Hankelise the elementary matrices by diagonal averaging.
3. Derive the timeseries components from the Henkel matrices.

While wikipedia suggest doing 1., 2., 3. the kaggle kernel actually does 2., 3., 1. (so it henkelises first, then extracts the timeseries and then it groups). 

It's unclear what the grouping strategy is for the wikipedia article, so we'll stick to 2., 3., 1. 

#### Henkelisation of the elementary matrices

We need to reconstruct a Henkel matrix (the matrix with diagonal lines)  as this is the thing we've started the deconstruction from, and as we've seen in the animation above adding the elementary matrices leads progressively to a henkel matrix. 

The problem is that we'd like to understand what these elementary matrices represent **as a timeseries component**. 

From a hankel matrix, you can extract the timeseries (`row[0] concat. column[-1]`). To see why this works, remember the shape we gave to X:

$$\displaystyle \mathbf {X} =[X_{1}:\ldots :X_{K}]=(x_{ij})_{i,j=1}^{L,K}={\begin{bmatrix}x_{1}&x_{2}&x_{3}&\ldots &x_{K}\\x_{2}&x_{3}&x_{4}&\ldots &x_{K+1}\\x_{3}&x_{4}&x_{5}&\ldots &x_{K+2}\\\vdots &\vdots &\vdots &\ddots &\vdots \\x_{L}&x_{L+1}&x_{L+2}&\ldots &x_{N}\\\end{bmatrix}}$$

To extract $[x_{1}:\ldots :x_{N}]$ from the henkel matrix $X$ you just take `X[0, :]` and concat that with `X[1:, -1]`.

It's because of this that we actually seek **to force** the elementary matrices to be henkel matrices. This can be done using a process called *diagonal averaging* where each values represents the mean of all the elements of its corresponding diagonal.


A simple and not so efficient way of computing the indices of the elements that compose the diagonal of a certain point is shown bellow


```python
elem, shape = (3, 5), (5, 10)

def diagonal(elem, shape):
    """
    Computes all the elements that compose the diagonal in which 
    the given point (x, y) is a member of.
    """
    def _is_valid(i, j):
        return (0<=i<n) and (0<=j<m)
    
    n, m = shape
    x, y = elem
     
    j_offset = abs(x - y)
    for i in range(max(n, m)):
        j = i + j_offset
        if _is_valid(i, j):
            yield i, j

def numpy_indices(diagonal_gen):
    """
    Just takes the [(x1,y1), ..] generator and reshapes it into [[x1, ...], [y1, ...]] 
    so these can be plugged into a numpy array index
    """
    return tuple(zip(*diagonal_gen))

a = np.zeros(shape)
a[numpy_indices(diagonal(elem, shape))] = 1
a[elem] = 2

plt.matshow(a)
plt.title(f"Showing the diagonal of the {elem} point")
plt.xticks([]), plt.yticks([]);
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_112_0.png)


It's maybe easyer to show these in a short animation


```python
elem, shape = (2, 2), (5, 10)

def init():
    a = np.zeros(shape)
    a[numpy_indices(diagonal(elem, shape))] = 1
    a[elem] = 2

    fig, ax = plt.subplots(figsize=(8, 5))
    plot = ax.matshow(a)
    plt.title(f"Showing the diagonal of the {elem} point")
    plt.xticks([]), plt.yticks([]);
    return fig, ax, plot
    
def animate(i, plot):
    _elem = (elem[0], elem[1] + i)
    
    a = np.zeros(shape)
    a[numpy_indices(diagonal(_elem, shape))] = 1
    a[_elem] = 2
    
    ax.set_title(f"Diagonal of element {_elem}")
    plot.set_data(a)
    return [plot]

fig, ax, plot = init()
anim = animation.FuncAnimation(fig, animate, fargs=(plot,), frames=8, interval=500, blit=False)
display(HTML(anim.to_html5_video()))
plt.close()
```


<video width="576" height="360" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAAXkG1kYXQAAAKtBgX//6ncRem9
5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTUyIHIyODU0IGU5YTU5MDMgLSBILjI2NC9NUEVHLTQg
QVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDE3IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcv
eDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9
MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVm
PTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6
b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9NiBsb29r
YWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFj
ZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJh
bWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdl
aWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49MiBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNo
PTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFw
bWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAACXlliIQA
Ff/+98nvwKbrW1u1v5PPSPwstz7CKj9uWxO06gAAAwAAAwAAEF1oMWAALbiBLMAAAPSUln/Nh6x8
QA9qraSqFN8BwhLiyqX6DIZVaz9CPJd6y0hhs6szLyB142q2AwdWBPQkLiw6ekESeZqPYjH8MgUy
TFTTUyBI3vMf35PS4FQDYFBs2zZtKQci27s6MzJg4ANUJlnzouldKsV56CbHEEJqe45qUbhpgoty
m7+1JMW0PBNHzg/S4htfynBweLarUAIIXrQWb8mtDf0dPwRaRaq7m6pvmq2QyFJsFUMRN0XW58fO
OflybwDc2UbdGsmBHYfHHIFepufuIxHACYj00h5+3LQqo/uT5NVqvMtONCTYp8olHll3ez0x7Oss
mjJw+auw3St3euxVmAKZ04bxGXXzRe1olDRj/d3KW0flnzdfKYdWqLlKiaigREQfh4MGuSBNLjpE
LjxgAHIwTFuu95TDxKrC8Lgp3PxMRs8MOu2hohL8beU3NacVBqkqBvQ+lkrISSY5bgip6p9qvlgS
JsJwLwlTSZSSBxtCvQImTwCPVvD0jUF6Ic7TtLYyPv568dv/dlw96snYqt3H2ycyviDvSH/AVJS3
iDyYTnEErZUgGxilCxlkFFsBkoUv8dZpEUGyRtZ3sIF2dChOPVOgOUY7g5SN2gl074fMJTqNBnB6
URrmW5O+c5Ktv4Ql6ns6wHW9vdaM+jrTJjQoKBdWZ/TrIAU4XskJJZ03/2FJcqi2PinugjkARBWq
/u0t01p4utzTRNE4t2KbpwrKNZOHbDgGAtRROT1B35CGzcc9AG16QSzcdQFjx5Q8wAMVPpdLcTRu
mttIT4mtmFx+eqnITGhYaJfdiDW3U3T+ADreYcJySby8yScYFQprti8pqiw4dZJycmjQrLAHrsUb
SlFcKA97WUoMdjF5ikiI2zDtIf/RUDq4dL2P5aC1+8faK1aeoshVwFTyPgw45sk5q0ySU6k1BL3f
v6dSh+KoLbdkvjZdmClj90Bp+2w9aapsnoKI+WKIbh4b9dXufs/RLoXb6XPPyxgNSfWpr69HXWrp
fquREz3hcLP57ZmDcDM744gYYDowmZ6JKVKuNF478LcEVJ7xc+k0VZL1BmcXHEtS+i4tZORdcPek
6S06tMvDWH0yWdNisHDK31IXrIp0BF/SiQE4GjeD+bLVNV4guvMnhSyqOZN2/jAOe9Y0RCaSO3eF
VpsgMxAo/DiY98k3544WRMWHGvVHtJpfQUwr2al9aZZ1xHwI25KCjBz3nJtGLV12x22WwTZgWMJH
1dHezvSavS45u3IckipVnZkk2jZkZdK5i3a0NDNaprVz1bWKnd5/Ygt93WVZ08k94FEcuIhlgpCJ
UQF8EHxHLK/kbfBnFi7aNoxLh2SqqUrqJeP8R149k5mg9uuaYDUAFTLQ7NiCp0cpAtw5qNqdOYA2
n76Gz8gJ64oNSogZn+3BoBfdib7Tiaj/o5d2c0n7DwBvQfA1KpzFrzhMQq5pfFU/et1gSZTJfyy/
8djGy/QraVZ6uowfnypyaf+1auh1rTYvxnboNNjr3TmqgjOSyY63nJbDlxR901EhEDO/qG6mn4Gy
tyNTMi/NyhdwsByRXpHpNzJ2S+8gzc9g4wOZ5kElO/gVfdpMH1xF7UeEIwPL1p82SkNvpTl45W+h
FkE+bgKEceuWllZV45TxIt4nX0Rf4JLP8sDohc56KhHOkFvylAD4vU5EqSw9SKQDeYnbmvWrXBYn
2v2XyT6fxVbdnFFnZkjjfVRsFD8rBWj0jI6hpJqhvu2vI85ShLukv4Rzu74iriJKT/dlBqJ4VAEM
L13VUsErpphA5KxpytAxQpuDv3hoozcT69buudgQEYpvANrQHRJhrVzWr7D0Rx5iE/9Zmn2LbuKH
zNt9Lw78vx8w4NJXEKHbwq8jxpa1Prdl5p8fL/jIY/kpTHJjtnFQvKVc71AtOw037XV9aKTLQLG6
f7/f6x1cxhStr+vLGvxWbOq1Jz/+UJDuZobtKt7eqX9rcK4b+/e8SEXYwB6bU0yuIjcQ4ALpKhfW
25hByzOdPawF/XWImace4zSCG5DfIOnDGQqrQQTQB+ELFjE2tduedtMIV8xWOf7De9+Rz4ZbVcnt
Q+HcVhOt5/LFqhIyYgE93xk0Feu175k/2eb8cvDevcYALa7/Ilf0sJ2N7vfYPxHNLcDj6Yj7YC2v
vHGvUGvxd0Cw2cZ3zPOgJhK4Q0R54GTVNFrAg9/yCxhZ7C+zbUHhlIpmfjShx3gopcPiWteBsUaQ
KakFn5UHsA8uQ4qXwC+mR3SrJ9rHGoVvxtkEikfRCpmcFIVwExh339hP3/FSwJaZbcP/gWhEiAq9
Jt6M5vJbI6/oN9zqSUa7A2hW8L5FEN0lTxA9d/6JsntKH8Tw8Ju5ssCrXCwbWATjVz5QENmSsIPF
mI+pGYy7MINna1otaxppxOiveW3qJScD8GOr4+H8NZNT2qpuPjHWSZ4X1DUHobrD/ChznE0V3qf5
zRyT9favNABHy0GxSbwVD64XKYgwrkAphhqPBomlSN154uXqChhqkT14qCRGi3i/OJ/5cZnKTOX+
5wu4cralvSmqiHxuwaP23MKHulQh8Pr6eb1YWtfovqTHatVTl3PjwMrn42RKoAicBd6bVjsIC5R2
aldS0QMVfLaP+F8INgXJKFnf6wb/W4/cTis4gN94MoE9aoJBP7twADtwLuMrCrl7oWveuSdbnlN5
bu5swFsYvZL3j0e0GKrH/f8c9wAXvT4PDMpUGeQZAkcFATI9nhEwA66cUu3hxfR+kmLYmKUjpnEj
Meu6ei+H8d3RFZdjKL/b4rcQNAPHwnndkqV25VHhajkQoULGLGordoiO7dJPtDPnvkh6tM49CsJF
ORcKIW8iLS70XeZzJbbRT94KlWJlgOawtH6vrL9bxHt0tOz63HY7LQEqoIVMcmmLgzP+IRtpLY38
Zy0k6FUtIwm/Lhu53M9K6wmF7BCdp4T4AwcItQB7lUXRj57anwU6/ysLEO5MUOCLWmk9eS92T4ns
RUPxmuPlhPjWFU334EwuEjpjwNaRfA/xFaTy42qcKTrEYbFlRdwIbXU1AHgy5a11HOpBZPMYzI7U
o8LZgnsaQ8434B7vF0/O9L/7XAQcV4GYEahiunAjUsGxTc/H6CGm5AABDw9OhL7/asfBh0SrjTjq
l15aF+bj3MUg/J3bGaSHKDQ6AAADAAADAAYFAAACakGaImxBX/7WpVABqNhv4AbeeJ88y/xQXdAt
6nDrlqUxaAn2DEGrgC56cGVk9UAl0xoDde4de7KrXXnMrH5NVaWlpfel9Xw69us27pGTg8e0/Yt3
Pd2YDIn7JiT+HgWi/B81n6rMLlEODGHNXkZJHMo2xeiLoAgd5ofMjY66Uz3HIRkgkjv6vylmhtHo
b/yfqf444QngU/bft62DEboC29nL2oXwwoYwFogOpewKdfUqUQ5c6CIIzG7S8ZQzX7C4AuZxUEmV
6xdqlyIpLide6xPNfIMj+pNDSRUwd3jgZmJNB98t4U3ffmZKpDWehi7VSezgFmVVcFkb23iGvG+s
T+RjzCPb5vnBxrCAsb5472izMAosd+Ymyh1Y5Qj8OvoxNAIdwamoP9b8R76zGqkHoS7tQkkYlA7A
7ZLvUotF2BJd/ZrdYl/feLCZre3UWCan6VWeDLCVF+5enY3/fQn7Aj94AW1tMwHZ4UDJ6rsWU0gU
0CeHucdJeraFnEcltxJKpsMIYQxB8fElN2YGqNzj3ds10kqk8AXIXvidnFFAVVV9uGM5d9aTKV7o
I0vMs3yb0RpLMAYdbvOyt+SGa2eTq5oL4DBzzpy9MdzDp859JDb537mm2fSl08PCXcGcHJeLTl7z
LDQYoSzmqQPqAWZpYeYIT7cAUZe0ahMxabj0kDxLKynynz4ZCzmRM4bQLwIKF/J6cEozgVRtm3V4
POOfm95k928fIgISFBGy3keJuwLGeU5brFZFTo2J87ncPI4xoiYDcxoa3UMvonyMVMqVOlf8WSDS
fcodN5vHJtottGy2FwqTKtYIoseduAAAAVUBnkF5BL8ABmlMw8aMnYAWAD9AHVfTktWH2tdz64VR
HkJjn7W5hkyscQaoqmZRHYwXLjTv2H7uW2L/YNQ3hPF7ODZknq/f1wM75kI+cGe35LjsxYGN7P55
vOgX1PLgBZbWFryN/nj5qq6RvTEVMEj0jToPtql33NmpBMNfes8GigWDBWY6xS9rs7T4RP8HpVeU
Vtche1ew+TPq1MVMxG+0P40wLd/L/1XCaASwWpVRzHwC175BJZY6YJDBZRiYe1TQBgmfPL2MgM1P
NNvREjdvC0eJQfFJVGLCUwUznKoTf7XGXO+o86p/HPIRgLKePrhh16qNdHM19y8R9DkzL7AaCTh1
TfqGrvPPr6C9YJGDYbZBzn/Ukh/B7HLaGCA/o8leJsRApgv+6N0M7ML/eiiVuw5hDmtjL9PMtlD0
Zeqb5YzW8SWRGNBJUxl95w4b6WiBrV5swQAAAhtBmkQ8IZMphBT//taMsALcBuoAAhPbhqPpAfbP
K1rr6hmqEJ9YZKxUloh5jO8dU5v5yTgp356G8FEZukBDIuQsAzBSaMDNA7Jgtcp0Su+jUl+x2q4l
rto2c64e8+WsRTU3JcJ/jSf/kZg0DmCfNwRj6a5QvWwMEuIkq2UwYsxVT3KQEQp+cPuAa55l/zoB
npzfDbdN/xgLCwAkTQiB6OqXb8NA57mJcegqat6riKNu7iBy/5nSxEGYFf3rEghg/pfVwNVlJ+Vh
MTQLGC4DYVMwhpLwdneG/309wjFvucC79sI3Hb4dlM6gp6nBdzKF8uHJz7ZWj39WNTuHgnT15a8w
9f57HSi4Wxut2zUu7R7PLDDGS9cLYphciVTLlKAb1YJqy9Sg2bnZ+m+g6VOR44OUmH0upG5iNRMX
bBKKr6HZFeXV9IkxxtsFnDBPvq4lZ5RWmmdyFjDicbLVKFgpE4pnDByDTqAHJzmMM3vi0RDAnkzj
kD70zVMYYlS6wOxHR09BAtzrNbvskbf8eXcF5rjOV7fNQAMDFDSlY1E/2f6hOtoPT3o7iV+qJ4AX
1p7YyBBTkje5vihbI8CH2xzWVU0icxj6L7gwxdrQ43Px1HQ3E4kT2eWeg9IHOCzJTAbxK5qeaUIS
XbQptfWrfIziK2Cfrd4jn8til90PiMynWP0MGkaus96yvhp1OyV8iMc3+SDERn/i/kinQDBeYAAA
AXgBnmNqQS8ABYRaxMEGAASdh4Bmq0kMdWHFfwaUd6nWpKEB3fKK77JeIOc/2YCfUEHR2MDMXflr
286cioJHzMcWOMoUipGZe3HkbMkfv7lcZ6K4XQdkxyNLvramR5mzdJkMvDJSHxs/QbPMIFe1tCQq
JjVC9mlbzV3g4xNS+yYJJ7ygpRrMVeeNKgkTHEUtA7D4oO2g4kl/B4wllr9/u+8VsPbgDni2rZJf
4KD2LdOcSJ2zOP+erbl1BQpeQ7+ABBzu4vR7rrmlPNvA1cAWzjk+/0m8MMp+7vo4A47KRYSJFqhq
m85C7zseo67lno88kg54M3nEdwHCLND6xkCG1dXEKyt8/mUrvJhSD0oYA87ZWwd8Z/y62G6FrJpy
JnEkbq13FXkSNiB6Me9gSr0E4Z0IXKOBnIVju6k7hhzRnJGwGLbpVq5oythPy33pdPC42suT/qCu
9R8BKkb+ApqmJzk7a8AXRd5tWUAXfdiSlYdlZanryz0tdwuDAAABPkGaZUnhDyZTAgp//taMsALc
FuDQAPktw1S5YPD1z7hZdzW6rSz39aKGexGLGrX6CX3/LcAJM8e1P7J5lV10fuyOkBTwJrYZ84lV
7LP21hnivMmBPeSY00KTlefMLBQGiH24WQ8c4pLfHD+A6FdkEKhwwVYYr4UsxKjJpfaFN52fNPWG
F/ZZMw4vBJ2UokoNU4tv26t5zLd1pEtLaxatjQ/Q5jZY8S/IUf5Y0vaumfsA2zXUaym2WgSxLD95
i8qpL+yOjS8IhZine1/Y+7WxRm5N0BN0e5zuDzy/MGZTQYV8/HOAot19u/UXL4YrkU4myMOFKBgv
visYWMEUAAGSM+WEzYQF4vLNTgzs6GFbfGaQR2paw3Kis265YeP6o/zTa8XibVFcAiL8Zp6SAtqR
LZ/sotHEqL/D9vj0vQ4h1QAAAYJBmoZJ4Q8mUwIKf/7WjLAC3ZSxeAAQuwABmoIn+4g+vkNZy3iL
6OE9T/BkQQ4i6AfDkxIiZMwxQCrVckEB6NOtDpsfVKeVV3DDgZfLOZYbY1mlcBH6MlmPB4AWPE9f
76m2HuLRVnE48bXpJlU4QL+e5k8cacgjKFkxMMc0wCuoXVx65W+h0DzFTBO4kirqiqfYwMXiJ8/w
77xmLA3ukES4Rv+MxpJEfBfHZ+OX01jmgUxuA501DQ+q9bMn+mxfZoQVIJPC9uk1eI4ji8yH6ISM
7JFvpYzBGrilHW/Cy+hdc9FNfmopRaX7d7TN3eVUEtIpDvaQh/enxmfUbIFfvhDWG8lZK4x9/2+K
GrEqvLPkodtvK89LjEMnwX+J8F7RumXaC7wka7IMIHt6FfcHtbPmohyOBND733/O1IivD8TfkWgc
QjoiJEqddAas9diz0/YNH8/39xQRMYg+OKZ4C2/RZ5/cKWdhug4eO0N/u0a3qYWhPiP6hhbwydAq
o9L9yBP7cwAAASxBmqdJ4Q8mUwIJf/61KoALKUWaACac5uu7eKCTh8Y/JjwbVeAEYQo3pfxYXCpm
haVVaP+HWVLjH5cROXO1WtD1Mxa/lTmPszKIwCocuGM2nIE87HNWqEwk0OVSU9NXN02+jhupVWk5
x71XjjpWLv1ftbRSa4aAzw/XoaaRU2G3SNqtnNm3+cfA4t0sVZngOnyutxagGR3eRh1cy3Yh2zcf
8djSeq9QyuVO3B/q7FuqNhfnPDQKHDDB9bKnaRvU0/FotcLY3z4MgW2EFAbxpllwXyR0QBWN4mwk
GTkVsHfdwHRVaY8bE8iqhCxvL2rvnd/5XExZPOrd5MxRhZC2Z4ZaWBjVyZHiCoaO/u2HekxTsp7N
6x++ZCX4QRVybXnuxkraGrA6iUH5MmDR44EAAAN/bW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD
6AAAD6AAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAql0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAA
AAABAAAAAAAAD6AAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAA
AABAAAAAAkAAAAFoAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAA+gAABAAAABAAAAAAIhbWRp
YQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAABAABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAA
AAAAAAAAAABWaWRlb0hhbmRsZXIAAAABzG1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5m
AAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAYxzdGJsAAAAtHN0c2QAAAAAAAAAAQAAAKRh
dmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAkABaABIAAAASAAAAAAAAAABAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAAMmF2Y0MBZAAW/+EAGWdkABas2UCQL/lhAAADAAEA
AAMABA8WLZYBAAZo6+PLIsAAAAAcdXVpZGtoQPJfJE/FujmlG88DI/MAAAAAAAAAGHN0dHMAAAAA
AAAAAQAAAAgAACAAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAABAY3R0cwAAAAAAAAAGAAAAAQAAQAAA
AAABAABgAAAAAAEAACAAAAAAAQAAYAAAAAABAAAgAAAAAAMAAEAAAAAAHHN0c2MAAAAAAAAAAQAA
AAEAAAAIAAAAAQAAADRzdHN6AAAAAAAAAAAAAAAIAAAMLgAAAm4AAAFZAAACHwAAAXwAAAFCAAAB
hgAAATAAAAAUc3RjbwAAAAAAAAABAAAALAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAA
AAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZm
NTcuODMuMTAw
">
  Your browser does not support the video tag.
</video>


What we actually want is to compute the **anti**diagonal. Which is...


```python
elem, shape = (2, 0), (5, 10)

def antidiagonal(elem, shape):
    """
    Computes the elements of the antidiagonal
    """
    def _is_valid(i, j):
        return (0<=i<n) and (0<=j<m)
    
    (x, y), (n, m) = elem, shape
    tot = x + y
    for i in range(tot + 1):
        j = tot - i
        if _is_valid(i, j):
            yield i, j

def init():
    a = np.zeros(shape)
    a[numpy_indices(diagonal(elem, shape))] = 1
    a[elem] = 2

    fig, ax = plt.subplots(figsize=(8, 5))
    plot = ax.matshow(a)
    plt.title(f"Showing the diagonal of the {elem} point")
    plt.xticks([]), plt.yticks([]);
    return fig, ax, plot

def animate(i, plot):
    _elem = (elem[0], elem[1] + i)
    
    a = np.zeros(shape)
    a[numpy_indices(antidiagonal(_elem, shape))] = 1
    a[_elem] = 2
    
    ax.set_title(f"Antidiagonal of element {_elem}")
    plot.set_data(a)
    return [plot]

fig, ax, plot = init()
anim = animation.FuncAnimation(fig, animate, fargs=(plot,), frames=10, interval=500, blit=False)
display(HTML(anim.to_html5_video()))
plt.close()
```


<video width="576" height="360" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAAZo21kYXQAAAKtBgX//6ncRem9
5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTUyIHIyODU0IGU5YTU5MDMgLSBILjI2NC9NUEVHLTQg
QVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDE3IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcv
eDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9
MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVm
PTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6
b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9NiBsb29r
YWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFj
ZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJh
bWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdl
aWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49MiBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNo
PTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFw
bWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAACUtliIQA
Ff/+98nvwKbrW1u1v5PPSPwstz7CKj9uWxO06gAAAwAAAwAAEF1oMWAALbiBLMAAAPScEn/DQuPi
ASZ3ch625U+5AW4x91prTlr6DB4+2td8JTzNzsgHMHVp8WJG0exO4K4gm0g2gsBhZ7a8JB3uT953
wxJEKkk0RFK2rFC1lujHZTy6SfJRZlOSCLXFmtdrQiQNmlP5yUrTddZIniTr3Bt/gPgmu25hRJvX
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
">
  Your browser does not support the video tag.
</video>


The henkelisation procedure is the following:
* we will scan all the (i,j) elements of the matrix $X_i$
* we will compute the indices D of the antidiagonal of which (i,j) is an element of
* we will compute the mean of the values of the elements found on D
* set the mean above as the (i,j) values in the henkelized $X_{hat}$ matrix.


```python
def henkelize(X):
    """
    Henkelisation procedure for the given matrix X.
    Returns a matrix where all the elements on the antidiagonal are equal.
    """
    X_hat = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            indices = numpy_indices(antidiagonal((i,j), X.shape))
            X_hat[i, j] = X[indices].mean()
    return X_hat
```


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1, ax2 = no_axis(ax1), no_axis(ax2)

ax1.matshow(X_[0])
ax1.set_title(r"Original elementary matrix $X_i$")

ax2.matshow(henkelize(X_[0]))
ax2.set_title(r"Henkelized elementary matrix $X_{hat}$")
fig.suptitle("Henkelisation result")
fig.tight_layout();
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_119_0.png)


Quoting the original source, 

>it is important to note that $\hat{\mathcal{H}}$ is a linear operator, i.e.  $\hat{\mathcal{H}}(\mathbf{A} + \mathbf{B}) = \hat{\mathcal{H}}\mathbf{A} + \hat{\mathcal{H}}\mathbf{B}$. Then, for a trajectory matrix $\mathbf{X}$, 
$$\begin{align*}
\hat{\mathcal{H}}\mathbf{X} & = \hat{\mathcal{H}} \left( \sum_{i=0}^{d-1} \mathbf{X}_i \right) \\
                            &  = \sum_{i=0}^{d-1} \hat{\mathcal{H}} \mathbf{X}_i \\
                            &  \equiv \sum_{i=0}^{d-1} \tilde{\mathbf{X}_i}
\end{align*}$$ 
As $\mathbf{X}$ is already a Hankel matrix, then by definition $\hat{\mathcal{H}}\mathbf{X} = \mathbf{X}$. Therefore, the trajectory matrix can be expressed in terms of its Hankelised elementary matrices:
$$\mathbf{X} = \sum_{i=0}^{d-1} \tilde{\mathbf{X}_i}$$

Which means that doing the henkelisation on all the elementary matrices X resulted from the decomposition will not destroy the additive property of the elementary matrices. We will still end up with a set of elementary matrices that add up to the original henkelized X that we started from. Only now, the great benefit of them being the henkel format is that we can extract the time series.


```python
X_hat = np.array([henkelize(X_i) for X_i in _X])
```


```python
show_elementary_matrices(X_hat, d, max_matices_to_show=15)
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_122_0.png)



```python
from matplotlib import animation, rc
from IPython.display import HTML

def init():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = no_axis(ax1), no_axis(ax2)

    plot = ax1.matshow(np.sum(X_hat[:d], axis=0))
    ax1.set_title(f"Partial reconstruction using only first {1} components")

    ax2.matshow(X)
    ax2.set_title("Original matrix")

    fig.tight_layout()
    return fig, ax1, plot

def animate(i, plot):
    ax1.set_title(f"Recon. with {i} comp.")
    plot.set_data(np.sum(X_hat[:i], axis=0))
    return [plot]

fig, ax1, plot = init()
anim = animation.FuncAnimation(fig, animate, fargs=(plot,), frames=min(d, 100), interval=100, blit=False)
display(HTML(anim.to_html5_video()))
plt.close()
```


<video width="720" height="360" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAGLMm1kYXQAAAKuBgX//6rcRem9
5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTUyIHIyODU0IGU5YTU5MDMgLSBILjI2NC9NUEVHLTQg
QVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDE3IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcv
eDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9
MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVm
PTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6
b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9NiBsb29r
YWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFj
ZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJh
bWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdl
aWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49MTAgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVz
aD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBx
cG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAG04ZYiE
ABD//veBvzLLXyK6yXH5530srM885DxyXYmuuNAAAAMAAAMAAAMB+8C/gSCcz9Z0IAAAAwM6AJCB
Rc//j9Mj+ACGuujus6YyxIOypmLONDZm55Vx0ZWWC1VK4Mn3oRxm4KceQiDBmlM6J+kWz/xItTMa
sOHGoV60qPpjiHGplKiprwrc76lpOcwgArWACQdWM0UebWffmtcN1QkvNiJlKVOg+2LvoNX8u3qm
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
AAAAAAPoAAAbWAABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAA
AEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAFkXRyYWsAAABcdGtoZAAAAAMAAAAA
AAAAAAAAAAEAAAAAAAAbWAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAA
AAAAAAAAAEAAAAAC0AAAAWgAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAG1gAAAgAAAEAAAAA
BQltZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAACgAAAEYAFXEAAAAAAAtaGRscgAAAAAAAAAAdmlk
ZQAAAAAAAAAAAAAAAFZpZGVvSGFuZGxlcgAAAAS0bWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAA
JGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAAEdHN0YmwAAAC0c3RzZAAAAAAAAAAB
AAAApGF2YzEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAC0AFoAEgAAABIAAAAAAAAAAEAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAY//8AAAAyYXZjQwFkABb/4QAZZ2QAFqzZQLQv+WEA
AAMAAQAAAwAUDxYtlgEABmjr48siwAAAABx1dWlka2hA8l8kT8W6OaUbzwMj8wAAAAAAAAAYc3R0
cwAAAAAAAAABAAAARgAABAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAAjBjdHRzAAAAAAAAAEQAAAAD
AAAIAAAAAAEAABQAAAAAAQAACAAAAAABAAAAAAAAAAEAAAQAAAAAAQAADAAAAAABAAAEAAAAAAEA
AAgAAAAAAQAADAAAAAABAAAEAAAAAAEAAAwAAAAAAQAABAAAAAABAAAUAAAAAAEAAAgAAAAAAQAA
AAAAAAABAAAEAAAAAAEAABQAAAAAAQAACAAAAAABAAAAAAAAAAEAAAQAAAAAAQAADAAAAAABAAAE
AAAAAAEAAAwAAAAAAQAABAAAAAABAAAUAAAAAAEAAAgAAAAAAQAAAAAAAAABAAAEAAAAAAEAAAwA
AAAAAQAABAAAAAABAAAUAAAAAAEAAAgAAAAAAQAAAAAAAAABAAAEAAAAAAEAABQAAAAAAQAACAAA
AAABAAAAAAAAAAEAAAQAAAAAAQAADAAAAAABAAAEAAAAAAEAABQAAAAAAQAACAAAAAABAAAAAAAA
AAEAAAQAAAAAAQAAFAAAAAABAAAIAAAAAAEAAAAAAAAAAQAABAAAAAABAAAUAAAAAAEAAAgAAAAA
AQAAAAAAAAABAAAEAAAAAAEAABQAAAAAAQAACAAAAAABAAAAAAAAAAEAAAQAAAAAAQAAFAAAAAAB
AAAIAAAAAAEAAAAAAAAAAQAABAAAAAABAAAUAAAAAAEAAAgAAAAAAQAAAAAAAAABAAAEAAAAAAEA
ABQAAAAAAQAACAAAAAABAAAAAAAAAAEAAAQAAAAAHHN0c2MAAAAAAAAAAQAAAAEAAABGAAAAAQAA
ASxzdHN6AAAAAAAAAAAAAABGAABv7gAABdUAAAZ/AAAJEgAAAzIAAAGkAAAA/QAADGoAAACcAAAK
XAAAFJIAAAC6AAAO6gAAAIUAAB29AAAAywAAAKsAAACAAAAQ0AAAAJ8AAADSAAAAqAAACaEAAABw
AAALDgAAAJcAABEUAAAAjgAAAIcAAACbAAAKGQAAAK4AAA5XAAAAewAAAJcAAACZAAAObgAAAI4A
AACAAAAAiQAAB1cAAACdAAAOAgAAAHsAAACBAAAAiQAACNEAAACKAAAAggAAAHIAAAfKAAAAfwAA
AKQAAABSAAAGoQAAAL0AAACHAAAAZAAABPIAAACXAAAAegAAAFUAAAHCAAAAYwAAAEwAAAB1AAAA
XwAAAFYAAABDAAAAegAAABRzdGNvAAAAAAAAAAEAAAAsAAAAYnVkdGEAAABabWV0YQAAAAAAAAAh
aGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAAB
AAAAAExhdmY1Ny44My4xMDA=
">
  Your browser does not support the video tag.
</video>



```python
plot_original_vs_rec(X, X_hat)
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_124_0.png)


#### Extracting the elementary timeseries

So now, let's just extract the time series components from each of the henkelized elementary matricies.


```python
def timeseries(X_):
    return np.concatenate((X_[0, :], X_[1:, -1]))

[plt.plot(timeseries(X_h)) for X_h in X_hat[:10]]
plt.legend([f"X_hat[{i}]" for i in range(10)], loc=(1.05,0.1));
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_127_0.png)


Visually we can assume that:

$$\begin{align*}
\tilde{\mathbf{X}}^{\text{(trend)}} & = \tilde{\mathbf{X}}_0 + \tilde{\mathbf{X}}_1 
    & \implies &  \tilde{F}^{\text{(trend)}} = \tilde{F}_0 + \tilde{F}_1 \\
\tilde{\mathbf{X}}^{\text{(periodic 1)}} & = \tilde{\mathbf{X}}_2 + \tilde{\mathbf{X}}_3 
    & \implies & \tilde{F}^{\text{(periodic 1)}} = \tilde{F}_2 + \tilde{F}_3  \\
\tilde{\mathbf{X}}^{\text{(periodic 2)}} & = \tilde{\mathbf{X}}_4 + \tilde{\mathbf{X}}_5 
    & \implies & \tilde{F}^{\text{(periodic 2)}} = \tilde{F}_4 + \tilde{F}_5\\
\tilde{\mathbf{X}}^{\text{(noise)}} & = \tilde{\mathbf{X}}_6 + \tilde{\mathbf{X}}_7 + \ldots + \tilde{\mathbf{X}}_{69}
    & \implies & \tilde{F}^{\text{(noise)}} = \tilde{F}_6 + \tilde{F}_7 + \ldots + \tilde{F}_{69}
\end{align*}$$


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(timeseries(X_hat[0])+timeseries(X_hat[1]))
ax1.plot(timeseries(X_hat[2])+timeseries(X_hat[3]))
ax1.plot(timeseries(X_hat[4])+timeseries(X_hat[5]))
ax1.plot(np.sum([timeseries(X_hat[i]) for i in range(6, d)], axis=0), alpha=0.2)
ax1.legend(["Trend", "Seasionality_1", "Seasionality_2", "Noise"], loc=(0.025,0.74))
ax1.set_title("Reconstructed components")

ax2.plot(apply(trend, T), alpha=0.4)
ax2.plot(apply(period_1, T), alpha=0.4)
ax2.plot(apply(period_2, T), alpha=0.4)
ax2.plot(apply(noise, T), alpha=0.4)
ax2.legend(["trend", "period_1", "period_2", "noise", "function"], loc='upper right')
ax2.set_title("Original components");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_129_0.png)


There are a few observations we can draw from the above image:
* The components are scaled / normalized (noise was in the range 0 and 2 originally, while the reconstructions puts it in the range -1 and 1)
* We're still left with some seasionality component into the noise, evidenced by the wiggly shape of it and by the fact that the second seasionality component has a lower amplitude than we've expected
* The first seasinoality component was not actually decomposed entierly. It seems that it is actually a composition of two shapes, in this case each from the other original seasionality element. This is evidenced by the uneven ampltides in it (which usually happens when you add up regular waves with different amplitudes and frequencies).

The kaggle kernel [we've kept refencing](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition) offers a quicker way of extracting the timeseries directly from the elementary matrices $X_i$ (bypassing the henkelisation steps). Let's compare our extractinos with theirs both visually and quantitavely.


```python
def timeseries_from_elementary_matrix(X_i):
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])
```


```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(timeseries(X_hat[0])+timeseries(X_hat[1]))
ax1.plot(timeseries(X_hat[2])+timeseries(X_hat[3]))
ax1.plot(timeseries(X_hat[4])+timeseries(X_hat[5]))
ax1.plot(np.sum([timeseries(X_hat[i]) for i in range(6, d)], axis=0), alpha=0.2)
ax1.legend(["Trend", "Seasionality_1", "Seasionality_2", "Noise"], loc=(0.025,0.74))
ax1.set_title("Step-by-step reconstruction")

ax2.plot(timeseries_from_elementary_matrix(X_[0])+timeseries_from_elementary_matrix(X_[1]))
ax2.plot(timeseries_from_elementary_matrix(X_[2])+timeseries_from_elementary_matrix(X_[3]))
ax2.plot(timeseries_from_elementary_matrix(X_[4])+timeseries_from_elementary_matrix(X_[5]))
ax2.plot(np.sum([timeseries_from_elementary_matrix(X_[i]) for i in range(6, d)], axis=0), alpha=0.2)
ax2.legend(["Trend", "Seasionality_1", "Seasionality_2", "Noise"], loc=(0.025,0.74))
ax2.set_title("Direct reconstruction")

ax3.plot(apply(trend, T), alpha=0.4)
ax3.plot(apply(period_1, T), alpha=0.4)
ax3.plot(apply(period_2, T), alpha=0.4)
ax3.plot(apply(noise, T), alpha=0.4)
ax3.legend(["trend", "period_1", "period_2", "noise", "function"], loc='upper right')
ax3.set_title("Original components");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_133_0.png)


Testing that we get the same reconstructed values from both methods.


```python
for i in range(d):
    assert np.allclose(timeseries(X_hat[i]), timeseries_from_elementary_matrix(X_[i]))
```

Yup, the same values!

### Step 3: Grouping 

Obviously grouping in such a manner as we did above (visually) is both error-prone and time consuming and we need a more automated / structured way of doing this. The kaggle kernel [we've kept refencing](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition) offers one such approach. 

It's based on the idea of:
* computing a similariy metric between each timeseries component extracted.
* use the resulted similarity matrix on deciding a way to group (still by manual inspection).


```python
components = np.array([timeseries(X_h) for X_h in X_hat])
last_relevant = relevant_elements(S, variance=0.97)[-1]
```


```python
def show_top_correlations(ax, component, top=10, variance=0.96):
    last_relevant = relevant_elements(S, variance=variance)[-1]
    top = min(top, last_relevant + 1)
    most_correlated = np.argsort(Wcorr[component, :top])[::-1][1:]
    ax.bar(np.arange(1, top), Wcorr[component, most_correlated])
    ax.set_xticks(np.arange(1, top))
    ax.set_xticklabels(most_correlated[:top])
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f"Related components to {component}")
    
    
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 4))
show_top_correlations(ax1, 0, 10)
show_top_correlations(ax2, 1, 10)
show_top_correlations(ax3, 2, 10)
show_top_correlations(ax4, 3, 10)
show_top_correlations(ax5, 4, 10)
fig.suptitle("Related components to the top 5 elements (by variance contribution)");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_140_0.png)


#### Computing a similarity matrix

We're going to explore multiple ways of comparing the timeseries components and choose one that seems more promissing.

##### Similarity method 1: Weighted dot product

The kernel uses a weighted, normalized dot product of two vectors as the similarity matrix.

>For two reconstructed time series, $\tilde{F}_i$ and $\tilde{F}_j$, of length $N$, and a window length $L$, we define the *weighted inner product*, $(\tilde{F}_i, \tilde{F}_j)_w$ as:
$$(\tilde{F}_i, \tilde{F}_j)_w = \sum_{k=0}^{N-1} w_k \tilde{f}_{i,k} \tilde{f}_{j,k}$$
where $\tilde{f}_{i,k}$ and $\tilde{f}_{j,k}$ are the $k$th values of $\tilde{F}_i$ and $\tilde{F}_j$, respectively

>Put simply, if $(\tilde{F}_i, \tilde{F}_j)_w = 0$, $\tilde{F}_i$ and $\tilde{F}_j$ are *w-orthogonal* and the time series components are separable. Of course, total w-orthogonality does not occur in real life, so instead we define a $d \times d$ ***weighted correlation*** matrix, $\mathbf{W}_{\text{corr}}$, which measures the deviation of the components $\tilde{F}_i$ and $\tilde{F}_j$ from w-orthogonality. The elements of $\mathbf{W}_{\text{corr}}$ are given by
$$W_{i,j} = \frac{(\tilde{F}_i, \tilde{F}_j)_w}{\lVert \tilde{F}_i \rVert_w \lVert \tilde{F}_j \rVert_w}$$
where $\lVert \tilde{F}_k \rVert_w = \sqrt{(\tilde{F}_k, \tilde{F}_k)_w}$ for $k = i,j$. The interpretation of $W_{i,j}$ is straightforward: if $\tilde{F}_i$ and $\tilde{F}_j$ are arbitrarily close together (but not identical), then $(\tilde{F}_i, \tilde{F}_j)_w \rightarrow \lVert \tilde{F}_i \rVert_w \lVert \tilde{F}_j \rVert_w$ and therefore $W_{i,j} \rightarrow 1$. Of course, if $\tilde{F}_i$ and $\tilde{F}_j$ are w-orthogonal, then $W_{i,j} = 0$. Moderate values of $W_{i,j}$ between 0 and 1, say $W_{i,j} \ge 0.3$, indicate components that may need to be grouped together.


The similarity metric looks awfully similar to a cosine similarity between two vectors, only that this time the inner product is weighted by the dimension of each antidiagonal (so by how many times each value in $x_i, x_j$ appears in their own henkelized matrix). 

It's important to note here that both $x_i$ and $x_j$ are values found **at the same** index (position) in theis respective timeseries component $F_i$ and $F_j$. It's because of this, that they have an equal weight (i.e. their multiplier in the henkelized matrix is given by the dimension of the antidiagonal that owns them, which is has the same value, because they share the same position). 

Let's compute the w array first (the weights).


```python
shape = (5, 10)
a = np.zeros(shape=shape, dtype=np.int)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        a[i, j] = len(list(antidiagonal((i, j), shape)))


fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5))
ax = no_axis(ax)

ax.matshow(a, cmap=plt.cm.Blues)

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        ax.text(j, i, str(a[i, j]), va='center', ha='center')
ax.set_title("Parent antidiagonal dimension owning each element")    


counts = timeseries(a) 
ax2.matshow([counts], cmap=plt.cm.Blues)
ax2.set_xticks(list(range(a.shape[0] + a.shape[1])))
for i in range(a.shape[0] + a.shape[1] - 1):
    ax2.text(i, 0, str(counts[i]))

ax2.get_yaxis().set_visible(False)
ax2.xaxis.tick_bottom()
ax2.set_xlim((-0.5, a.shape[0] + a.shape[1] - 1.5))
ax2.set_title("Unrolled dimension of antidiagonal containing each element (the W array)");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_147_0.png)



```python
def compute_weights_array(L, N_K):
    return np.array(list(np.arange(L)+1) + [L ]*(N-L-1) + list(np.arange(L)+1)[::-1])

w = compute_weights_array(5, 10)
w
```




    array([1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1])



We're going to compute the $W_{corr}$ matrix and only retain the elements that ensure 97% of the variance.


```python
w = compute_weights_array(70, 200 - 70)

# Calculate the individual weighted norms, ||F_i||_w, first, then take inverse square-root so we don't have to later.
F_wnorms = np.array([w.dot(components[i]**2) for i in range(d)])
F_wnorms = F_wnorms**-0.5

# Calculate the w-corr matrix. The diagonal elements are equal to 1, so we can start with an identity matrix
# and iterate over all pairs of i's and j's (i != j), noting that Wij = Wji.
Wcorr = np.identity(d)
for i in range(d):
    for j in range(i+1,d):
        Wcorr[i,j] = abs(w.dot(components[i]*components[j]) * F_wnorms[i] * F_wnorms[j])
        Wcorr[j,i] = Wcorr[i,j]
        
plt.imshow(
    Wcorr[:last_relevant, :last_relevant],
    vmin=0,
    vmax=1
)
plt.title("Wcorr similarity matrix");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_150_0.png)


##### Similarity method 2: Covariance matrix

Given that the above is a weighted correlation measurement, and since the weights basically don't do much beside penalizing the edge values (a bit) we could approximate the computation by using a plaint Perason correlation.

Sure we won't have the edges penalized anymore but the series is a long one anyway, compared to the window size, so the vast majority of values will have an equal weight. 

If this works reasonalby well (the results look somewhat similar) we could get away with fewer code and a cleaner architecture.


```python
Corr_perason = np.corrcoef(components)
Corr_perason[Corr_perason < 0] = 0
plt.imshow(
    Corr_perason[:last_relevant, :last_relevant],
    vmin=0,
    vmax=1
)
plt.title("Perason correlation similarity matrix");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_153_0.png)


##### Similarity method 3: Cosine similarity 

Since we've noted before that 
>The similarity metric looks awfully similar to a cosine similarity between two vectors, only that this time the inner product is weighted by the dimension of each antidiagonal (so by how many times each value in $x_i, x_j$ appears in their own henkelized matrix). 

we'll try that as well to see how it behaves.


```python
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(components)
cos_sim[cos_sim < 0] = 0
plt.imshow(
    cos_sim[:last_relevant, :last_relevant],
    vmin=0,
    vmax=1
)
plt.title("Cosine similarity matrix");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_156_0.png)


##### Comparing the metrics and choosing one


```python
def _diff(sim1, sim2):
    return np.round(np.power(sim1[:last_relevant, :last_relevant] - sim2[:last_relevant, :last_relevant], 2), 3)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1, ax2, ax3 = no_axis(ax1), no_axis(ax2), no_axis(ax3)

ax1.imshow(
    _diff(Wcorr, Corr_perason),
    vmin=0,
    vmax=1
)
ax1.set_title(r"$|Wcorr - Corr\_pearson|^2$")

ax2.imshow(
    _diff(Wcorr, cos_sim),
    vmin=0,
    vmax=1    
)
ax2.set_title(r"$|Wcorr - cos\_sim|^2$")

ax3.imshow(
    _diff(cos_sim, Corr_perason),
    vmin=0,
    vmax=1
)
ax3.set_title(r"$|cos\_sim - Corr\_pearson|^2$")

fig.suptitle("Comparision between the three similarity metrics");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_158_0.png)


Plotting the distance between each similarity method pair we can see that:
* Perason correlation and cosine similarity yield very similar results
* Pearson correlation seem the better capture the correlation between the first two components (so is better in
this respect to the cosine similarity)
    * The main difference is between components `0` and `1`. On the Pearson correlation, the similarity between them is really strong, while on the custom similarity there isn't nearly as much similarity. Since `0` and `1` by visual inspection should be grouped togheter it seems a better performer would be the Perason correlation (at least on this example). 
* The weighted correalation is stronger on the second group (2, 3, 4) than the other two.

It's thus a tough call but due to the simpler code the Perason correlation might give the best tradeoffs. 

#### Clustering the components using the similarity matrix

Next we will cluster the elements using the similarity matrix and AffinityPropagation.


```python
from sklearn.cluster import AffinityPropagation

similarity_matrix = Corr_perason
last_relevant = relevant_elements(S, variance=0.96)[-1]
relevant_block_matrix = similarity_matrix[:last_relevant+1, :last_relevant+1]
clusters = AffinityPropagation(affinity='precomputed').fit_predict(relevant_block_matrix)
clusters
```




    array([0, 0, 1, 1, 2, 2, 2])



## Results

Ok, we've come this far, and... let's see what we've got!


```python
# Plotting of the clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

unique_clusters = sorted(np.unique(clusters))
for cluster in unique_clusters:
    series = np.array([timeseries(X_hat[i]) for i in np.where(clusters == cluster)[0]])
    ax1.plot(series.sum(axis=0))

ax1.plot(np.sum([timeseries(X_hat[i]) for i in range(last_relevant+1, d)], axis=0), alpha=0.2)
ax1.legend([f"F_{i}" for i in unique_clusters] + ["noise"], loc=(0.025,0.74))
ax1.set_title("Reconstructed components")
ax1.set_ylim(-2, 10)

ax2.plot(apply(trend, T), alpha=0.3)
ax2.plot(apply(period_1, T), alpha=0.3)
ax2.plot(apply(period_2, T), alpha=0.3)
ax2.plot(apply(noise, T), alpha=0.3)
plt.plot(apply(f, T), alpha=0.9)
ax2.legend(["trend", "period_1", "period_2", "noise", "Target function"], loc='upper right')
ax2.set_title("Original components")
ax2.set_ylim(-2, 10)
fig.suptitle("Automatically computed groupings using unsupervised clustering");
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_165_0.png)


Comparing the seasonality of the reconstruction and the original seasionality (2)


```python
plt.plot(apply(period_1, T), "--", alpha=0.4)
series = np.array([timeseries(X_hat[i]) for i in np.where(clusters == 1)[0]])
plt.plot(series.sum(axis=0))
plt.title("Comparing the original seasionality with the reconstruction")
plt.xticks([]); plt.yticks([]);
```


![png](../assets/images/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_files/2019-11-11-SSA_%28Singular_Spectrum_analisys%29_167_0.png)


# Conclusions

SSA seems to be a powerfull technique! We were able to reconstruct a large part of the components. Overall we'd like to point out the following:
* when the noise is to strong (like in our case) not all components can be reconstructed
* SSA (as descibed here):
    * works on timeseries data
    * relies on transforming it into a 2D (trajectory) matrix 
    * is using SVD on the trajectory matrix to decopose it into a sum of elementary matrices
    * forces the elementary matrices into henkel form via diagonal averaging (which preserves the addition property)
    * extracts the timeseries components from the diagonalized elementary matrices
* a grouping of the timeseries components is necessary to reconstruct the original components
* the grouping is based on defining a good similarity metric between components, possible metrics being:
    * anti-diagonal weighed correlation
    * Perason correlation
    * cosine similarity 
* in our case, Person correlation proved to be both performant and less verbose
* clustering can be used to **automatically** group the elementary components (via the similarity matirx).
* hyperparameters that need to be choosen well:
    * L - the window length when building the trajectory matrix
        * start with larger values (max is N/2) and decrease until a suitable result
    * variance we need to retain before grouping (similar to setting the noise threshold). 
Retaining a higher variance (e.g. 99%) leads to many elementary components (with a low contribution) to be included in the clustering, leading to more grouped components than necessary 

*Based on*: 
1. [Wikipedia article](https://en.wikipedia.org/wiki/Singular_spectrum_analysis)
2. [Kaggle kernel](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition#3.-Time-Series-Component-Separation-and-Grouping)
