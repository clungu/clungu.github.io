---
tags:
    - optimisation
    - ml
mathjax: true
comments: true
title:  SGD update and the intuition behind it
header:
  teaser: 
---

[#ml](/tags/#ml) [#optimisation](/tags/#optimisation)

20200905172135

---


The SGD update rule is:

$$x = x - \lambda * \Delta x$$

Keep this in mind for a moment.

## Incremental mean

The incremental mean of a sequence $$x_1, x_2, x_3, …$$ expressed as $$\mu_1, \mu_2, \mu_3, ..$$ can be computed incrementally:

$$\mu_k = \frac{1}{k}\sum_{j=1}^kx_j$$ 

$$\mu_k = \frac{1}{k}(x_k+\sum_{j=1}^{k-1}x_j)$$

$$\mu_k = \frac{1}{k}(x_k+\frac{\sum_{j=1}^{k-1}x_j*(k-1)}{k-1})$$

$$\mu_k = \frac{1}{k}(x_k+(k-1)\frac{\sum_{j=1}^{k-1}x_j}{k-1})$$

$$\mu_k = \frac{1}{k}(x_k + (k-1)\mu_{k-1})$$ 

$$\color{red}\mu_k = \mu_{k-1} + \frac{1}{k}(x_k - \mu_{k-1})$$

* In this, $$(x_k - \mu_{k-1})$$ is an "error" term where $$\mu_{k-1}$$ is the "prediction" and $$x_k$$ is the current observation.  
* $$\frac{1}{k}$$ is some "learning" rate, saying over how many previous samples we want to compute the "moving average". When $$k$$ is equal to the number of all the samples, this is exactly the incremental mean.

In non-stationary environment (not changing in time), it can be useful to compute a moving (running) average (mean) over a finite number of previous states, and not on all of them. This is done by having $$\frac{1}{N(s)}$$ be replaced with $$\color{red}\alpha$$ in the following form
$$\mu_k = \mu_{k-1} + {\color{red}\alpha}(x_k - \mu_{k-1})$$ 

Under this interpretation:
* A really small $$\color{red}\alpha$$ means that we keep incorporating information from **lots more** previous states
    * if $$\alpha$$ is small, this means $$\alpha = \frac{1}{large\_nr\_steps}$$

* A large $$\color{red}\alpha$$ means that we've incorporated information from only a **few** previous states
    * if $$\alpha$$ is large, this means $$\alpha = \frac{1}{small\_nr\_steps}$$

## Estimating the derivative analytically

If we are to use $$\mu_{k-1}$$ above in an update rule, this would look like:

$$\mu = \mu + {\color{red}\alpha}(x - \mu)$$ 

$$\mu = \mu - {\color{red}\alpha}(\mu - x)$$ 

which says that the previous value $$\mu$$ (the lvalue of the assignment), which was a moving average over a period in time - whose length is defined by $$\alpha$$ - was updated with the moving average value over computed on the shifted sequence, the one that also includes $$x$$ now (the rvalue of the assignment).

This really explains the intuition of why the SDG update step really looks the way it does. In it, the term $$\mu - x$$ is an approximation of the $$\Delta \mu_x$$ (derived from [Numerical Differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation)) ,  which is a means of analytically computing the error term, saying in which direction we need to change the moving average towards the end goal.


This intuition is also presented in [[20200729171147]] RL Course by David Silver - Lesson 4.

On the other hand, you can find a different explanation (or intuition) for why SGD update works in [Yann LeCun's Lesson 4](https://www.youtube.com/watch?v=--NZb480zlg&feature=youtu.be)  ( [[20201201214511]] Yann LeCun video course)