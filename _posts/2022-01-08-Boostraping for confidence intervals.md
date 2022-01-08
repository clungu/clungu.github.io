---
tags:
    - statistics
    - statistical
    - statistic
mathjax: true
comments: true
title:  Bootstraping as a technique for building confidence intervals
header:
  teaser: /assets/images/2022-01-08-Boostraping_for_confidence_intervals_files/stats_boostraping_video.png
---

[#statistic](/tags/#statistic) [#statistics](/tags/#statistics) [#statistical](/tags/#statistical)

20210114223129

---


When you want to extract a value from a series of data (like a `mean`, a prediction from multiple weak classifiers - `RandomForests` - etc.. ) you may also need to know a confidence interval for that number.

One analytical way of doing this is by employing a technique called Bootstrapping. [This](https://www.youtube.com/watch?v=O_Fj4q8lgmc) is a good whiteboard explanation for it.

![stats_boostraping_video.png](/assets/images/2022-01-08-Boostraping_for_confidence_intervals_files/stats_boostraping_video.png)

The procedure is roughly the following:
* do multiple times (at least 10 000) the following:
    * resample *with replacement* a number of $$n$$ elements from the original dataset (presumed to be of size $$n$$ as well)
    * on the resulted series compute your desired metric

* at the end of the above loop you should have a series of metrics computed on a list of *synthetic* (resampled) data akin to multiple Monte-Carlo simulations.
* On the above series of metrics, compute the histogram as it should resemble a *normal distribution* (bell shaped) - by the Law Of Large Numbers.
* Compute the parameters of this normal distribution, the *mean* and the *std* to see 
    * where your confidence interval starts and end ( [-2*std, +2*std] encompass 96.7% observations, [-3*std, 3*std] encompass 99.7% observations, etc..)


Find also [here](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals_print.html) a refresher on confidence intervals computed directly on the *std* of a normal class  


This [video from Khan academy ](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample/more-significance-testing-videos/v/hypothesis-testing-and-p-values)is also great at explaining p-value calculation

See also:
- https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample

## Backlinks

> - [Statistics course on Khan academy](Statistics course on Khan accademy.md)
>   - [[20210114223129]] Boostraping for confidence intervals

_Backlinks last generated 2022-01-08 20:08:39_