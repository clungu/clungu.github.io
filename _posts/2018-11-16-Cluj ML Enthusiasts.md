---
categories: 
    - course ml
tags:
    - analysis
    - real data
    - correlation
---

My last post received quite some responses, but before actually settling on the curricula of the upcoming ML course, I've decided to "eat my own lunch" and use ML to analyze what the data is telling me.

More specifically I'm interested in the classes of ML enthusiasts that exist in Cluj-Napoca, and try to identify what those classes represent. I'd like to understand who my audience is, what they know, and what they'd like to learn next.

I'd also like to get an insight into why are they interested in joining this course.

The basic strategy that I'm going to use, is:
* data cleaning 
* feature engineering
* feature analysis
* clustering
* cluster analysis

In the end I'll end up with the insight I'm seeking, so let's proceed!

# Load the data

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
%matplotlib inline
import sklearn 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

</details>

I have a `WorkAt` field that I'm going to drop it as it contains sensitive information.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df = pd.read_csv("Cluj_ML_Enthusiasts.csv", sep='\t', na_values=['-']).drop(columns=['Email address', 'Name'])
df.drop(columns=['WorkAt']).head()
```

</details>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Knowledge</th>
      <th>Effort</th>
      <th>Degree</th>
      <th>IsResearcher</th>
      <th>IsML</th>
      <th>IsDeveloper</th>
      <th>IsTeaching</th>
      <th>IsStudent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29/10/2018 16:05:09</td>
      <td>15</td>
      <td>16</td>
      <td>4</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29/10/2018 16:11:16</td>
      <td>7</td>
      <td>32</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29/10/2018 17:18:45</td>
      <td>6</td>
      <td>16</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29/10/2018 17:21:18</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29/10/2018 17:25:51</td>
      <td>10</td>
      <td>4</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The `Knowledge` field contains values that represent, in ascending order, the self-assessment of ML knowledge. The actual number-to-question relation is found bellow: 

```
1 - I heard about ML
2 - I understand what ML is and should be doing
3 - I've used some packaged libraries / API's that used ML in the background 
4 - I regularly read news articles or blogs about ML and wish to transition into ML
5 - I've started online ML courses but gave up before finishing
6 - I've finished 1-2 courses about ML (at university or online)
7 - I've followed a few tutorials and executed their code ony my machine / forked the GitHub project
8 - I've registered on Kaggle and ran some kernels other people wrote
9 - I've wrote some ML hobby projects / played with some Kaggle datasets myself
10- I've been recently employed on an ML position
11- I've finished a PoC ML code for my employer but the project is young
12- I know how to debug my ML model, I understand what it does and what are it's shortcomings
13- I know the academic name, SoTA, previous work for the class of problems that I work on.
14- I've competed and finished in top 10% in at least one Kaggle competition / I have active ML code in production
15- I read ML academic papers and am trying to reproduce their results / Pursuing or have a PhD in AI
16- I'm able to reproduce ML academic papers and can evaluate the correctness of their claims
17- When I read a paper I can easily see where this fits in the current SoTA, and can anticipate some future directions of research
18- I write ML papers with code released online that gets published in recognized venues / I work at DeepMind, FAIR or Microsoft Research
```

The `Degree` field is a 4 value list for the following enumeration:

```
1 - Student
2 - BsC
3 - MsC
4 - PhD
```

The other `Is..` fields are just expansions of the information contained in `Knowledge`, `Degree` and `WorkAt` columns taken as a whole which I've filled them myself.



# Clean the data

Making the `WorkAt` column a categorical column. This both hides the sensitive information while at the same time, converts everything into numbers. ML models love numbers!

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df.WorkAt = df.WorkAt.astype('category').cat.as_ordered().cat.codes + 1
```

</details>

Making a numerical copy of the dataframe and anonymising the data

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df_num = df.copy()
df_num.WorkAt = df.WorkAt.astype('category').cat.as_ordered().cat.codes + 1
df_num.tail()
```

</details>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Knowledge</th>
      <th>Effort</th>
      <th>Degree</th>
      <th>IsResearcher</th>
      <th>IsML</th>
      <th>IsDeveloper</th>
      <th>IsTeaching</th>
      <th>IsStudent</th>
      <th>WorkAt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>01/11/2018 16:26:09</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>21</td>
    </tr>
    <tr>
      <th>68</th>
      <td>02/11/2018 09:20:57</td>
      <td>1</td>
      <td>16</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>22</td>
    </tr>
    <tr>
      <th>69</th>
      <td>02/11/2018 22:25:43</td>
      <td>2</td>
      <td>16</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>20</td>
    </tr>
    <tr>
      <th>70</th>
      <td>02/11/2018 22:29:13</td>
      <td>2</td>
      <td>16</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>13</td>
    </tr>
    <tr>
      <th>71</th>
      <td>06/11/2018 22:14:40</td>
      <td>5</td>
      <td>16</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



Converting the `Timestamp` column to `datatime`. This needs some attention and this case a specific `format` parameter was required for pandas. It usually infers the date format correctly but in this instance some of the November entries were detected as being from February.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df_num.Timestamp = pd.to_datetime(df_num.Timestamp, dayfirst=True, format="%d/%m/%Y %H:%M:%S")
```

</details>

# Some feature engineering

Adding an `HoursElapsed` column, infered from the `Timestamp` column. The `HoursElapsed` represent a synthetic feature that counts the number of hours passed between me publishing the news and the actual registration. It's usefull to know how fast a certain user reacted when seeing the event.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
import numpy as np
hours_elapsed = df_num.Timestamp.astype(np.int64) // ((10 ** 9) * 3600) # hours
hours_elapsed -= min(hours_elapsed)
df_num['HoursElapsed'] = hours_elapsed
df_num.head()
```

</details>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Knowledge</th>
      <th>Effort</th>
      <th>Degree</th>
      <th>IsResearcher</th>
      <th>IsML</th>
      <th>IsDeveloper</th>
      <th>IsTeaching</th>
      <th>IsStudent</th>
      <th>WorkAt</th>
      <th>HoursElapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-10-29 16:05:09</td>
      <td>15</td>
      <td>16</td>
      <td>4</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-10-29 16:11:16</td>
      <td>7</td>
      <td>32</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-10-29 17:18:45</td>
      <td>6</td>
      <td>16</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-10-29 17:21:18</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-10-29 17:25:51</td>
      <td>10</td>
      <td>4</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>18</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The `HoursElapsed` per se, are just a quick proxy for measuring how enthusiastic is a certain user about the idea of an ML course. Arguably, it's also a proxy of how much time he/she spends on Facebook or Twitter, but I'll assume the former.

I'll do a simple transformation on the `HourElapsed` by which most recent ones will have a high number and the more distant ones will fade to 0 enthusiasm.

Modeling the enthusiasm of people by assuming that the velocity of their response is relative to a positive Gaussian kernel, that defines the `Enthusiasm`

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
from math import pi
plt.rcParams['figure.figsize'] = [10, 10]

def gaussian(x, std): 
    return np.exp(-0.5*((x/std)**2)) / (std * np.sqrt(2*pi))


from matplotlib import pyplot as plt
plt.scatter(df_num.HoursElapsed, gaussian(df_num.HoursElapsed, 40))
```

</details>






![png](../../assets/images/2018-11-16-Cluj_ML_Enthusiasts_files/2018-11-16-Cluj_ML_Enthusiasts_23_1.png)


We are going to add the `Enthusiasm` values to our data, bellow.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
kernel = gaussian(df_num.HoursElapsed, 40)
kernel = (kernel - kernel.mean()) / kernel.std()
kernel

df_num['Enthusiasm'] = kernel
df_num.tail()
```

</details>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Knowledge</th>
      <th>Effort</th>
      <th>Degree</th>
      <th>IsResearcher</th>
      <th>IsML</th>
      <th>IsDeveloper</th>
      <th>IsTeaching</th>
      <th>IsStudent</th>
      <th>WorkAt</th>
      <th>HoursElapsed</th>
      <th>Enthusiasm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>2018-11-01 16:26:09</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>21</td>
      <td>72</td>
      <td>-1.789470</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2018-11-02 09:20:57</td>
      <td>1</td>
      <td>16</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>22</td>
      <td>89</td>
      <td>-2.122546</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2018-11-02 22:25:43</td>
      <td>2</td>
      <td>16</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>20</td>
      <td>102</td>
      <td>-2.255501</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2018-11-02 22:29:13</td>
      <td>2</td>
      <td>16</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>13</td>
      <td>102</td>
      <td>-2.255501</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2018-11-06 22:14:40</td>
      <td>5</td>
      <td>16</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>21</td>
      <td>198</td>
      <td>-2.368870</td>
    </tr>
  </tbody>
</table>
</div>



Now that we've added the `Enthusiasm` column, we can remove the `HoursElapsed` and `Timestamp` columns since they are correlated with the `Enthusiasm`

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df_num = df_num.drop(columns=['Timestamp', 'HoursElapsed'])
```

</details>

# Feature analisys

Do a feature analysis and remove highly correlated features

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
def plot_correlation(df_num):
    plt.rcParams['figure.figsize'] = [140, 105]
    corr = df_num.corr().abs()
    plt.matshow(df_num.corr())
    plt.xticks(ticks=np.arange(len(df_num.columns.values)), labels=df_num.columns)
    plt.yticks(ticks=np.arange(len(df_num.columns.values)), labels=df_num.columns)
    plt.show()
    return corr

corr = plot_correlation(df_num)
```

</details>

![png](../../assets/images/2018-11-16-Cluj_ML_Enthusiasts_files/2018-11-16-Cluj_ML_Enthusiasts_30_0.png)


{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

get_top_abs_correlations(df_num, n=20)
```

</details>


    IsResearcher  IsTeaching      0.624758
    Knowledge     IsML            0.585793
    IsResearcher  IsML            0.535484
    IsDeveloper   IsStudent       0.452602
    IsTeaching    Enthusiasm      0.422794
    IsDeveloper   IsTeaching      0.408248
    Knowledge     IsResearcher    0.402911
    Degree        IsStudent       0.384491
    IsML          IsTeaching      0.369175
    WorkAt        Enthusiasm      0.363629
    IsResearcher  IsDeveloper     0.324617
    Degree        IsTeaching      0.315283
    IsDeveloper   WorkAt          0.293584
    Degree        IsResearcher    0.282929
    IsResearcher  IsStudent       0.266398
    Effort        IsML            0.254194
    Knowledge     Degree          0.253271
    Degree        IsML            0.239952
    IsTeaching    IsStudent       0.234521
    IsResearcher  Enthusiasm      0.219462
    dtype: float64



We can see that some of the pairs are highly correlated with the other. From the mostly correlated pairs we're going to drop one of them, as it contains redundant information.

Decided on dropping the following columns:

```
IsML, Degree, IsResearcher
```

In addition, `WorkAt` brings no evident value, drop it.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df_num = df_num.drop(columns=['IsML', 'Degree', 'IsResearcher', 'WorkAt'])
df_num.head()
```

</details>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Knowledge</th>
      <th>Effort</th>
      <th>IsDeveloper</th>
      <th>IsTeaching</th>
      <th>IsStudent</th>
      <th>Enthusiasm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558944</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>32</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
    </tr>
  </tbody>
</table>
</div>




# Data plotting 

We will attempt to plot the data that we have on a 2D plot. In order to do this, we will reduce the dimensionality of it using PCA. This is useful because we want to get a rough idea of how many clusters do we have.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans
```


```python
from sklearn.decomposition import PCA


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('dim_reducer', PCA(n_components=2))
])


reduced = pipe.fit_transform(df_num.values.astype(np.float32))

plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(reduced[:, 0], reduced[:, 1])

for i, (x, y) in enumerate(reduced[:, [0, 1]]):
    plt.text(x, y, "Name " + str(df.index[i]))
```

</details>

![png](../../assets/images/2018-11-16-Cluj_ML_Enthusiasts_files/2018-11-16-Cluj_ML_Enthusiasts_38_0.png)


If you squint, it seems we have roughly 4 generic clusters (with a couple of outliers). We will use KMeans clustering with this value, but only after we normalize all the features. Let's see what we get!

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df_ = df_num.copy()
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('cluster', KMeans(n_clusters=4))
])

df_['Cluster'] = pipe.fit_predict(df_num)
df_.sort_values('Cluster')
```

</details>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Knowledge</th>
      <th>Effort</th>
      <th>IsDeveloper</th>
      <th>IsTeaching</th>
      <th>IsStudent</th>
      <th>Enthusiasm</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558944</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>12</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558944</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>-2.122546</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>7</td>
      <td>32</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558944</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>3</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.246598</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>7</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.333842</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>3</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>9</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.246598</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>7</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>0.555286</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.555286</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.555286</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>4</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>-2.255501</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>9</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.544341</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.550721</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>32</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558944</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>10</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.333842</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.544341</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>-2.122546</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>-2.255501</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.555286</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.555286</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.550721</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
      <td>16</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>0.555286</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>3</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>-0.681784</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>0.536159</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>4</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>7</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.277015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.360152</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.514453</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>0.514453</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>0.514453</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>0.536159</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.536159</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.544341</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>0.558029</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1</td>
      <td>4</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>0.526190</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>15</td>
      <td>4</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>-1.789470</td>
      <td>2</td>
    </tr>
    <tr>
      <th>67</th>
      <td>12</td>
      <td>4</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>-1.789470</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2</td>
      <td>16</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>-2.255501</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>15</td>
      <td>4</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>0.526190</td>
      <td>2</td>
    </tr>
    <tr>
      <th>65</th>
      <td>5</td>
      <td>16</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>-1.231252</td>
      <td>2</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2</td>
      <td>16</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>-2.255501</td>
      <td>2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3</td>
      <td>16</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>-1.231252</td>
      <td>2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>6</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>-2.368870</td>
      <td>3</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.526190</td>
      <td>3</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.550721</td>
      <td>3</td>
    </tr>
    <tr>
      <th>44</th>
      <td>14</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.558029</td>
      <td>3</td>
    </tr>
    <tr>
      <th>43</th>
      <td>5</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.558029</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.526190</td>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.526190</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.550721</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.558029</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.558029</td>
      <td>3</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.526190</td>
      <td>3</td>
    </tr>
    <tr>
      <th>71</th>
      <td>5</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>-2.368870</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 7 columns</p>
</div>



From the initial looks of it, the clusters have the following characteristics:
* 0 - High commitment Developers (36,1%)
* 1 - Quick win Developers (36,1%)
* 2 - Academics / Researchers (11,1%)
* 3 - Unemployed students (16,7%)


`AffinityPropagation` is a clustering algo that doesn't require a number of clusters as its inputs. I'm going to use it for estimating the number of clusters, because maybe I'm missing some clusters. 

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
nr_clusters = np.unique(AffinityPropagation().fit_predict(df_num))
nr_clusters
```

</details>



    array([0, 1, 2, 3, 4])



It seems that there's roughly the same amount of clusters that it finds. After some inspection, it seems that the model decided to split the 'Accademics' into high and low commitment ones. 

* 1 - Quick win Developers ( 36,1% )
  - < 4h work / week
  - employed

* 2 - Unemployed students ( 16,7% )

* 3 - High commitment Developers ( 36,1% )
  - 16+ hours / week
  - employed
    
* 4 - Academics ( 11,1% )
  - high commitment (> 16h/week) ( 5,6% )
  - low commitment  (< 4h/week) ( 5,6% )

So there you go! I won't draw any conclusions to the above findings, I'll leave the interpretation of the result up to you, but people seem generally interested in ML ;) 
