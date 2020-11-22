# Distilling a Random Forest with a single DecisionTree

On HackerNews there was a topic discussed at some point about ways to distill knowledge from a complex (almost black box) large tree ensamble (a RandomForest with lots of sub-trees was used as an example).

You would like to do this for multiple resones, but one of them is model exaplainability, so a way to understand how that complex model behaves so you can draw conclusions and improve it (or guard against its failures).

One comment really caught my eye:

> An alternative is to re-lable data with the ensemble's outputs and then learn a decision tree over that. ([source](https://news.ycombinator.com/item?id=25121998))


This post is my attempt of testing this strategy out.


# Get a dataset

I'll first use a clean and small dataset not to make things to complicated.


```
from sklearn.datasets import load_iris
import pandas as pd

dataset = load_iris()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

df.head()
```




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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Then I'm going to split this dataset into a training set (random 70% of the data) and a test set (the other 15% of the data).


```
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3)
df_train.shape, df_test.shape, df_train.shape[0] / (df_train.shape[0] + df_test.shape[0])
```




    ((105, 5), (45, 5), 0.7)



## Train a Random Forest model

And then I'm going to train a `RandomForestClassifier` that will solve this problem. I'm not going to be too concerned on the model performence (so I'm not going to really make it generalize well) because I'm really more interested in achieving the same behaviour (either good or bad as it is) with a `DecisionTree` proxy. 


```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
rf.fit(df_train.drop(columns='target'), df_train['target'])

print(classification_report(df_test['target'], rf.predict(df_test.drop(columns='target'))))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        10
               1       1.00      0.95      0.98        21
               2       0.93      1.00      0.97        14
    
        accuracy                           0.98        45
       macro avg       0.98      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    


## Retraining a single decision tree that would approximate the RandomForest

> An alternative is to re-lable data with the ensemble's outputs and then learn a decision tree over that. ([source](https://news.ycombinator.com/item?id=25121998))



What this actually means is doing the following steps:
* Train a large, complex ensamble (the RandomForest model above)
* Take **all the data** (including the test set) and add the predictions to it (even for the test set)
* Overfitt a single DecisionTree over the **all the data** but training on the predictions of the RandomForests.

What this basically does is creating a single DecisionTree that can predict exactly the same stuff as the RandomForest. We know that a `DecisionTree` has the ability to overfitt perfectly the training data if it is allowed to (for example, if we leave it to grow until leaf nodes contain only one datapoint).


```
def __inp(df, exclude_columns=['target']):
    return df.drop(columns=list(set(exclude_columns) & set(df.columns)))

def __out(df, target_column='target'):
    return df[target_column]

def relable(df, model):
    df = df.copy()
    df['relabel'] = model.predict(__inp(df))
    return df

# relable everything
df_train_tree = relable(df_train, rf)
df_test_tree = relable(df_test, rf)
df_tree = relable(df, rf)

df_train_tree.head()
```




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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
      <th>relabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>6.2</td>
      <td>2.9</td>
      <td>4.3</td>
      <td>1.3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>114</th>
      <td>5.8</td>
      <td>2.8</td>
      <td>5.1</td>
      <td>2.4</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>125</th>
      <td>7.2</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>1.8</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>110</th>
      <td>6.5</td>
      <td>3.2</td>
      <td>5.1</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>113</th>
      <td>5.7</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



We *want* to overfit the training data with the decision tree because what we are actually looking for is a single condensed tree that behaves exactly the same as the original `RandomForest`. And we want the DecisionTree to behave **exactly** the same as the RandomForest on the test data as well. That's why we train on the full dataset here.


```
from sklearn.tree import DecisionTreeClassifier
from functools import partial
from sklearn.metrics import f1_score

__inp = partial(__inp, exclude_columns=['target', 'relabel'])
__rel = partial(__out, target_column='relabel')
__f1_score = partial(f1_score, average="macro")


dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_impurity_split=0)
dt.fit(__inp(df_tree), __rel(df_tree))

print(f"This should show that we've completely overfitted the relables (i.e. F1 score == 1.0). \nSo we've mimiked the RandomForest's behaviour perfectly!")
print(classification_report(__rel(df_train_tree), dt.predict(__inp(df_train_tree))))
assert __f1_score(__rel(df_train_tree), dt.predict(__inp(df_train_tree))) == 1.0

print("\n\n")
print(f"This shows the performance on the actual `target` values of the test set (never seen).")
print(classification_report(__out(df_test_tree), dt.predict(__inp(df_test_tree))))
assert __f1_score(__out(df_test), rf.predict(__inp(df_test_tree))) == __f1_score(__out(df_test), dt.predict(__inp(df_test_tree)))
```

    This should show that we've completely overfitted the relables (i.e. F1 score == 1.0). 
    So we've mimiked the RandomForest's behaviour perfectly!
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        40
               1       1.00      1.00      1.00        29
               2       1.00      1.00      1.00        36
    
        accuracy                           1.00       105
       macro avg       1.00      1.00      1.00       105
    weighted avg       1.00      1.00      1.00       105
    
    
    
    
    This shows the performance on the actual `target` values of the test set (never seen).
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        10
               1       1.00      0.95      0.98        21
               2       0.93      1.00      0.97        14
    
        accuracy                           0.98        45
       macro avg       0.98      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    


    /usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py:301: FutureWarning: The min_impurity_split parameter is deprecated. Its default value will change from 1e-7 to 0 in version 0.23, and it will be removed in 0.25. Use the min_impurity_decrease parameter instead.
      FutureWarning)


So this did work, we have a `DecisionTree` that behaves exactly as the `RandomForest`. The only problem is that this applies only on the seen data (so it may be possible that the RandomForest generalises well on new / other data, but the DecisionTree will not, because it is a perfect aproximator trained only on the data available at that moment of training). We will test this in the following section.

Neverthless, what tree did we got?


```
!apt-get update; apt-get -y install graphviz
!pip install dtreeviz
```

    Ign:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Hit:2 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease
    Ign:3 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:4 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
    Hit:5 http://security.ubuntu.com/ubuntu bionic-security InRelease
    Hit:6 http://archive.ubuntu.com/ubuntu bionic InRelease
    Hit:7 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease
    Hit:8 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Hit:9 http://archive.ubuntu.com/ubuntu bionic-updates InRelease
    Hit:10 http://archive.ubuntu.com/ubuntu bionic-backports InRelease
    Hit:11 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
    Reading package lists... Done
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    graphviz is already the newest version (2.40.1-2).
    0 upgraded, 0 newly installed, 0 to remove and 57 not upgraded.
    Requirement already satisfied: dtreeviz in /usr/local/lib/python3.6/dist-packages (1.1.2)
    Requirement already satisfied: pyspark in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (3.0.1)
    Requirement already satisfied: graphviz>=0.9 in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (0.10.1)
    Requirement already satisfied: colour in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (0.1.5)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (0.22.2.post1)
    Requirement already satisfied: pytest in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (3.6.4)
    Requirement already satisfied: xgboost in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (0.90)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (1.18.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (3.2.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from dtreeviz) (1.1.4)
    Requirement already satisfied: py4j==0.10.9 in /usr/local/lib/python3.6/dist-packages (from pyspark->dtreeviz) (0.10.9)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->dtreeviz) (0.17.0)
    Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->dtreeviz) (1.4.1)
    Requirement already satisfied: atomicwrites>=1.0 in /usr/local/lib/python3.6/dist-packages (from pytest->dtreeviz) (1.4.0)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.6/dist-packages (from pytest->dtreeviz) (20.2.0)
    Requirement already satisfied: pluggy<0.8,>=0.5 in /usr/local/lib/python3.6/dist-packages (from pytest->dtreeviz) (0.7.1)
    Requirement already satisfied: more-itertools>=4.0.0 in /usr/local/lib/python3.6/dist-packages (from pytest->dtreeviz) (8.6.0)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from pytest->dtreeviz) (1.15.0)
    Requirement already satisfied: py>=1.5.0 in /usr/local/lib/python3.6/dist-packages (from pytest->dtreeviz) (1.9.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from pytest->dtreeviz) (50.3.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->dtreeviz) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->dtreeviz) (1.3.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->dtreeviz) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->dtreeviz) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->dtreeviz) (2018.9)



```
from dtreeviz.trees import *
viz = dtreeviz(dt, 
               __inp(df_tree),
               __out(df_tree),
               target_name='species',
               feature_names=__inp(df_test_tree).columns, 
               class_names=dataset.target_names.tolist()
              )
viz
```




    
![svg](../assets/images/2020-11-22-Distilling_a_Random_Forest_with_a_single_DecisionTree_files/2020-11-22-Distilling_a_Random_Forest_with_a_single_DecisionTree_19_0.svg)
    



## Does this aproximation hold for unseen data?

As I've said before, the only problem is that this strategy applies strictly only on the seen/available data (so it may be possible that the RandomForest generalisez well on new / other data, but the DecisionTree will not, because it is a perfect aproximator trained only on the data available at that moment of training).

To thest this out we will do a three-way split, leaving the third unseen to the DecitionTree and we will replicate the experiment above


```
from sklearn.model_selection import train_test_split

df_train, df_rest = train_test_split(df, test_size=0.3)
df_test, df_future = train_test_split(df_rest, test_size=0.5)
df_all = pd.concat((df_train, df_test))

df_train.shape, df_test.shape,  df_future.shape, df_test.shape[0] / (df_train.shape[0] + df_test.shape[0] + df_future.shape[0])
```




    ((105, 5), (22, 5), (23, 5), 0.14666666666666667)



We now have:
* `df_train` - 70% of the data
* `df_test` - 15% of the data
* `df_all` = `df_train + df_test`
* `df_future` - 15% of (simulated) future data


```
# train a "generizable" RandomForest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
rf.fit(__inp(df_train), __out(df_train))

print(f"RandomForest performance:")
print(classification_report(__out(df_test), rf.predict(__inp(df_test))))

# relable **current** data
df_train_tree = relable(df_train, rf)
df_test_tree = relable(df_test, rf)
df_tree = relable(df_all, rf)

# train DecisionTree aproximator
dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_impurity_split=0)
dt.fit(__inp(df_tree), __rel(df_tree))

print("\n\n")
print(f"This should show that we've completely overfitted the predictions (i.e. F1 score == 1.0). \nSo we've mimiked the RandomForest's behaviour perfectly!")
print(classification_report(__rel(df_train_tree), dt.predict(__inp(df_train_tree))))
assert __f1_score(__rel(df_train_tree), dt.predict(__inp(df_train_tree))) == 1.0

print("\n\n")
print(f"This shows the performance on the actual `target` values of the test set and \nthat they are equal to the performance of the RandomForest model on the same data.")
print(classification_report(__out(df_test_tree), dt.predict(__inp(df_test_tree))))
assert __f1_score(__out(df_test), rf.predict(__inp(df_test_tree))) == __f1_score(__out(df_test), dt.predict(__inp(df_test_tree)))
```

    RandomForest performance:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         9
               1       1.00      1.00      1.00         6
               2       1.00      1.00      1.00         7
    
        accuracy                           1.00        22
       macro avg       1.00      1.00      1.00        22
    weighted avg       1.00      1.00      1.00        22
    
    
    
    
    This should show that we've completely overfitted the predictions (i.e. F1 score == 1.0). 
    So we've mimiked the RandomForest's behaviour perfectly!
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        37
               1       1.00      1.00      1.00        33
               2       1.00      1.00      1.00        35
    
        accuracy                           1.00       105
       macro avg       1.00      1.00      1.00       105
    weighted avg       1.00      1.00      1.00       105
    
    
    
    
    This shows the performance on the actual `target` values of the test set and 
    that they are equal to the performance of the RandomForest model on the same data.
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         9
               1       1.00      1.00      1.00         6
               2       1.00      1.00      1.00         7
    
        accuracy                           1.00        22
       macro avg       1.00      1.00      1.00        22
    weighted avg       1.00      1.00      1.00        22
    


    /usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py:301: FutureWarning: The min_impurity_split parameter is deprecated. Its default value will change from 1e-7 to 0 in version 0.23, and it will be removed in 0.25. Use the min_impurity_decrease parameter instead.
      FutureWarning)


Let's see how we do on the future data now!


```
print("Random Forest performance on future data:")
print(classification_report(__out(df_future), rf.predict(__inp(df_future))))

print("DecisionTree aproximator on future data")
print(classification_report(__out(df_future), dt.predict(__inp(df_future))))
```

    Random Forest performance on future data:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         4
               1       1.00      0.82      0.90        11
               2       0.80      1.00      0.89         8
    
        accuracy                           0.91        23
       macro avg       0.93      0.94      0.93        23
    weighted avg       0.93      0.91      0.91        23
    
    DecisionTree aproximator on future data
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         4
               1       1.00      0.82      0.90        11
               2       0.80      1.00      0.89         8
    
        accuracy                           0.91        23
       macro avg       0.93      0.94      0.93        23
    weighted avg       0.93      0.91      0.91        23
    


From the performance of the two models above, you can see that they indeed reach the same performance (quite surprisingly I would say).

Now, the code above is **not** determinstic so if you run all the cells from the beggining up until this point multiple times, you will see that the `RandomForest` has a different accuracy each time. 

Having said that, this particular comparision that we are interested (the performance of the `RandomForest` on the future data, compared on the perfirmance of the `DecisionTree` aproximator in the future data) is **almost** always the same. Almost (9 ot of 10), but not **always**. I've seen ocasional runs where the `RandomForest` outperformed slightly the `DecisionTree`.

# Using a more challenging dataset

The [iris dataset](https://scikit-learn.org/stable/datasets/index.html#iris-dataset) is a [toy dataset](https://scikit-learn.org/stable/datasets/index.html#toy-datasets) from scikit-learn because it has only 150 datapoints and very few lables.

To thest the above approach more thoroughly we need to use a more [plausible dataset](https://scikit-learn.org/stable/datasets/index.html#real-world-datasets) (still for classification, to keep things consistent) with lots more features and datapoints.

For this we've choosen the [forest covertype dataset](https://scikit-learn.org/stable/datasets/index.html#forest-covertypes) where we have 581012 datapoints, each with 54 features describing some 30x30m measurements of a plot of land. We need to predict the correct category of vegetation for each plot.




```
from sklearn.datasets import fetch_covtype 
import pandas as pd

dataset = fetch_covtype()
df = pd.DataFrame(data=dataset.data, columns=["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"])
df['target'] = dataset.target

df.head()
```




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
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>Wilderness_Area2</th>
      <th>Wilderness_Area3</th>
      <th>Wilderness_Area4</th>
      <th>Soil_Type1</th>
      <th>Soil_Type2</th>
      <th>Soil_Type3</th>
      <th>Soil_Type4</th>
      <th>Soil_Type5</th>
      <th>Soil_Type6</th>
      <th>Soil_Type7</th>
      <th>Soil_Type8</th>
      <th>Soil_Type9</th>
      <th>Soil_Type10</th>
      <th>Soil_Type11</th>
      <th>Soil_Type12</th>
      <th>Soil_Type13</th>
      <th>Soil_Type14</th>
      <th>Soil_Type15</th>
      <th>Soil_Type16</th>
      <th>Soil_Type17</th>
      <th>Soil_Type18</th>
      <th>Soil_Type19</th>
      <th>Soil_Type20</th>
      <th>Soil_Type21</th>
      <th>Soil_Type22</th>
      <th>Soil_Type23</th>
      <th>Soil_Type24</th>
      <th>Soil_Type25</th>
      <th>Soil_Type26</th>
      <th>Soil_Type27</th>
      <th>Soil_Type28</th>
      <th>Soil_Type29</th>
      <th>Soil_Type30</th>
      <th>Soil_Type31</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596.0</td>
      <td>51.0</td>
      <td>3.0</td>
      <td>258.0</td>
      <td>0.0</td>
      <td>510.0</td>
      <td>221.0</td>
      <td>232.0</td>
      <td>148.0</td>
      <td>6279.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590.0</td>
      <td>56.0</td>
      <td>2.0</td>
      <td>212.0</td>
      <td>-6.0</td>
      <td>390.0</td>
      <td>220.0</td>
      <td>235.0</td>
      <td>151.0</td>
      <td>6225.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804.0</td>
      <td>139.0</td>
      <td>9.0</td>
      <td>268.0</td>
      <td>65.0</td>
      <td>3180.0</td>
      <td>234.0</td>
      <td>238.0</td>
      <td>135.0</td>
      <td>6121.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785.0</td>
      <td>155.0</td>
      <td>18.0</td>
      <td>242.0</td>
      <td>118.0</td>
      <td>3090.0</td>
      <td>238.0</td>
      <td>238.0</td>
      <td>122.0</td>
      <td>6211.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595.0</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>153.0</td>
      <td>-1.0</td>
      <td>391.0</td>
      <td>220.0</td>
      <td>234.0</td>
      <td>150.0</td>
      <td>6172.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



We will again save a `future` dataset for later use.


```
from sklearn.model_selection import train_test_split

df_train, df_rest = train_test_split(df, test_size=0.3)
df_test, df_future = train_test_split(df_rest, test_size=0.5)
df_all = pd.concat((df_train, df_test))

df_train.shape, df_test.shape,  df_future.shape, df_test.shape[0] / (df_train.shape[0] + df_test.shape[0] + df_future.shape[0])
```




    ((406708, 55), (87152, 55), (87152, 55), 0.1500003442269695)




```
# train a "generizable" RandomForest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
rf.fit(__inp(df_train), __out(df_train))

print(f"RandomForest performance:")
print(classification_report(__out(df_test), rf.predict(__inp(df_test))))

# relable **current** data
df_train_tree = relable(df_train, rf)
df_test_tree = relable(df_test, rf)
df_tree = relable(df_all, rf)

# train DecisionTree aproximator
dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_impurity_split=0)
dt.fit(__inp(df_tree), __rel(df_tree))

print("\n\n")
print(f"This should show that we've completely overfitted the predictions (i.e. F1 score == 1.0). \nSo we've mimiked the RandomForest's behaviour perfectly!")
print(classification_report(__rel(df_train_tree), dt.predict(__inp(df_train_tree))))
assert __f1_score(__rel(df_train_tree), dt.predict(__inp(df_train_tree))) == 1.0

print("\n\n")
print(f"This shows the performance on the actual `target` values of the test set and \nthat they are equal to the performance of the RandomForest model on the same data.")
print(classification_report(__out(df_test_tree), dt.predict(__inp(df_test_tree))))
assert __f1_score(__out(df_test), rf.predict(__inp(df_test_tree))) == __f1_score(__out(df_test), dt.predict(__inp(df_test_tree)))
```

    RandomForest performance:


    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


                  precision    recall  f1-score   support
    
               1       0.64      0.74      0.69     31900
               2       0.71      0.76      0.74     42493
               3       0.62      0.65      0.63      5254
               4       0.00      0.00      0.00       414
               5       0.00      0.00      0.00      1397
               6       0.00      0.00      0.00      2673
               7       0.00      0.00      0.00      3021
    
        accuracy                           0.68     87152
       macro avg       0.28      0.31      0.29     87152
    weighted avg       0.62      0.68      0.65     87152
    


    /usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py:301: FutureWarning: The min_impurity_split parameter is deprecated. Its default value will change from 1e-7 to 0 in version 0.23, and it will be removed in 0.25. Use the min_impurity_decrease parameter instead.
      FutureWarning)


    
    
    
    This should show that we've completely overfitted the predictions (i.e. F1 score == 1.0). 
    So we've mimiked the RandomForest's behaviour perfectly!
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00    169056
               2       1.00      1.00      1.00    211823
               3       1.00      1.00      1.00     25829
    
        accuracy                           1.00    406708
       macro avg       1.00      1.00      1.00    406708
    weighted avg       1.00      1.00      1.00    406708
    
    
    
    
    This shows the performance on the actual `target` values of the test set and 
    that they are equal to the performance of the RandomForest model on the same data.
                  precision    recall  f1-score   support
    
               1       0.64      0.74      0.69     31900
               2       0.71      0.76      0.74     42493
               3       0.62      0.65      0.63      5254
               4       0.00      0.00      0.00       414
               5       0.00      0.00      0.00      1397
               6       0.00      0.00      0.00      2673
               7       0.00      0.00      0.00      3021
    
        accuracy                           0.68     87152
       macro avg       0.28      0.31      0.29     87152
    weighted avg       0.62      0.68      0.65     87152
    


    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


Now that we have everything prepared, let's just test what happens with the two models on this more plausible dataset.


```
print("Random Forest performance on future data:")
print(classification_report(__out(df_future), rf.predict(__inp(df_future))))

print("DecisionTree aproximator on future data")
print(classification_report(__out(df_future), dt.predict(__inp(df_future))))
```

    Random Forest performance on future data:


    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


                  precision    recall  f1-score   support
    
               1       0.65      0.74      0.69     31869
               2       0.72      0.76      0.74     42488
               3       0.62      0.64      0.63      5331
               4       0.00      0.00      0.00       402
               5       0.00      0.00      0.00      1389
               6       0.00      0.00      0.00      2597
               7       0.00      0.00      0.00      3076
    
        accuracy                           0.68     87152
       macro avg       0.28      0.31      0.29     87152
    weighted avg       0.62      0.68      0.65     87152
    
    DecisionTree aproximator on future data
                  precision    recall  f1-score   support
    
               1       0.65      0.74      0.69     31869
               2       0.72      0.76      0.74     42488
               3       0.62      0.64      0.63      5331
               4       0.00      0.00      0.00       402
               5       0.00      0.00      0.00      1389
               6       0.00      0.00      0.00      2597
               7       0.00      0.00      0.00      3076
    
        accuracy                           0.68     87152
       macro avg       0.28      0.31      0.29     87152
    weighted avg       0.62      0.68      0.65     87152
    


Again, we have exactly the same performance so this seems to work OK, but..

## Is this generalizable to other tree-ensamble methods, like XGBoost?

We will train a XGBoost model and try to reproduce the results above with it so see if is possible to distill the XGBoost model into a single decision tree.

We will use the same `covtype` dataset since it's large and tune the training and instantiation of the model to obtain a nice performant model. While in the previous experiment we didn't really bother optimising the model, it is possible that a great generalizable model will show some differences when applying this process.


```
from sklearn.datasets import fetch_covtype 
import pandas as pd

dataset = fetch_covtype()
df = pd.DataFrame(data=dataset.data, columns=["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"])
df['target'] = dataset.target

df.head()
```




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
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>Wilderness_Area2</th>
      <th>Wilderness_Area3</th>
      <th>Wilderness_Area4</th>
      <th>Soil_Type1</th>
      <th>Soil_Type2</th>
      <th>Soil_Type3</th>
      <th>Soil_Type4</th>
      <th>Soil_Type5</th>
      <th>Soil_Type6</th>
      <th>Soil_Type7</th>
      <th>Soil_Type8</th>
      <th>Soil_Type9</th>
      <th>Soil_Type10</th>
      <th>Soil_Type11</th>
      <th>Soil_Type12</th>
      <th>Soil_Type13</th>
      <th>Soil_Type14</th>
      <th>Soil_Type15</th>
      <th>Soil_Type16</th>
      <th>Soil_Type17</th>
      <th>Soil_Type18</th>
      <th>Soil_Type19</th>
      <th>Soil_Type20</th>
      <th>Soil_Type21</th>
      <th>Soil_Type22</th>
      <th>Soil_Type23</th>
      <th>Soil_Type24</th>
      <th>Soil_Type25</th>
      <th>Soil_Type26</th>
      <th>Soil_Type27</th>
      <th>Soil_Type28</th>
      <th>Soil_Type29</th>
      <th>Soil_Type30</th>
      <th>Soil_Type31</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596.0</td>
      <td>51.0</td>
      <td>3.0</td>
      <td>258.0</td>
      <td>0.0</td>
      <td>510.0</td>
      <td>221.0</td>
      <td>232.0</td>
      <td>148.0</td>
      <td>6279.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590.0</td>
      <td>56.0</td>
      <td>2.0</td>
      <td>212.0</td>
      <td>-6.0</td>
      <td>390.0</td>
      <td>220.0</td>
      <td>235.0</td>
      <td>151.0</td>
      <td>6225.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804.0</td>
      <td>139.0</td>
      <td>9.0</td>
      <td>268.0</td>
      <td>65.0</td>
      <td>3180.0</td>
      <td>234.0</td>
      <td>238.0</td>
      <td>135.0</td>
      <td>6121.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785.0</td>
      <td>155.0</td>
      <td>18.0</td>
      <td>242.0</td>
      <td>118.0</td>
      <td>3090.0</td>
      <td>238.0</td>
      <td>238.0</td>
      <td>122.0</td>
      <td>6211.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595.0</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>153.0</td>
      <td>-1.0</td>
      <td>391.0</td>
      <td>220.0</td>
      <td>234.0</td>
      <td>150.0</td>
      <td>6172.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```
from sklearn.model_selection import train_test_split

df_train, df_rest = train_test_split(df, test_size=0.3)
df_test, df_future = train_test_split(df_rest, test_size=0.5)
df_all = pd.concat((df_train, df_test))

df_train.shape, df_test.shape,  df_future.shape, df_test.shape[0] / (df_train.shape[0] + df_test.shape[0] + df_future.shape[0])
```




    ((406708, 55), (87152, 55), (87152, 55), 0.1500003442269695)




```
from xgboost import XGBClassifier

# train a "generizable" RandomForest
xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=20, 
    n_jobs=-1, 
    verbosity=1, 
    booster='gbtree',
    objective='mlogloss',
    # num_class=len(np.unique(__out(df)))
)

xgb.fit(
    X=__inp(df_train), 
    y=__out(df_train), 
    verbose=True, 
    eval_set=[(__inp(df_test), __out(df_test))],
    eval_metric=['mlogloss'],
    early_stopping_rounds=4,
)

print(f"XGBoost performance:")
print(classification_report(__out(df_test), xgb.predict(__inp(df_test))))

# relable **current** data
df_train_tree = relable(df_train, xgb)
df_test_tree = relable(df_test, xgb)
df_tree = relable(df_all, xgb)

# train DecisionTree aproximator
dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_impurity_split=0)
dt.fit(__inp(df_tree), __rel(df_tree))

print("\n\n")
print(f"This should show that we've completely overfitted the predictions (i.e. F1 score == 1.0). \nSo we've mimiked the RandomForest's behaviour perfectly!")
print(classification_report(__rel(df_train_tree), dt.predict(__inp(df_train_tree))))
assert __f1_score(__rel(df_train_tree), dt.predict(__inp(df_train_tree))) == 1.0

print("\n\n")
print(f"This shows the performance on the actual `target` values of the test set and \nthat they are equal to the performance of the RandomForest model on the same data.")
print(classification_report(__out(df_test_tree), dt.predict(__inp(df_test_tree))))
assert __f1_score(__out(df_test), xgb.predict(__inp(df_test_tree))) == __f1_score(__out(df_test), dt.predict(__inp(df_test_tree)))
```

    [0]	validation_0-mlogloss:1.76519
    Will train until validation_0-mlogloss hasn't improved in 4 rounds.
    [1]	validation_0-mlogloss:1.62483
    [2]	validation_0-mlogloss:1.50959
    [3]	validation_0-mlogloss:1.41254
    [4]	validation_0-mlogloss:1.32897
    [5]	validation_0-mlogloss:1.25724
    [6]	validation_0-mlogloss:1.19438
    [7]	validation_0-mlogloss:1.13841
    [8]	validation_0-mlogloss:1.08877
    [9]	validation_0-mlogloss:1.04369
    [10]	validation_0-mlogloss:1.00364
    [11]	validation_0-mlogloss:0.967579
    [12]	validation_0-mlogloss:0.934488
    [13]	validation_0-mlogloss:0.905293
    [14]	validation_0-mlogloss:0.878637
    [15]	validation_0-mlogloss:0.854214
    [16]	validation_0-mlogloss:0.831893
    [17]	validation_0-mlogloss:0.81107
    [18]	validation_0-mlogloss:0.792415
    [19]	validation_0-mlogloss:0.77495
    [20]	validation_0-mlogloss:0.759048
    [21]	validation_0-mlogloss:0.744567
    [22]	validation_0-mlogloss:0.731076
    [23]	validation_0-mlogloss:0.718687
    [24]	validation_0-mlogloss:0.707045
    [25]	validation_0-mlogloss:0.695962
    [26]	validation_0-mlogloss:0.685939
    [27]	validation_0-mlogloss:0.676211
    [28]	validation_0-mlogloss:0.667509
    [29]	validation_0-mlogloss:0.659425
    [30]	validation_0-mlogloss:0.651654
    [31]	validation_0-mlogloss:0.644297
    [32]	validation_0-mlogloss:0.637633
    [33]	validation_0-mlogloss:0.631391
    [34]	validation_0-mlogloss:0.625518
    [35]	validation_0-mlogloss:0.61975
    [36]	validation_0-mlogloss:0.614328
    [37]	validation_0-mlogloss:0.609253
    [38]	validation_0-mlogloss:0.604604
    [39]	validation_0-mlogloss:0.600179
    [40]	validation_0-mlogloss:0.595672
    [41]	validation_0-mlogloss:0.591668
    [42]	validation_0-mlogloss:0.587884
    [43]	validation_0-mlogloss:0.584106
    [44]	validation_0-mlogloss:0.580601
    [45]	validation_0-mlogloss:0.576968
    [46]	validation_0-mlogloss:0.573537
    [47]	validation_0-mlogloss:0.570581
    [48]	validation_0-mlogloss:0.568026
    [49]	validation_0-mlogloss:0.565185
    [50]	validation_0-mlogloss:0.562348
    [51]	validation_0-mlogloss:0.559119
    [52]	validation_0-mlogloss:0.556793
    [53]	validation_0-mlogloss:0.55439
    [54]	validation_0-mlogloss:0.552006
    [55]	validation_0-mlogloss:0.549798
    [56]	validation_0-mlogloss:0.547591
    [57]	validation_0-mlogloss:0.545573
    [58]	validation_0-mlogloss:0.543886
    [59]	validation_0-mlogloss:0.542113
    [60]	validation_0-mlogloss:0.540389
    [61]	validation_0-mlogloss:0.537976
    [62]	validation_0-mlogloss:0.536513
    [63]	validation_0-mlogloss:0.534732
    [64]	validation_0-mlogloss:0.533412
    [65]	validation_0-mlogloss:0.53167
    [66]	validation_0-mlogloss:0.530359
    [67]	validation_0-mlogloss:0.52876
    [68]	validation_0-mlogloss:0.527035
    [69]	validation_0-mlogloss:0.525793
    [70]	validation_0-mlogloss:0.524423
    [71]	validation_0-mlogloss:0.52235
    [72]	validation_0-mlogloss:0.521258
    [73]	validation_0-mlogloss:0.519966
    [74]	validation_0-mlogloss:0.518731
    [75]	validation_0-mlogloss:0.517765
    [76]	validation_0-mlogloss:0.516584
    [77]	validation_0-mlogloss:0.515267
    [78]	validation_0-mlogloss:0.514166
    [79]	validation_0-mlogloss:0.513287
    [80]	validation_0-mlogloss:0.512135
    [81]	validation_0-mlogloss:0.510947
    [82]	validation_0-mlogloss:0.510011
    [83]	validation_0-mlogloss:0.509051
    [84]	validation_0-mlogloss:0.508119
    [85]	validation_0-mlogloss:0.507152
    [86]	validation_0-mlogloss:0.506181
    [87]	validation_0-mlogloss:0.505228
    [88]	validation_0-mlogloss:0.504472
    [89]	validation_0-mlogloss:0.503728
    [90]	validation_0-mlogloss:0.502491
    [91]	validation_0-mlogloss:0.501492
    [92]	validation_0-mlogloss:0.500893
    [93]	validation_0-mlogloss:0.499965
    [94]	validation_0-mlogloss:0.498986
    [95]	validation_0-mlogloss:0.49769
    [96]	validation_0-mlogloss:0.497148
    [97]	validation_0-mlogloss:0.496566
    [98]	validation_0-mlogloss:0.495456
    [99]	validation_0-mlogloss:0.494433
    XGBoost performance:
                  precision    recall  f1-score   support
    
               1       0.78      0.77      0.77     31761
               2       0.80      0.84      0.82     42533
               3       0.75      0.86      0.80      5299
               4       0.88      0.78      0.83       423
               5       0.81      0.23      0.36      1392
               6       0.68      0.38      0.49      2640
               7       0.90      0.75      0.81      3104
    
        accuracy                           0.79     87152
       macro avg       0.80      0.66      0.70     87152
    weighted avg       0.79      0.79      0.78     87152
    


    /usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py:301: FutureWarning: The min_impurity_split parameter is deprecated. Its default value will change from 1e-7 to 0 in version 0.23, and it will be removed in 0.25. Use the min_impurity_decrease parameter instead.
      FutureWarning)


    
    
    
    This should show that we've completely overfitted the predictions (i.e. F1 score == 1.0). 
    So we've mimiked the RandomForest's behaviour perfectly!
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00    145591
               2       1.00      1.00      1.00    210335
               3       1.00      1.00      1.00     28306
               4       1.00      1.00      1.00      1826
               5       1.00      1.00      1.00      1975
               6       1.00      1.00      1.00      6585
               7       1.00      1.00      1.00     12090
    
        accuracy                           1.00    406708
       macro avg       1.00      1.00      1.00    406708
    weighted avg       1.00      1.00      1.00    406708
    
    
    
    
    This shows the performance on the actual `target` values of the test set and 
    that they are equal to the performance of the RandomForest model on the same data.
                  precision    recall  f1-score   support
    
               1       0.78      0.77      0.77     31761
               2       0.80      0.84      0.82     42533
               3       0.75      0.86      0.80      5299
               4       0.88      0.78      0.83       423
               5       0.81      0.23      0.36      1392
               6       0.68      0.38      0.49      2640
               7       0.90      0.75      0.81      3104
    
        accuracy                           0.79     87152
       macro avg       0.80      0.66      0.70     87152
    weighted avg       0.79      0.79      0.78     87152
    



```
print("XGBoost performance on future data:")
print(classification_report(__out(df_future), xgb.predict(__inp(df_future))))

print("DecisionTree aproximator on future data")
print(classification_report(__out(df_future), dt.predict(__inp(df_future))))
```

    XGBoost performance on future data:
                  precision    recall  f1-score   support
    
               1       0.78      0.76      0.77     31881
               2       0.79      0.84      0.82     42520
               3       0.76      0.86      0.81      5326
               4       0.85      0.73      0.79       421
               5       0.82      0.23      0.35      1358
               6       0.70      0.38      0.49      2623
               7       0.88      0.74      0.80      3023
    
        accuracy                           0.79     87152
       macro avg       0.80      0.65      0.69     87152
    weighted avg       0.79      0.79      0.78     87152
    
    DecisionTree aproximator on future data
                  precision    recall  f1-score   support
    
               1       0.78      0.76      0.77     31881
               2       0.79      0.84      0.82     42520
               3       0.75      0.86      0.80      5326
               4       0.84      0.73      0.79       421
               5       0.81      0.23      0.36      1358
               6       0.69      0.37      0.48      2623
               7       0.88      0.74      0.80      3023
    
        accuracy                           0.79     87152
       macro avg       0.79      0.65      0.69     87152
    weighted avg       0.78      0.79      0.78     87152
    


If you look closely then you will see some diffrences in performance between the two, but overall they seem pretty close actually!

# Conclusions

Yes, it seems plausible that you can indeed aproximate a tree ensamble (either a RandomForest or an XGBoost - and most likely a GradientBoostedTree or a LigtGBM model bu I haven't tested these) with into a single tree that you can later inspect and debug. 

There may be some performance drops between the two, but in my experiments, the distillation process *mostly* yielded the same results regardless if it was a XGBoost model or a RandomFores, or if we had a big or small dataset to train on.

One advice I'd give is that if you go on this route you need to actually compare the performance of the two datasets with a fresh dataset (either kept asside from the beggining or gathered anew) because **there is a difference between the two models**.

If you combine this [with model exporting to code](http://www.clungu.com/Converting_a_DecisionTree_into_python_code/), you get quite a nice dependency free deloyment process of a large an powerfull model.

In a future post I'd like to discuss also the GRANT option mentioned in the same HackerNews thread to see how it compares and performs.

>Graft, Reassemble, Answer delta, Neighbour sensitivity, Training delta (GRANT)



You can open this notebook in Colab by using the button bellow:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YDHkVQHhVxS5Kzy-Uxn4UAjP-d3e_Wxp?usp=sharing)

