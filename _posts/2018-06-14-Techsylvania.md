
# Goal: Monitoring media for terrorisim with AI

SIGINT


Summary:
* Take a text only dataset
* Rough segmentation of the data
* Data cleaning and enriching
* Embeddings
* Cluster the similar stuff togheter

# Setup

* setup a virtualenv for python3.5

* install the Jupyter notebook  

    `pip install jupyter`

* install the jupyter extensions  
    `pip install jupyter_contrib_nbextensions`
    `jupyter contrib nbextension install --user`


* start the notebook  
    `` `which python` `which jupyter-notebook` --no-browser --ip 127.0.0.1 --port 8888``

* install scipy, numpy, pandas, tensorflow, keras, scikit-learn  
    `pip install scipy numpy pandas tensorflow keras scikit-learn`

# Clasiffy adverse media articles

For this talk we will be using the [Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd/data) available on [Kaggle](www.kaggle.com).

Suppose we want to monitor all the media in the world, and only want to process the articles that relate to terrorism (exclude Donald Trumps, or <insert-here-other-political-figure-that-tries-to-destroy-a-country\>) 

## Load the dataset


```python
import pandas as pd
from IPython.display import display
table = pd.read_csv('./terrorism.csv', encoding = "ISO-8859-1")
display(table.describe()), display(table.head())
```

    /home/cristi/Envs/techsylvania/lib/python3.5/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (4,6,31,33,53,61,62,63,76,79,90,92,94,96,114,115,121) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



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
      <th>eventid</th>
      <th>iyear</th>
      <th>imonth</th>
      <th>iday</th>
      <th>extended</th>
      <th>country</th>
      <th>region</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>specificity</th>
      <th>...</th>
      <th>ransomamt</th>
      <th>ransomamtus</th>
      <th>ransompaid</th>
      <th>ransompaidus</th>
      <th>hostkidoutcome</th>
      <th>nreleased</th>
      <th>INT_LOG</th>
      <th>INT_IDEO</th>
      <th>INT_MISC</th>
      <th>INT_ANY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.703500e+05</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
      <td>165744.000000</td>
      <td>165744.000000</td>
      <td>170346.000000</td>
      <td>...</td>
      <td>1.279000e+03</td>
      <td>4.960000e+02</td>
      <td>7.070000e+02</td>
      <td>487.000000</td>
      <td>9911.000000</td>
      <td>9322.000000</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
      <td>170350.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.001776e+11</td>
      <td>2001.709997</td>
      <td>6.474365</td>
      <td>15.466845</td>
      <td>0.043634</td>
      <td>132.526669</td>
      <td>7.091441</td>
      <td>23.399774</td>
      <td>26.350909</td>
      <td>1.454428</td>
      <td>...</td>
      <td>3.224502e+06</td>
      <td>4.519918e+05</td>
      <td>3.849663e+05</td>
      <td>272.462012</td>
      <td>4.624458</td>
      <td>-28.717335</td>
      <td>-4.583387</td>
      <td>-4.510555</td>
      <td>0.091083</td>
      <td>-3.975128</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.314444e+09</td>
      <td>13.144146</td>
      <td>3.392364</td>
      <td>8.817929</td>
      <td>0.204279</td>
      <td>112.848161</td>
      <td>2.949206</td>
      <td>18.844885</td>
      <td>58.570068</td>
      <td>1.009005</td>
      <td>...</td>
      <td>3.090625e+07</td>
      <td>6.070186e+06</td>
      <td>2.435027e+06</td>
      <td>3130.068208</td>
      <td>2.041008</td>
      <td>58.737198</td>
      <td>4.542694</td>
      <td>4.630440</td>
      <td>0.583166</td>
      <td>4.691492</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.970000e+11</td>
      <td>1970.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>-53.154613</td>
      <td>-176.176447</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-9.900000e+01</td>
      <td>-9.900000e+01</td>
      <td>-9.900000e+01</td>
      <td>-99.000000</td>
      <td>1.000000</td>
      <td>-99.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.990053e+11</td>
      <td>1990.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>75.000000</td>
      <td>5.000000</td>
      <td>11.263580</td>
      <td>2.396199</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-9.900000e+01</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>-99.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
      <td>0.000000</td>
      <td>-9.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.007121e+11</td>
      <td>2007.000000</td>
      <td>6.000000</td>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>98.000000</td>
      <td>6.000000</td>
      <td>31.472680</td>
      <td>43.130000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.420000e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.014023e+11</td>
      <td>2014.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
      <td>0.000000</td>
      <td>160.000000</td>
      <td>10.000000</td>
      <td>34.744167</td>
      <td>68.451297</td>
      <td>1.000000</td>
      <td>...</td>
      <td>4.000000e+05</td>
      <td>0.000000e+00</td>
      <td>7.356800e+02</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.017013e+11</td>
      <td>2016.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
      <td>1.000000</td>
      <td>1004.000000</td>
      <td>12.000000</td>
      <td>74.633553</td>
      <td>179.366667</td>
      <td>5.000000</td>
      <td>...</td>
      <td>1.000000e+09</td>
      <td>1.320000e+08</td>
      <td>4.100000e+07</td>
      <td>48000.000000</td>
      <td>7.000000</td>
      <td>1201.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 77 columns</p>
</div>



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
      <th>eventid</th>
      <th>iyear</th>
      <th>imonth</th>
      <th>iday</th>
      <th>approxdate</th>
      <th>extended</th>
      <th>resolution</th>
      <th>country</th>
      <th>country_txt</th>
      <th>region</th>
      <th>...</th>
      <th>addnotes</th>
      <th>scite1</th>
      <th>scite2</th>
      <th>scite3</th>
      <th>dbsource</th>
      <th>INT_LOG</th>
      <th>INT_IDEO</th>
      <th>INT_MISC</th>
      <th>INT_ANY</th>
      <th>related</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>197000000001</td>
      <td>1970</td>
      <td>7</td>
      <td>2</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>58</td>
      <td>Dominican Republic</td>
      <td>2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PGIS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>197000000002</td>
      <td>1970</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>130</td>
      <td>Mexico</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PGIS</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>197001000001</td>
      <td>1970</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>160</td>
      <td>Philippines</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PGIS</td>
      <td>-9</td>
      <td>-9</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>197001000002</td>
      <td>1970</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>78</td>
      <td>Greece</td>
      <td>8</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PGIS</td>
      <td>-9</td>
      <td>-9</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197001000003</td>
      <td>1970</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>101</td>
      <td>Japan</td>
      <td>4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PGIS</td>
      <td>-9</td>
      <td>-9</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 135 columns</p>
</div>





    (None, None)



* Train classifier

First, let's see how many types of terrorist attacks we have documented, in how many datapoints


```python
import numpy as np
summaries = table['summary'].values
empty_summaries = np.array([isinstance(summ, float) and np.isnan(summ) for summ in summaries], dtype=bool)
sum(empty_summaries)
```




    66138



How does one summary look like?


```python
summ = table['summary'].tolist()[11]
summ
```




    '1/6/1970: Unknown perpetrators threw a Molotov cocktail into an Army Recruiting Station in Denver, Colorado, United States.  There were no casualties but damages to the station were estimated at $305.'




```python
sum(np.isnan(table['attacktype1'].values))
```




    0



So we have 66k text descriptions and all of them have an attacktype specified.

What are all the attack types that we have labeled?


```python
set(table['attacktype1_txt'].tolist())
```




    {'Armed Assault',
     'Assassination',
     'Bombing/Explosion',
     'Facility/Infrastructure Attack',
     'Hijacking',
     'Hostage Taking (Barricade Incident)',
     'Hostage Taking (Kidnapping)',
     'Unarmed Assault',
     'Unknown'}



We should actually convert those to indexed values so we can deal with numbers instead of strings.

Fortunately, the dataset already gracefully provides the indexed labels from above


```python
set(table['attacktype1'].tolist())
```




    {1, 2, 3, 4, 5, 6, 7, 8, 9}




```python
classtype = {classname: classvalue for classname, classvalue in table[['attacktype1_txt', 'attacktype1']].values}
classindx = dict(zip(classtype.values(), classtype.keys()))
classindx = [''] + [classindx[i] for i in range(1, len(classindx)+1)]
classtype, classindx
```




    ({'Armed Assault': 2,
      'Assassination': 1,
      'Bombing/Explosion': 3,
      'Facility/Infrastructure Attack': 7,
      'Hijacking': 4,
      'Hostage Taking (Barricade Incident)': 5,
      'Hostage Taking (Kidnapping)': 6,
      'Unarmed Assault': 8,
      'Unknown': 9},
     ['',
      'Assassination',
      'Armed Assault',
      'Bombing/Explosion',
      'Hijacking',
      'Hostage Taking (Barricade Incident)',
      'Hostage Taking (Kidnapping)',
      'Facility/Infrastructure Attack',
      'Unarmed Assault',
      'Unknown'])



## Build the classification dataset

We now want to build a classifier that can quickly sort out the things we are/are not interested in.


```python
raw_classification_lables = table['attacktype1'].values
raw_classification_inputs = table['summary'].values
mask_for_non_empty_summaries = np.array([not (isinstance(summ, float) and np.isnan(summ)) for summ in summaries], dtype=bool)

classification_inputs = raw_classification_inputs[mask_for_non_empty_summaries]
classification_labels = raw_classification_lables[mask_for_non_empty_summaries]

assert classification_inputs.shape[0] == classification_labels.shape[0]

classification_inputs[0], classification_labels[0] 
```




    ('1/1/1970: Unknown African American assailants fired several bullets at police headquarters in Cairo, Illinois, United States.  There were no casualties, however, one bullet narrowly missed several police officers.  This attack took place during heightened racial tensions, including a Black boycott of White-owned businesses, in Cairo Illinois.',
     2)



## Train, test split

Now that we have a dataset, we will shuffle it and split it into training and test sets.


```python
import numpy as np

# suffle
perm = np.random.permutation(classification_inputs.shape[0])
classification_inputs = classification_inputs[perm]
classification_labels = classification_labels[perm]

from sklearn.model_selection import train_test_split
classification_train_inputs, classification_test_inputs, classification_train_lables, classification_test_lables = train_test_split(classification_inputs, classification_labels, test_size=0.33)

assert classification_train_inputs.shape[0] == classification_train_lables.shape[0]
assert classification_train_inputs.shape[0] + classification_test_inputs.shape[0] == classification_inputs.shape[0]
```

## Build a really quick classification model 

For this task we will use [scikit-learn](http://scikit-learn.org/stable/index.html).  

We will:
    * build a simple data pipeline 
    * use stop-words to trim out the frequent (useless words) out of our vocabulary
    * use the [td-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (term frequency - inverse document frequency) method to vectorize the articles.
    * use the final vectors to train a [Bayesian classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) on top of the data we have.
    * use [grid-search](https://en.wikipedia.org/wiki/Hyperparameter_optimization) for optimizind the hyperparamters and fitting a better model instance.


```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

classifier = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])
classifier.fit(X=classification_inputs, y=classification_labels)
```




    Pipeline(steps=[('vectorizer', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=Tr...      vocabulary=None)), ('classifier', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])



Scikit-lean has a nice module for computing the metrics that we want. We're just going to use that to see how we did.


```python
from sklearn.metrics import classification_report
print(classification_report(classification_test_lables, classifier.predict(classification_test_inputs), target_names=classindx[1:]))
```

                                         precision    recall  f1-score   support
    
                          Assassination       0.96      0.01      0.02      2095
                          Armed Assault       0.62      0.63      0.63      8126
                      Bombing/Explosion       0.72      0.99      0.84     18199
                              Hijacking       0.00      0.00      0.00        89
    Hostage Taking (Barricade Incident)       0.00      0.00      0.00       115
            Hostage Taking (Kidnapping)       0.98      0.33      0.50      2496
         Facility/Infrastructure Attack       0.92      0.11      0.20      1826
                        Unarmed Assault       0.00      0.00      0.00       183
                                Unknown       0.00      0.00      0.00      1261
    
                            avg / total       0.71      0.71      0.64     34390
    


    /home/cristi/Envs/techsylvania/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


So the results are somewhat bad for for the classes where we only have ~100 examples each but on the frequent classes (and the ones that we're interested on 'Bombing/Explosions') it's not that bad. 

Nevertheless, some optimisations are required.  
What we can do is tune the [hyperparamters](https://en.wikipedia.org/wiki/Hyperparameter_optimization) so that we find the best overall model.

**NOTE!! The code bellow take some minutes to run, so we'll not run it. I've ran it for you before the talk so we can see the results bellow**


```python
from sklearn.model_selection import GridSearchCV
parameters = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__use_idf': (True, False),
    'classifier__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(classifier, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(classification_train_inputs, classification_train_lables)
```

What was the best combination found?


```python
gs_clf.best_params_
```




    {'classifier__alpha': 0.01,
     'vectorizer__ngram_range': (1, 2),
     'vectorizer__use_idf': False}




```python
from sklearn.metrics import classification_report
print(classification_report(classification_test_lables, gs_clf.predict(classification_test_inputs), target_names=classindx[1:]))
```

                                         precision    recall  f1-score   support
    
                          Assassination       0.98      0.92      0.95      4184
                          Armed Assault       0.94      0.96      0.95     16525
                      Bombing/Explosion       0.99      0.98      0.98     37182
                              Hijacking       0.98      0.54      0.70       191
    Hostage Taking (Barricade Incident)       0.98      0.60      0.74       236
            Hostage Taking (Kidnapping)       0.97      0.98      0.98      4886
         Facility/Infrastructure Attack       0.92      0.95      0.93      3710
                        Unarmed Assault       1.00      0.76      0.86       362
                                Unknown       0.80      0.88      0.84      2546
    
                            avg / total       0.96      0.96      0.96     69822
    


## Conclusion

So using the above classifier we can filter out really fast, any media article that we're not interested in and then focus on the specific use case we want to handle.

There's A TON of other approached that you could try to improve the above result. Nowadays, in NPL you don't want to do '[bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)' models as we just did above, but instead you would vectorize the text using [word vectors](https://en.wikipedia.org/wiki/Word_embedding).  

You might also want to try some other advanced stuff like:
* Vectorize by passing the word vectors through an RNN
* Use a bidirectional RNN for better state-of-the-art results
* Use a stacked CNN on top of the word vectors for a different type of vectorisation
* Use an attention mechanism right before the classification output, etc..

# Extract Named Entities

Now that we have only the articles that we're interested in (i.e. we have a classifier that we can use to select them), we need, for each one to reason on it's content.  

The most important thing we can do is to parse the text for interesting tokens.  
Usually, these are: people names, geographical location (countries, cities), landmarks (eg. eiffel tour), dates, etc..  

In academia, this is a fairly well established problem that is known as [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition).   

There numerous papers, strategies and datasets that you can use to train an ML model for this.  
There are also some pretty decent libraries that come included with pretrained NER model, one of which is [spacy](https://spacy.io/usage/models).


```python
import spacy
```

Download the english language model from scipy repo  
`python -m spacy download en`


```python
nlp = spacy.load('en')
```

Let's see one example of what this does:


```python
doc = nlp(summ)
doc, [(ent.label_, ent.text) for ent in doc.ents] 
```




    (1/6/1970: Unknown perpetrators threw a Molotov cocktail into an Army Recruiting Station in Denver, Colorado, United States.  There were no casualties but damages to the station were estimated at $305.,
     [('DATE', '1/6/1970'),
      ('ORG', 'Army Recruiting Station'),
      ('GPE', 'Denver'),
      ('GPE', 'Colorado'),
      ('GPE', 'United States'),
      ('MONEY', '305')])



Example entity [types](https://spacy.io/usage/linguistic-features#section-named-entities) (values found in .label_):

     ORG    = organization
     GPE    = geo-political entity
     PERSON = person (may be fictional!)
     ...

We will build a class that will take a text string and return an processed `event`


```python
class Process:
    def __init__(self, language_model):
        self.model = language_model
        
    def text(self, data):
        data = str(data)
        results = {'TEXT': data}
        for ent in self.model(data).ents:
            results.setdefault(ent.label_, set()).add(ent.text)
        return results
    
process = Process(nlp)
process.text(summ)
```




    {'DATE': {'1/6/1970'},
     'GPE': {'Colorado', 'Denver', 'United States'},
     'MONEY': {'305'},
     'ORG': {'Army Recruiting Station'},
     'TEXT': '1/6/1970: Unknown perpetrators threw a Molotov cocktail into an Army Recruiting Station in Denver, Colorado, United States.  There were no casualties but damages to the station were estimated at $305.'}




```python
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
```


    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    


Let's see some examples


```python
extracted = table['summary'][:10].progress_apply(process.text)
```


    HBox(children=(IntProgress(value=0, max=10), HTML(value='')))


    



```python
extracted.tolist()
```




    [{'TEXT': 'nan'},
     {'TEXT': 'nan'},
     {'TEXT': 'nan'},
     {'TEXT': 'nan'},
     {'TEXT': 'nan'},
     {'CARDINAL': {'one'},
      'DATE': {'1/1/1970'},
      'GPE': {'Cairo', 'Illinois', 'United States'},
      'NORP': {'African American'},
      'ORG': {'White-'},
      'TEXT': '1/1/1970: Unknown African American assailants fired several bullets at police headquarters in Cairo, Illinois, United States.  There were no casualties, however, one bullet narrowly missed several police officers.  This attack took place during heightened racial tensions, including a Black boycott of White-owned businesses, in Cairo Illinois.'},
     {'TEXT': 'nan'},
     {'CARDINAL': {'Three'},
      'DATE': {'1/2/1970'},
      'GPE': {'California', 'Oakland', 'United States'},
      'MONEY': {'an estimated $20,000 to $25,000'},
      'ORG': {'the Pacific Gas & Electric Company'},
      'TEXT': '1/2/1970: Unknown perpetrators detonated explosives at the Pacific Gas & Electric Company Edes substation in Oakland, California, United States.  Three transformers were damaged costing an estimated $20,000 to $25,000.  There were no casualties.'},
     {'DATE': {'1/2/1970'},
      'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
      'MONEY': {'around $60,000'},
      'ORG': {'the New Years Gang',
       'the Old Red Gym',
       'the University of Wisconsin'},
      'PERSON': {'Karl Armstrong', 'Madison'},
      'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'},
     {'DATE': {'1/3/1970'},
      'GPE': {'United States', 'Wisconsin'},
      'ORDINAL': {'first'},
      'ORG': {'Selective Service Headquarters',
       'the New Years Gang',
       "the University of Wisconsin's"},
      'PERSON': {'Armstrong', 'Karl Armstrong', 'Madison'},
      'TEXT': "1/3/1970: Karl Armstrong, a member of the New Years Gang, broke into the University of Wisconsin's Primate Lab and set a fire on the first floor of the building.  Armstrong intended to set fire to the Madison, Wisconsin, United States, Selective Service Headquarters across the street but mistakenly confused the building with the Primate Lab.  The fire caused slight damages and was extinguished almost immediately."}]




```python
sample_events = [event for event in extracted.tolist() if event['TEXT'] != 'nan']
sample_events, len(sample_events)
```




    ([{'CARDINAL': {'one'},
       'DATE': {'1/1/1970'},
       'GPE': {'Cairo', 'Illinois', 'United States'},
       'NORP': {'African American'},
       'ORG': {'White-'},
       'TEXT': '1/1/1970: Unknown African American assailants fired several bullets at police headquarters in Cairo, Illinois, United States.  There were no casualties, however, one bullet narrowly missed several police officers.  This attack took place during heightened racial tensions, including a Black boycott of White-owned businesses, in Cairo Illinois.'},
      {'CARDINAL': {'Three'},
       'DATE': {'1/2/1970'},
       'GPE': {'California', 'Oakland', 'United States'},
       'MONEY': {'an estimated $20,000 to $25,000'},
       'ORG': {'the Pacific Gas & Electric Company'},
       'TEXT': '1/2/1970: Unknown perpetrators detonated explosives at the Pacific Gas & Electric Company Edes substation in Oakland, California, United States.  Three transformers were damaged costing an estimated $20,000 to $25,000.  There were no casualties.'},
      {'DATE': {'1/2/1970'},
       'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
       'MONEY': {'around $60,000'},
       'ORG': {'the New Years Gang',
        'the Old Red Gym',
        'the University of Wisconsin'},
       'PERSON': {'Karl Armstrong', 'Madison'},
       'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'},
      {'DATE': {'1/3/1970'},
       'GPE': {'United States', 'Wisconsin'},
       'ORDINAL': {'first'},
       'ORG': {'Selective Service Headquarters',
        'the New Years Gang',
        "the University of Wisconsin's"},
       'PERSON': {'Armstrong', 'Karl Armstrong', 'Madison'},
       'TEXT': "1/3/1970: Karl Armstrong, a member of the New Years Gang, broke into the University of Wisconsin's Primate Lab and set a fire on the first floor of the building.  Armstrong intended to set fire to the Madison, Wisconsin, United States, Selective Service Headquarters across the street but mistakenly confused the building with the Primate Lab.  The fire caused slight damages and was extinguished almost immediately."}],
     4)




```python
sample_event = sample_events[2]
sample_event
```




    {'DATE': {'1/2/1970'},
     'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
     'MONEY': {'around $60,000'},
     'ORG': {'the New Years Gang',
      'the Old Red Gym',
      'the University of Wisconsin'},
     'PERSON': {'Karl Armstrong', 'Madison'},
     'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'}



## Process all the sumaries

Let's process all the data that we have in the database.. **but it takes roughly one 1.5 hours :)**


```python
all_events = table['summary'].progress_apply(process.text)
```


    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    



    HBox(children=(IntProgress(value=0, max=170350), HTML(value='')))


    



```python
import numpy as np
np.savez_compressed(
    "./table_data.npz",
    summaries=np.array(table['summary'].tolist()),
    all_events=np.array(all_events.tolist()),
    attacktype=np.array(table['attacktype1'].tolist()),
    attacktype_txt=np.array(table['attacktype1_txt'].tolist())
)
```

Better load them up from a backup


```python
import numpy as np
with np.load("./table_data.npz") as store:
    all_events = store['all_events']
```


```python
len(all_events), type(all_events)
```




    (170350, numpy.ndarray)



Count all the non-empty events in the dataset


```python
sum(1 for event in all_events.tolist() if event['TEXT'] != 'nan')
```




    104212



## Conclusion

We now have the means, from the filtered media to parse the article content and extract `events` of structured data.  
This enables us to creat structured queries on the information that we collect.

# Enrich data

The above parsing stage, although usefull still doesn't provide enough information suited for our SIGINT bosses.  
We're actually leaving a lof of usable information on the table, and we can do better on exracting it and making it queryable.

We will implement bellow some `enhancer` modules that will transform our `events` into content rich elments.

## Adress enricher

The first thing we can do is transform the 'GPE' elements into rich addresses datastructures.   

[geopy](https://github.com/geopy/geopy) - Geocoding library for Python. Based on OSM. (there are other examples available)

`pip install geopy`


```python
from geopy.geocoders import Nominatim
```


```python
geolocator = Nominatim()
```


```python
geolocator.geocode('Cluj Napoca', timeout=5, addressdetails=True).raw
```




    {'address': {'city': 'Cluj-Napoca',
      'country': 'România',
      'country_code': 'ro',
      'county': 'Cluj',
      'postcode': '400133'},
     'boundingbox': ['46.6093367', '46.9293367', '23.4300604', '23.7500604'],
     'class': 'place',
     'display_name': 'Cluj-Napoca, Cluj, 400133, România',
     'icon': 'https://nominatim.openstreetmap.org/images/mapicons/poi_place_city.p.20.png',
     'importance': 0.37419916164485,
     'lat': '46.7693367',
     'licence': 'Data © OpenStreetMap contributors, ODbL 1.0. https://osm.org/copyright',
     'lon': '23.5900604',
     'osm_id': '32591050',
     'osm_type': 'node',
     'place_id': '195112',
     'type': 'city'}



If you're interested you in `.raw` you have lot more information.


```python
geolocator.geocode('Oakland', addressdetails=True, geometry='geojson').raw
```




    {'address': {'city': 'Oakland',
      'country': 'United States of America',
      'country_code': 'us',
      'county': 'Alameda County',
      'state': 'California'},
     'boundingbox': ['37.632226', '37.885368', '-122.355881', '-122.114672'],
     'class': 'place',
     'display_name': 'Oakland, Alameda County, California, United States of America',
     'geojson': {'coordinates': [[[-122.355881, 37.835727],
        [-122.3500919, 37.8201616],
        [-122.3468451, 37.8114822],
        [-122.3465852, 37.8108476],
        [-122.340281, 37.800628],
        [-122.33516, 37.799448],
        [-122.3198, 37.795908],
        [-122.31468, 37.794728],
        [-122.312471, 37.794484],
        [-122.305305, 37.793692],
        [-122.29478, 37.792528],
        [-122.29388, 37.792628],
        [-122.292678, 37.792778],
        [-122.286379, 37.793565],
        [-122.28428, 37.793828],
        [-122.282679, 37.793548],
        [-122.27788, 37.792708],
        [-122.27628, 37.792428],
        [-122.275951, 37.792272],
        [-122.274964, 37.791804],
        [-122.274636, 37.791648],
        [-122.274544, 37.791584],
        [-122.274271, 37.791392],
        [-122.27418, 37.791328],
        [-122.27302, 37.790668],
        [-122.26954, 37.788688],
        [-122.26838, 37.788028],
        [-122.26596, 37.787152],
        [-122.264285, 37.786545],
        [-122.260868, 37.785308],
        [-122.258703, 37.784525],
        [-122.25788, 37.784228],
        [-122.25668, 37.785428],
        [-122.25558, 37.786328],
        [-122.25478, 37.786728],
        [-122.253844, 37.786594],
        [-122.25058, 37.786127],
        [-122.246179, 37.783426],
        [-122.245606, 37.782233],
        [-122.244978, 37.780925],
        [-122.244817, 37.779123],
        [-122.24488, 37.777354],
        [-122.244385, 37.776997],
        [-122.242569, 37.775687],
        [-122.240684, 37.774327],
        [-122.238506, 37.772954],
        [-122.236566, 37.771731],
        [-122.236488, 37.771696],
        [-122.235308, 37.77114],
        [-122.231769, 37.769474],
        [-122.23059, 37.768919],
        [-122.230537, 37.768899],
        [-122.230514, 37.768891],
        [-122.230438, 37.768864],
        [-122.230377, 37.768842],
        [-122.230325, 37.768823],
        [-122.229152, 37.768077],
        [-122.226947, 37.766675],
        [-122.225986, 37.765454],
        [-122.225154, 37.764397],
        [-122.225127, 37.764363],
        [-122.224662, 37.762402],
        [-122.224639, 37.762302],
        [-122.224151, 37.760241],
        [-122.223983, 37.75953],
        [-122.2239157, 37.7578529],
        [-122.223859, 37.75644],
        [-122.2238617, 37.7546503],
        [-122.223939, 37.754148],
        [-122.224419, 37.753308],
        [-122.224579, 37.753028],
        [-122.224842, 37.752361],
        [-122.225631, 37.75036],
        [-122.225895, 37.749694],
        [-122.226029, 37.749335],
        [-122.226432, 37.748261],
        [-122.2266426, 37.747496],
        [-122.226634, 37.7456907],
        [-122.2266, 37.745501],
        [-122.226174, 37.745402],
        [-122.225782, 37.745347],
        [-122.225491, 37.745305],
        [-122.22512, 37.745253],
        [-122.225221, 37.742207],
        [-122.2252677, 37.7407975],
        [-122.2253386, 37.7386598],
        [-122.225524, 37.733071],
        [-122.225562, 37.731959],
        [-122.225653, 37.731766],
        [-122.225785, 37.731462],
        [-122.226171, 37.731039],
        [-122.22664, 37.730371],
        [-122.226574, 37.729607],
        [-122.226379, 37.727316],
        [-122.226314, 37.726553],
        [-122.226663, 37.726615],
        [-122.22771, 37.726804],
        [-122.22806, 37.726867],
        [-122.228084, 37.726841],
        [-122.228158, 37.726763],
        [-122.228183, 37.726738],
        [-122.228379, 37.726529],
        [-122.228625, 37.726277],
        [-122.231853, 37.72297],
        [-122.232267, 37.722546],
        [-122.235965, 37.718759],
        [-122.236101, 37.71862],
        [-122.236902, 37.71902],
        [-122.2396303, 37.7209887],
        [-122.244042, 37.724172],
        [-122.244359, 37.724401],
        [-122.246139, 37.725685],
        [-122.247919, 37.726969],
        [-122.248308, 37.72725],
        [-122.248827, 37.727808],
        [-122.257477, 37.720448],
        [-122.261513, 37.717015],
        [-122.251514, 37.710788],
        [-122.255398, 37.707621],
        [-122.269663, 37.70795],
        [-122.28178, 37.70823],
        [-122.28088, 37.70723],
        [-122.27852, 37.704389],
        [-122.268864, 37.692763],
        [-122.265646, 37.688888],
        [-122.262912, 37.685596],
        [-122.254712, 37.675722],
        [-122.251979, 37.672431],
        [-122.250919, 37.671091],
        [-122.247739, 37.667071],
        [-122.246679, 37.665731],
        [-122.243359, 37.661693],
        [-122.233402, 37.649579],
        [-122.230083, 37.645542],
        [-122.228172, 37.643217],
        [-122.222441, 37.636244],
        [-122.220531, 37.63392],
        [-122.220252, 37.633581],
        [-122.219416, 37.632564],
        [-122.219138, 37.632226],
        [-122.210477, 37.639931],
        [-122.192169, 37.656224],
        [-122.185139, 37.662923],
        [-122.184717, 37.663288],
        [-122.184314, 37.663638],
        [-122.181977, 37.665731],
        [-122.176816, 37.670712],
        [-122.1805886, 37.6706514],
        [-122.1915576, 37.6833296],
        [-122.1922955, 37.6827235],
        [-122.2002987, 37.6922869],
        [-122.2097346, 37.7110809],
        [-122.2095614, 37.7111215],
        [-122.2021718, 37.7129762],
        [-122.1970363, 37.7158231],
        [-122.1944265, 37.7158239],
        [-122.1942729, 37.7159061],
        [-122.1947225, 37.7175321],
        [-122.1934095, 37.7178236],
        [-122.1929432, 37.716788],
        [-122.1928462, 37.7168523],
        [-122.1894321, 37.7178203],
        [-122.1894285, 37.7178213],
        [-122.1894253, 37.7178224],
        [-122.1894228, 37.7178232],
        [-122.1894201, 37.7178242],
        [-122.1894108, 37.7178278],
        [-122.1894087, 37.7178288],
        [-122.189406, 37.71783],
        [-122.1894018, 37.7178319],
        [-122.1893984, 37.7178337],
        [-122.1893947, 37.7178356],
        [-122.1893912, 37.7178375],
        [-122.189387, 37.71784],
        [-122.1893849, 37.7178413],
        [-122.1893818, 37.7178432],
        [-122.1893791, 37.7178451],
        [-122.1893764, 37.7178469],
        [-122.1893707, 37.7178512],
        [-122.1893678, 37.7178535],
        [-122.1893662, 37.7178549],
        [-122.1893639, 37.7178568],
        [-122.1893615, 37.717859],
        [-122.1893586, 37.7178617],
        [-122.1893527, 37.7178678],
        [-122.1893497, 37.717871],
        [-122.1893472, 37.717874],
        [-122.1893434, 37.7178788],
        [-122.1893419, 37.717881],
        [-122.1893392, 37.7178848],
        [-122.189338, 37.7178867],
        [-122.1893366, 37.7178889],
        [-122.1893348, 37.7178919],
        [-122.1893334, 37.7178943],
        [-122.1893319, 37.7178972],
        [-122.1893308, 37.7178994],
        [-122.1893291, 37.717903],
        [-122.189328, 37.7179055],
        [-122.1893271, 37.7179076],
        [-122.1893262, 37.7179102],
        [-122.1893248, 37.7179141],
        [-122.1893234, 37.7179185],
        [-122.1893226, 37.7179213],
        [-122.1893217, 37.717925],
        [-122.1893211, 37.717928],
        [-122.1893207, 37.7179305],
        [-122.1893196, 37.717938],
        [-122.1893197, 37.7179608],
        [-122.1893212, 37.7179704],
        [-122.1893216, 37.7179724],
        [-122.1893221, 37.7179745],
        [-122.1893225, 37.7179763],
        [-122.189323, 37.7179782],
        [-122.1893239, 37.7179814],
        [-122.1893248, 37.7179842],
        [-122.1893256, 37.7179866],
        [-122.1893265, 37.7179889],
        [-122.189328, 37.7179927],
        [-122.1893296, 37.7179962],
        [-122.1893306, 37.7179983],
        [-122.1893323, 37.7180018],
        [-122.1893338, 37.7180045],
        [-122.1893364, 37.718009],
        [-122.1893376, 37.7180109],
        [-122.1893388, 37.7180128],
        [-122.1893406, 37.7180155],
        [-122.1893417, 37.7180171],
        [-122.1893446, 37.7180209],
        [-122.1893468, 37.7180237],
        [-122.1893481, 37.7180252],
        [-122.1893496, 37.718027],
        [-122.1893509, 37.7180285],
        [-122.1893534, 37.7180312],
        [-122.1893559, 37.7180339],
        [-122.189359, 37.7180369],
        [-122.1896191, 37.7182671],
        [-122.1896771, 37.7183184],
        [-122.189748, 37.7183811],
        [-122.1899005, 37.718516],
        [-122.191689, 37.7200451],
        [-122.1941388, 37.7221476],
        [-122.1962817, 37.7233039],
        [-122.197404, 37.7252566],
        [-122.1977541, 37.7251652],
        [-122.1980629, 37.7262216],
        [-122.1978788, 37.7262982],
        [-122.1968054, 37.7267427],
        [-122.1961528, 37.7259974],
        [-122.195463, 37.7262654],
        [-122.1952163, 37.726399],
        [-122.1949018, 37.726564],
        [-122.1928563, 37.7276793],
        [-122.1928297, 37.7276936],
        [-122.1919357, 37.7272817],
        [-122.1913198, 37.7281285],
        [-122.1912718, 37.7281991],
        [-122.190983, 37.7280678],
        [-122.1908757, 37.7282159],
        [-122.1903833, 37.7288823],
        [-122.1900052, 37.728747],
        [-122.189116, 37.7285063],
        [-122.1890531, 37.7285924],
        [-122.1891571, 37.7286201],
        [-122.1886069, 37.7285014],
        [-122.1884775, 37.7284654],
        [-122.1882633, 37.7284198],
        [-122.187447, 37.7279691],
        [-122.1868899, 37.7278166],
        [-122.1864619, 37.7275827],
        [-122.1857298, 37.7267982],
        [-122.1852401, 37.7266435],
        [-122.184695, 37.7262223],
        [-122.1842884, 37.7260474],
        [-122.1841962, 37.7259949],
        [-122.1841369, 37.7259411],
        [-122.1838394, 37.7256708],
        [-122.1836141, 37.7256091],
        [-122.1832994, 37.7255463],
        [-122.1831021, 37.725507],
        [-122.1826651, 37.7254199],
        [-122.1825451, 37.7253829],
        [-122.1823968, 37.7253572],
        [-122.1820668, 37.7253502],
        [-122.1819839, 37.7253574],
        [-122.1811768, 37.7254445],
        [-122.180697, 37.7256497],
        [-122.1805141, 37.7257279],
        [-122.1802995, 37.7257374],
        [-122.1799921, 37.7257062],
        [-122.1794241, 37.7253692],
        [-122.1793917, 37.7252428],
        [-122.1791452, 37.7251794],
        [-122.1788456, 37.7252161],
        [-122.1785759, 37.7252682],
        [-122.178237, 37.7253295],
        [-122.1778926, 37.7253829],
        [-122.1775416, 37.7253275],
        [-122.1768537, 37.7252809],
        [-122.176356, 37.7252455],
        [-122.1758113, 37.7252247],
        [-122.1754525, 37.7252967],
        [-122.1753517, 37.7253779],
        [-122.1750214, 37.7259275],
        [-122.1744362, 37.7265507],
        [-122.1743485, 37.7266437],
        [-122.1738415, 37.7267323],
        [-122.1729987, 37.7267497],
        [-122.1727106, 37.7268091],
        [-122.1725622, 37.7269876],
        [-122.1720073, 37.7276392],
        [-122.1716714, 37.7278924],
        [-122.1713014, 37.7280003],
        [-122.1708563, 37.7278968],
        [-122.1708263, 37.7277853],
        [-122.1706258, 37.727692],
        [-122.1706573, 37.727815],
        [-122.1703211, 37.7277266],
        [-122.171207, 37.7285001],
        [-122.1714323, 37.7286925],
        [-122.1737968, 37.7307587],
        [-122.1738722, 37.7308246],
        [-122.1742323, 37.7311405],
        [-122.1738568, 37.7312221],
        [-122.173863, 37.7312275],
        [-122.1705046, 37.7318622],
        [-122.1702545, 37.7319799],
        [-122.1700554, 37.7320735],
        [-122.1697571, 37.7322138],
        [-122.1703336, 37.7331004],
        [-122.1707125, 37.7336827],
        [-122.1710111, 37.7341418],
        [-122.1710381, 37.7341835],
        [-122.1711699, 37.7343861],
        [-122.1713923, 37.7347281],
        [-122.1715635, 37.7349913],
        [-122.1682431, 37.7365495],
        [-122.165791, 37.7377],
        [-122.1656193, 37.7377586],
        [-122.1655284, 37.7376586],
        [-122.1654178, 37.7375084],
        [-122.1642214, 37.7358999],
        [-122.161921, 37.7369679],
        [-122.1609643, 37.7374147],
        [-122.1608255, 37.7374678],
        [-122.1604836, 37.7375729],
        [-122.160071, 37.7376831],
        [-122.1585568, 37.7381037],
        [-122.1570203, 37.7385325],
        [-122.1550746, 37.7390725],
        [-122.1533904, 37.7395494],
        [-122.1519704, 37.7399295],
        [-122.1511386, 37.740167],
        [-122.1502105, 37.7404248],
        [-122.1487167, 37.7408628],
        [-122.1483868, 37.7409321],
        [-122.1476606, 37.7411349],
        [-122.1475912, 37.7411636],
        [-122.1474508, 37.7412684],
        [-122.1472756, 37.7414609],
        [-122.1470727, 37.7417074],
        [-122.1468491, 37.7420152],
        [-122.1466794, 37.7422401],
        [-122.1465062, 37.7424619],
        [-122.1460456, 37.7427526],
        [-122.1458247, 37.7425391],
        [-122.1456067, 37.7422994],
        [-122.145477, 37.7421425],
        [-122.1453595, 37.7419917],
        [-122.1451901, 37.7417617],
        [-122.1448595, 37.741327],
        [-122.1443557, 37.7406551],
        [-122.144205, 37.7404666],
        [-122.1439762, 37.7402061],
        [-122.1438009, 37.7400238],
        [-122.1436543, 37.739884],
        [-122.1434267, 37.7396819],
        [-122.1432682, 37.7395519],
        [-122.1420797, 37.7386136],
        [-122.1417613, 37.7383506],
        [-122.1416026, 37.7382127],
        [-122.1413447, 37.7379698],
        [-122.1411159, 37.7377086],
        [-122.1409432, 37.7374868],
        [-122.1408622, 37.737375],
        [-122.1406368, 37.7370106],
        [-122.1405013, 37.736741],
        [-122.1403938, 37.7364936],
        [-122.1403626, 37.7364099],
        [-122.1403036, 37.7362418],
        [-122.1402398, 37.7360181],
        [-122.1401847, 37.7357933],
        [-122.1398947, 37.7344861],
        [-122.13953, 37.7329106],
        [-122.1395115, 37.7328269],
        [-122.1392806, 37.7319666],
        [-122.138994, 37.7309489],
        [-122.1389854, 37.7309187],
        [-122.1382252, 37.7309656],
        [-122.1381012, 37.7310036],
        [-122.1376605, 37.7313624],
        [-122.137082, 37.7316072],
        [-122.1367081, 37.7318518],
        [-122.1364564, 37.732118],
        [-122.1360903, 37.7327194],
        [-122.1357155, 37.7331767],
        [-122.1354654, 37.7336162],
        [-122.1348556, 37.7343023],
        [-122.1344755, 37.7342905],
        [-122.1344466, 37.7342896],
        [-122.1343822, 37.7342775],
        [-122.1335268, 37.7341159],
        [-122.133186, 37.7340516],
        [-122.132825, 37.7339832],
        [-122.1320072, 37.733661],
        [-122.1312331, 37.7335812],
        [-122.1306313, 37.7336884],
        [-122.1304013, 37.7334194],
        [-122.130499, 37.7327637],
        [-122.1308741, 37.7323743],
        [-122.1311207, 37.7322618],
        [-122.1321873, 37.7317934],
        [-122.1323536, 37.7317204],
        [-122.1315875, 37.7308785],
        [-122.1312351, 37.7304213],
        [-122.131073, 37.7303271],
        [-122.1306578, 37.7301158],
        [-122.1303637, 37.7300731],
        [-122.1298457, 37.7299995],
        [-122.129349, 37.729954],
        [-122.1281305, 37.7298786],
        [-122.1270106, 37.7293253],
        [-122.126866, 37.729587],
        [-122.126824, 37.729576],
        [-122.126651, 37.729532],
        [-122.126416, 37.729477],
        [-122.126072, 37.729434],
        [-122.125546, 37.729396],
        [-122.125491, 37.72938],
        [-122.125429, 37.729352],
        [-122.125387, 37.729352],
        [-122.125173, 37.729297],
        [-122.125131, 37.729275],
        [-122.125055, 37.729253],
        [-122.124931, 37.729204],
        [-122.124813, 37.729149],
        [-122.12473, 37.729122],
        [-122.12471, 37.729111],
        [-122.124688, 37.7291],
        [-122.124557, 37.72905],
        [-122.124426, 37.729012],
        [-122.124384, 37.729006],
        [-122.124121, 37.729007],
        [-122.124038, 37.729023],
        [-122.124004, 37.729034],
        [-122.123887, 37.72908],
        [-122.12377, 37.729127],
        [-122.123676, 37.729174],
        [-122.122721, 37.72966],
        [-122.122403, 37.729823],
        [-122.122322, 37.729873],
        [-122.122082, 37.730027],
        [-122.122002, 37.730078],
        [-122.121584, 37.730417],
        [-122.120997, 37.730894],
        [-122.120268, 37.731348],
        [-122.119812, 37.731633],
        [-122.117984, 37.732559],
        [-122.117891, 37.732668],
        [-122.116765, 37.734006],
        [-122.116492, 37.734329],
        [-122.115618, 37.735886],
        [-122.115366, 37.736704],
        [-122.1155, 37.738461],
        [-122.11556, 37.739233],
        [-122.114984, 37.739994],
        [-122.114847, 37.740178],
        [-122.114672, 37.740334],
        [-122.114694, 37.740381],
        [-122.11471, 37.740414],
        [-122.114714, 37.740533],
        [-122.114716, 37.740586],
        [-122.114962, 37.742155],
        [-122.114983, 37.742284],
        [-122.115701, 37.746865],
        [-122.115948, 37.748436],
        [-122.115987, 37.748725],
        [-122.116105, 37.749593],
        [-122.116145, 37.749883],
        [-122.116584, 37.750715],
        [-122.117145, 37.751777],
        [-122.117438, 37.752331],
        [-122.117903, 37.753212],
        [-122.11823, 37.753831],
        [-122.118343, 37.754045],
        [-122.118424, 37.754156],
        [-122.118457, 37.754256],
        [-122.118547, 37.754543],
        [-122.120237, 37.757223],
        [-122.12232, 37.760528],
        [-122.125794, 37.76687],
        [-122.126196, 37.767603],
        [-122.128166, 37.769701],
        [-122.1292499, 37.7708957],
        [-122.129465, 37.771127],
        [-122.130074, 37.771795],
        [-122.130075, 37.771797],
        [-122.1301207, 37.771847],
        [-122.130382, 37.772133],
        [-122.1312789, 37.7734601],
        [-122.132361, 37.775061],
        [-122.132768, 37.775611],
        [-122.132837, 37.775681],
        [-122.132915, 37.775759],
        [-122.134276, 37.777127],
        [-122.133852, 37.777409],
        [-122.133676, 37.777527],
        [-122.134376, 37.778327],
        [-122.134608, 37.778227],
        [-122.135076, 37.778027],
        [-122.136872, 37.779955],
        [-122.142263, 37.78574],
        [-122.142634, 37.786138],
        [-122.143312, 37.786865],
        [-122.144061, 37.787669],
        [-122.143517, 37.78839],
        [-122.142207, 37.790129],
        [-122.142645, 37.790431],
        [-122.143389, 37.790945],
        [-122.143527, 37.790831],
        [-122.143943, 37.79049],
        [-122.144082, 37.790377],
        [-122.144368, 37.790625],
        [-122.144885, 37.791075],
        [-122.145227, 37.791372],
        [-122.145514, 37.791622],
        [-122.14627, 37.79234],
        [-122.149924, 37.792398],
        [-122.151171, 37.792418],
        [-122.151473, 37.796394],
        [-122.156874, 37.796469],
        [-122.156874, 37.796866],
        [-122.157874, 37.797361],
        [-122.158575, 37.796957],
        [-122.159276, 37.797452],
        [-122.159078, 37.797647],
        [-122.158576, 37.798147],
        [-122.160077, 37.799735],
        [-122.16187, 37.799934],
        [-122.162229, 37.800745],
        [-122.161961, 37.800896],
        [-122.161868, 37.800949],
        [-122.16192, 37.801356],
        [-122.164266, 37.803229],
        [-122.164289, 37.803286],
        [-122.164733, 37.804395],
        [-122.164477, 37.804424],
        [-122.164779, 37.805111],
        [-122.166682, 37.805586],
        [-122.166936, 37.805635],
        [-122.167022, 37.805652],
        [-122.17059, 37.804734],
        [-122.170192, 37.802315],
        [-122.169399, 37.801071],
        [-122.1686, 37.801161],
        [-122.168502, 37.800652],
        [-122.168703, 37.800346],
        [-122.168703, 37.800044],
        [-122.168705, 37.799025],
        [-122.17081, 37.799696],
        [-122.171733, 37.8003],
        [-122.172321, 37.800587],
        [-122.17569, 37.802234],
        [-122.175286, 37.803162],
        [-122.175758, 37.803357],
        [-122.176581, 37.803697],
        [-122.17736, 37.803316],
        [-122.177527, 37.803455],
        [-122.178031, 37.803874],
        [-122.178199, 37.804014],
        [-122.178182, 37.804028],
        [-122.178134, 37.80407],
        [-122.178118, 37.804085],
        [-122.177736, 37.804721],
        [-122.176592, 37.806632],
        [-122.176477, 37.806825],
        [-122.176129, 37.807208],
        [-122.176057, 37.807225],
        [-122.175995, 37.807241],
        [-122.17593, 37.807264],
        [-122.17585, 37.807297],
        [-122.17583, 37.807306],
        [-122.175785, 37.80733],
        [-122.175599, 37.807773],
        [-122.175577, 37.807827],
        [-122.175424, 37.808193],
        [-122.175111, 37.808929],
        [-122.175036, 37.809103],
        [-122.174849, 37.809547],
        [-122.174477, 37.810427],
        [-122.174562, 37.810982],
        [-122.175077, 37.814327],
        [-122.175903, 37.815153],
        [-122.176977, 37.816227],
        [-122.179093, 37.817778],
        [-122.181333, 37.8194193],
        [-122.181477, 37.819526],
        [-122.185977, 37.820726],
        [-122.18604, 37.820979],
        [-122.186677, 37.823526],
        [-122.186611, 37.823677],
        [-122.186417, 37.82413],
        [-122.186377, 37.824226],
        [-122.186355, 37.824283],
        [-122.186144, 37.824829],
        [-122.186036, 37.825113],
        [-122.185877, 37.825526],
        [-122.186413, 37.826381],
        [-122.186725, 37.826877],
        [-122.186752, 37.826972],
        [-122.186759, 37.826993],
        [-122.186784, 37.827043],
        [-122.186807, 37.827065],
        [-122.186858, 37.827075],
        [-122.186973, 37.827146],
        [-122.187058, 37.827199],
        [-122.187068, 37.827329],
        [-122.187094, 37.827665],
        [-122.187177, 37.828726],
        [-122.187134, 37.829062],
        [-122.187077, 37.829526],
        [-122.186811, 37.830003],
        [-122.186577, 37.830426],
        [-122.186166, 37.831013],
        [-122.185977, 37.831226],
        [-122.18589, 37.831357],
        [-122.185589, 37.831813],
        [-122.185584, 37.831818],
        [-122.185544, 37.831859],
        [-122.185525, 37.831878],
        [-122.185516, 37.831889],
        [-122.185468, 37.831945],
        [-122.185409, 37.832042],
        [-122.185375, 37.832107],
        [-122.185336, 37.83219],
        [-122.185276, 37.832297],
        [-122.185251, 37.832343],
        [-122.185207, 37.832434],
        [-122.185141, 37.832542],
        [-122.185085, 37.832612],
        [-122.18501, 37.832709],
        [-122.184802, 37.833012],
        [-122.184179, 37.833922],
        [-122.184177, 37.833926],
        [-122.184271, 37.834277],
        [-122.184305, 37.834406],
        [-122.184408, 37.834793],
        [-122.184443, 37.834923],
        [-122.18449, 37.835103],
        [-122.184516, 37.835199],
        [-122.184575, 37.835427],
        [-122.184628, 37.835646],
        [-122.184674, 37.835828],
        [-122.184703, 37.835887],
        [-122.184707, 37.835893],
        [-122.184783, 37.836047],
        [-122.185114, 37.836703],
        [-122.185225, 37.836923],
        [-122.185277, 37.837026],
        [-122.185903, 37.837143],
        [-122.186877, 37.837326],
        [-122.188132, 37.837514],
        [-122.188877, 37.837626],
        [-122.188886, 37.83763],
        [-122.188915, 37.837645],
        [-122.188925, 37.83765],
        [-122.189298, 37.837836],
        [-122.189644, 37.83801],
        [-122.189679, 37.838027],
        [-122.19031, 37.838559],
        [-122.190318, 37.838565],
        [-122.19063, 37.838828],
        [-122.190812, 37.838981],
        [-122.191358, 37.83944],
        [-122.19154, 37.839594],
        [-122.191578, 37.839626],
        [-122.191778, 37.839826],
        [-122.191794, 37.839833],
        [-122.191963, 37.839906],
        [-122.193433, 37.840545],
        [-122.193923, 37.840758],
        [-122.193917, 37.840834],
        [-122.193917, 37.840859],
        [-122.193917, 37.840963],
        [-122.193947, 37.841028],
        [-122.194028, 37.841126],
        [-122.194109, 37.841159],
        [-122.194123, 37.841164],
        [-122.194124, 37.841193],
        [-122.194191, 37.841284],
        [-122.194304, 37.841366],
        [-122.194452, 37.841426],
        [-122.194458, 37.841427],
        [-122.1946, 37.841468],
        [-122.194771, 37.841487],
        [-122.194988, 37.841524],
        [-122.195124, 37.841553],
        [-122.195214, 37.841593],
        [-122.195268, 37.84163],
        [-122.195335, 37.841688],
        [-122.195623, 37.841913],
        [-122.195666, 37.841936],
        [-122.195709, 37.84196],
        [-122.195831, 37.842004],
        [-122.195972, 37.842012],
        [-122.196101, 37.842005],
        [-122.19593, 37.842174],
        [-122.195418, 37.842685],
        [-122.195315, 37.842789],
        [-122.195293, 37.842799],
        [-122.195287, 37.842817],
        [-122.195282, 37.843113],
        [-122.1952772, 37.8431795],
        [-122.195278, 37.843312],
        [-122.195274, 37.843587],
        [-122.196209, 37.844361],
        [-122.196575, 37.844663],
        [-122.196593, 37.844678],
        [-122.196703, 37.844765],
        [-122.196778, 37.844825],
        [-122.19704, 37.844983],
        [-122.197059, 37.844994],
        [-122.19718, 37.845068],
        [-122.197279, 37.845123],
        [-122.197383, 37.845197],
        [-122.197528, 37.845307],
        [-122.198541, 37.846069],
        [-122.198879, 37.846324],
        [-122.199255, 37.846566],
        [-122.19999, 37.84704],
        [-122.200278, 37.847225],
        [-122.200371, 37.847309],
        [-122.200678, 37.847589],
        [-122.200702, 37.847612],
        [-122.200837, 37.847734],
        [-122.201242, 37.848103],
        [-122.201378, 37.848226],
        [-122.201744, 37.848684],
        [-122.202843, 37.850058],
        [-122.20321, 37.850516],
        [-122.203809, 37.851106],
        [-122.204094, 37.851387],
        [-122.206218, 37.851474],
        [-122.207063, 37.85151],
        [-122.207232, 37.851511],
        [-122.207307, 37.851526],
        [-122.207378, 37.851555],
        [-122.207564, 37.851649],
        [-122.207784, 37.851737],
        [-122.207861, 37.85175],
        [-122.207967, 37.851751],
        [-122.208063, 37.851754],
        [-122.208174, 37.85174],
        [-122.208486, 37.85169],
        [-122.209044, 37.852325],
        [-122.210718, 37.85423],
        [-122.210978, 37.854526],
        [-122.21081, 37.854946],
        [-122.210778, 37.855026],
        [-122.210957, 37.855322],
        [-122.211629, 37.856431],
        [-122.211854, 37.856801],
        [-122.211862, 37.85682],
        [-122.211885, 37.85687],
        [-122.21189, 37.85688],
        [-122.211902, 37.8569],
        [-122.211907, 37.856909],
        [-122.211926, 37.856937],
        [-122.211933, 37.856946],
        [-122.212012, 37.857013],
        [-122.21238, 37.857234],
        [-122.212494, 37.857269],
        [-122.212973, 37.857503],
        [-122.213135, 37.857583],
        [-122.213136, 37.857583],
        [-122.213176, 37.857623],
        [-122.213175, 37.857658],
        [-122.213173, 37.85778],
        [-122.213172, 37.857822],
        [-122.213194, 37.857835],
        [-122.213263, 37.857874],
        [-122.213778, 37.858026],
        [-122.213874, 37.858546],
        [-122.214736, 37.85951],
        [-122.215452, 37.860147],
        [-122.217531, 37.861998],
        [-122.218616, 37.8629304],
        [-122.220389, 37.864427],
        [-122.221488, 37.865026],
        [-122.220315, 37.866144],
        [-122.219181, 37.867226],
        [-122.217999, 37.867815],
        [-122.217897, 37.867866],
        [-122.217778, 37.867925],
        [-122.217583, 37.868061],
        [-122.217227, 37.868316],
        [-122.217075, 37.868424],
        [-122.216995, 37.868464],
        [-122.216783, 37.868571],
        [-122.216769, 37.868577],
        [-122.216276, 37.868822],
        [-122.216738, 37.870043],
        [-122.217376, 37.871724],
        [-122.218721, 37.872936],
        [-122.220211, 37.874276],
        [-122.2213955, 37.8753179],
        [-122.22156, 37.875543],
        [-122.221573, 37.875558],
        [-122.221711, 37.875723],
        [-122.222448, 37.876609],
        [-122.223878, 37.878326],
        [-122.225567, 37.879125],
        [-122.225778, 37.879225],
        [-122.226831, 37.87965],
        [-122.227116, 37.879764],
        [-122.227117, 37.879765],
        [-122.227661, 37.879984],
        [-122.230149, 37.880989],
        [-122.230979, 37.881325],
        [-122.231999, 37.881595],
        [-122.234378, 37.882225],
        [-122.235068, 37.882384],
        [-122.235679, 37.882525],
        [-122.236029, 37.88262],
        [-122.236038, 37.882623],
        [-122.236095, 37.882638],
        [-122.236382, 37.882714],
        [-122.237249, 37.882946],
        [-122.237538, 37.883024],
        [-122.237543, 37.883026],
        [-122.237597, 37.88304],
        [-122.237916, 37.883124],
        [-122.238679, 37.883325],
        [-122.239042, 37.883203],
        [-122.239353, 37.8831],
        [-122.239413, 37.88308],
        [-122.239506, 37.883048],
        [-122.239785, 37.882956],
        [-122.239879, 37.882925],
        [-122.240261, 37.882739],
        [-122.240273, 37.882732],
        [-122.241459, 37.882157],
        [-122.241855, 37.881966],
        [-122.241971, 37.881927],
        [-122.241979, 37.881924],
        [-122.244079, 37.883224],
        [-122.244114, 37.883252],
        [-122.244615, 37.883646],
        [-122.244635, 37.883662],
        [-122.244803, 37.883794],
        [-122.24531, 37.884192],
        [-122.245479, 37.884325],
        [-122.245716, 37.884511],
        [-122.24643, 37.885069],
        [-122.246586, 37.885191],
        [-122.2469126, 37.885368],
        [-122.2468285, 37.8846227],
        [-122.2468285, 37.8845997],
        [-122.246797, 37.8845137],
        [-122.246782, 37.8843897],
        [-122.2468341, 37.8841026],
        [-122.246759, 37.8840032],
        [-122.2466999, 37.8835531],
        [-122.2466768, 37.8833083],
        [-122.2465613, 37.8822776],
        [-122.2465292, 37.8819701],
        [-122.2464823, 37.8816],
        [-122.2464564, 37.8813696],
        [-122.2458657, 37.8760803],
        [-122.2457429, 37.8749799],
        [-122.2451848, 37.8699791],
        [-122.2451769, 37.8699366],
        [-122.245128, 37.8694772],
        [-122.2451033, 37.8693142],
        [-122.2450873, 37.8691475],
        [-122.2450221, 37.8685854],
        [-122.2450019, 37.8684585],
        [-122.2449901, 37.8683318],
        [-122.2449425, 37.8679253],
        [-122.2449283, 37.8676847],
        [-122.2448268, 37.8675822],
        [-122.244603, 37.8674008],
        [-122.2447496, 37.8672107],
        [-122.244814, 37.8670511],
        [-122.2450153, 37.8669188],
        [-122.2450636, 37.8668702],
        [-122.2447049, 37.8666633],
        [-122.2447907, 37.8665555],
        [-122.2448185, 37.8664858],
        [-122.2448283, 37.8663874],
        [-122.244818, 37.8662784],
        [-122.2455346, 37.8662374],
        [-122.2451916, 37.8657979],
        [-122.2447332, 37.8657883],
        [-122.2446211, 37.8647213],
        [-122.2448737, 37.8647056],
        [-122.2451046, 37.8643863],
        [-122.2452505, 37.8642577],
        [-122.2455253, 37.8640367],
        [-122.2450246, 37.8633557],
        [-122.2450077, 37.8632025],
        [-122.2449498, 37.8630586],
        [-122.244766, 37.8628556],
        [-122.2444098, 37.8627113],
        [-122.2443424, 37.8620745],
        [-122.244415, 37.8620742],
        [-122.2444112, 37.8620459],
        [-122.2444734, 37.8618713],
        [-122.2441376, 37.8614963],
        [-122.244273, 37.8614233],
        [-122.244235, 37.8610662],
        [-122.2442521, 37.8609628],
        [-122.2440452, 37.8607118],
        [-122.2441741, 37.8605911],
        [-122.2442395, 37.8604995],
        [-122.2442608, 37.8604491],
        [-122.2443073, 37.8602233],
        [-122.2444704, 37.8592479],
        [-122.2445203, 37.8591076],
        [-122.2445697, 37.8590004],
        [-122.2440523, 37.8589753],
        [-122.2440171, 37.8589649],
        [-122.2439845, 37.8589455],
        [-122.2439634, 37.8589249],
        [-122.243947, 37.8588967],
        [-122.2439222, 37.8586397],
        [-122.2438106, 37.8586425],
        [-122.2426025, 37.8587208],
        [-122.2425214, 37.8579348],
        [-122.2427189, 37.8579331],
        [-122.2425515, 37.8577131],
        [-122.2423989, 37.8576347],
        [-122.2423854, 37.8576321],
        [-122.2421966, 37.8575302],
        [-122.2420782, 37.8574885],
        [-122.2419728, 37.8574367],
        [-122.2418497, 37.8573448],
        [-122.2414906, 37.8573743],
        [-122.2417604, 37.857253],
        [-122.2416854, 37.8571477],
        [-122.2416432, 37.8570529],
        [-122.2411547, 37.8572623],
        [-122.2412131, 37.8573975],
        [-122.2407935, 37.8574323],
        [-122.2405321, 37.8575453],
        [-122.2404511, 37.8574342],
        [-122.2403659, 37.8572964],
        [-122.2397933, 37.8575393],
        [-122.2397198, 37.8575681],
        [-122.2396459, 37.8575023],
        [-122.2394198, 37.85751],
        [-122.2391063, 37.8577233],
        [-122.2389204, 37.8575487],
        [-122.2386662, 37.8575528],
        [-122.2382307, 37.8576782],
        [-122.2381736, 37.8576897],
        [-122.2381011, 37.8575496],
        [-122.2380282, 37.8575704],
        [-122.2376997, 37.8575905],
        [-122.2371982, 37.857637],
        [-122.2368218, 37.8576585],
        [-122.23681, 37.8575811],
        [-122.2367865, 37.8575818],
        [-122.2367561, 37.8575772],
        [-122.236729, 37.8575682],
        [-122.236715, 37.8575604],
        [-122.2367415, 37.8576506],
        [-122.2364297, 37.8576683],
        [-122.2362043, 37.8576914],
        [-122.2347612, 37.8577907],
        [-122.2346034, 37.8563505],
        [-122.2345931, 37.8562144],
        [-122.2345495, 37.8557916],
        [-122.2345302, 37.8556829],
        [-122.2345004, 37.8554109],
        [-122.2344928, 37.8554112],
        [-122.2344212, 37.8546582],
        [-122.2343896, 37.8543809],
        [-122.2342719, 37.8532878],
        [-122.2342364, 37.8529745],
        [-122.2341962, 37.8525906],
        [-122.2342067, 37.8525929],
        [-122.2342669, 37.8524675],
        [-122.2344038, 37.8523568],
        [-122.2351037, 37.8523109],
        [-122.2361216, 37.8522499],
        [-122.2362043, 37.8522462],
        [-122.2383598, 37.8521323],
        [-122.2417934, 37.8519192],
        [-122.241932, 37.8519069],
        [-122.243606, 37.851807],
        ...],
       [[-122.249374, 37.823649],
        [-122.249123, 37.823883],
        [-122.248373, 37.824586],
        [-122.248324, 37.824632],
        [-122.248118, 37.824815],
        [-122.248098, 37.824894],
        [-122.24804, 37.825131],
        [-122.248022, 37.825211],
        [-122.247953, 37.825417],
        [-122.247748, 37.826034],
        [-122.24768, 37.826241],
        [-122.247236, 37.826496],
        [-122.245945, 37.827239],
        [-122.245907, 37.827263],
        [-122.245476, 37.827538],
        [-122.245308, 37.827648],
        [-122.244805, 37.827978],
        [-122.244638, 37.828089],
        [-122.244548, 37.828135],
        [-122.244417, 37.828205],
        [-122.244292, 37.828293],
        [-122.24426, 37.828317],
        [-122.244367, 37.8285009],
        [-122.2444357, 37.8286569],
        [-122.244279, 37.828839],
        [-122.243557, 37.829776],
        [-122.243348, 37.830048],
        [-122.243317, 37.830088],
        [-122.243311, 37.830083],
        [-122.243293, 37.830068],
        [-122.243288, 37.830063],
        [-122.243025, 37.830012],
        [-122.242915, 37.830007],
        [-122.242825, 37.830009],
        [-122.242715, 37.830021],
        [-122.242402, 37.830094],
        [-122.242351, 37.830107],
        [-122.241229, 37.830406],
        [-122.241087, 37.830439],
        [-122.240935, 37.830458],
        [-122.240781, 37.830466],
        [-122.239748, 37.830478],
        [-122.239668, 37.830479],
        [-122.239064, 37.830503],
        [-122.238852, 37.830537],
        [-122.238596, 37.830572],
        [-122.238456, 37.830581],
        [-122.238227, 37.830596],
        [-122.237892, 37.830623],
        [-122.237723, 37.830656],
        [-122.237552, 37.830696],
        [-122.237327, 37.830807],
        [-122.237299, 37.830817],
        [-122.237251, 37.830836],
        [-122.237151, 37.830842],
        [-122.237048, 37.830823],
        [-122.236966, 37.830786],
        [-122.236921, 37.830761],
        [-122.236656, 37.830846],
        [-122.236059, 37.831038],
        [-122.235855, 37.831072],
        [-122.235581, 37.831118],
        [-122.23533, 37.831163],
        [-122.23458, 37.831302],
        [-122.234331, 37.831348],
        [-122.233055, 37.831582],
        [-122.232279, 37.831726],
        [-122.230566, 37.8322557],
        [-122.2302181, 37.8323633],
        [-122.229317, 37.832642],
        [-122.228079, 37.833026],
        [-122.227873, 37.832887],
        [-122.227258, 37.832473],
        [-122.227054, 37.832335],
        [-122.227016, 37.832247],
        [-122.226966, 37.832155],
        [-122.226962, 37.832148],
        [-122.226919, 37.832109],
        [-122.226873, 37.832079],
        [-122.22682, 37.832052],
        [-122.226752, 37.832029],
        [-122.22646, 37.832],
        [-122.226408, 37.831982],
        [-122.226392, 37.831977],
        [-122.22628, 37.831929],
        [-122.226229, 37.831895],
        [-122.226139, 37.831847],
        [-122.225871, 37.831703],
        [-122.225782, 37.831655],
        [-122.2256897, 37.8316049],
        [-122.225631, 37.831573],
        [-122.225178, 37.83133],
        [-122.225028, 37.831249],
        [-122.224919, 37.831046],
        [-122.224594, 37.83044],
        [-122.224486, 37.830239],
        [-122.224083, 37.829695],
        [-122.223721, 37.829206],
        [-122.222798, 37.828126],
        [-122.222741, 37.828059],
        [-122.222666, 37.827972],
        [-122.222591, 37.827884],
        [-122.222359, 37.827613],
        [-122.222307, 37.827539],
        [-122.2222597, 37.8274239],
        [-122.222151, 37.827319],
        [-122.2221, 37.827246],
        [-122.221949, 37.82696],
        [-122.2216251, 37.8263302],
        [-122.2215874, 37.8262758],
        [-122.221499, 37.826102],
        [-122.221349, 37.825817],
        [-122.220777, 37.825364],
        [-122.2194664, 37.8243268],
        [-122.2193598, 37.8242424],
        [-122.219061, 37.824006],
        [-122.218785, 37.823788],
        [-122.218489, 37.823554],
        [-122.218212, 37.823432],
        [-122.217381, 37.823067],
        [-122.2172545, 37.8230185],
        [-122.217104, 37.822946],
        [-122.217006, 37.822895],
        [-122.216911, 37.822845],
        [-122.216714, 37.822742],
        [-122.216617, 37.822692],
        [-122.216491, 37.822673],
        [-122.216473, 37.822671],
        [-122.216302, 37.822604],
        [-122.216185, 37.822514],
        [-122.216172, 37.822487],
        [-122.216144, 37.822425],
        [-122.216127, 37.82237],
        [-122.215999, 37.822242],
        [-122.2157672, 37.8220028],
        [-122.215615, 37.821861],
        [-122.215488, 37.821734],
        [-122.215278, 37.82162],
        [-122.214649, 37.821281],
        [-122.21444, 37.821168],
        [-122.214013, 37.820911],
        [-122.212732, 37.820141],
        [-122.212719, 37.8201173],
        [-122.212305, 37.819885],
        [-122.212135, 37.819791],
        [-122.211627, 37.819512],
        [-122.211458, 37.819419],
        [-122.211195, 37.81928],
        [-122.21041, 37.818866],
        [-122.210148, 37.818728],
        [-122.210724, 37.81832],
        [-122.211212, 37.817981],
        [-122.21161, 37.817707],
        [-122.211929, 37.817457],
        [-122.212129, 37.817268],
        [-122.212327, 37.81704],
        [-122.21248, 37.81683],
        [-122.212596, 37.816646],
        [-122.212731, 37.816384],
        [-122.2128, 37.816227],
        [-122.212848, 37.816077],
        [-122.21291, 37.815812],
        [-122.212988, 37.815202],
        [-122.213017, 37.81506],
        [-122.213054, 37.814913],
        [-122.213104, 37.814779],
        [-122.213108, 37.814772],
        [-122.21319, 37.814636],
        [-122.213623, 37.814122],
        [-122.213673, 37.814058],
        [-122.213728, 37.813968],
        [-122.213802, 37.813822],
        [-122.213847, 37.813712],
        [-122.213863, 37.813669],
        [-122.213901, 37.813591],
        [-122.213959, 37.813499],
        [-122.213978, 37.813473],
        [-122.214015, 37.813423],
        [-122.214044, 37.813383],
        [-122.214086, 37.813336],
        [-122.214237, 37.813201],
        [-122.214354, 37.813107],
        [-122.214476, 37.813024],
        [-122.21459, 37.812934],
        [-122.214645, 37.812892],
        [-122.214791, 37.812749],
        [-122.214857, 37.812695],
        [-122.214882, 37.812668],
        [-122.214993, 37.812554],
        [-122.215021, 37.81252],
        [-122.215111, 37.812381],
        [-122.215134, 37.812347],
        [-122.215174, 37.812276],
        [-122.215445, 37.81237],
        [-122.215945, 37.812543],
        [-122.216164, 37.812646],
        [-122.21625, 37.812677],
        [-122.216521, 37.812776],
        [-122.21676, 37.812818],
        [-122.2171962, 37.812898],
        [-122.217478, 37.812945],
        [-122.2175523, 37.8129548],
        [-122.217718, 37.812988],
        [-122.218206, 37.813096],
        [-122.2182949, 37.8131104],
        [-122.21967, 37.813421],
        [-122.2196935, 37.8134296],
        [-122.220159, 37.81353],
        [-122.220355, 37.813583],
        [-122.220944, 37.813744],
        [-122.2210062, 37.8137476],
        [-122.221141, 37.813798],
        [-122.221439, 37.813845],
        [-122.222336, 37.813986],
        [-122.222635, 37.814034],
        [-122.22291, 37.814097],
        [-122.223735, 37.814286],
        [-122.224011, 37.81435],
        [-122.224256, 37.81442],
        [-122.224992, 37.814632],
        [-122.2250842, 37.8146599],
        [-122.225238, 37.814703],
        [-122.225912, 37.814855],
        [-122.227934, 37.815313],
        [-122.228609, 37.815466],
        [-122.2289513, 37.8155298],
        [-122.229332, 37.815613],
        [-122.231504, 37.816054],
        [-122.232228, 37.816202],
        [-122.232438, 37.81627],
        [-122.233067, 37.816476],
        [-122.233278, 37.816545],
        [-122.233549, 37.816621],
        [-122.2341225, 37.8167768],
        [-122.234363, 37.81685],
        [-122.234635, 37.816927],
        [-122.234761, 37.816961],
        [-122.235139, 37.817064],
        [-122.235265, 37.817099],
        [-122.2355, 37.817143],
        [-122.236208, 37.817277],
        [-122.236444, 37.817322],
        [-122.236673, 37.817341],
        [-122.237361, 37.817399],
        [-122.237591, 37.817419],
        [-122.237934, 37.817503],
        [-122.238966, 37.817756],
        [-122.23931, 37.817841],
        [-122.23957, 37.817892],
        [-122.24035, 37.818045],
        [-122.24061, 37.818096],
        [-122.241007, 37.818187],
        [-122.242198, 37.81846],
        [-122.242596, 37.818552],
        [-122.242747, 37.818592],
        [-122.2432, 37.818712],
        [-122.2431735, 37.8187221],
        [-122.243351, 37.818753],
        [-122.243518, 37.818807],
        [-122.244022, 37.818972],
        [-122.244191, 37.819027],
        [-122.244287, 37.819074],
        [-122.244554, 37.819203],
        [-122.244578, 37.819215],
        [-122.244676, 37.819262],
        [-122.244851, 37.819377],
        [-122.245378, 37.819722],
        [-122.245554, 37.819838],
        [-122.245579, 37.819819],
        [-122.245604, 37.81981],
        [-122.245643, 37.819813],
        [-122.245688, 37.819831],
        [-122.245753, 37.819874],
        [-122.245788, 37.819901],
        [-122.245841, 37.819942],
        [-122.245992, 37.820079],
        [-122.246123, 37.820215],
        [-122.246322, 37.820499],
        [-122.246395, 37.820602],
        [-122.246468, 37.820725],
        [-122.246667, 37.820899],
        [-122.247267, 37.821424],
        [-122.247414, 37.821552],
        [-122.247468, 37.821599],
        [-122.248041, 37.82201],
        [-122.249336, 37.822939],
        [-122.248953, 37.8233],
        [-122.248859, 37.82339],
        [-122.249374, 37.823649]]],
      'type': 'Polygon'},
     'icon': 'https://nominatim.openstreetmap.org/images/mapicons/poi_place_city.p.20.png',
     'importance': 0.26134751598304,
     'lat': '37.8044557',
     'licence': 'Data © OpenStreetMap contributors, ODbL 1.0. https://osm.org/copyright',
     'lon': '-122.2713563',
     'osm_id': '2833530',
     'osm_type': 'relation',
     'place_id': '178751800',
     'type': 'city'}



We will implement an Enricher class that will query 'GPE' elements with geopy and copy some of the most usefull stuff into our event.  
It's not implemented bellow but I suggest you use a caching mechanism to reduce the bandwith.


```python
from copy import deepcopy
class AddressEnricher():
    def __init__(self):
        self.geolocator = Nominatim()
    
    def _extract_address(self, location, query):
        return {
            'query': query,
            'address': location['address'],
            'boundingbox': [(float(location['boundingbox'][2]), float(location['boundingbox'][0])), (float(location['boundingbox'][3]), float(location['boundingbox'][1]))],
            'coord': (float(location['lon']), float(location['lat']))
        }
            
    def _emit_addresses(self, event):
        for location in event.get('GPE', []):
            try:
                info = self.geolocator.geocode(location, addressdetails=True).raw        
                yield self._extract_address(info, location)
            except:
                print("Query failed for %s" % location)
                yield event
        
    def enrich(self, event):
        event = deepcopy(event)
        for address in self._emit_addresses(event):
            event.setdefault('ADDRESS', []).append(address)
        if 'GPE' in event: 
            del event['GPE']
            
        yield event
list(AddressEnricher().enrich(sample_event))
```




    [{'ADDRESS': [{'address': {'city': '서울특별시',
         'city_district': '공릉2동',
         'country': '대한민국',
         'country_code': 'kr',
         'military': '학군단',
         'town': '노원구',
         'village': '공릉동'},
        'boundingbox': [(127.0809946, 37.6285013), (127.0815729, 37.6288716)],
        'coord': (127.081364331109, 37.628668),
        'query': 'R.O.T.C.'},
       {'address': {'country': 'United States of America',
         'country_code': 'us',
         'state': 'Wisconsin'},
        'boundingbox': [(-92.8893149, 42.4919436), (-86.249548, 47.3025)],
        'coord': (-89.6884637, 44.4308975),
        'query': 'Wisconsin'},
       {'address': {'country': 'United States of America', 'country_code': 'us'},
        'boundingbox': [(-180.0, -14.7608358), (180.0, 71.6048217)],
        'coord': (-100.4458825, 39.7837304),
        'query': 'United States'}],
      'DATE': {'1/2/1970'},
      'MONEY': {'around $60,000'},
      'ORG': {'the New Years Gang',
       'the Old Red Gym',
       'the University of Wisconsin'},
      'PERSON': {'Karl Armstrong', 'Madison'},
      'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'}]



## Date enricher

Another thing that we can do is interpret the `DATE` elements and replace them with `(year, month, day)` triplets.


```python
import dateparser
date = dateparser.parse("1/3/1970")
date
```




    datetime.datetime(1970, 1, 3, 0, 0)




```python
date.year, date.month, date.day
```




    (1970, 1, 3)




```python
import traceback
import dateparser

class DateEnricher():
    def enrich(self, event):
        event = deepcopy(event)
        dates = event.get('DATE', set())
        for unparsed_date in set(dates):
            try:
                date = dateparser.parse(unparsed_date)
                event.setdefault('TIME', set()).add((date.year, date.month, date.day))
                dates.remove(unparsed_date)
            except:
                pass
        if not dates and 'DATE' in event: 
            del event['DATE']
        yield event
    
list(DateEnricher().enrich(sample_event))
```




    [{'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
      'MONEY': {'around $60,000'},
      'ORG': {'the New Years Gang',
       'the Old Red Gym',
       'the University of Wisconsin'},
      'PERSON': {'Karl Armstrong', 'Madison'},
      'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.',
      'TIME': {(1970, 1, 2)}}]



## Associate enricher

We see that events usually have more than one `PERSON` elements into them and we like the reason about individuals, not groups.  
On the other hand, we would like to keep the information that a certain person was at one point involved in the same event as the others so what keep this grouping in the `ASSOCIATES` field.  


```python
class AssociateEnricher():        
    def enrich(self, event):
        associates = {name for name in event.get('PERSON', [])}
        if not associates: return
        for associate in associates:
            new_event = deepcopy(event)
            del new_event['PERSON']
            new_event['NAME'] = associate
            new_event['ASSOCIATES'] = associates - {associate}
            yield new_event

list(AssociateEnricher().enrich(sample_event))
```




    [{'ASSOCIATES': {'Karl Armstrong'},
      'DATE': {'1/2/1970'},
      'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
      'MONEY': {'around $60,000'},
      'NAME': 'Madison',
      'ORG': {'the New Years Gang',
       'the Old Red Gym',
       'the University of Wisconsin'},
      'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'},
     {'ASSOCIATES': {'Madison'},
      'DATE': {'1/2/1970'},
      'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
      'MONEY': {'around $60,000'},
      'NAME': 'Karl Armstrong',
      'ORG': {'the New Years Gang',
       'the Old Red Gym',
       'the University of Wisconsin'},
      'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'}]



## Gender enricher

Names contain a lots of embedded infromation into them, one of which is the gender.

Chicksexer is a python package that can detect genders based on names

`pip install chicksexer`


```python
import chicksexer
chicksexer.predict_gender('Cristian Lungu')
```

    2018-06-11 10:11:25,193 - chicksexer.api - INFO - Loading model (only required for the initial prediction)...





    {'female': 0.00036776065826416016, 'male': 0.9996322393417358}




```python
sample_event_2 = {'ASSOCIATES': {'Armstrong ', 'Madison'},
  'DATE': {'1/3/1970'},
  'GPE': {'United States', 'Wisconsin'},
  'NAME': 'Karl Armstrong',
  'ORDINAL': {'first '},
  'ORG': {'Selective Service Headquarters ',
   'the New Years Gang',
   "the University of Wisconsin's "},
  'TEXT': "1/3/1970: Karl Armstrong, a member of the New Years Gang, broke into the University of Wisconsin's Primate Lab and set a fire on the first floor of the building.  Armstrong intended to set fire to the Madison, Wisconsin, United States, Selective Service Headquarters across the street but mistakenly confused the building with the Primate Lab.  The fire caused slight damages and was extinguished almost immediately."}
```

Extract the gender of a single name


```python
max([(score, gender) for gender, score in chicksexer.predict_gender('Cristian Lungu').items()])[1]
```




    'male'




```python
import chicksexer
class GenderEnricher():
    def enrich(self, event):
        event = deepcopy(event)
        gender = max([(score, gender) for gender, score in chicksexer.predict_gender(event['NAME']).items()])[1]
        event['GENDER'] = gender
        yield event
        
next(GenderEnricher().enrich(sample_event_2))
```




    {'ASSOCIATES': {'Armstrong ', 'Madison'},
     'DATE': {'1/3/1970'},
     'GENDER': 'male',
     'GPE': {'United States', 'Wisconsin'},
     'NAME': 'Karl Armstrong',
     'ORDINAL': {'first '},
     'ORG': {'Selective Service Headquarters ',
      'the New Years Gang',
      "the University of Wisconsin's "},
     'TEXT': "1/3/1970: Karl Armstrong, a member of the New Years Gang, broke into the University of Wisconsin's Primate Lab and set a fire on the first floor of the building.  Armstrong intended to set fire to the Madison, Wisconsin, United States, Selective Service Headquarters across the street but mistakenly confused the building with the Primate Lab.  The fire caused slight damages and was extinguished almost immediately."}



## Ethnicity enricher

Names have embedded in them, beside the gender information, also the ethnicity (usulally). We can train a classifier that can learn patterns of names and associate them to certain ethnicities. "Ion" for example is mostly romanian surname.

Fortunately there are pretrained models already available for this task so we can use those.

We will be using [Ethnea](https://www.ideals.illinois.edu/handle/2142/88927).

[NamePrism](http://www.name-prism.com/) is a recent famous example, but a paid one


```python
import requests
```


```python
data = requests.get("http://abel.lis.illinois.edu/cgi-bin/ethnea/search.py", params={"Fname": "Cristi Lungu", "format": "json"})
data.text
```




    "{'Genni': 'F', 'Ethnea': 'KOREAN-ROMANIAN', 'Last': 'X', 'First': 'Cristi Lungu'}\n"




```python
import json
json.loads(data.text.replace("'", '"'))['Ethnea']
```




    'KOREAN-ROMANIAN'




```python
import time
import json
import requests

class EthnicityEnhancer():
    def _get_ethnicity(self, name):
        data = requests.get("http://abel.lis.illinois.edu/cgi-bin/ethnea/search.py", params={"Fname": name, "format": "json"})
        ethnicity = json.loads(data.text.replace("'", '"'))['Ethnea']
        time.sleep(1)
        return ethnicity
    
    def enrich(self, event):
        event = deepcopy(event)
        name = event['NAME']
        ethnicity = self._get_ethnicity(name)
        event['ETHNICITY'] = ethnicity
        yield event
        
EthnicityEnhancer()._get_ethnicity('Karl Armstrong'), next(EthnicityEnhancer().enrich(sample_event_2))
```




    ('NORDIC',
     {'ASSOCIATES': {'Armstrong ', 'Madison'},
      'DATE': {'1/3/1970'},
      'ETHNICITY': 'NORDIC',
      'GPE': {'United States', 'Wisconsin'},
      'NAME': 'Karl Armstrong',
      'ORDINAL': {'first '},
      'ORG': {'Selective Service Headquarters ',
       'the New Years Gang',
       "the University of Wisconsin's "},
      'TEXT': "1/3/1970: Karl Armstrong, a member of the New Years Gang, broke into the University of Wisconsin's Primate Lab and set a fire on the first floor of the building.  Armstrong intended to set fire to the Madison, Wisconsin, United States, Selective Service Headquarters across the street but mistakenly confused the building with the Primate Lab.  The fire caused slight damages and was extinguished almost immediately."})



## Putting it all togheter


```python
def run(enricher, events):
    new_events = []
    for event in tqdm(events):
        for new_event in enricher(event):
            new_events.append(new_event)
    return new_events

def enriched_events(events):
    enrichers = [
        AddressEnricher(),
        DateEnricher(),
        AssociateEnricher(),
        GenderEnricher(),
        EthnicityEnhancer()
    ]
    
    iterator = tqdm(enrichers, total=len(enrichers))
    for enricher in iterator:
        iterator.set_description(enricher.__class__.__name__)
        events = run(enricher.enrich, events)
    return events
        
e_events = enriched_events(sample_events)
```


    HBox(children=(IntProgress(value=0, max=5), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=4), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=4), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=4), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=5), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=5), HTML(value='')))


    



```python
len(e_events), e_events[0]
```




    (5,
     {'ADDRESS': [{'address': {'city': '서울특별시',
         'city_district': '공릉2동',
         'country': '대한민국',
         'country_code': 'kr',
         'military': '학군단',
         'town': '노원구',
         'village': '공릉동'},
        'boundingbox': [(127.0809946, 37.6285013), (127.0815729, 37.6288716)],
        'coord': (127.081364331109, 37.628668),
        'query': 'R.O.T.C.'},
       {'address': {'country': 'United States of America',
         'country_code': 'us',
         'state': 'Wisconsin'},
        'boundingbox': [(-92.8893149, 42.4919436), (-86.249548, 47.3025)],
        'coord': (-89.6884637, 44.4308975),
        'query': 'Wisconsin'},
       {'address': {'country': 'United States of America', 'country_code': 'us'},
        'boundingbox': [(-180.0, -14.7608358), (180.0, 71.6048217)],
        'coord': (-100.4458825, 39.7837304),
        'query': 'United States'}],
      'ASSOCIATES': {'Karl Armstrong'},
      'ETHNICITY': 'ENGLISH',
      'GENDER': 'female',
      'MONEY': {'around $60,000'},
      'NAME': 'Madison',
      'ORG': {'the New Years Gang',
       'the Old Red Gym',
       'the University of Wisconsin'},
      'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.',
      'TIME': {(1970, 1, 2)}})



Let's do this with all the events.  
**NOTE: RUNS AWFULLY SLOW. BETTER LOAD!**


```python
all_enriched_events = enriched_events(all_events.tolist())
```


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=170350), HTML(value='')))


    



```python
len(all_enriched_events)
```




    69600




```python
np.savez_compressed("./all_enriched_events.npz", all_enriched_events=all_enriched_events)
```

Load from backup


```python
import numpy as np
with np.loads("./all_enriched_events.npz") as store:
    all_enriched_events = store["all_enriched_events"]
```

## Conclusion

We've showed some of the ways in which you can enrich a profile with more information. Here are some other things that you can try:
* derive general topic (politics, education, sports, etc..)
* extract phone numbers
    * phone numbers like names, have lots of embedded information in them that can be extracted (carrier network, region, country, etc..)
* Age retrieval
* Geo triangularisation of addresses, etc..

# Event2Vec

Now that we have rich events, we may need to be able to index them and have the ones that share a common theme grouped togheter.

This is usefull when looking for insights and "not-so-obvious" links between people's interests. Usually, such a grouping reveals a common agenda like:
    * a common terrorisit cell
    * a common corruption ring
    * an organized crime cartell

Were going to do this by deriving [event embeddings](https://machinelearningmastery.com/what-are-word-embeddings/).

They will act as coordinated in a multidimensional space that we can later use fo cluster the similar ones together.

## Model discussions

One approach for getting these embeddings is replicate the [skip-gram model](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b) published by Google in teir famous [Word2Vec paper](https://arxiv.org/pdf/1310.4546) but with some modifications.

* We will use all the elements of an event that look like unique name identifiers as "words". 
* We define "context" to be the set of "words" found in a single event.
* We will build a model whose task is, given a pair of "words" to:
    * output 1, if they appear in the same "context" 
    * output 0, if the "words" don't share a "context"
    * this is unlike the original model where they use a hyerarchical softmax approach.
* The training will happen in the embedding layer, where "words" (ids) will be converted to a multidimensional array that will model the latent variables of where "words".

* The final step will be, for each "event", to add all the "word" embeddings up. The result it the "event" embedding.

We will implement this model in [keras](https://keras.io).

## Usefull tokens 


```python
embedding_keys = {'ASSOCIATES', 'GPE', 'ORG', 'LOC', 'FAC', 'EVENT', 'PRODUCT'} # + 'NAME'
```


```python
all_enriched_events[1]
```




    {'ASSOCIATES': {'Madison'},
     'DATE': {'1/2/1970'},
     'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
     'MONEY': {'around $60,000'},
     'NAME': 'Karl Armstrong',
     'ORG': {'the New Years Gang',
      'the Old Red Gym',
      'the University of Wisconsin'},
     'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'}



Using all the events will make the computation much more demanding so for this demonstration we will use only the first 1000 of events. 


```python
event_data = all_enriched_events[:1000]
```

## Preprocessing the named tokens

We first define some helper function that extract and clean out the names of an event (in the hope that doing this will reduce some duplicate names and spelling errors).


```python
from keras.preprocessing.text import text_to_word_sequence

event = event_data[1]

def normalize(token):
    return " ".join([word for word in text_to_word_sequence(token)])

def enumerate_names(event):
    for embedding_key in embedding_keys & event.keys():
        for name in event[embedding_key]:
            yield normalize(name)
    yield normalize(event['NAME'])

set(enumerate_names(event))
```

    Using TensorFlow backend.





    {'karl armstrong',
     'madison',
     'r o t c',
     'the new years gang',
     'the old red gym',
     'the university of wisconsin',
     'united states',
     'wisconsin'}



Collect all the name tokens to get an idea of how large is our name vocabulary.


```python
name_tokens = set()
for event in tqdm(event_data):
    name_tokens |= set(enumerate_names(event))
len(name_tokens)
```


    HBox(children=(IntProgress(value=0, max=1000), HTML(value='')))


    





    1713



Show some name examples.


```python
sorted(list(name_tokens))[500:510]
```




    ['hanukkah',
     'hardcastle realty',
     'hare krishna',
     'harford county',
     'harold mciver',
     'harold nelson',
     'harrison',
     'harry j candee',
     'hato rey',
     'hawaii']



## Token vocabulary

I'm implementing a quick class to store and query by name and by id the names.  
We will convert all the names to ids for training but we also need to convert back ids to string values for debugging purposes.


```python
class Vocabulary(dict):
    def __init__(self):
        self.index = []
        
    def add(self, item):
        if item not in self:
            self[item] = len(self.index)
            self.index.append(item)
        return self[item]
    
    def value(self, idx):
        assert 0 <= index <= len(self.index)
        return self.index[idx]
    
v = Vocabulary()
v.add('a')
v.add('c')
v.add('b')
v.add('c')

v.index, v, v.add('a'), v.add('d')
```




    (['a', 'c', 'b', 'd'], {'a': 0, 'b': 2, 'c': 1, 'd': 3}, 0, 3)




```python
vocabulary = Vocabulary()
for token in tqdm(name_tokens):
    vocabulary.add(token)
```


    HBox(children=(IntProgress(value=0, max=1713), HTML(value='')))


    



```python
len(vocabulary.index)
```




    1713



## Build the training data based on event context


```python
from itertools import combinations
tokens = set(enumerate_names(event))
list(combinations(tokens, 4))
```




    [('', 'armstrong', 'skinhead', 'the west end synagogue'),
     ('', 'armstrong', 'skinhead', 'nashville'),
     ('', 'armstrong', 'skinhead', 'the ku klux klan'),
     ('', 'armstrong', 'skinhead', 'united states'),
     ('', 'armstrong', 'skinhead', 'leonard william armstrong'),
     ('', 'armstrong', 'skinhead', 'tennessee'),
     ('', 'armstrong', 'skinhead', 'white knights'),
     ('', 'armstrong', 'the west end synagogue', 'nashville'),
     ('', 'armstrong', 'the west end synagogue', 'the ku klux klan'),
     ('', 'armstrong', 'the west end synagogue', 'united states'),
     ('', 'armstrong', 'the west end synagogue', 'leonard william armstrong'),
     ('', 'armstrong', 'the west end synagogue', 'tennessee'),
     ('', 'armstrong', 'the west end synagogue', 'white knights'),
     ('', 'armstrong', 'nashville', 'the ku klux klan'),
     ('', 'armstrong', 'nashville', 'united states'),
     ('', 'armstrong', 'nashville', 'leonard william armstrong'),
     ('', 'armstrong', 'nashville', 'tennessee'),
     ('', 'armstrong', 'nashville', 'white knights'),
     ('', 'armstrong', 'the ku klux klan', 'united states'),
     ('', 'armstrong', 'the ku klux klan', 'leonard william armstrong'),
     ('', 'armstrong', 'the ku klux klan', 'tennessee'),
     ('', 'armstrong', 'the ku klux klan', 'white knights'),
     ('', 'armstrong', 'united states', 'leonard william armstrong'),
     ('', 'armstrong', 'united states', 'tennessee'),
     ('', 'armstrong', 'united states', 'white knights'),
     ('', 'armstrong', 'leonard william armstrong', 'tennessee'),
     ('', 'armstrong', 'leonard william armstrong', 'white knights'),
     ('', 'armstrong', 'tennessee', 'white knights'),
     ('', 'skinhead', 'the west end synagogue', 'nashville'),
     ('', 'skinhead', 'the west end synagogue', 'the ku klux klan'),
     ('', 'skinhead', 'the west end synagogue', 'united states'),
     ('', 'skinhead', 'the west end synagogue', 'leonard william armstrong'),
     ('', 'skinhead', 'the west end synagogue', 'tennessee'),
     ('', 'skinhead', 'the west end synagogue', 'white knights'),
     ('', 'skinhead', 'nashville', 'the ku klux klan'),
     ('', 'skinhead', 'nashville', 'united states'),
     ('', 'skinhead', 'nashville', 'leonard william armstrong'),
     ('', 'skinhead', 'nashville', 'tennessee'),
     ('', 'skinhead', 'nashville', 'white knights'),
     ('', 'skinhead', 'the ku klux klan', 'united states'),
     ('', 'skinhead', 'the ku klux klan', 'leonard william armstrong'),
     ('', 'skinhead', 'the ku klux klan', 'tennessee'),
     ('', 'skinhead', 'the ku klux klan', 'white knights'),
     ('', 'skinhead', 'united states', 'leonard william armstrong'),
     ('', 'skinhead', 'united states', 'tennessee'),
     ('', 'skinhead', 'united states', 'white knights'),
     ('', 'skinhead', 'leonard william armstrong', 'tennessee'),
     ('', 'skinhead', 'leonard william armstrong', 'white knights'),
     ('', 'skinhead', 'tennessee', 'white knights'),
     ('', 'the west end synagogue', 'nashville', 'the ku klux klan'),
     ('', 'the west end synagogue', 'nashville', 'united states'),
     ('', 'the west end synagogue', 'nashville', 'leonard william armstrong'),
     ('', 'the west end synagogue', 'nashville', 'tennessee'),
     ('', 'the west end synagogue', 'nashville', 'white knights'),
     ('', 'the west end synagogue', 'the ku klux klan', 'united states'),
     ('',
      'the west end synagogue',
      'the ku klux klan',
      'leonard william armstrong'),
     ('', 'the west end synagogue', 'the ku klux klan', 'tennessee'),
     ('', 'the west end synagogue', 'the ku klux klan', 'white knights'),
     ('', 'the west end synagogue', 'united states', 'leonard william armstrong'),
     ('', 'the west end synagogue', 'united states', 'tennessee'),
     ('', 'the west end synagogue', 'united states', 'white knights'),
     ('', 'the west end synagogue', 'leonard william armstrong', 'tennessee'),
     ('', 'the west end synagogue', 'leonard william armstrong', 'white knights'),
     ('', 'the west end synagogue', 'tennessee', 'white knights'),
     ('', 'nashville', 'the ku klux klan', 'united states'),
     ('', 'nashville', 'the ku klux klan', 'leonard william armstrong'),
     ('', 'nashville', 'the ku klux klan', 'tennessee'),
     ('', 'nashville', 'the ku klux klan', 'white knights'),
     ('', 'nashville', 'united states', 'leonard william armstrong'),
     ('', 'nashville', 'united states', 'tennessee'),
     ('', 'nashville', 'united states', 'white knights'),
     ('', 'nashville', 'leonard william armstrong', 'tennessee'),
     ('', 'nashville', 'leonard william armstrong', 'white knights'),
     ('', 'nashville', 'tennessee', 'white knights'),
     ('', 'the ku klux klan', 'united states', 'leonard william armstrong'),
     ('', 'the ku klux klan', 'united states', 'tennessee'),
     ('', 'the ku klux klan', 'united states', 'white knights'),
     ('', 'the ku klux klan', 'leonard william armstrong', 'tennessee'),
     ('', 'the ku klux klan', 'leonard william armstrong', 'white knights'),
     ('', 'the ku klux klan', 'tennessee', 'white knights'),
     ('', 'united states', 'leonard william armstrong', 'tennessee'),
     ('', 'united states', 'leonard william armstrong', 'white knights'),
     ('', 'united states', 'tennessee', 'white knights'),
     ('', 'leonard william armstrong', 'tennessee', 'white knights'),
     ('armstrong', 'skinhead', 'the west end synagogue', 'nashville'),
     ('armstrong', 'skinhead', 'the west end synagogue', 'the ku klux klan'),
     ('armstrong', 'skinhead', 'the west end synagogue', 'united states'),
     ('armstrong',
      'skinhead',
      'the west end synagogue',
      'leonard william armstrong'),
     ('armstrong', 'skinhead', 'the west end synagogue', 'tennessee'),
     ('armstrong', 'skinhead', 'the west end synagogue', 'white knights'),
     ('armstrong', 'skinhead', 'nashville', 'the ku klux klan'),
     ('armstrong', 'skinhead', 'nashville', 'united states'),
     ('armstrong', 'skinhead', 'nashville', 'leonard william armstrong'),
     ('armstrong', 'skinhead', 'nashville', 'tennessee'),
     ('armstrong', 'skinhead', 'nashville', 'white knights'),
     ('armstrong', 'skinhead', 'the ku klux klan', 'united states'),
     ('armstrong', 'skinhead', 'the ku klux klan', 'leonard william armstrong'),
     ('armstrong', 'skinhead', 'the ku klux klan', 'tennessee'),
     ('armstrong', 'skinhead', 'the ku klux klan', 'white knights'),
     ('armstrong', 'skinhead', 'united states', 'leonard william armstrong'),
     ('armstrong', 'skinhead', 'united states', 'tennessee'),
     ('armstrong', 'skinhead', 'united states', 'white knights'),
     ('armstrong', 'skinhead', 'leonard william armstrong', 'tennessee'),
     ('armstrong', 'skinhead', 'leonard william armstrong', 'white knights'),
     ('armstrong', 'skinhead', 'tennessee', 'white knights'),
     ('armstrong', 'the west end synagogue', 'nashville', 'the ku klux klan'),
     ('armstrong', 'the west end synagogue', 'nashville', 'united states'),
     ('armstrong',
      'the west end synagogue',
      'nashville',
      'leonard william armstrong'),
     ('armstrong', 'the west end synagogue', 'nashville', 'tennessee'),
     ('armstrong', 'the west end synagogue', 'nashville', 'white knights'),
     ('armstrong', 'the west end synagogue', 'the ku klux klan', 'united states'),
     ('armstrong',
      'the west end synagogue',
      'the ku klux klan',
      'leonard william armstrong'),
     ('armstrong', 'the west end synagogue', 'the ku klux klan', 'tennessee'),
     ('armstrong', 'the west end synagogue', 'the ku klux klan', 'white knights'),
     ('armstrong',
      'the west end synagogue',
      'united states',
      'leonard william armstrong'),
     ('armstrong', 'the west end synagogue', 'united states', 'tennessee'),
     ('armstrong', 'the west end synagogue', 'united states', 'white knights'),
     ('armstrong',
      'the west end synagogue',
      'leonard william armstrong',
      'tennessee'),
     ('armstrong',
      'the west end synagogue',
      'leonard william armstrong',
      'white knights'),
     ('armstrong', 'the west end synagogue', 'tennessee', 'white knights'),
     ('armstrong', 'nashville', 'the ku klux klan', 'united states'),
     ('armstrong', 'nashville', 'the ku klux klan', 'leonard william armstrong'),
     ('armstrong', 'nashville', 'the ku klux klan', 'tennessee'),
     ('armstrong', 'nashville', 'the ku klux klan', 'white knights'),
     ('armstrong', 'nashville', 'united states', 'leonard william armstrong'),
     ('armstrong', 'nashville', 'united states', 'tennessee'),
     ('armstrong', 'nashville', 'united states', 'white knights'),
     ('armstrong', 'nashville', 'leonard william armstrong', 'tennessee'),
     ('armstrong', 'nashville', 'leonard william armstrong', 'white knights'),
     ('armstrong', 'nashville', 'tennessee', 'white knights'),
     ('armstrong',
      'the ku klux klan',
      'united states',
      'leonard william armstrong'),
     ('armstrong', 'the ku klux klan', 'united states', 'tennessee'),
     ('armstrong', 'the ku klux klan', 'united states', 'white knights'),
     ('armstrong', 'the ku klux klan', 'leonard william armstrong', 'tennessee'),
     ('armstrong',
      'the ku klux klan',
      'leonard william armstrong',
      'white knights'),
     ('armstrong', 'the ku klux klan', 'tennessee', 'white knights'),
     ('armstrong', 'united states', 'leonard william armstrong', 'tennessee'),
     ('armstrong', 'united states', 'leonard william armstrong', 'white knights'),
     ('armstrong', 'united states', 'tennessee', 'white knights'),
     ('armstrong', 'leonard william armstrong', 'tennessee', 'white knights'),
     ('skinhead', 'the west end synagogue', 'nashville', 'the ku klux klan'),
     ('skinhead', 'the west end synagogue', 'nashville', 'united states'),
     ('skinhead',
      'the west end synagogue',
      'nashville',
      'leonard william armstrong'),
     ('skinhead', 'the west end synagogue', 'nashville', 'tennessee'),
     ('skinhead', 'the west end synagogue', 'nashville', 'white knights'),
     ('skinhead', 'the west end synagogue', 'the ku klux klan', 'united states'),
     ('skinhead',
      'the west end synagogue',
      'the ku klux klan',
      'leonard william armstrong'),
     ('skinhead', 'the west end synagogue', 'the ku klux klan', 'tennessee'),
     ('skinhead', 'the west end synagogue', 'the ku klux klan', 'white knights'),
     ('skinhead',
      'the west end synagogue',
      'united states',
      'leonard william armstrong'),
     ('skinhead', 'the west end synagogue', 'united states', 'tennessee'),
     ('skinhead', 'the west end synagogue', 'united states', 'white knights'),
     ('skinhead',
      'the west end synagogue',
      'leonard william armstrong',
      'tennessee'),
     ('skinhead',
      'the west end synagogue',
      'leonard william armstrong',
      'white knights'),
     ('skinhead', 'the west end synagogue', 'tennessee', 'white knights'),
     ('skinhead', 'nashville', 'the ku klux klan', 'united states'),
     ('skinhead', 'nashville', 'the ku klux klan', 'leonard william armstrong'),
     ('skinhead', 'nashville', 'the ku klux klan', 'tennessee'),
     ('skinhead', 'nashville', 'the ku klux klan', 'white knights'),
     ('skinhead', 'nashville', 'united states', 'leonard william armstrong'),
     ('skinhead', 'nashville', 'united states', 'tennessee'),
     ('skinhead', 'nashville', 'united states', 'white knights'),
     ('skinhead', 'nashville', 'leonard william armstrong', 'tennessee'),
     ('skinhead', 'nashville', 'leonard william armstrong', 'white knights'),
     ('skinhead', 'nashville', 'tennessee', 'white knights'),
     ('skinhead',
      'the ku klux klan',
      'united states',
      'leonard william armstrong'),
     ('skinhead', 'the ku klux klan', 'united states', 'tennessee'),
     ('skinhead', 'the ku klux klan', 'united states', 'white knights'),
     ('skinhead', 'the ku klux klan', 'leonard william armstrong', 'tennessee'),
     ('skinhead',
      'the ku klux klan',
      'leonard william armstrong',
      'white knights'),
     ('skinhead', 'the ku klux klan', 'tennessee', 'white knights'),
     ('skinhead', 'united states', 'leonard william armstrong', 'tennessee'),
     ('skinhead', 'united states', 'leonard william armstrong', 'white knights'),
     ('skinhead', 'united states', 'tennessee', 'white knights'),
     ('skinhead', 'leonard william armstrong', 'tennessee', 'white knights'),
     ('the west end synagogue', 'nashville', 'the ku klux klan', 'united states'),
     ('the west end synagogue',
      'nashville',
      'the ku klux klan',
      'leonard william armstrong'),
     ('the west end synagogue', 'nashville', 'the ku klux klan', 'tennessee'),
     ('the west end synagogue', 'nashville', 'the ku klux klan', 'white knights'),
     ('the west end synagogue',
      'nashville',
      'united states',
      'leonard william armstrong'),
     ('the west end synagogue', 'nashville', 'united states', 'tennessee'),
     ('the west end synagogue', 'nashville', 'united states', 'white knights'),
     ('the west end synagogue',
      'nashville',
      'leonard william armstrong',
      'tennessee'),
     ('the west end synagogue',
      'nashville',
      'leonard william armstrong',
      'white knights'),
     ('the west end synagogue', 'nashville', 'tennessee', 'white knights'),
     ('the west end synagogue',
      'the ku klux klan',
      'united states',
      'leonard william armstrong'),
     ('the west end synagogue', 'the ku klux klan', 'united states', 'tennessee'),
     ('the west end synagogue',
      'the ku klux klan',
      'united states',
      'white knights'),
     ('the west end synagogue',
      'the ku klux klan',
      'leonard william armstrong',
      'tennessee'),
     ('the west end synagogue',
      'the ku klux klan',
      'leonard william armstrong',
      'white knights'),
     ('the west end synagogue', 'the ku klux klan', 'tennessee', 'white knights'),
     ('the west end synagogue',
      'united states',
      'leonard william armstrong',
      'tennessee'),
     ('the west end synagogue',
      'united states',
      'leonard william armstrong',
      'white knights'),
     ('the west end synagogue', 'united states', 'tennessee', 'white knights'),
     ('the west end synagogue',
      'leonard william armstrong',
      'tennessee',
      'white knights'),
     ('nashville',
      'the ku klux klan',
      'united states',
      'leonard william armstrong'),
     ('nashville', 'the ku klux klan', 'united states', 'tennessee'),
     ('nashville', 'the ku klux klan', 'united states', 'white knights'),
     ('nashville', 'the ku klux klan', 'leonard william armstrong', 'tennessee'),
     ('nashville',
      'the ku klux klan',
      'leonard william armstrong',
      'white knights'),
     ('nashville', 'the ku klux klan', 'tennessee', 'white knights'),
     ('nashville', 'united states', 'leonard william armstrong', 'tennessee'),
     ('nashville', 'united states', 'leonard william armstrong', 'white knights'),
     ('nashville', 'united states', 'tennessee', 'white knights'),
     ('nashville', 'leonard william armstrong', 'tennessee', 'white knights'),
     ('the ku klux klan',
      'united states',
      'leonard william armstrong',
      'tennessee'),
     ('the ku klux klan',
      'united states',
      'leonard william armstrong',
      'white knights'),
     ('the ku klux klan', 'united states', 'tennessee', 'white knights'),
     ('the ku klux klan',
      'leonard william armstrong',
      'tennessee',
      'white knights'),
     ('united states', 'leonard william armstrong', 'tennessee', 'white knights')]



We're going to define a function for positive and one for negative sample generation.


```python
from itertools import combinations

def make_positive_samples(tokens):
    return [list(comb) for comb in combinations(tokens, 2)]

make_positive_samples([1, 2, 3, 4])
```




    [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]




```python
import random
def make_negative_sample(vocabulary_size):
    return [random.randint(0, vocabulary_size-1), random.randint(0, vocabulary_size-1)]
make_negative_sample(len(vocabulary.index))
```




    [1187, 344]



## Build the training data

We're going to replace all the names from an event with the indices from the built vocabulary before using them to build the training data.


```python
tokens = {vocabulary[token] for token in enumerate_names(event)}
tokens
```




    {0, 78, 625, 815, 827, 1090, 1446, 1517, 1533, 1598}




```python
positive = []
negative = []

for i, event in tqdm(enumerate(event_data), total=len(event_data)):
    tokens = {vocabulary[token] for token in enumerate_names(event)}
    positive += make_positive_samples(tokens)

vocabulary_size = len(vocabulary.index)
for _ in range(len(positive) * 2):
    negative.append(make_negative_sample(vocabulary_size))

labels = ([1] * len(positive)) + ([0] * len(negative))
```


    HBox(children=(IntProgress(value=0, max=1000), HTML(value='')))


    


Merge the positive, negative and shuffle them up along with their labels.


```python
inputs = np.array(positive + negative)
labels = np.array(labels)
perm = np.random.permutation(len(positive) + len(negative))

inputs = inputs[perm]
labels = labels[perm]
```

How much training data did we generate?


```python
inputs.shape, labels.shape
```




    ((125031, 2), (125031,))




```python
np.savez_compressed(
    "./data_embedding.npz",
    inputs=inputs,
    labels=labels
)
```

## The embeddings model


```python
import keras
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import merge, Lambda, Reshape, Dense, Dot
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
```


```python
inp = Input(shape=(1,), dtype='int32')
lbl = Input(shape=(1,), dtype='int32')

emb = Embedding(input_dim=len(vocabulary.index), output_dim=(10))

inp_emb = Reshape((10, 1))(emb(inp))
trg_emb = Reshape((10, 1))(emb(lbl))


dot = Dot(axes=1)([inp_emb, trg_emb])
dot = Reshape((1,))(dot)

# out = Dense(1, activation='sigmoid')(dot)
out = Activation(activation='sigmoid')(dot)

model = Model([inp, lbl], out)
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            (None, 1)            0                                            
    __________________________________________________________________________________________________
    input_4 (InputLayer)            (None, 1)            0                                            
    __________________________________________________________________________________________________
    embedding_2 (Embedding)         (None, 1, 10)        17130       input_3[0][0]                    
                                                                     input_4[0][0]                    
    __________________________________________________________________________________________________
    reshape_3 (Reshape)             (None, 10, 1)        0           embedding_2[0][0]                
    __________________________________________________________________________________________________
    reshape_4 (Reshape)             (None, 10, 1)        0           embedding_2[1][0]                
    __________________________________________________________________________________________________
    dot_1 (Dot)                     (None, 1, 1)         0           reshape_3[0][0]                  
                                                                     reshape_4[0][0]                  
    __________________________________________________________________________________________________
    reshape_5 (Reshape)             (None, 1)            0           dot_1[0][0]                      
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 1)            0           reshape_5[0][0]                  
    ==================================================================================================
    Total params: 17,130
    Trainable params: 17,130
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
from keras_tqdm import TQDMNotebookCallback
model.fit(x=[inputs[:, 0], inputs[:, 1]], y=labels, epochs=1000, batch_size=1024, callbacks=[TQDMNotebookCallback()], verbose=0)
```


    HBox(children=(IntProgress(value=0, description='Training', max=1000), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 0', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 1', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 2', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 3', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 4', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 5', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 6', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 7', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 8', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 9', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 10', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 11', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 12', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 13', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 14', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 15', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 16', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 17', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 18', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 19', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 20', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 21', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 22', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 23', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 24', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 25', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 26', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 27', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 28', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 29', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 30', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 31', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 32', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 33', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 34', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 35', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 36', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 37', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 38', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 39', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 40', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 41', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 42', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 43', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 44', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 45', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 46', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 47', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 48', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 49', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 50', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 51', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 52', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 53', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 54', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 55', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 56', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 57', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 58', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 59', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 60', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 61', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 62', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 63', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 64', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 65', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 66', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 67', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 68', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 69', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 70', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 71', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 72', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 73', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 74', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 75', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 76', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 77', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 78', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 79', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 80', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 81', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 82', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 83', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 84', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 85', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 86', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 87', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 88', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 89', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 90', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 91', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 92', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 93', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 94', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 95', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 96', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 97', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 98', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 99', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 100', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 101', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 102', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 103', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 104', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 105', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 106', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 107', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 108', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 109', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 110', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 111', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 112', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 113', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 114', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 115', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 116', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 117', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 118', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 119', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 120', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 121', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 122', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 123', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 124', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 125', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 126', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 127', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 128', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 129', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 130', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 131', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 132', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 133', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 134', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 135', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 136', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 137', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 138', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 139', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 140', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 141', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 142', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 143', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 144', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 145', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 146', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 147', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 148', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 149', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 150', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 151', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 152', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 153', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 154', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 155', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 156', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 157', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 158', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 159', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 160', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 161', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 162', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 163', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 164', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 165', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 166', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 167', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 168', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 169', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 170', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 171', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 172', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 173', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 174', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 175', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 176', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 177', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 178', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 179', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 180', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 181', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 182', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 183', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 184', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 185', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 186', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 187', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 188', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 189', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 190', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 191', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 192', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 193', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 194', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 195', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 196', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 197', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 198', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 199', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 200', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 201', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 202', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 203', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 204', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 205', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 206', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 207', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 208', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 209', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 210', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 211', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 212', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 213', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 214', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 215', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 216', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 217', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 218', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 219', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 220', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 221', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 222', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 223', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 224', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 225', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 226', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 227', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 228', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 229', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 230', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 231', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 232', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 233', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 234', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 235', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 236', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 237', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 238', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 239', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 240', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 241', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 242', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 243', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 244', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 245', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 246', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 247', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 248', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 249', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 250', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 251', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 252', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 253', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 254', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 255', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 256', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 257', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 258', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 259', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 260', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 261', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 262', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 263', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 264', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 265', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 266', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 267', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 268', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 269', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 270', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 271', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 272', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 273', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 274', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 275', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 276', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 277', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 278', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 279', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 280', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 281', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 282', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 283', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 284', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 285', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 286', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 287', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 288', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 289', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 290', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 291', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 292', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 293', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 294', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 295', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 296', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 297', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 298', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 299', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 300', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 301', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 302', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 303', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 304', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 305', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 306', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 307', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 308', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 309', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 310', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 311', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 312', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 313', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 314', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 315', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 316', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 317', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 318', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 319', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 320', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 321', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 322', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 323', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 324', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 325', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 326', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 327', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 328', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 329', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 330', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 331', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 332', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 333', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 334', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 335', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 336', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 337', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 338', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 339', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 340', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 341', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 342', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 343', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 344', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 345', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 346', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 347', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 348', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 349', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 350', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 351', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 352', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 353', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 354', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 355', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 356', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 357', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 358', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 359', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 360', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 361', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 362', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 363', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 364', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 365', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 366', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 367', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 368', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 369', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 370', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 371', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 372', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 373', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 374', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 375', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 376', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 377', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 378', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 379', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 380', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 381', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 382', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 383', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 384', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 385', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 386', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 387', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 388', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 389', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 390', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 391', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 392', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 393', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 394', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 395', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 396', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 397', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 398', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 399', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 400', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 401', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 402', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 403', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 404', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 405', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 406', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 407', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 408', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 409', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 410', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 411', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 412', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 413', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 414', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 415', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 416', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 417', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 418', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 419', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 420', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 421', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 422', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 423', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 424', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 425', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 426', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 427', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 428', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 429', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 430', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 431', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 432', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 433', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 434', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 435', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 436', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 437', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 438', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 439', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 440', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 441', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 442', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 443', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 444', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 445', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 446', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 447', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 448', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 449', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 450', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 451', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 452', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 453', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 454', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 455', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 456', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 457', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 458', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 459', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 460', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 461', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 462', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 463', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 464', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 465', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 466', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 467', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 468', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 469', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 470', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 471', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 472', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 473', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 474', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 475', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 476', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 477', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 478', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 479', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 480', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 481', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 482', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 483', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 484', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 485', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 486', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 487', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 488', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 489', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 490', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 491', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 492', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 493', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 494', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 495', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 496', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 497', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 498', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 499', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 500', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 501', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 502', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 503', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 504', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 505', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 506', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 507', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 508', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 509', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 510', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 511', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 512', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 513', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 514', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 515', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 516', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 517', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 518', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 519', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 520', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 521', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 522', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 523', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 524', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 525', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 526', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 527', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 528', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 529', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 530', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 531', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 532', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 533', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 534', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 535', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 536', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 537', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 538', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 539', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 540', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 541', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 542', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 543', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 544', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 545', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 546', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 547', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 548', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 549', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 550', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 551', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 552', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 553', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 554', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 555', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 556', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 557', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 558', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 559', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 560', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 561', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 562', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 563', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 564', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 565', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 566', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 567', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 568', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 569', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 570', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 571', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 572', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 573', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 574', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 575', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 576', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 577', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 578', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 579', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 580', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 581', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 582', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 583', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 584', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 585', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 586', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 587', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 588', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 589', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 590', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 591', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 592', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 593', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 594', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 595', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 596', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 597', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 598', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 599', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 600', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 601', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 602', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 603', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 604', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 605', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 606', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 607', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 608', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 609', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 610', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 611', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 612', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 613', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 614', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 615', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 616', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 617', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 618', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 619', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 620', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 621', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 622', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 623', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 624', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 625', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 626', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 627', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 628', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 629', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 630', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 631', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 632', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 633', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 634', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 635', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 636', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 637', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 638', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 639', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 640', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 641', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 642', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 643', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 644', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 645', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 646', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 647', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 648', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 649', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 650', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 651', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 652', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 653', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 654', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 655', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 656', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 657', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 658', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 659', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 660', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 661', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 662', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 663', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 664', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 665', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 666', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 667', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 668', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 669', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 670', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 671', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 672', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 673', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 674', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 675', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 676', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 677', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 678', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 679', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 680', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 681', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 682', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 683', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 684', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 685', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 686', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 687', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 688', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 689', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 690', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 691', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 692', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 693', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 694', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 695', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 696', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 697', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 698', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 699', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 700', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 701', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 702', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 703', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 704', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 705', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 706', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 707', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 708', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 709', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 710', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 711', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 712', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 713', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 714', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 715', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 716', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 717', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 718', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 719', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 720', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 721', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 722', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 723', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 724', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 725', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 726', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 727', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 728', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 729', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 730', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 731', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 732', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 733', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 734', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 735', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 736', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 737', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 738', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 739', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 740', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 741', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 742', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 743', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 744', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 745', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 746', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 747', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 748', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 749', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 750', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 751', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 752', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 753', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 754', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 755', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 756', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 757', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 758', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 759', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 760', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 761', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 762', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 763', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 764', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 765', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 766', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 767', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 768', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 769', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 770', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 771', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 772', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 773', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 774', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 775', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 776', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 777', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 778', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 779', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 780', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 781', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 782', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 783', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 784', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 785', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 786', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 787', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 788', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 789', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 790', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 791', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 792', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 793', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 794', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 795', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 796', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 797', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 798', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 799', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 800', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 801', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 802', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 803', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 804', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 805', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 806', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 807', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 808', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 809', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 810', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 811', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 812', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 813', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 814', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 815', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 816', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 817', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 818', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 819', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 820', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 821', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 822', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 823', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 824', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 825', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 826', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 827', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 828', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 829', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 830', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 831', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 832', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 833', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 834', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 835', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 836', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 837', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 838', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 839', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 840', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 841', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 842', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 843', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 844', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 845', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 846', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 847', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 848', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 849', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 850', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 851', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 852', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 853', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 854', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 855', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 856', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 857', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 858', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 859', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 860', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 861', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 862', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 863', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 864', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 865', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 866', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 867', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 868', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 869', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 870', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 871', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 872', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 873', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 874', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 875', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 876', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 877', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 878', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 879', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 880', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 881', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 882', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 883', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 884', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 885', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 886', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 887', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 888', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 889', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 890', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 891', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 892', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 893', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 894', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 895', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 896', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 897', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 898', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 899', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 900', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 901', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 902', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 903', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 904', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 905', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 906', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 907', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 908', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 909', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 910', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 911', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 912', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 913', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 914', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 915', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 916', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 917', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 918', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 919', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 920', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 921', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 922', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 923', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 924', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 925', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 926', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 927', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 928', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 929', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 930', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 931', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 932', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 933', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 934', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 935', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 936', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 937', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 938', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 939', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 940', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 941', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 942', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 943', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 944', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 945', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 946', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 947', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 948', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 949', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 950', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 951', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 952', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 953', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 954', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 955', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 956', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 957', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 958', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 959', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 960', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 961', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 962', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 963', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 964', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 965', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 966', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 967', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 968', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 969', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 970', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 971', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 972', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 973', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 974', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 975', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 976', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 977', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 978', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 979', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 980', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 981', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 982', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 983', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 984', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 985', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 986', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 987', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 988', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 989', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 990', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 991', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 992', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 993', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 994', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 995', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 996', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 997', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 998', max=125031), HTML(value='')))



    HBox(children=(IntProgress(value=0, description='Epoch 999', max=125031), HTML(value='')))


    





    <keras.callbacks.History at 0x7f5589efcba8>



Getting the weights form the keras model


```python
emb.get_weights()[0].shape
```




    (1713, 10)




```python
[e['NAME'] for e in event_data[:100]]
```




    ['Madison',
     'Karl Armstrong',
     'Madison',
     'Armstrong',
     'Karl Armstrong',
     'James Madison High School',
     'Judith Bissell',
     'Patrolmen William Kivlehan',
     'Ralph Bax',
     'Bax',
     'Joseph Blik',
     'Officer Blik',
     'Gables',
     'John Abercrombie',
     'Harold Nelson',
     'Strike',
     'Dore',
     'Fred Dore',
     'Leslie Jewel',
     'John Murtagh',
     'Anti-Vietnam',
     'Murtagh',
     'James C. Perrill',
     'Ithaca',
     'Karl Armstrong',
     'Keyes',
     'Frank Schaeffer',
     'Schaeffer',
     'Brown',
     'White Racists',
     "H. Rap Brown's",
     'H. Rap Brown',
     'William Payne',
     'Ralph Featherstone',
     'Black',
     'H. Rap Brown',
     'S. I. Hayakawa',
     'Samuel Ichiye Hayakawa',
     'Clyde William McKay Jr.',
     'Leonard Glatkowski',
     'Glatkowski',
     'Gregory',
     'Burton I. Gordin',
     'Richard Nixon',
     'Joe',
     "Auguste Rodin's",
     'Thinker',
     'William Calley',
     'Curtis W. Tarr',
     'Ithaca',
     'Africana Studies',
     'Ithaca',
     'Castro',
     'David G. Sprague',
     'Free',
     'Lawrence',
     'Molotov Cocktails',
     'T-9',
     'Cumulatively',
     'Cumulatively',
     'Stanley Sierakowski',
     'Patrolman Donald Sager',
     'Cheng Tzu-tsai',
     'James Ziede',
     'Chiang',
     'Chiang Ching-kuo',
     'Peter Huang Wen-hsiung',
     'John McKinney',
     'Dorchester',
     'The Burger King',
     'Edgar Hoults',
     'Edgar Hoults',
     'Joe Schock',
     'Gables',
     'Torah Scroll',
     'Dorchester',
     'Bernard Bennett',
     'Lloyd Smothers',
     'Ku Klux Klan',
     'James Rudder',
     'Larry G. Ward',
     'Larry Clark',
     'Black Panther',
     'Ronald Reed',
     'James Sackett',
     'Sackett',
     'Owen Warehouse',
     'Torah Scroll',
     'Dorchester',
     'Torah',
     'Dorchester',
     'Barr',
     'William G. Barr',
     'Levin P. West',
     ' ',
     'Marion Troutt',
     'Kenneth Kaner',
     'Bruce Sharp',
     'William Redwine',
     'Radetich']




```python
from scipy.spatial.distance import euclidean, cosine

def best_embeddings(target_embedding, all_embeddings, top=20):
    distances = [cosine(target_embedding, candidate) for i, candidate in enumerate(all_embeddings)]
    return np.argsort(distances)[:top]
    
def best_match(name):
    ne = emb.get_weights()[0]
    e = ne[vocabulary[name]]
    best_ids = best_embeddings(e, ne)
    print(name,":", [vocabulary.index[best_id] for best_id in best_ids])
    
best_match("karl armstrong"), best_match('madison'), best_match('armstrong')
```

    karl armstrong : ['karl armstrong', 'mathews', 'johnnie veal', 'david lane', 'kim holland', 'ashkelon', 'lane', 'denver', 'civic center', 'the jewish armed resistance assault team', 'black pyramid courts', "the northern illinois women's center", 'richard scutari', 'hezbollah', 'billy joel oglesby', 'power authority', 'iran', 'kuwait', 'gerald gordon', 'the army recruiting station']
    madison : ['madison', 'seabrook', 'the los angeles international airport', 'new hampshire', 'maryland', 'michael donald bray', 'james sommerville', 'langley way', 'chiang', 'twin lakes high school', 'annapolis', 'william cann', 'selective service headquarters', 'gant', 'cloverdale', 'everett c carlson', 'thomas spinks', 'charles lawrence', 'frank schaeffer', "the metropolitan medical and women's center"]
    armstrong : ['armstrong', 'bon marche', 'premises', 'decatur', 'the army recruiting station', 'kim holland', 'bhagwan shree rajneesh', 'fried chicken', "the northern illinois women's center", 'adams street', 'robert mathews', 'army', 'oregon', 'marion troutt', 'kenneth blapper', 'john joseph kaiser ii', 'the dorchester army national guard', 'rosalie zevallos', 'mississippi', 'james rudder']





    (None, None, None)




```python
# model.save_weights("./embedding_model.hdf5")
```


```python
# model.save_weights("./embedding_model_without_dense_2.hdf5")
```


```python
model.load_weights("./embedding_model.hdf5")
```

## Event embeddings

So now, all we need to do to compute an event embedding is add all the embeddings togheter.


```python
def compute_event_embedding(event):
    event_emb = np.zeros(10)
    ne = emb.get_weights()[0]
    for name in enumerate_names(event):
        event_emb += ne[vocabulary[name]]
    return event_emb

sample = 0
event_data[sample], compute_event_embedding(event_data[sample])
```




    ({'ASSOCIATES': {'Karl Armstrong'},
      'DATE': {'1/2/1970'},
      'GPE': {'R.O.T.C.', 'United States', 'Wisconsin'},
      'MONEY': {'around $60,000'},
      'NAME': 'Madison',
      'ORG': {'the New Years Gang',
       'the Old Red Gym',
       'the University of Wisconsin'},
      'TEXT': '1/2/1970: Karl Armstrong, a member of the New Years Gang, threw a firebomb at R.O.T.C. offices located within the Old Red Gym at the University of Wisconsin in Madison, Wisconsin, United States.  There were no casualties but the fire caused around $60,000 in damages to the building.'},
     array([  4.03297613,  -5.36483431,   1.85935935,  -1.71195513,
             12.51382726,  -1.84665674,   6.86580369,   3.14632877,
              8.94560277,  -4.70104777]))



Build an event_embeddings array.


```python
event_embeddings = np.zeros((len(event_data), 10))
for i, event in enumerate(event_data):
    event_embeddings[i, :] = compute_event_embedding(event)
```


```python
def best_event_match(event_id):
    matches = best_embeddings(event_embeddings[event_id], event_embeddings, 10)
    print(event_data[event_id]['NAME'], ":", [event_data[match]['NAME'] for match in matches], matches)
    
best_event_match(3)
```

    Armstrong : ['Madison', 'Armstrong', 'Karl Armstrong', 'Madison', 'Karl Armstrong', 'Kaleidoscope', 'Edward P. Gullion', 'Richard J. Picariello', 'Joseph Aceto', 'Everett C. Carlson'] [  2   3   4   0   1 150 457 458 459 460]



```python
np.savez_compressed("./event_embeddings.npz", 
    event_embeddings=event_embeddings,
    event_data=np.array(event_data),
    name_embeddings=emb.get_weights()[0]
)
```

## Conclusion

We've trained a model to derive name embeddings that we latter used to assemble "event embeddings". 

These can be used as indexes in a database, similar elements being close to one another.



The interesting thing about embeddings right now is that we can also use them to make inteligent "Google-like" queries:
    - "Air force" + "New York" + "Bombbings" + "1980" -> "John Malcom"

# Clustering the event_embeddings

The final step in our jurney is to cluster all the embeddings into common groups.

Most of the known clustering algorithms require us to input the desired number of clusters beforehad, which obviously is not the case for us. 

Foretunately there are a couple of algorithms that automatically estimate the "best" number of clusters.

We will be using AffinityPropagation, but [MeanShift](http://www.clungu.com/Mean-Shift/) is another good approach whose internals I've [described previously on my blog](http://www.clungu.com/Mean-Shift/).


```python
import sklearn
from sklearn.cluster import MeanShift, AffinityPropagation
```


```python
clusterer = AffinityPropagatoion(damping=0.5)
clusters = clusterer.fit_predict(event_embeddings)
clusters
```




    array([ 0,  0,  0,  0,  0,  3, 40,  3,  3,  3,  3,  3, 73,  0,  0, 95, 76,
           76, 91,  3,  3,  3, 40,  8,  0, 40, 91, 91,  1,  1,  1,  1,  1,  1,
           40, 40,  4,  4,  2,  2,  2,  4, 40, 97,  3, 91, 91, 97, 97,  8,  8,
            8,  4, 40,  3,  4,  4, 97, 72,  5, 40, 40,  6,  6,  6,  6,  6, 91,
            8, 72, 91, 91, 97, 83,  8,  8, 97, 97, 97, 40, 40,  7,  7,  7,  7,
            7, 40,  8,  8,  8,  8, 97, 97, 40,  0,  0,  4,  4,  4,  4,  4, 95,
           52, 52, 52, 52,  8,  4,  3,  3,  9,  9,  9,  9,  9, 91, 91,  4,  4,
            4, 53, 53, 97, 97, 97, 97,  0,  0,  0,  0,  3,  4,  4, 45, 53, 53,
           53, 40, 40,  4,  4, 40, 97, 95, 10, 10, 10, 10, 10, 97, 95, 91, 91,
           40, 40, 40, 97, 40, 40, 97,  3,  3, 40, 40, 91, 11, 11, 11, 11, 12,
           12, 12, 12, 97, 91, 73, 73, 73,  0, 97, 97, 40, 40, 91,  3, 40, 97,
           95, 95, 95, 97, 13, 13, 13, 13, 97,  8, 97, 14, 14, 14, 91, 91, 91,
            4,  4,  4,  4,  4,  4,  4,  4, 97, 97, 87, 87, 87,  4,  4,  4,  5,
           97, 40,  4, 97, 97, 15, 15, 15,  3,  4,  4,  4,  4, 97, 97, 97, 97,
           97, 97,  3,  3,  3,  3, 97, 97,  4,  4,  4,  4,  4,  4,  3,  3,  3,
           72, 72,  3,  3,  4,  4,  0,  0,  0, 95, 95,  3,  3,  3, 16, 16, 16,
           16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 45, 45, 45, 18, 19,
           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 30, 30, 30, 30, 30, 30,
           46, 46, 46, 45, 45, 45, 31, 31, 31, 31, 31, 31, 31, 92, 92, 92, 32,
           32, 32, 32, 32, 32, 32, 45, 32, 32, 32, 32, 32, 53, 53, 32, 32, 32,
           32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 31, 31, 31, 34, 34, 34,
           34, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 35, 35, 35, 35,
           35, 35, 35, 35, 33, 33, 33, 33, 33, 33, 36, 36, 36, 36, 36, 36, 36,
           36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 92, 45, 45, 92,
           92, 92, 92, 92, 45, 45, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
           38, 38, 72, 45, 39, 39, 39, 39, 40, 41, 41, 41, 41, 41, 40, 45, 45,
            3,  3,  3, 72, 82,  8,  8,  8, 97, 97,  4,  4, 40, 40,  3, 42, 42,
           42, 42, 43, 43, 43, 43, 45, 45, 43, 45, 40, 92, 72, 44, 44, 44, 44,
           44, 45,  5, 45, 45, 46, 97, 45, 46, 46, 46, 46, 47, 47, 53, 53, 53,
           53, 38, 38, 43, 43,  4, 45, 46, 46, 46, 46, 72, 72, 48, 48, 48, 48,
           43, 43, 97, 97, 97, 49, 49, 49, 49, 49,  4,  4,  4, 50, 50, 50, 50,
            3,  4,  4, 51, 51, 51, 51, 53, 53, 53, 53, 53, 45, 45, 46, 52, 52,
           52, 52, 52, 43, 53, 53, 53, 53, 95, 97, 97, 97, 97, 54, 54, 54, 54,
           54, 54, 54, 40, 43, 53, 53, 46, 46, 45, 45, 45, 55, 56, 57, 58, 59,
           60, 61, 62, 63, 64, 65, 51, 51, 51, 66, 66, 66, 66, 66, 66, 67, 67,
           67, 67, 67, 68, 68, 68, 92, 92, 92, 69, 69, 69, 69, 70, 70, 70, 70,
            4,  5,  5, 38, 97, 97,  9,  9,  9, 72, 72, 45, 91, 91, 91,  4,  4,
           72, 97, 97, 71, 71, 71, 53, 53, 53, 40, 72, 85, 85, 73, 73, 73, 73,
           95, 95,  3,  3,  3, 91, 72, 74, 74, 74, 74, 74, 74, 74, 74, 74, 72,
           97, 72,  4,  4,  3, 75, 75, 75, 75, 88, 88, 88, 88, 88,  3, 95, 72,
           53, 53, 53, 83, 83, 83, 83,  3, 70, 70, 70, 95, 95, 95, 45, 76, 76,
           76, 77, 77, 77, 78, 78, 78, 78,  0,  0,  0,  0, 82, 82, 82, 76, 76,
           76, 76, 84, 84, 84, 84, 84, 84, 76, 76, 76, 76, 76, 76, 76, 76, 87,
           87,  5, 45,  5, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79,
           79, 76, 76, 76, 80, 80, 80, 80, 80, 40, 40, 45, 81, 81, 81, 81, 81,
           81, 81, 83, 83, 83, 83, 82, 82, 82, 82, 82, 82, 45, 45, 40, 40, 40,
           84, 84, 84, 82, 82, 82, 82, 82, 82, 72, 72, 82, 82, 82, 83, 83, 83,
           83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 84, 84, 84, 72,  4,  5,
            5, 85, 85, 95, 95, 88, 88, 88, 93, 93, 86, 86, 86, 88, 88, 88, 88,
           88, 88, 88, 88, 97, 87, 87, 87, 87, 87, 92, 92, 92, 92, 91, 91, 91,
            5, 72, 72, 72, 76, 76, 88, 88, 88,  4,  4, 72, 72, 72, 93, 93, 91,
           93, 93, 93, 89, 89, 89, 97, 93, 90, 90, 90, 90, 90, 72,  5,  5, 72,
           72, 87, 87, 87, 87, 87, 40, 40, 87, 87, 87, 87, 87, 87, 40, 97, 91,
            4, 91, 97, 97, 97, 72, 91,  4, 40, 14, 14, 92, 94, 94, 94, 94, 94,
           93, 72, 95, 95, 95, 40, 93, 93, 72, 38, 72, 93, 66, 66, 66, 66,  4,
            4, 93, 97, 94, 94, 94,  4, 94, 94, 94, 94, 72, 72, 72, 46, 46, 95,
           97,  4, 45, 97, 92, 46, 94, 94,  4,  4,  4, 52, 52,  5,  5, 12, 97,
           95, 95,  5,  5, 96, 96, 96, 96, 93, 72, 95, 95, 45, 45, 45, 93, 93,
           92, 87, 87, 87,  3,  3, 97,  4, 73, 73, 73, 73, 72, 91])



We will build the event groupings from the above result and ses some results. 


```python
groups = dict()
for event_id, cluster_id in enumerate(clusters):
    groups.setdefault(cluster_id, []).append(event_id)

group = 8
groups[group], [event_data[event_id]['NAME'] for event_id in groups[group]]
```




    ([23, 49, 50, 51, 68, 74, 75, 87, 88, 89, 90, 106, 196, 447, 448, 449],
     ['Ithaca',
      'Ithaca',
      'Africana Studies',
      'Ithaca',
      'Dorchester',
      'Torah Scroll',
      'Dorchester',
      'Torah',
      'Torah Scroll',
      'Dorchester',
      'Dorchester',
      'Dorchester',
      'Ithaca',
      'Carol Ann Manning',
      'Ray Luc Levasseur',
      'Thomas Manning'])




```python
event_data[23], event_data[196]
```




    ({'ASSOCIATES': set(),
      'CARDINAL': {'2/22/1970'},
      'FAC': {'the Wari House Dormitory'},
      'GPE': {'New York', 'United States'},
      'NAME': 'Ithaca',
      'ORG': {'Cornell University', "the Black Women's Cooperative"},
      'TEXT': "2/22/1970: Unknown perpetrators threw kerosene flare pots at the Wari House Dormitory which housed the Black Women's Cooperative at Cornell University in Ithaca, New York, United States.  The incendiary tossed at the dormitory failed to ignite, but an incendiary thrown through the window of a car parked in front of the dormitory burst into flames and caused minor damages to the vehicle.  There were no casualties."},
     {'ASSOCIATES': set(),
      'GPE': {'New York', 'United States'},
      'NAME': 'Ithaca',
      'ORG': {'Cornell University', 'the Air Force R.O.T.C.'},
      'TEXT': '3/17/1971: Unknown perpetrators set fire to a classroom used by the Air Force R.O.T.C. at Cornell University in Ithaca, New York, United States.  There were no casualties and the fire caused only minor damage.'})



# Conclusions

To recap our jurney today:
    * Trained a "terrorism" media filter (to weed out all the can-can stories)
    * Parsed the text into structured format
    * Enriched data with external or implicit information
    * Derived event embeddings for querying and search
    * Clustered similar events into groups

* Media articles contain a rich source of information
* Machine Learning allows us to process this information into queryiable format
* There are multiple frameworks and strategies that we can use for this
    * usually a blend of them is the most pragmatic choice
* We can also build and train our own models to better suit our needs.

* These approaches can greatly enhance your decision making process, be it
    * compliance with law
    * insights gathering
    * monitoring for certain events
    * investing
