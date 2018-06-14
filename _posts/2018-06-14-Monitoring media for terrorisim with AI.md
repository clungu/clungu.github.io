
# Goal: SIGINT (i.e. Signals Inteligence)


Summary:
* Take a text only dataset
* Rough segmentation of the data
* Data cleaning and enriching
* Embeddings
* Cluster the similar stuff together

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

There's A TON of other approached that you could try to improve the above result. Nowadays, in NLP you don't want to do '[bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)' models as we just did above, but instead you would vectorize the text using [word vectors](https://en.wikipedia.org/wiki/Word_embedding).  

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
        ...
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



## Putting it all together


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
* Geo triangularization of addresses, etc..

# Event2Vec

Now that we have rich events, we may need to be able to index them and have the ones that share a common theme grouped together.

This is useful when looking for insights and "not-so-obvious" links between people's interests. Usually, such a grouping reveals a common agenda like:
    * a common terrorist cell
    * a common corruption ring
    * an organized crime cartel

Were going to do this by deriving [event embeddings](https://machinelearningmastery.com/what-are-word-embeddings/).

They will act as coordinated in a multidimensional space that we can later use of cluster the similar ones together.

## Model discussions

One approach for getting these embeddings is replicate the [skip-gram model](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b) published by Google in their famous [Word2Vec paper](https://arxiv.org/pdf/1310.4546) but with some modifications.

* We will use all the elements of an event that look like unique name identifiers as "words". 
* We define "context" to be the set of "words" found in a single event.
* We will build a model whose task is, given a pair of "words" to:
    * output 1, if they appear in the same "context" 
    * output 0, if the "words" don't share a "context"
    * this is unlike the original model where they use a hierarchical softmax approach.
* The training will happen in the embedding layer, where "words" (ids) will be converted to a multidimensional array that will model the latent variables of where "words".

* The final step will be, for each "event", to add all the "word" embeddings up. The result it the "event" embedding.

We will implement this model in [keras](https://keras.io).

## Useful tokens 


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
     ...
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



    HBox(children=(IntProgress(value=0, description='Epoch 999', max=125031), HTML(value='')))




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

So now, all we need to do to compute an event embedding is add all the embeddings together.


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



The interesting thing about embeddings right now is that we can also use them to make intelligent "Google-like" queries:
    - "Air force" + "New York" + "Bombings" + "1980" -> "John Malcom"

# Clustering the event_embeddings

The final step in our journey is to cluster all the embeddings into common groups.

Most of the known clustering algorithms require us to input the desired number of clusters beforehand, which obviously is not the case for us. 

Fortunately there are a couple of algorithms that automatically estimate the "best" number of clusters.

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

To recap our journey today:
* Trained a "terrorism" media filter (to weed out all the can-can stories)
* Parsed the text into structured format
* Enriched data with external or implicit information
* Derived event embeddings for querying and search
* Clustered similar events into groups

Key takeaways:
* Media articles are a rich source of information
* Machine Learning allows us to process this information into queryiable format
* There are multiple frameworks and strategies that we can use for this
    * usually a blend of them is the most pragmatic choice
* We can also build and train our own models to better suit our needs.

* These approaches can greatly enhance your decision making process, be it
    * compliance with law
    * insights gathering
    * monitoring for certain events
    * investing
