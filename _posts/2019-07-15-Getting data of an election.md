---
categories: 
    - application
tags:
    - real data
---

# Getting data of an election

![image.png](https://www.ifri.org/sites/default/files/styles/image_contenu_article/public/thumbnails/image/image_site_-_etude_collective_-_elections_europeennes.png)

On the 26th of May, we had the european parliamentary elections. In Romania, the results and progress of the vote were published online in real time on the [official electoral site](https://prezenta.bec.ro/europarlamentare26052019).

As far as I know it's the first time we had such data exposed to the public, and with such granularity. 

Since my daily work involves working closely with data, I couldn't miss the oportunity to get my hands on that dataset. Of course, the site is not documented and there aren't any publicly available API's to begin with. So I spent some hours debugging the underlying stack to see how I can query it and compile it in a usable format.

As far as I see it's built with React, using some NoSQL as the backend. I'm betting on NoSQL because, while doing the ETL I've found some schema inconsistencies that shouldn't normally happen if the data sat on top of a SQL DB.

# Understanding the API

Maybe there's better way to do this, but what I did was start the developer console of the browser, refresh the election page and look for a request that seemed to contain the data that I was looking for. Using this approach I've found the following endopoints that I could query.

An example of how you interogate the BEC site, for the `presence` data. You need to know the `county` (in this case AR, code for ARAD).

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
!curl 'https://prezenta.bec.ro/europarlamentare26052019/data/pv/json//pv_AR.json' -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-pv-part' -H 'authority: prezenta.bec.ro' --compressed -o "_data/AR.json"
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 73871    0 73871    0     0   319k      0 --:--:-- --:--:-- --:--:--  319k

</details>

There is also an endpoint publishing the `presence` count (the number of people that voted so far ). Again, we also need to query this for each `county`.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
!curl 'https://prezenta.bec.ro/europarlamentare26052019/data/presence/json/presence_AR_now.json' -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-precincts' -H 'authority: prezenta.bec.ro' --compressed -o "_data/AR-presence.json"
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 60786    0 60786    0     0   690k      0 --:--:-- --:--:-- --:--:--  690k

</details>

There is also another csv that we can use, which contains the `presence` data in a single big file.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
!curl 'https://prezenta.bec.ro/europarlamentare26052019/data/presence/csv/presence_now.csv' -H 'Referer: https://prezenta.bec.ro/europarlamentare26052019/abroad-pv-part' --compressed -o "_data/all_presence.csv"
```

</details>

# Fetching the data

Getting all the data ouf of the site. Each county has a dedicated page which contains information about its stats. By looking over the source of the site we can compile a list of all counties that we need to inspect. It's instersting that the S1..S6 (Bucharest's sectors) were modeled as counties. 

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
counties = ["AR", "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "BZ", "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN", "B", "SR", "S1", "S2", "S3", "S4", "S5", "S6"]
len(counties)
```

</details>




The `vote` information is stored on the `data/pv/json/` route, specific for each county. In order not to make multiple queries while testing, we first cache all the results localy and we can refer to them later on.

Above we've deduced the counties we can have, but I've found that there are slight asymetries for certain cases (mostly regarding the expat data and the way Bucharest is represented).

It's because of this that we need to handle the counties list in a case-by-case fashion.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  

```python
import json
from tqdm import tqdm_notebook as tqdm
```


```python
counties = ["AR", "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "BZ", "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN", "S1", "S2", "S3", "S4", "S5", "S6"]
for county in tqdm(counties):
    !curl 'https://prezenta.bec.ro/europarlamentare26052019/data/pv/json//pv_{county}.json' -H 'accept-encoding: gzip, deflate, br'  -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-pv-part' -H 'authority: prezenta.bec.ro' --compressed -o "_data/{county}.json"
```
    
</details>

The `presence` data is (as above) stored in a different (`data/presence/json/`) route specific to each county. Again, we fetch everything an cache localy.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  

```python
counties = ["AR", "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "BZ", "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN", "B", "SR"]
for county in tqdm(counties[-8:-6]):
    !curl 'https://prezenta.bec.ro/europarlamentare26052019/data/presence/json/presence_{county}_now.json' -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-precincts' -H 'authority: prezenta.bec.ro' --compressed -o "_data/{county}-presence.json"
```

</details>


Let's also get the `all in one` data about the `presence`.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  

```python
!curl 'https://prezenta.bec.ro/europarlamentare26052019/data/presence/csv/presence_now.csv' -H 'Referer: https://prezenta.bec.ro/europarlamentare26052019/abroad-pv-part' --compressed -o "_data/all_presence.csv"
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 2618k    0 2618k    0     0  5729k      0 --:--:-- --:--:-- --:--:-- 5729k

</details>

# Compiling the data

## Loading a `presence` file

When reading the `presence` file, there's some manipulation that we need to do because the original returned json contains lots of information that seemed either useless or redundant (info we already had in other places), or information that I didn't know how to interpret. 

There was also the `age_ranges` field which was contained actually a list of values, that I needed to exapend into individual columns, by using a transform function.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
def process_row(row):
    return tuple(row.age_ranges.values())

def load_presence(presence_file_name):
    _json = read_json_file(presence_file_name)
    _df = pd.DataFrame.from_records(_json['precinct'])
    _df[["men_18_24", "men_25_34", "men_35_44", "men_45_64", "men_65+", "women_18_24", "women_25_34", "women_35_44", "women_45_64", "women_65+"]] = _df.apply(process_row, axis=1, result_type='expand')
    _df.drop(columns=['age_ranges'], inplace=True)
    _df.columns = [
        'liste_permanente', 
        'lista_suplimentare', 
        'total', 
        'urna_mobila', 
        'county_code',
        'county_name',
        'id_county',
        'id_locality',
        'id_precinct',
        'id_uat',
        'initial_count',
        'latitude',
        'locality_name',
        'longitude',
        'medium',
        'precinct_name',
        'precinct_nr',
        'presence',
        'siruta',
        'uat_code',
        'uat_name',
        'men_18_24',
        'men_25_34',
        'men_35_44',
        'men_45_64',
        'men_65+',
        'women_18_24',
        'women_25_34',
        'women_35_44',
        'women_45_64',
        'women_65+',
    ]
    return _df

tulcea = load_presence("_data/TL-presence.json")
tulcea.head()
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
      <th>liste_permanente</th>
      <th>lista_suplimentare</th>
      <th>total</th>
      <th>urna_mobila</th>
      <th>county_code</th>
      <th>county_name</th>
      <th>id_county</th>
      <th>id_locality</th>
      <th>id_precinct</th>
      <th>id_uat</th>
      <th>...</th>
      <th>men_18_24</th>
      <th>men_25_34</th>
      <th>men_35_44</th>
      <th>men_45_64</th>
      <th>men_65+</th>
      <th>women_18_24</th>
      <th>women_25_34</th>
      <th>women_35_44</th>
      <th>women_45_64</th>
      <th>women_65+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>409</td>
      <td>31</td>
      <td>440</td>
      <td>0</td>
      <td>TL</td>
      <td>TULCEA</td>
      <td>38</td>
      <td>8884</td>
      <td>15848</td>
      <td>2882</td>
      <td>...</td>
      <td>13</td>
      <td>23</td>
      <td>36</td>
      <td>84</td>
      <td>29</td>
      <td>17</td>
      <td>22</td>
      <td>55</td>
      <td>110</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>471</td>
      <td>73</td>
      <td>544</td>
      <td>0</td>
      <td>TL</td>
      <td>TULCEA</td>
      <td>38</td>
      <td>8884</td>
      <td>15849</td>
      <td>2882</td>
      <td>...</td>
      <td>11</td>
      <td>28</td>
      <td>55</td>
      <td>90</td>
      <td>71</td>
      <td>10</td>
      <td>40</td>
      <td>62</td>
      <td>92</td>
      <td>85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>20</td>
      <td>34</td>
      <td>0</td>
      <td>TL</td>
      <td>TULCEA</td>
      <td>38</td>
      <td>8909</td>
      <td>15943</td>
      <td>2892</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>134</td>
      <td>35</td>
      <td>169</td>
      <td>0</td>
      <td>TL</td>
      <td>TULCEA</td>
      <td>38</td>
      <td>8914</td>
      <td>15948</td>
      <td>2894</td>
      <td>...</td>
      <td>5</td>
      <td>6</td>
      <td>17</td>
      <td>37</td>
      <td>21</td>
      <td>4</td>
      <td>9</td>
      <td>7</td>
      <td>37</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>638</td>
      <td>70</td>
      <td>708</td>
      <td>0</td>
      <td>TL</td>
      <td>TULCEA</td>
      <td>38</td>
      <td>8912</td>
      <td>15946</td>
      <td>2894</td>
      <td>...</td>
      <td>19</td>
      <td>42</td>
      <td>46</td>
      <td>185</td>
      <td>78</td>
      <td>17</td>
      <td>42</td>
      <td>53</td>
      <td>142</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



## Getting all the individual `result` files for all `precint_nr`

The voting results (what voted for what, in which place) are stored in the `data/pv` route. The problem is that if we only use the `county` file, downloaded in step 3, there isn't any keys that will link the results to the presence rows. To be more exact, the results are listed per voting facilty, whereas the presence is aggregated at a locality level. This means that we can't really jon the two.

Fortuantely, I've found that if you know the id of a specific voting facilty, you could ask for the resuls of that specific facility through a version of the original `results` API.

So the final strategy that worked was something along the following lines:

* For all `countyes`, we will load the `presence` file associated to it
* inspect all the `precint_nr` that is contains
* individually query the api for the results of that `precint_nr`. 

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
counties = ["AR", "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "BZ", "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN", "SR"]
for county in tqdm(counties):
    df_county = load_presence(f"_data/{county}-presence.json")
    for precinct_nr in tqdm(df_county['precinct_nr'].values, leave=False):
        file_name = f"_data/{county}_results_{precinct_nr}.json"
        if not os.path.exists(file_name): 
            !curl 'https://prezenta.bec.ro/europarlamentare26052019/data/pv/csv/pv_{county}_{precinct_nr}_EUP_PART.csv' -H 'accept-encoding: gzip, deflate, br' -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36' -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-pv-part' -H 'authority: prezenta.bec.ro' -H 'cookie: _ga=GA1.2.772980748.1558943895; _gid=GA1.2.1466959792.1561374632' --compressed --silent -o "_data/{county}_results_{precinct_nr}.json"
        else:
            with open(file_name) as f:
                file_contents = f.read()
            if "Cod birou electoral" not in file_contents:
                print(f"File: {file_name} has bad content {file_contents[:50]}. Will retry")
                os.remove(file_name)
                !curl 'https://prezenta.bec.ro/europarlamentare26052019/data/pv/csv/pv_{county}_{precinct_nr}_EUP_PART.csv' -H 'accept-encoding: gzip, deflate, br' -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36' -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-pv-part' -H 'authority: prezenta.bec.ro' -H 'cookie: _ga=GA1.2.772980748.1558943895; _gid=GA1.2.1466959792.1561374632' --compressed --silent -o "_data/{county}_results_{precinct_nr}.json"
```

</details>


Bucharest is a special case. It's treated as a county but the results are stored by sectors so we need to do things a bit different.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
county = "B"
df_county = load_presence(f"_data/{county}-presence.json")
for id_sector in tqdm(df_county.id_locality.unique()):
    sector = f"S{int(id_sector) - 9910 + 1}"
    print(f"Processing: {sector}")
    county = sector
    for precinct_nr in tqdm(df_county[df_county.id_locality == id_sector]['precinct_nr'].values, leave=False):
        file_name = f"_data/{county}_results_{precinct_nr}.json"
        if not os.path.exists(file_name): 
            !curl 'https://prezenta.bec.ro/europarlamentare26052019/data/pv/csv/pv_{county}_{precinct_nr}_EUP_PART.csv' -H 'accept-encoding: gzip, deflate, br' -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36' -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-pv-part' -H 'authority: prezenta.bec.ro' -H 'cookie: _ga=GA1.2.772980748.1558943895; _gid=GA1.2.1466959792.1561374632' --compressed --silent -o "_data/{county}_results_{precinct_nr}.json"
        else:
            with open(file_name) as f:
                file_contents = f.read()
            if "Cod birou electoral" not in file_contents:
                print(f"File: {file_name} has bad content {file_contents[:50]}. Will retry")
                os.remove(file_name)
                !curl 'https://prezenta.bec.ro/europarlamentare26052019/data/pv/csv/pv_{county}_{precinct_nr}_EUP_PART.csv' -H 'accept-encoding: gzip, deflate, br' -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36' -H 'accept: */*' -H 'referer: https://prezenta.bec.ro/europarlamentare26052019/romania-pv-part' -H 'authority: prezenta.bec.ro' -H 'cookie: _ga=GA1.2.772980748.1558943895; _gid=GA1.2.1466959792.1561374632' --compressed --silent -o "_data/{county}_results_{precinct_nr}.json"
```

</details>


In `SR` we have data about the foreign offices.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
load_presence("_data/SR-presence.json").head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>liste_permanente</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lista_suplimentare</th>
      <td>18</td>
      <td>57</td>
      <td>865</td>
      <td>79</td>
      <td>1330</td>
    </tr>
    <tr>
      <th>total</th>
      <td>18</td>
      <td>57</td>
      <td>865</td>
      <td>79</td>
      <td>1330</td>
    </tr>
    <tr>
      <th>urna_mobila</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>county_code</th>
      <td>SR</td>
      <td>SR</td>
      <td>SR</td>
      <td>SR</td>
      <td>SR</td>
    </tr>
    <tr>
      <th>county_name</th>
      <td>STRAINATATE</td>
      <td>STRAINATATE</td>
      <td>STRAINATATE</td>
      <td>STRAINATATE</td>
      <td>STRAINATATE</td>
    </tr>
    <tr>
      <th>id_county</th>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
    </tr>
    <tr>
      <th>id_locality</th>
      <td>10244</td>
      <td>10178</td>
      <td>10334</td>
      <td>10206</td>
      <td>9996</td>
    </tr>
    <tr>
      <th>id_precinct</th>
      <td>18619</td>
      <td>18627</td>
      <td>19096</td>
      <td>18716</td>
      <td>18723</td>
    </tr>
    <tr>
      <th>id_uat</th>
      <td>3230</td>
      <td>3218</td>
      <td>3245</td>
      <td>3219</td>
      <td>3186</td>
    </tr>
    <tr>
      <th>initial_count</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>locality_name</th>
      <td>KABUL</td>
      <td>BUENOS AIRES</td>
      <td>RENNES</td>
      <td>TBILISI</td>
      <td>STUTTGART</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>medium</th>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
    </tr>
    <tr>
      <th>precinct_name</th>
      <td>Eticheta Credentiale/Tableta - 1</td>
      <td>Eticheta Credentiale/Tableta - 10</td>
      <td>Eticheta Credentiale/Tableta - 100</td>
      <td>Eticheta Credentiale/Tableta - 101</td>
      <td>Eticheta Credentiale/Tableta - 102</td>
    </tr>
    <tr>
      <th>precinct_nr</th>
      <td>1</td>
      <td>10</td>
      <td>100</td>
      <td>101</td>
      <td>102</td>
    </tr>
    <tr>
      <th>presence</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>siruta</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>uat_code</th>
      <td>AF</td>
      <td>AR</td>
      <td>FR</td>
      <td>GE</td>
      <td>DE</td>
    </tr>
    <tr>
      <th>uat_name</th>
      <td>AFGANISTAN</td>
      <td>ARGENTINA</td>
      <td>FRANȚA</td>
      <td>GEORGIA</td>
      <td>GERMANIA</td>
    </tr>
    <tr>
      <th>men_18_24</th>
      <td>0</td>
      <td>1</td>
      <td>50</td>
      <td>1</td>
      <td>71</td>
    </tr>
    <tr>
      <th>men_25_34</th>
      <td>4</td>
      <td>3</td>
      <td>155</td>
      <td>9</td>
      <td>258</td>
    </tr>
    <tr>
      <th>men_35_44</th>
      <td>11</td>
      <td>4</td>
      <td>172</td>
      <td>18</td>
      <td>270</td>
    </tr>
    <tr>
      <th>men_45_64</th>
      <td>3</td>
      <td>7</td>
      <td>141</td>
      <td>14</td>
      <td>249</td>
    </tr>
    <tr>
      <th>men_65+</th>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>women_18_24</th>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>1</td>
      <td>50</td>
    </tr>
    <tr>
      <th>women_25_34</th>
      <td>0</td>
      <td>10</td>
      <td>119</td>
      <td>13</td>
      <td>182</td>
    </tr>
    <tr>
      <th>women_35_44</th>
      <td>0</td>
      <td>13</td>
      <td>93</td>
      <td>11</td>
      <td>128</td>
    </tr>
    <tr>
      <th>women_45_64</th>
      <td>0</td>
      <td>11</td>
      <td>76</td>
      <td>9</td>
      <td>112</td>
    </tr>
    <tr>
      <th>women_65+</th>
      <td>0</td>
      <td>7</td>
      <td>14</td>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## Reading all the presence data into a single `DataFrame`

We now have all the `presence` data cached, and we'll read it into a single dataframe.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
counties = ["AR", "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "BZ", "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN", "SR", "B"]
 
df_precints = pd.concat((load_presence(f) for f in tqdm(glob("_data/*-presence.json"))), ignore_index=True)
df_precints.shape
```

</details>



    (19171, 31)


{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  

```python
df_precints.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>liste_permanente</th>
      <td>696</td>
      <td>140</td>
      <td>501</td>
      <td>571</td>
      <td>680</td>
    </tr>
    <tr>
      <th>lista_suplimentare</th>
      <td>63</td>
      <td>10</td>
      <td>25</td>
      <td>41</td>
      <td>55</td>
    </tr>
    <tr>
      <th>total</th>
      <td>759</td>
      <td>150</td>
      <td>526</td>
      <td>612</td>
      <td>736</td>
    </tr>
    <tr>
      <th>urna_mobila</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>county_code</th>
      <td>VS</td>
      <td>VS</td>
      <td>VS</td>
      <td>VS</td>
      <td>VS</td>
    </tr>
    <tr>
      <th>county_name</th>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>VASLUI</td>
    </tr>
    <tr>
      <th>id_county</th>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>id_locality</th>
      <td>9015</td>
      <td>9015</td>
      <td>9006</td>
      <td>9006</td>
      <td>9006</td>
    </tr>
    <tr>
      <th>id_precinct</th>
      <td>16128</td>
      <td>16187</td>
      <td>16086</td>
      <td>16087</td>
      <td>16088</td>
    </tr>
    <tr>
      <th>id_uat</th>
      <td>2936</td>
      <td>2936</td>
      <td>2933</td>
      <td>2933</td>
      <td>2933</td>
    </tr>
    <tr>
      <th>initial_count</th>
      <td>1470</td>
      <td>1840</td>
      <td>1354</td>
      <td>1375</td>
      <td>1570</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>46.6389853639550</td>
      <td>46.6421141774663</td>
      <td>46.2240238056566</td>
      <td>46.2278431009305</td>
      <td>46.2278431009305</td>
    </tr>
    <tr>
      <th>locality_name</th>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>BÂRLAD</td>
      <td>BÂRLAD</td>
      <td>BÂRLAD</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>27.7326775437114</td>
      <td>27.7289502189002</td>
      <td>27.6775710052581</td>
      <td>27.6686353095150</td>
      <td>27.6686353095150</td>
    </tr>
    <tr>
      <th>medium</th>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
    </tr>
    <tr>
      <th>precinct_name</th>
      <td>CASA DE CULTURĂ A SINDICATELOR ”CONSTANTIN TĂN...</td>
      <td>ȘCOALA GIMNAZIALĂ ”CONSTANTIN PARFENE”</td>
      <td>ŞCOALA GIMNAZIALĂ ”VICTOR IOAN POPA”</td>
      <td>CASA DE CULTURĂ A SINDICATELOR ”GEORGE TUTOVEANU”</td>
      <td>CASA DE CULTURĂ A SINDICATELOR ”GEORGE TUTOVEANU”</td>
    </tr>
    <tr>
      <th>precinct_nr</th>
      <td>1</td>
      <td>10</td>
      <td>100</td>
      <td>101</td>
      <td>102</td>
    </tr>
    <tr>
      <th>presence</th>
      <td>51.6327</td>
      <td>8.1522</td>
      <td>38.8479</td>
      <td>44.5091</td>
      <td>46.879</td>
    </tr>
    <tr>
      <th>siruta</th>
      <td>161954</td>
      <td>161954</td>
      <td>161801</td>
      <td>161801</td>
      <td>161801</td>
    </tr>
    <tr>
      <th>uat_code</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>uat_name</th>
      <td>MUNICIPIUL VASLUI</td>
      <td>MUNICIPIUL VASLUI</td>
      <td>MUNICIPIUL BÂRLAD</td>
      <td>MUNICIPIUL BÂRLAD</td>
      <td>MUNICIPIUL BÂRLAD</td>
    </tr>
    <tr>
      <th>men_18_24</th>
      <td>13</td>
      <td>4</td>
      <td>18</td>
      <td>15</td>
      <td>11</td>
    </tr>
    <tr>
      <th>men_25_34</th>
      <td>31</td>
      <td>5</td>
      <td>25</td>
      <td>29</td>
      <td>40</td>
    </tr>
    <tr>
      <th>men_35_44</th>
      <td>66</td>
      <td>17</td>
      <td>60</td>
      <td>52</td>
      <td>60</td>
    </tr>
    <tr>
      <th>men_45_64</th>
      <td>110</td>
      <td>28</td>
      <td>104</td>
      <td>113</td>
      <td>158</td>
    </tr>
    <tr>
      <th>men_65+</th>
      <td>115</td>
      <td>20</td>
      <td>43</td>
      <td>72</td>
      <td>81</td>
    </tr>
    <tr>
      <th>women_18_24</th>
      <td>22</td>
      <td>6</td>
      <td>17</td>
      <td>22</td>
      <td>14</td>
    </tr>
    <tr>
      <th>women_25_34</th>
      <td>34</td>
      <td>8</td>
      <td>33</td>
      <td>32</td>
      <td>46</td>
    </tr>
    <tr>
      <th>women_35_44</th>
      <td>78</td>
      <td>15</td>
      <td>64</td>
      <td>55</td>
      <td>52</td>
    </tr>
    <tr>
      <th>women_45_64</th>
      <td>171</td>
      <td>28</td>
      <td>117</td>
      <td>127</td>
      <td>178</td>
    </tr>
    <tr>
      <th>women_65+</th>
      <td>119</td>
      <td>19</td>
      <td>45</td>
      <td>95</td>
      <td>96</td>
    </tr>
  </tbody>
</table>
</div>



The `all_presence.csv` file contains information about age groups, more granular than the bucketed info found in the county files. We will merge it with the current dataframe.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
_all_df = pd.read_csv("_data/all_presence.csv")
```

```python
_all_df.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Judet</th>
      <td>AB</td>
      <td>AB</td>
      <td>AB</td>
      <td>AB</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>UAT</th>
      <td>MUNICIPIUL ALBA IULIA</td>
      <td>MUNICIPIUL ALBA IULIA</td>
      <td>MUNICIPIUL SEBEŞ</td>
      <td>MUNICIPIUL SEBEŞ</td>
      <td>MUNICIPIUL SEBEŞ</td>
    </tr>
    <tr>
      <th>Localitate</th>
      <td>ALBA IULIA</td>
      <td>ALBA IULIA</td>
      <td>SEBEŞ</td>
      <td>SEBEŞ</td>
      <td>SEBEŞ</td>
    </tr>
    <tr>
      <th>Siruta</th>
      <td>1026</td>
      <td>1026</td>
      <td>1883</td>
      <td>1883</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>Nr sectie de votare</th>
      <td>1</td>
      <td>10</td>
      <td>100</td>
      <td>101</td>
      <td>102</td>
    </tr>
    <tr>
      <th>Nume sectie de votare</th>
      <td>CENTRUL DE ZI PENTRU PERSOANE VÂRSTNICE</td>
      <td>COLEGIUL NAŢIONAL „HOREA CLOŞCA ŞI CRIŞAN”</td>
      <td>ŞCOALA GIMNAZIALĂ NR. 2 SEBEŞ</td>
      <td>COLEGIUL NAŢIONAL ”LUCIAN BLAGA” SEBEŞ</td>
      <td>COLEGIUL NAŢIONAL ”LUCIAN BLAGA” SEBEŞ</td>
    </tr>
    <tr>
      <th>Mediu</th>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
    </tr>
    <tr>
      <th>Votanti lista</th>
      <td>1612</td>
      <td>1443</td>
      <td>1415</td>
      <td>1303</td>
      <td>1362</td>
    </tr>
    <tr>
      <th>LP</th>
      <td>901</td>
      <td>648</td>
      <td>769</td>
      <td>697</td>
      <td>765</td>
    </tr>
    <tr>
      <th>LS</th>
      <td>45</td>
      <td>143</td>
      <td>66</td>
      <td>73</td>
      <td>27</td>
    </tr>
    <tr>
      <th>UM</th>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LT</th>
      <td>946</td>
      <td>791</td>
      <td>849</td>
      <td>812</td>
      <td>792</td>
    </tr>
    <tr>
      <th>Barbati 18-24</th>
      <td>26</td>
      <td>38</td>
      <td>34</td>
      <td>24</td>
      <td>31</td>
    </tr>
    <tr>
      <th>Barbati 25-34</th>
      <td>58</td>
      <td>68</td>
      <td>56</td>
      <td>56</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Barbati 35-44</th>
      <td>88</td>
      <td>69</td>
      <td>84</td>
      <td>70</td>
      <td>82</td>
    </tr>
    <tr>
      <th>Barbati 45-64</th>
      <td>165</td>
      <td>128</td>
      <td>157</td>
      <td>136</td>
      <td>136</td>
    </tr>
    <tr>
      <th>Barbati 65+</th>
      <td>102</td>
      <td>71</td>
      <td>88</td>
      <td>78</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Femei 18-24</th>
      <td>32</td>
      <td>40</td>
      <td>30</td>
      <td>30</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Femei 25-34</th>
      <td>72</td>
      <td>56</td>
      <td>61</td>
      <td>61</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Femei 35-44</th>
      <td>107</td>
      <td>79</td>
      <td>76</td>
      <td>92</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Femei 45-64</th>
      <td>178</td>
      <td>161</td>
      <td>163</td>
      <td>167</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Femei 65+</th>
      <td>118</td>
      <td>81</td>
      <td>100</td>
      <td>98</td>
      <td>85</td>
    </tr>
    <tr>
      <th>Barbati 18</th>
      <td>7</td>
      <td>9</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Barbati 19</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Barbati 20</th>
      <td>5</td>
      <td>7</td>
      <td>6</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Barbati 21</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Barbati 22</th>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Barbati 23</th>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Barbati 24</th>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Barbati 25</th>
      <td>5</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Femei 91</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 92</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 93</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 94</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 95</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 96</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 97</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 98</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 99</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 100</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 101</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 102</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 103</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 104</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 105</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 106</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 107</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 108</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 109</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 110</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 111</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 112</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 113</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 114</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 115</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 116</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 117</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 118</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 119</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 120</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>228 rows × 5 columns</p>
</div>



It's interesting that this file contains presence data on a per year-of-birth grouping (which is more granular than the 10 years buckets we had prior).

## Reading all the results data into a single dataframe.

The individual results files we've got from two steps above, we will load them into a single big pandas `DataFrame`

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
df_results = pd.concat((pd.read_csv(f) for f in tqdm(glob("_data/*_results_*"))), ignore_index=True)
df_results.shape
```

    (19171, 36)


```python
df_results.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cod birou electoral</th>
      <td>22</td>
      <td>5</td>
      <td>35</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Județ</th>
      <td>HUNEDOARA</td>
      <td>BIHOR</td>
      <td>SUCEAVA</td>
      <td>SUCEAVA</td>
      <td>BISTRIŢA-NĂSĂUD</td>
    </tr>
    <tr>
      <th>Uat</th>
      <td>ORAŞ CĂLAN</td>
      <td>CEICA</td>
      <td>VICOVU DE JOS</td>
      <td>MUNICIPIUL SUCEAVA</td>
      <td>BUDEŞTI</td>
    </tr>
    <tr>
      <th>Localitate</th>
      <td>CĂLAN</td>
      <td>BUCIUM</td>
      <td>VICOVU DE JOS</td>
      <td>SUCEAVA</td>
      <td>BUDEŞTI-FÂNAŢE</td>
    </tr>
    <tr>
      <th>Secție</th>
      <td>ŞCOALA GIMNAZIALĂ</td>
      <td>ȘCOALA BUCIUM</td>
      <td>SCOALA CU CLASELE I-VIII IOAN VICOVEANU</td>
      <td>GRĂDINIŢA CU PROGRAM NORMAL NR.7</td>
      <td>ŞCOALA PRIMARĂ BUDEŞTI-FÎNAŢE</td>
    </tr>
    <tr>
      <th>Nr</th>
      <td>190</td>
      <td>320</td>
      <td>532</td>
      <td>61</td>
      <td>97</td>
    </tr>
    <tr>
      <th>Tip</th>
      <td>Europarlamentare</td>
      <td>Europarlamentare</td>
      <td>Europarlamentare</td>
      <td>Europarlamentare</td>
      <td>Europarlamentare</td>
    </tr>
    <tr>
      <th>a</th>
      <td>1471</td>
      <td>172</td>
      <td>1344</td>
      <td>1393</td>
      <td>256</td>
    </tr>
    <tr>
      <th>a1</th>
      <td>1471</td>
      <td>172</td>
      <td>1344</td>
      <td>1393</td>
      <td>256</td>
    </tr>
    <tr>
      <th>a2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>534</td>
      <td>116</td>
      <td>520</td>
      <td>625</td>
      <td>162</td>
    </tr>
    <tr>
      <th>b1</th>
      <td>505</td>
      <td>88</td>
      <td>479</td>
      <td>560</td>
      <td>141</td>
    </tr>
    <tr>
      <th>b2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b3</th>
      <td>29</td>
      <td>28</td>
      <td>41</td>
      <td>65</td>
      <td>21</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1600</td>
      <td>188</td>
      <td>1500</td>
      <td>1500</td>
      <td>300</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1066</td>
      <td>72</td>
      <td>980</td>
      <td>875</td>
      <td>138</td>
    </tr>
    <tr>
      <th>e</th>
      <td>514</td>
      <td>113</td>
      <td>504</td>
      <td>605</td>
      <td>153</td>
    </tr>
    <tr>
      <th>f</th>
      <td>20</td>
      <td>3</td>
      <td>16</td>
      <td>20</td>
      <td>9</td>
    </tr>
    <tr>
      <th>h</th>
      <td>0</td>
      <td>0</td>
      <td>NU ESTE CAZUL</td>
      <td>NU ESTE CAZUL</td>
      <td>0</td>
    </tr>
    <tr>
      <th>i</th>
      <td>0</td>
      <td>0</td>
      <td>FOARTE BUNA</td>
      <td>FOARTE BUNA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>g1</th>
      <td>111</td>
      <td>46</td>
      <td>174</td>
      <td>128</td>
      <td>18</td>
    </tr>
    <tr>
      <th>g2</th>
      <td>86</td>
      <td>14</td>
      <td>51</td>
      <td>126</td>
      <td>29</td>
    </tr>
    <tr>
      <th>g3</th>
      <td>29</td>
      <td>5</td>
      <td>34</td>
      <td>43</td>
      <td>7</td>
    </tr>
    <tr>
      <th>g4</th>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>g5</th>
      <td>176</td>
      <td>28</td>
      <td>162</td>
      <td>153</td>
      <td>62</td>
    </tr>
    <tr>
      <th>g6</th>
      <td>18</td>
      <td>4</td>
      <td>19</td>
      <td>27</td>
      <td>10</td>
    </tr>
    <tr>
      <th>g7</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>g8</th>
      <td>25</td>
      <td>5</td>
      <td>29</td>
      <td>55</td>
      <td>11</td>
    </tr>
    <tr>
      <th>g9</th>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>g10</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>g11</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>g12</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>g13</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>g14</th>
      <td>18</td>
      <td>0</td>
      <td>4</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>g15</th>
      <td>9</td>
      <td>2</td>
      <td>9</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>g16</th>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>22</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Joining the results with the presence data

Some code cleanup are neede. In order to join the two dataframes we need to make slight conversions to make all the keys from both side match.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
df_results.loc[df_results.Județ == "STRĂINĂTATE", "Județ"] = "STRAINATATE"
df_results.loc[df_results.Uat == "OMAN", "Uat"] = "SULTANATUL OMAN"
df_results.loc[df_results.Județ == "SECTOR 1", "Județ"] = "MUNICIPIUL BUCUREŞTI"
df_results.loc[df_results.Județ == "SECTOR 2", "Județ"] = "MUNICIPIUL BUCUREŞTI"
df_results.loc[df_results.Județ == "SECTOR 3", "Județ"] = "MUNICIPIUL BUCUREŞTI"
df_results.loc[df_results.Județ == "SECTOR 4", "Județ"] = "MUNICIPIUL BUCUREŞTI"
df_results.loc[df_results.Județ == "SECTOR 5", "Județ"] = "MUNICIPIUL BUCUREŞTI"
df_results.loc[df_results.Județ == "SECTOR 6", "Județ"] = "MUNICIPIUL BUCUREŞTI"
```
</details>

Now, if we merge the two we will get a single big dataframe with the same number of rows but double the columns.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
df_precint_with_results = pd.merge(left=df_precints, right=df_results, left_on=["county_name", "uat_name", "precinct_nr"], right_on=["Județ", "Uat", "Nr"])
df_precint_with_results.shape
```

</details>



    (19171, 67)



Let's print one example of how one entry this looks like in practice.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
dict(df_precint_with_results.iloc[0])
```

</details>



    {'liste_permanente': 696,
     'lista_suplimentare': 63,
     'total': 759,
     'urna_mobila': 0,
     'county_code': 'VS',
     'county_name': 'VASLUI',
     'id_county': '39',
     'id_locality': '9015',
     'id_precinct': '16128',
     'id_uat': '2936',
     'initial_count': 1470,
     'latitude': '46.6389853639550',
     'locality_name': 'VASLUI',
     'longitude': '27.7326775437114',
     'medium': 'U',
     'precinct_name': 'CASA DE CULTURĂ A SINDICATELOR ”CONSTANTIN TĂNASE”',
     'precinct_nr': 1,
     'presence': 51.6327,
     'siruta': '161954',
     'uat_code': None,
     'uat_name': 'MUNICIPIUL VASLUI',
     'men_18_24': 13,
     'men_25_34': 31,
     'men_35_44': 66,
     'men_45_64': 110,
     'men_65+': 115,
     'women_18_24': 22,
     'women_25_34': 34,
     'women_35_44': 78,
     'women_45_64': 171,
     'women_65+': 119,
     'Cod birou electoral': 39,
     'Județ': 'VASLUI',
     'Uat': 'MUNICIPIUL VASLUI',
     'Localitate': 'VASLUI',
     'Secție': 'CASA DE CULTURĂ A SINDICATELOR ”CONSTANTIN TĂNASE”',
     'Nr': 1,
     'Tip': 'Europarlamentare',
     'a': 1470,
     'a1': 1470,
     'a2': 0,
     'b': 759,
     'b1': 695,
     'b2': 0,
     'b3': 64,
     'c': 1500,
     'd': 741,
     'e': 741,
     'f': 18,
     'h': 0,
     'i': 0,
     'g1': 185,
     'g2': 232,
     'g3': 51,
     'g4': 0,
     'g5': 118,
     'g6': 37,
     'g7': 2,
     'g8': 68,
     'g9': 0,
     'g10': 4,
     'g11': 5,
     'g12': 3,
     'g13': 2,
     'g14': 18,
     'g15': 9,
     'g16': 7}



We will also join the data with the `all_presence.csv` file.


{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
df_full = pd.merge(left=df_precint_with_results, right=_all_df, left_on=["county_code", "uat_name", "precinct_nr"], right_on=["Judet", "UAT", "Nr sectie de votare"])
df_full.shape
```


    (19171, 295)

</details>


## Applying the legend

Some of the columns in the above dataframe are not quite obvious (`g1`, .., `g16`, etc..). These are party names that I was only able to find in a legend in the dropdown of a button in the UI of the site. I've copied it here, along with explanations of some fields that I've been able to figure out by looking over the PDF's of official scanned documents.

We also need to convert these column names into more meaningfull lables.


{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
df_full.rename(columns={
    "g1": "PSD",
    "g2": "USR-PLUS",
    "g3": "PRO Romania",
    "g4": "UDMR",
    "g5": "PNL",
    "g6": "ALDE",
    "g7": "PRODEMO",
    "g8": "PMP",
    "g9": "Partidul Socialist Roman",
    "g10": "Partidul Social Democrat Independent",
    "g11": "Partidul Romania Unita",
    "g12": "Uniunea Nationala Pentur Progresul Romaniei",
    "g13": "Blocul Unitatii Nationale",
    "g14": "Gregoriana-Carmen Tudoran",
    "g15": "George-Nicaolae Simion",
    "g16": "Peter Costea",
    "a": "Total alegatori",
    "a1": "Total lista permanenta",
    "a2": "Total urna mobila",
    "b": "Total prezenti",
    "b1": "Prezenti lista permanenta",
    "b2": "Prezenti urna mobila",
    "b3": "Prezenti lista suplimentara",
    "c": "Total voturi",
    "d": "Voturi nefolosite",
    "e": "Voturi valabile",
    "f": "Voturi anulate",
    "h": "Contestatii",
    "i": "Starea sigiliilor"
}, inplace=True)
```

</details>

Ok, let's check for the amount of missing data


{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
na_series = df_full.isna().sum()
na_series[na_series != 0]
```

</details>


    latitude               479
    longitude              479
    siruta                 441
    uat_code             18730
    Contestatii             19
    Starea sigiliilor        8
    Siruta                 441



## Removing duplicate columns

Because we've basically merged two types of `presence` datasets (the `per-county` one and the `all_presence.csv` one) we ended up with some duplicate columns in the joined dataframe. We also have as duplicates the `join on` columns, and columns that contained the same type of information.  

We want to eliminate those. We will find the duplicated columns by:
* using the `pandas.duplicated` method (used on the transposed matix - duplicated only works on rows)
* looking at the correlation matrix of the resulting columns and get the pairs of columns that have the highes correlation.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  

```python
duplicate_columns = df_full.columns[df_full.T.duplicated(keep=False)]
duplicate_columns
```

</details>


    Index(['liste_permanente', 'lista_suplimentare', 'total', 'urna_mobila',
           'county_code', 'county_name', 'initial_count', 'locality_name',
           'medium', 'precinct_name', 'precinct_nr', 'uat_name', 'men_18_24',
           'men_25_34', 'men_35_44', 'men_45_64', 'men_65+', 'women_18_24',
           'women_25_34', 'women_35_44', 'women_45_64', 'women_65+', 'Județ',
           'Uat', 'Nr', 'Judet', 'UAT', 'Localitate_y', 'Nr sectie de votare',
           'Nume sectie de votare', 'Mediu', 'Votanti lista', 'LP', 'LS', 'UM',
           'LT', 'Barbati 18-24', 'Barbati 25-34', 'Barbati 35-44',
           'Barbati 45-64', 'Barbati 65+', 'Femei 18-24', 'Femei 25-34',
           'Femei 35-44', 'Femei 45-64', 'Femei 65+', 'Barbati 104', 'Barbati 106',
           'Barbati 108', 'Barbati 109', 'Barbati 110', 'Barbati 112',
           'Barbati 113', 'Barbati 114', 'Barbati 115', 'Barbati 116',
           'Barbati 117', 'Barbati 118', 'Barbati 119', 'Barbati 120', 'Femei 105',
           'Femei 106', 'Femei 107', 'Femei 108', 'Femei 110', 'Femei 111',
           'Femei 112', 'Femei 113', 'Femei 114', 'Femei 115', 'Femei 116',
           'Femei 117', 'Femei 118', 'Femei 119', 'Femei 120'],
          dtype='object')



With these, we will compare each with each and see what searies are equals. This will results in a long list of pairs of columns that are duplicates of one another.

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
_pairs = set()
for i, _a in enumerate(duplicate_columns):
    for _b in duplicate_columns[i+1:]:
        if (df_full[_a] == df_full[_b]).all():
            _pairs.add(tuple(sorted([_a, _b])))
_pairs
```

</details>


    {('Barbati 104', 'Barbati 106'),
     ('Barbati 104', 'Barbati 108'),
     ('Barbati 104', 'Barbati 109'),
     ...
     ('Barbati 120', 'Femei 120'),
     ('Barbati 18-24', 'men_18_24'),
     ('Barbati 25-34', 'men_25_34'),
     ('Barbati 35-44', 'men_35_44'),
     ('Barbati 45-64', 'men_45_64'),
     ('Barbati 65+', 'men_65+'),
     ('Femei 105', 'Femei 106'),
     ('Femei 105', 'Femei 107'),
     ...
     ('Femei 119', 'Femei 120'),
     ('Femei 18-24', 'women_18_24'),
     ('Femei 25-34', 'women_25_34'),
     ('Femei 35-44', 'women_35_44'),
     ('Femei 45-64', 'women_45_64'),
     ('Femei 65+', 'women_65+'),
     ('Judet', 'county_code'),
     ('Județ', 'county_name'),
     ('LP', 'liste_permanente'),
     ('LS', 'lista_suplimentare'),
     ('LT', 'total'),
     ('Localitate_y', 'locality_name'),
     ('Mediu', 'medium'),
     ('Nr', 'Nr sectie de votare'),
     ('Nr', 'precinct_nr'),
     ('Nr sectie de votare', 'precinct_nr'),
     ('Nume sectie de votare', 'precinct_name'),
     ('UAT', 'Uat'),
     ('UAT', 'uat_name'),
     ('UM', 'urna_mobila'),
     ('Uat', 'uat_name'),
     ('Votanti lista', 'initial_count')}



There's only one more step that we need to do: find the groups of columns that have the same information. There are cases where the columns are not only duplicated but triplicated, which results in (A == B), (B == C), (C == A) pairs in the analisys above.

This is the perfect job of the disjoint-set datastructure. 

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  

```python
_groups = DisjointSets()
for _a, _b in _pairs:
    _groups.union(_a, _b)
_groups.sets()
```

</details>



    [['Barbati 109',
      'Femei 118',
      'Femei 110',
      'Femei 114',
      'Barbati 104',
      'Femei 115',
      'Barbati 116',
      'Barbati 117',
      'Barbati 114',
      'Femei 107',
      'Femei 119',
      'Femei 105',
      'Barbati 115',
      'Femei 108',
      'Barbati 118',
      'Barbati 108',
      'Barbati 119',
      'Femei 120',
      'Femei 116',
      'Barbati 112',
      'Femei 113',
      'Barbati 113',
      'Barbati 120',
      'Femei 117',
      'Barbati 106',
      'Barbati 110',
      'Femei 106',
      'Femei 112',
      'Femei 111'],
     ['LP', 'liste_permanente'],
     ['Femei 35-44', 'women_35_44'],
     ['LT', 'total'],
     ['Nr sectie de votare', 'precinct_nr', 'Nr'],
     ['UM', 'urna_mobila'],
     ['Mediu', 'medium'],
     ['Barbati 65+', 'men_65+'],
     ['Barbati 35-44', 'men_35_44'],
     ['Femei 18-24', 'women_18_24'],
     ['Votanti lista', 'initial_count'],
     ['Femei 25-34', 'women_25_34'],
     ['Barbati 25-34', 'men_25_34'],
     ['UAT', 'Uat', 'uat_name'],
     ['Barbati 18-24', 'men_18_24'],
     ['Barbati 45-64', 'men_45_64'],
     ['Localitate_y', 'locality_name'],
     ['Femei 45-64', 'women_45_64'],
     ['Nume sectie de votare', 'precinct_name'],
     ['Judet', 'county_code'],
     ['Femei 65+', 'women_65+'],
     ['Județ', 'county_name'],
     ['LS', 'lista_suplimentare']]



From the list above we know we choose to drop the following columns:

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>

```python
drop_columns = ['Barbati 104', 'Barbati 106',
       'Barbati 108', 'Barbati 109', 'Barbati 110', 'Barbati 112',
       'Barbati 113', 'Barbati 114', 'Barbati 115', 'Barbati 116',
       'Barbati 117', 'Barbati 118', 'Barbati 119', 'Barbati 120', 'Femei 105',
       'Femei 106', 'Femei 107', 'Femei 108', 'Femei 110', 'Femei 111',
       'Femei 112', 'Femei 113', 'Femei 114', 'Femei 115', 'Femei 116',
       'Femei 117', 'Femei 118', 'Femei 119', 'Femei 120', 'LP', 'Femei 35-44', 'LT', 'Nr sectie de votare', 'Nr', 'UM', 
       'Mediu', 'Barbati 65+', 'Barbati 35-44','Femei 18-24',  'initial_count', 'Femei 25-34', 'Barbati 25-34', 
       'UAT', 'Uat', 'Barbati 18-24', 'Barbati 45-64', 'Localitate_y', 'Femei 45-64', 'Femei 45-64', 'Nume sectie de votare',
       'Judet', 'Femei 65+', 'Județ', 'LS', 
]
```

```python
df_final = df_full.drop(columns=drop_columns)
df_final.columns
```

</details>



    Index(['liste_permanente', 'lista_suplimentare', 'total', 'urna_mobila',
           'county_code', 'county_name', 'id_county', 'id_locality', 'id_precinct',
           'id_uat',
           ...
           'Femei 96', 'Femei 97', 'Femei 98', 'Femei 99', 'Femei 100',
           'Femei 101', 'Femei 102', 'Femei 103', 'Femei 104', 'Femei 109'],
          dtype='object', length=242)



And we end up with..

{::options parse_block_html="true" /}
<details><summary markdown='span'>Code</summary>
  
```python
df_final.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>liste_permanente</th>
      <td>696</td>
      <td>140</td>
      <td>501</td>
      <td>571</td>
      <td>680</td>
    </tr>
    <tr>
      <th>lista_suplimentare</th>
      <td>63</td>
      <td>10</td>
      <td>25</td>
      <td>41</td>
      <td>55</td>
    </tr>
    <tr>
      <th>total</th>
      <td>759</td>
      <td>150</td>
      <td>526</td>
      <td>612</td>
      <td>736</td>
    </tr>
    <tr>
      <th>urna_mobila</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>county_code</th>
      <td>VS</td>
      <td>VS</td>
      <td>VS</td>
      <td>VS</td>
      <td>VS</td>
    </tr>
    <tr>
      <th>county_name</th>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>VASLUI</td>
    </tr>
    <tr>
      <th>id_county</th>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>id_locality</th>
      <td>9015</td>
      <td>9015</td>
      <td>9006</td>
      <td>9006</td>
      <td>9006</td>
    </tr>
    <tr>
      <th>id_precinct</th>
      <td>16128</td>
      <td>16187</td>
      <td>16086</td>
      <td>16087</td>
      <td>16088</td>
    </tr>
    <tr>
      <th>id_uat</th>
      <td>2936</td>
      <td>2936</td>
      <td>2933</td>
      <td>2933</td>
      <td>2933</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>46.6389853639550</td>
      <td>46.6421141774663</td>
      <td>46.2240238056566</td>
      <td>46.2278431009305</td>
      <td>46.2278431009305</td>
    </tr>
    <tr>
      <th>locality_name</th>
      <td>VASLUI</td>
      <td>VASLUI</td>
      <td>BÂRLAD</td>
      <td>BÂRLAD</td>
      <td>BÂRLAD</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>27.7326775437114</td>
      <td>27.7289502189002</td>
      <td>27.6775710052581</td>
      <td>27.6686353095150</td>
      <td>27.6686353095150</td>
    </tr>
    <tr>
      <th>medium</th>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
    </tr>
    <tr>
      <th>precinct_name</th>
      <td>CASA DE CULTURĂ A SINDICATELOR ”CONSTANTIN TĂN...</td>
      <td>ȘCOALA GIMNAZIALĂ ”CONSTANTIN PARFENE”</td>
      <td>ŞCOALA GIMNAZIALĂ ”VICTOR IOAN POPA”</td>
      <td>CASA DE CULTURĂ A SINDICATELOR ”GEORGE TUTOVEANU”</td>
      <td>CASA DE CULTURĂ A SINDICATELOR ”GEORGE TUTOVEANU”</td>
    </tr>
    <tr>
      <th>precinct_nr</th>
      <td>1</td>
      <td>10</td>
      <td>100</td>
      <td>101</td>
      <td>102</td>
    </tr>
    <tr>
      <th>presence</th>
      <td>51.6327</td>
      <td>8.1522</td>
      <td>38.8479</td>
      <td>44.5091</td>
      <td>46.879</td>
    </tr>
    <tr>
      <th>siruta</th>
      <td>161954</td>
      <td>161954</td>
      <td>161801</td>
      <td>161801</td>
      <td>161801</td>
    </tr>
    <tr>
      <th>uat_code</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>uat_name</th>
      <td>MUNICIPIUL VASLUI</td>
      <td>MUNICIPIUL VASLUI</td>
      <td>MUNICIPIUL BÂRLAD</td>
      <td>MUNICIPIUL BÂRLAD</td>
      <td>MUNICIPIUL BÂRLAD</td>
    </tr>
    <tr>
      <th>men_18_24</th>
      <td>13</td>
      <td>4</td>
      <td>18</td>
      <td>15</td>
      <td>11</td>
    </tr>
    <tr>
      <th>men_25_34</th>
      <td>31</td>
      <td>5</td>
      <td>25</td>
      <td>29</td>
      <td>40</td>
    </tr>
    <tr>
      <th>men_35_44</th>
      <td>66</td>
      <td>17</td>
      <td>60</td>
      <td>52</td>
      <td>60</td>
    </tr>
    <tr>
      <th>men_45_64</th>
      <td>110</td>
      <td>28</td>
      <td>104</td>
      <td>113</td>
      <td>158</td>
    </tr>
    <tr>
      <th>men_65+</th>
      <td>115</td>
      <td>20</td>
      <td>43</td>
      <td>72</td>
      <td>81</td>
    </tr>
    <tr>
      <th>women_18_24</th>
      <td>22</td>
      <td>6</td>
      <td>17</td>
      <td>22</td>
      <td>14</td>
    </tr>
    <tr>
      <th>women_25_34</th>
      <td>34</td>
      <td>8</td>
      <td>33</td>
      <td>32</td>
      <td>46</td>
    </tr>
    <tr>
      <th>women_35_44</th>
      <td>78</td>
      <td>15</td>
      <td>64</td>
      <td>55</td>
      <td>52</td>
    </tr>
    <tr>
      <th>women_45_64</th>
      <td>171</td>
      <td>28</td>
      <td>117</td>
      <td>127</td>
      <td>178</td>
    </tr>
    <tr>
      <th>women_65+</th>
      <td>119</td>
      <td>19</td>
      <td>45</td>
      <td>95</td>
      <td>96</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Femei 76</th>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Femei 77</th>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 78</th>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Femei 79</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Femei 80</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Femei 81</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 82</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Femei 83</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Femei 84</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Femei 85</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 86</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 87</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Femei 88</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 89</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Femei 90</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 91</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 92</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 93</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 94</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 95</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 96</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 97</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 98</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 99</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 100</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 101</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 102</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 103</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 104</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Femei 109</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>242 rows × 5 columns</p>
</div>



## Save the data to csv format

We're almost done. We only need to save the dataset on disk and start using it (to be continued, in a future post)!


```python
df_final.to_csv("_data/final.csv")
```
