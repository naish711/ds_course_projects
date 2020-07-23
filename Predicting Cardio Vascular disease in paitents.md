
# Predicting the Cardio-Vascular paitent

## Course Project - Phase 2 (MATH 2319)

Authors : Manikanta Naishadu Devabhakthuni ,
          Varsha Shankar 

The objective of this project is to predict whether an individual is a cardiovascular paitent using the data provided from a kaggler. The dataset used in this study is from Kaggle [2]. This report is organised as follows the work done in Phase-1 of the project.
This dataset has one target feature named as cardio , has two classes as '1' if the person is cardio and '0' if the person is not a cardio vascular paitent. The descriptive features include 5 numerical and 6 nominal values.

## Overview

We have used the below mentioned three binary classifiers to predicit the target feature.

1. K-Nearest Neighbors

2. Naive Bayes Classifier

3. Decision Trees

## Methodology

We started data transformation with the preprocessed data (data cleaning) from phase 1. This transformation approach includes label encoding for all features with more than two levels. Essentially we scaled the data to get all descriptive features into a single range of numerical value more likely between 0 to 1 to facilitate the feature selection.

Then, we move on to feature selection ,in which we have used most popular Random Forrest Importance (RFI) with estimator of 100 trees with 10 maximum limited features.

We have data instances more than 68k which more for the task. Hence we have chosen 30k instances for this study. We sampled the data randomly and split the data with stratification method in order to ensure the target sample is selected with equally distributed(ratio) class lablels .

As process of hyperparameter tuning ,with the help of pipeline technique , we stack the RFI features to grid serach function in order to estimate the best parameters for all three classifiers. This helps to select an optimal model for each of the three classifier and to avoid overfitting of the each model.

In the process of model selection , we considered 10 fold cross validation and paired t-test to check with statistical significance between three classifiers and chosen the best classifier according to the highest recall values which is genereated by classification report of each of the three classifiers.

## Data Preparation


```python
#importing the required library
import pandas as pd
import numpy as np
import sklearn as skl
import os
```


```python
# Chossing the working directory
os.chdir('D:\Machine learning\ML Course Project')
```

As mentioned earlier in the report, we have just imported the dataset that has been preprocessed in the phase 1. To recap, during data preprocessing we found some missing values, impossibles values and obvious typos in different features. We have successfully manipulated these values and dropped few columns (approx. 8 rows) which are obvious error and a column named as "ID".


```python
#importing pre processed data from Phase-I of course project

cardio = pd.read_csv("PreData.csv")
```


```python
# the dataset has 11 descriptive features and cardio as the target varaible
cardio.head(10)
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
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>cholesterol</th>
      <th>gluc</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cardio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>1</td>
      <td>168</td>
      <td>62.0</td>
      <td>110</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>0</td>
      <td>156</td>
      <td>85.0</td>
      <td>140</td>
      <td>90</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52</td>
      <td>0</td>
      <td>165</td>
      <td>64.0</td>
      <td>130</td>
      <td>70</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>1</td>
      <td>169</td>
      <td>82.0</td>
      <td>150</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48</td>
      <td>0</td>
      <td>156</td>
      <td>56.0</td>
      <td>100</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>0</td>
      <td>151</td>
      <td>67.0</td>
      <td>120</td>
      <td>80</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>61</td>
      <td>0</td>
      <td>157</td>
      <td>93.0</td>
      <td>130</td>
      <td>80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>62</td>
      <td>1</td>
      <td>178</td>
      <td>95.0</td>
      <td>130</td>
      <td>90</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>48</td>
      <td>0</td>
      <td>158</td>
      <td>71.0</td>
      <td>110</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>54</td>
      <td>0</td>
      <td>164</td>
      <td>68.0</td>
      <td>110</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(cardio.shape)
```

    (68992, 12)
    


```python
cardio.columns.values
```




    array(['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
           'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'],
          dtype=object)



## Summary Statistics

Summary of statistics is shown as below:


```python
cardio.describe(include='all')
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
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>cholesterol</th>
      <th>gluc</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cardio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
      <td>68992.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.324849</td>
      <td>0.348678</td>
      <td>164.359317</td>
      <td>74.119079</td>
      <td>126.241405</td>
      <td>81.620550</td>
      <td>1.364376</td>
      <td>1.225852</td>
      <td>0.087851</td>
      <td>0.053586</td>
      <td>0.803267</td>
      <td>0.494898</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.768156</td>
      <td>0.476555</td>
      <td>8.203868</td>
      <td>14.327062</td>
      <td>15.677191</td>
      <td>8.214059</td>
      <td>0.678672</td>
      <td>0.571797</td>
      <td>0.283080</td>
      <td>0.225200</td>
      <td>0.397532</td>
      <td>0.499978</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>55.000000</td>
      <td>21.000000</td>
      <td>90.000000</td>
      <td>65.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>48.000000</td>
      <td>0.000000</td>
      <td>159.000000</td>
      <td>65.000000</td>
      <td>120.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>54.000000</td>
      <td>0.000000</td>
      <td>165.000000</td>
      <td>72.000000</td>
      <td>120.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>58.000000</td>
      <td>1.000000</td>
      <td>170.000000</td>
      <td>82.000000</td>
      <td>140.000000</td>
      <td>90.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>65.000000</td>
      <td>1.000000</td>
      <td>250.000000</td>
      <td>200.000000</td>
      <td>170.000000</td>
      <td>105.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cardio.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 68992 entries, 0 to 68991
    Data columns (total 12 columns):
    age            68992 non-null int64
    gender         68992 non-null int64
    height         68992 non-null int64
    weight         68992 non-null float64
    ap_hi          68992 non-null int64
    ap_lo          68992 non-null int64
    cholesterol    68992 non-null int64
    gluc           68992 non-null int64
    smoke          68992 non-null int64
    alco           68992 non-null int64
    active         68992 non-null int64
    cardio         68992 non-null int64
    dtypes: float64(1), int64(11)
    memory usage: 6.3 MB
    

As nominal features like 'cholesterol','gluc' which has more than two level are in numerical format. In order to create encode them , we first them converted into categorical variables. 


```python
cardio['cholesterol'] = cardio['cholesterol'].astype(object)
cardio['gluc'] = cardio['gluc'].astype(object)
```

Spliting the target variable apart from the features and named as 'target' and the rest descriptive features are named as 'features'.


```python
from sklearn import preprocessing
features = cardio.drop(columns='cardio')
target = cardio['cardio']
target.value_counts()
target = preprocessing.LabelEncoder().fit_transform(target)
```


```python
features_cols = cardio.columns[cardio.dtypes==object].tolist()

features_cols 
```




    ['cholesterol', 'gluc']



We make use of get_dummies function from pandas library in order to get the dummy variables for categorical varaibles having more than two levels. Particulaly in this dataset , most of the categorical variables having two level '1' and '0' which is more like TRUE or FALSE expect two features 'cholesterol' and 'gluc'. <br> As in the data description from source website, it is mentioned that 1,2,3 constitutes the degree of the substance present in the body/person. 


```python
# use one-hot-encoding for categorical features with >2 levels
features = pd.get_dummies(features)
```


```python
features.sample(5, random_state=999)
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
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cholesterol_1</th>
      <th>cholesterol_2</th>
      <th>cholesterol_3</th>
      <th>gluc_1</th>
      <th>gluc_2</th>
      <th>gluc_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53732</th>
      <td>64</td>
      <td>0</td>
      <td>172</td>
      <td>76.0</td>
      <td>130</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15127</th>
      <td>49</td>
      <td>1</td>
      <td>182</td>
      <td>79.0</td>
      <td>140</td>
      <td>100</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41950</th>
      <td>48</td>
      <td>1</td>
      <td>181</td>
      <td>91.0</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19715</th>
      <td>48</td>
      <td>0</td>
      <td>161</td>
      <td>70.0</td>
      <td>120</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14991</th>
      <td>58</td>
      <td>0</td>
      <td>158</td>
      <td>60.0</td>
      <td>120</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Scaling the features

In order to perform algorithms like KNN , Descion Tree etc., Scaling is the most important step in data preprocessing. Especially will performing the feature selection , all descriptive features must be on one scale i.e, in same range. <br> In this study, the least numerical value from all features is 0 and highest is 252(ap_hi) . Hence, we decided to perform scaling in order to proceed further for feature selection.


```python
from sklearn import preprocessing

features_copy = features.copy()

scaler = preprocessing.MinMaxScaler()
scaler.fit(features)
features = scaler.fit_transform(features)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:323: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:323: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)
    


```python
pd.DataFrame(features, columns=features_copy.columns).sample(5, random_state=999)
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
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cholesterol_1</th>
      <th>cholesterol_2</th>
      <th>cholesterol_3</th>
      <th>gluc_1</th>
      <th>gluc_2</th>
      <th>gluc_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53732</th>
      <td>0.971429</td>
      <td>0.0</td>
      <td>0.600000</td>
      <td>0.307263</td>
      <td>0.500</td>
      <td>0.375</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15127</th>
      <td>0.542857</td>
      <td>1.0</td>
      <td>0.651282</td>
      <td>0.324022</td>
      <td>0.625</td>
      <td>0.875</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>41950</th>
      <td>0.514286</td>
      <td>1.0</td>
      <td>0.646154</td>
      <td>0.391061</td>
      <td>0.375</td>
      <td>0.375</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19715</th>
      <td>0.514286</td>
      <td>0.0</td>
      <td>0.543590</td>
      <td>0.273743</td>
      <td>0.375</td>
      <td>0.375</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14991</th>
      <td>0.800000</td>
      <td>0.0</td>
      <td>0.528205</td>
      <td>0.217877</td>
      <td>0.375</td>
      <td>0.375</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Selection

In this study, we have use Random Forrest Algorithm to get the feature importance. At a glance, 


```python
# Ensemble model for feature selection

from sklearn.ensemble import RandomForestClassifier

num_features = 10
model_rfi = RandomForestClassifier(n_estimators=100)
```


```python
model_rfi.fit(features, target)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
fs_indices_rfi = np.argsort(model_rfi.feature_importances_)[::-1][0:num_features]
```


```python
best_features_rfi = features_copy.columns[fs_indices_rfi].values
```


```python
best_features_rfi
```




    array(['weight', 'height', 'ap_hi', 'age', 'ap_lo', 'cholesterol_1',
           'cholesterol_3', 'gender', 'active', 'smoke'], dtype=object)




```python

```


```python
n_samples = 30000

features_sample = pd.DataFrame(features).sample(n=n_samples, random_state=9).values
target_sample = pd.DataFrame(target).sample(n=n_samples, random_state=9).values

print(features_sample.shape)
print(target_sample.shape)
```

    (30000, 15)
    (30000, 1)
    


```python
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(features_sample, target_sample, 
                                                    test_size = 0.4, random_state=999,
                                                    stratify = target_sample)

print(X_train.shape)
print(y_test.shape)
```

    (18000, 15)
    (12000, 1)
    


```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv_method = StratifiedKFold(n_splits=5, random_state=999)
```


```python
from sklearn.base import BaseEstimator, TransformerMixin

# custom function for RFI feature selection inside a pipeline
# here we use n_estimators=100
class RFIFeatureSelector(BaseEstimator, TransformerMixin):
    
    # class constructor 
    # make sure class attributes end with a "_"
    # per scikit-learn convention to avoid errors
    def __init__(self, n_features_=10):
        self.n_features_ = n_features_
        self.fs_indices_ = None

    # override the fit function
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from numpy import argsort
        model_rfi = RandomForestClassifier(n_estimators=100)
        model_rfi.fit(X, y)
        self.fs_indices_ = argsort(model_rfi.feature_importances_)[::-1][0:self.n_features_] 
        return self 
    
    # override the transform function
    def transform(self, X, y=None):
        return X[:, self.fs_indices_]
```


```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

pipe_KNN = Pipeline(steps=[('rfi_fs', RFIFeatureSelector()), 
                           ('knn', KNeighborsClassifier())])

params_pipe_KNN = {'rfi_fs__n_features_': [10, 20, features.shape[1]],
                   'knn__n_neighbors': [1, 10, 20, 40, 60, 100],
                   'knn__p': [1, 2, 5]}

gs_pipe_KNN = GridSearchCV(estimator=pipe_KNN, 
                           param_grid=params_pipe_KNN, 
                           cv=cv_method,
                           refit=True,
                           n_jobs=-2,
                           scoring='roc_auc',
                           verbose=1) 
```


```python
gs_pipe_KNN.fit(X_train, y_train);
```

    Fitting 5 folds for each of 54 candidates, totalling 270 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 11 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  28 tasks      | elapsed:   23.0s
    [Parallel(n_jobs=-2)]: Done 178 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=-2)]: Done 270 out of 270 | elapsed:  5.1min finished
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\pipeline.py:267: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      self._final_estimator.fit(Xt, y, **fit_params)
    


```python
gs_pipe_KNN.best_params_
```




    {'knn__n_neighbors': 60, 'knn__p': 1, 'rfi_fs__n_features_': 10}




```python
gs_pipe_KNN.best_score_
```




    0.7841814821750296




```python
from sklearn.preprocessing import PowerTransformer
X_train_transformed = PowerTransformer().fit_transform(X_train)
```


```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV

pipe_NB = Pipeline([('rfi_fs', RFIFeatureSelector()), 
                     ('nb', GaussianNB())])

params_pipe_NB = {'rfi_fs__n_features_': [10, 20, features.shape[1]],
                  'nb__var_smoothing': np.logspace(1,-3, num=200)}

n_iter_search = 20
gs_pipe_NB = RandomizedSearchCV(estimator=pipe_NB, 
                          param_distributions=params_pipe_NB, 
                          cv=cv_method,
                          refit=True,
                          n_jobs=-2,
                          scoring='roc_auc',
                          n_iter=n_iter_search,
                          verbose=1) 


```


```python
gs_pipe_NB.fit(X_train_transformed, y_train);
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 11 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  28 tasks      | elapsed:    5.0s
    [Parallel(n_jobs=-2)]: Done 100 out of 100 | elapsed:   15.6s finished
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    


```python
gs_pipe_NB.best_params_
```




    {'rfi_fs__n_features_': 10, 'nb__var_smoothing': 6.593188271333546}




```python
gs_pipe_NB.best_score_
```




    0.7770914742255031




```python
from sklearn.tree import DecisionTreeClassifier

pipe_DT = Pipeline([('rfi_fs', RFIFeatureSelector()),
                    ('dt', DecisionTreeClassifier(criterion='gini'))])

params_pipe_DT = {'rfi_fs__n_features_': [10],
                  'dt__max_depth': [3, 4, 5,6,7,8,9,10],
                  'dt__min_samples_split': [2, 5]}

gs_pipe_DT = GridSearchCV(estimator=pipe_DT, 
                          param_grid=params_pipe_DT, 
                          cv=cv_method,
                          refit=True,
                          n_jobs=-2,
                          scoring='roc_auc',
                          verbose=1) 
```


```python
gs_pipe_DT.fit(X_train, y_train);
```

    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    

    [Parallel(n_jobs=-2)]: Using backend LokyBackend with 11 concurrent workers.
    [Parallel(n_jobs=-2)]: Done  28 tasks      | elapsed:    5.1s
    [Parallel(n_jobs=-2)]: Done  80 out of  80 | elapsed:   12.9s finished
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:19: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    


```python
gs_pipe_DT.best_params_
```




    {'dt__max_depth': 6, 'dt__min_samples_split': 2, 'rfi_fs__n_features_': 10}




```python
gs_pipe_DT.best_score_
```




    0.7814834180463645




```python
from sklearn.model_selection import cross_val_score

cv_method_ttest = StratifiedKFold(n_splits=10, random_state=999)

cv_results_KNN = cross_val_score(estimator=gs_pipe_KNN.best_estimator_,
                                 X=X_test,
                                 y=y_test, 
                                 cv=cv_method_ttest, 
                                 n_jobs=-2,
                                 scoring='roc_auc')

```


```python
cv_results_KNN.mean()
```




    0.7923123891692838




```python
X_test_transformed = PowerTransformer().fit_transform(X_test)

cv_results_NB = cross_val_score(estimator=gs_pipe_NB.best_estimator_,
                                X=X_test_transformed,
                                y=y_test, 
                                cv=cv_method_ttest, 
                                n_jobs=-2,
                                scoring='roc_auc')

```


```python
cv_results_NB.mean()
```




    0.789046075861584




```python
cv_results_DT = cross_val_score(estimator=gs_pipe_DT.best_estimator_,
                                X=X_test,
                                y=y_test, 
                                cv=cv_method_ttest, 
                                n_jobs=-2,
                                scoring='roc_auc')

```


```python
cv_results_DT.mean()
```




    0.7917491402840351




```python
from scipy import stats

print(stats.ttest_rel(cv_results_KNN, cv_results_NB))
print(stats.ttest_rel(cv_results_DT, cv_results_KNN))
print(stats.ttest_rel(cv_results_DT, cv_results_NB))
```

    Ttest_relResult(statistic=1.948739990342999, pvalue=0.08313476401277939)
    Ttest_relResult(statistic=-0.2466478599013119, pvalue=0.8107145437359558)
    Ttest_relResult(statistic=1.0539745273359413, pvalue=0.3193667587118561)
    


```python
pred_KNN = gs_pipe_KNN.predict(X_test)
```


```python
Data_test_transformed = PowerTransformer().fit_transform(X_test)
pred_NB = gs_pipe_NB.predict(Data_test_transformed)
```


```python
pred_DT = gs_pipe_DT.predict(X_test)
```


```python
from sklearn import metrics
print("\nClassification report for K-Nearest Neighbor") 
print(metrics.classification_report(y_test, pred_KNN))
print("\nClassification report for Naive Bayes") 
print(metrics.classification_report(y_test, pred_NB))
print("\nClassification report for Decision Tree") 
print(metrics.classification_report(y_test, pred_DT))
```

    
    Classification report for K-Nearest Neighbor
                  precision    recall  f1-score   support
    
               0       0.72      0.78      0.75      6048
               1       0.76      0.69      0.72      5952
    
       micro avg       0.73      0.73      0.73     12000
       macro avg       0.74      0.73      0.73     12000
    weighted avg       0.74      0.73      0.73     12000
    
    
    Classification report for Naive Bayes
                  precision    recall  f1-score   support
    
               0       0.65      0.88      0.75      6048
               1       0.81      0.52      0.63      5952
    
       micro avg       0.70      0.70      0.70     12000
       macro avg       0.73      0.70      0.69     12000
    weighted avg       0.73      0.70      0.69     12000
    
    
    Classification report for Decision Tree
                  precision    recall  f1-score   support
    
               0       0.71      0.79      0.75      6048
               1       0.76      0.68      0.72      5952
    
       micro avg       0.73      0.73      0.73     12000
       macro avg       0.74      0.73      0.73     12000
    weighted avg       0.74      0.73      0.73     12000
    
    


```python
from sklearn import metrics
print("\nConfusion matrix for K-Nearest Neighbor") 
print(metrics.confusion_matrix(y_test, pred_KNN))
print("\nConfusion matrix for Naive Bayes") 
print(metrics.confusion_matrix(y_test, pred_NB))
print("\nConfusion matrix for Decision Tree") 
print(metrics.confusion_matrix(y_test, pred_DT))
```

    
    Confusion matrix for K-Nearest Neighbor
    [[4720 1328]
     [1858 4094]]
    
    Confusion matrix for Naive Bayes
    [[5315  733]
     [2878 3074]]
    
    Confusion matrix for Decision Tree
    [[4754 1294]
     [1912 4040]]
    

On contradict to paired t-test results, classification report supports decision tree with high value of recall score compared to other two classifer. As true predicition of cardio vascular paitent can be identified sligtly higher by decision tree classifier to predict the target feature.

## Limitations

Our modeling strategy has a two major limitations. First, since we have performed randomly selected raw predictive performance over interpretability.

Second, we have chosen only a small subset of the full dataset considering the shorter run times as priority at this instant, both for training and testing. Since data is availble ie., more than half of the data is available, we could perform experiments with the entire data with apt consideration of spliting data set and validtion techniques.

In future, we will continue our work to investigate the hidden patterns of the data,best approach to select important features and techniques to optimize the classifers in terms of parameters and data sampling.

## Conclusion

The K-near Neighbours model with 100 n_neighbours and 10 of the best features selected by Random Forest Importance (RFI) produces the highest cross-validated AUC score on the training data. In addition, when evaluated on the test set, the KNN model again outperforms both Naive Bayes and Decision tree models with respect to AUC scores. However, the Decision Tree model gives higher recall score on the test data.

## Dataset


Cardiovascular Disease dataset-https://www.kaggle.com/sulianova/cardiovascular-disease-dataset
