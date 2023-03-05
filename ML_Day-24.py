#!/usr/bin/env python
# coding: utf-8

# # 16-02-2023
# # BaggingCLassifier 
# # DecisionTreeClassifier

# In[2]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values
# Split data in train and test data
X = array[:,0:8]
Y = array[:,8]
seed = 7


kfold = KFold(n_splits =10, random_state=seed,shuffle=True)
cart = DecisionTreeClassifier()
num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)  #classifier
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())


# In[3]:


results


# # Random Forest Classifiers

# In[4]:


# Random Forest Classifiers
from pandas import read_csv
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

array = dataframe.values
# Split data in train and test data
X = array[:,0:8]
Y = array[:,8]

num_trees = 100
max_features = 3 
kfold = KFold(n_splits =10)

model = RandomForestClassifier(n_estimators=num_trees, max_features = max_features)  #classifier
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())


# # AdaBoost Classifiers

# In[5]:


# AdaBoost Classifiers
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

array = dataframe.values
# Split data in train and test data
X = array[:,0:8]
Y = array[:,8]

num_trees = 10 
seed = 7
kfold = KFold(n_splits =10, random_state = seed, shuffle=True)

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)  #classifier
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())


# # Stacking Ensemble for Classification

# In[7]:


# Stacking Ensemble for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.ensemble import VotingClassifier

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10)

# create the sub models
estimators = [] # create empty list for different algorithms

model1 = LogisticRegression(max_iter=500)
estimators.append(('logistic', model1))

model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))

model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())


# In[8]:


estimators


# In[ ]:




