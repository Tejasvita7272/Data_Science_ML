#!/usr/bin/env python
# coding: utf-8

# # KNN Classification
# # 17-02-2023

# In[19]:


# KNN Classification
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[20]:


filename = "pima-indians-diabetes.data.csv"
names = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]


# In[21]:


dataframe


# In[4]:


X


# In[5]:


X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))


# In[6]:


X


# In[7]:


num_folds = 10
kfold = KFold(n_splits=10)


# In[8]:


model = KNeighborsClassifier(n_neighbors=17)  #K = 17
results = cross_val_score(model,X,Y,cv=kfold)


# In[9]:


print(results.mean())


# # Grid search for algorithm Tuning

# In[10]:


import numpy 
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[11]:


filename = "pima-indians-diabetes.data.csv"
names = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]


# In[12]:


n_neighbors = np.array(range(1,40))  # K =1 to  39
param_grid = dict(n_neighbors = n_neighbors)


# In[13]:


param_grid


# In[14]:


n_neighbors


# In[15]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X,Y)


# In[16]:


print(grid.best_score_)
print(grid.best_params_)


# # Visualizing the CV Results

# In[17]:


# search for an optimal value of K for KNN

# range of k we want to try
k_range = range(1, 41)
# empty list to store scores
k_scores = []

#we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())


# k_scores=grid.best_score_
pd.Series(k_scores).sort_values(ascending=False)


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []

# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10)
    k_scores.append(scores.mean())

# plot to see clearly
plt.plot(k_range, k_scores)


plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[ ]:




