#!/usr/bin/env python
# coding: utf-8

# # 13-02-2023
# # Model_Validation

# # 1) Univariate Statistical Tests (Chi-squared for classification)

# In[24]:


# Evaluate using a train and a test set
from pandas import read_csv
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)
modle = LogisticRegression()
model.fit(X_train,Y_train)
result = model.score(X_test, Y_test)


# In[25]:


result


# In[26]:


result*100.0


# In[27]:


dataframe


# # 2) Evaluate using Cross Validation 

# In[28]:


# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values
# Split data in train and test data
X = array[:,0:8]
Y = array[:,8]

num_folds = 10
seed = 7


kfold = KFold(n_splits = num_folds)
model = LogisticRegression(max_iter = 200)
results = cross_val_score(model, X, Y, cv = kfold)


# In[29]:


results  # Accuracy of 10 models


# In[30]:


results.mean()*100.0   # Final accuracy is the mean of all accuracies


# In[31]:


results.std()*100.0
# + or - 5% standard deviation for accuracy. If Std is very high means models are very inconsistent for this dataset


# # 3) Evaluate using Leave One out Cross Validation

# In[32]:


from pandas import read_csv
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

loocv = LeaveOneOut()
model = LogisticRegression(max_iter=300)
results = cross_val_score(model, X, Y, cv = loocv)


# In[33]:


results  # 1 indicates 100 % accuracy and 0 indicates 0 % accuracy here


# In[34]:


results.mean()


# In[35]:


results.mean()*100.0


# In[36]:


results.std()*100.0
# Here accuracy is either 0% or 100% so we are getting high std.
# So don't consider Std. here


# In[37]:


import numpy as np 
np.array([100,100,100,0,0]).std() # check std of values 100 to 0


# In[ ]:




