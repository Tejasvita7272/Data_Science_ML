#!/usr/bin/env python
# coding: utf-8

# # 08-02-2023      
# # Decision Tree C5.0  CART  
# 

# In[301]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[302]:


# import some data to play with
iris = pd.read_csv('iris.csv',index_col=0)
iris.head()


# In[303]:


iris['Species'].value_counts()


# In[304]:


# Complete Iris dataset
label_encoder = preprocessing.LabelEncoder()
iris['Species']= label_encoder.fit_transform(iris['Species']) 
iris


# In[305]:


iris['Species'].value_counts()


# In[306]:


x=iris.iloc[:,:4]
y=iris['Species']


# In[307]:


x


# In[308]:


y


# In[309]:


colnames = list(iris.columns)
colnames


# In[310]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)


# In[311]:


x_train


#  # Building Decision Tree classifier using Entropy Criteria 

# In[312]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[313]:


# plot the Decision tree
tree.plot_tree(model);


# In[314]:


y_train.value_counts()


# In[315]:


fn = ['Sepal length(CM)','Sepal width (CM)','Petal length (CM)','Petal width(CM)']
cn = ['Setosa','Versicolor','Virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[316]:


# Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[317]:


preds


# In[318]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[319]:


# Accuracy
np.mean(preds==y_test)


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[320]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion = 'gini',max_depth=3)


# In[321]:


model_gini.fit(x_train,y_train)


# In[322]:


# Prediction and computing the accuracy
pred = model_gini.predict(x_test)
np.mean(pred==y_test)


# # Decision Tree Regression Example

# In[323]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[324]:


array = iris.values
X = array[:,0:3]
y = array[:,3]


# In[325]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[326]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[327]:


# Find the Accuracy
model.score(X_test, y_test)


# In[ ]:




