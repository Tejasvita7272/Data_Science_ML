#!/usr/bin/env python
# coding: utf-8

# # SVM 
# # 20-02-2023

# In[1]:


# SVM Classification
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# from sklearn import SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


filename = "pima-indians-diabetes.data.csv"
names = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)


# In[4]:


import matplotlib.pyplot as plt 
plt.scatter(dataframe['plas'], dataframe['pedi'], c=dataframe['class'])
# try plot with other features as well mass and pedi


# In[14]:


clf = SVC(kernel='rbf',gamma=0.0001)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred) * 100
print("Accuracy= ",acc)
confusion_matrix(y_test,y_pred)


# # Grid search CV

# In[18]:


# to get optimal value of gamma use grid search cv
clf = SVC()
# can put ['linear','rbf','poly'],give range for gamma i.e. C as a regularization parameter. Best out of it will be 
# selected by algorithm.
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5,1,0.0001],'C':[1,15,14,13,12,11,10,0.1]}]
# 8 X 6 =48 models will be created and will give best of it.
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[19]:


gsv.best_params_,gsv.best_score_


# In[20]:


clf = SVC(C=1,gamma=0.0001,kernel='rbf')  # can change kernel and check accuracy 
clf.fit(X_train,y_train) # build model 
y_pred = clf.predict(X_test) # predict on test dataset
acc = accuracy_score(y_test,y_pred) * 100
print("Accuracy =",acc)
confusion_matrix(y_test,y_pred)


# In[21]:


(142+36) / (142+36+45+8)   # accuracy by confusion matrix


# In[ ]:




