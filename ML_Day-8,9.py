#!/usr/bin/env python
# coding: utf-8

# # 20-01-2023 

# In[14]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as snf

from sklearn.linear_model import LogisticRegression


# In[15]:


# read/load the data
claimants = pd.read_csv("claimants.csv")
claimants.head()


# In[16]:


claimants.shape


# In[17]:


# drop the case number col. not required 
claimants.drop(["CASENUM"],inplace=True,axis = 1)


# In[18]:


claimants.shape


# In[19]:


# remove Na values in data
claimants = claimants.dropna()
claimants.shape


# In[20]:


# Dividing our data into input and output variables
X = claimants.iloc[:,1:]
Y = claimants.iloc[:,0]


# In[24]:


# Logistic regression  and fit the model
classifier = LogisticRegression()
# 1st create object 'classifier' for class LogisticRegression
classifier.fit(X,Y)


# In[26]:


# Predict for X dataset 
y_pred = classifier.predict(X)
y_pred # y values for x


# In[28]:


y_pred_df = pd.DataFrame({'actual':Y,
                         'predicted_prob': classifier.predict(X)})
y_pred_df


# In[30]:


# Confusion Matrix  for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print(confusion_matrix)


# In[31]:


((381+395)/(381+197+123+395))*100  # Accuracy


# In[32]:


# ROC curve


# In[42]:


from sklearn.metrics import roc_curve   #roc-receiver characteristics
from sklearn.metrics import roc_auc_score    # auc-area under curve

fpr,tpr, thresholds = roc_curve(Y,classifier.predict_proba (X)[:,1])

auc = roc_auc_score(Y,y_pred) # compute roc_auc_score based on y and y predict

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False positive rate or [1- true negative rate]')
plt.ylabel('True positivr rate')
plt.show()


# In[ ]:





# In[ ]:




