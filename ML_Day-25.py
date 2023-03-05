#!/usr/bin/env python
# coding: utf-8

# # 16-02-2023
# # XGBM Coding parts

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier,StackingClassifier,BaggingClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("Wine.csv")
df


# In[3]:


df['Customer_Segment'].value_counts()


# In[4]:


df.describe()


# In[5]:


sns.heatmap(df.isna())# to visualise null values. There are no null values


# In[6]:


#Define X and Y
x=df.iloc[:,:-1] # all rows, all columns except last column
y=df['Customer_Segment']


# In[7]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


# In[8]:


#Build individual model first. Check whether single model performing well or when we bag is it is performing well.
lg=LogisticRegression()
lg.fit(xtrain,ytrain)
ypred=lg.predict(xtest)
print(classification_report(ytest,ypred))#get confusion matrix
print('Train Accuracy: ',lg.score(xtrain,ytrain))
print('Test Accuracy: ',lg.score(xtest,ytest))

# model is overfitted
# # Bagging Classifier

# In[22]:


#bg=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20)
bg=BaggingClassifier(DecisionTreeClassifier())
#base_estimator: for which algorithm you want to creat bagging classifier like knn or logistic
#n_estimators: for base estimator algo. how many models you want to create. Its a hypert parameter
#base estimator is same for all algorithm.


# In[23]:


bg.fit(xtrain,ytrain)
ypred=bg.predict(xtest)
print(classification_report(ytest,ypred))
print('Train Accuracy: ',bg.score(xtrain,ytrain))
print('Test Accuracy: ',bg.score(xtest,ytest))


# In[26]:


#as we are doing same for diff algorithms so create a function which will build model and print accuracy
# write prediction function
def predictor(model):
  model.fit(xtrain,ytrain)
  ypred=model.predict(xtest)
  print('Train Accuracy: ',model.score(xtrain,ytrain))
  print('Test Accuracy: ',model.score(xtest,ytest))
  print(classification_report(ytest,ypred))


# In[27]:


predictor(DecisionTreeClassifier())


# In[28]:


predictor(BaggingClassifier(DecisionTreeClassifier()))


# In[29]:


predictor(LogisticRegression())


# In[30]:


predictor(KNeighborsClassifier()) # No feature scaling done so less accuracy


# In[31]:


predictor(BaggingClassifier(KNeighborsClassifier())) # may be after bagging (for 10 KNN algorithms) accuracy will increase


# In[32]:


predictor(AdaBoostClassifier())
# huge difference between train and test accuracy - uses stump


# In[33]:


predictor(GradientBoostingClassifier())# much better than adaboost - as uses fully grown tree, it works on residuals (tries to correct previous errors)
# this is only on one train and test data. Try for kfold


# # K-Fold Cross Validation

# In[34]:


kf=KFold(n_splits=10)
score=cross_val_score(GradientBoostingClassifier(),x,y,cv=kf)
score


# In[35]:


# gradient boosting is performing well. Final accuracy will be avg of all
score.mean()


# In[46]:


predictor(XGBClassifier()) # some may get error.
# in target column, Customer_Segment we have class numbers as 1,2,3
#new version reguires classification should start from 0. It expects class as 0,1,2


# In[42]:


# check class of datset
sns.countplot(df['Customer_Segment'])


# In[43]:


# to change 1,2,3 to 0,1,2 perform label encoding
le = LabelEncoder()
y=le.fit_transform(y)
y


# In[44]:


predictor(XGBClassifier())


# # Voting and Stacking Classifier

# In[47]:


# create a list of algorithms
models=[]
models.append(('lr',LogisticRegression()))
models.append(('dt',DecisionTreeClassifier()))
models.append(('dt1',DecisionTreeClassifier(criterion='entropy')))
models.append(('knn',KNeighborsClassifier()))
models.append(('rf',RandomForestClassifier()))


# In[48]:


predictor(VotingClassifier(estimators=models))


# In[50]:


predictor(StackingClassifier(estimators=models,final_estimator=RandomForestClassifier()))
#suppose we have x,y and we are using M1,M2,M3. Outputs of these models are say y1,y2,y3
# if you are using final model as Random Forest model then it will use y, y1,y2,y3 as x variables


# In[ ]:




