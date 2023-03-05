#!/usr/bin/env python
# coding: utf-8

# # 10-02-2023
# # EDA2

# In[73]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd


# In[74]:


data = pd.read_csv("iris.csv",index_col=0)
data.head()


# In[75]:


# changes does not effect the "data" dataframe
data1=data.copy()


# In[76]:


labelencoder = LabelEncoder()  # we want to label the species column i.e. y variable 
data1.iloc[:, -1] = labelencoder.fit_transform(data1.iloc[:,-1])   # -1: last column
data1


# # One Hot Encoder

# # Using sklearn

# In[77]:


from sklearn.preprocessing import OneHotEncoder


# In[78]:


data2=pd.read_csv("iris.csv",index_col=0)
data2


# In[79]:


# creating instance of one-hot-encoder
OHE = OneHotEncoder(handle_unknown='ignore') # Specifies the way unknown categories are handle


# In[80]:


# passing bridge-types-cat column (label encoded values of bridge_types)
# convert it to array - to columns
enc_df = pd.DataFrame(OHE.fit_transform(data2[['Species']]).toarray())
enc_df


# In[81]:


# merge with main df
data_final = data2.iloc[:,0:4].join(enc_df)
data_final


# # Using pandas

# In[82]:


import pandas as pd


# In[83]:


data3 = pd.read_csv("iris.csv",index_col=0)


# In[84]:


data_encoded = pd.get_dummies(data3)
data_encoded


# # Isolation Forest

# In[85]:


from sklearn.ensemble import IsolationForest
import pandas as pd


# In[86]:


data =pd.read_csv("iris.csv",index_col=0)
data_encoded=pd.get_dummies(data)


# In[87]:


#  training the model
clf = IsolationForest(random_state=10,contamination=.01)
# contanimation : how much percentage of outliers you are expecting in dataset
# eg. in health care domain it will be very low .01 or .001
clf.fit(data_encoded)


# In[88]:


# predictions
y_pred_outliers = clf.predict(data_encoded)


# In[89]:


#-1 for outliers and 1 for inliers.
y_pred_outliers


# In[90]:


data_encoded


# In[91]:


## Let us add a new data point which is outlier
data_encoded.loc[150]=[20,40,30,50,1,0,0]
data_encoded


# In[92]:


# training the model
clf = IsolationForest(random_state=10,contamination=.01)
clf.fit(data_encoded)
# predictions
y_pred_outliers = clf.predict(data_encoded)
y_pred_outliers


# In[93]:


data_encoded['scores']=clf.decision_function(data_encoded)


# In[94]:


data_encoded['anomaly']=clf.predict(data_encoded.iloc[:,0:7])
# we can pinpoint those outliers exactly by applying this filtering score
data_encoded


# In[95]:


# print the outlier data points
data_encoded[data_encoded['anomaly']==-1] # scores are given by decision tree


# # PPS

# In[96]:


get_ipython().system('pip install ppscore')


# In[97]:


import ppscore as pps


# In[98]:


data.head()


# In[99]:


#pps.score(df, "feature_column", "target_column")  syntax
pps.score(data, "Sepal.Length", "Petal.Length")#ppscore:0.55 so ok kind of score


# In[ ]:





# In[ ]:




