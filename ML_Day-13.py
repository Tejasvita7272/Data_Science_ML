#!/usr/bin/env python
# coding: utf-8

# # DBSCAN Clustering    27-01-2023   

# In[43]:


#Import the libraries
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dir(sklearn.cluster.DBSCAN)
# help(sklearn.cluster.DBSCAN)


# In[44]:


# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("Wholesale customers data.csv");

print(df.head())


# In[45]:


print(df.info())


# In[46]:


df.drop(['Channel','Region'],axis=1,inplace=True) 
#drop channel and region column as they don't add any value to dataset 


# In[47]:


array=df.values


# In[48]:


array


# In[49]:


stscaler = StandardScaler().fit(array) # apply standardization on dataset
X = stscaler.transform(array)


# In[50]:


X


# In[51]:


dbscan = DBSCAN(eps=0.8, min_samples=6)  # creating instance  dbscan of DBSCAN class with eps and min_samples parameters
dbscan.fit(X)
#dbscan = DBSCAN(eps=2, min_samples=7) # creating instance  dbscan of DBSCAN class with eps and min_samples parameters
#dbscan.fit (X)


# In[52]:


#Noisy samples are given the label -1.
dbscan.labels_ # first 2 data points belongs to cluster 0. Here only 1 cluster is formed i.e. 0


# In[53]:


cluster =pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[54]:


cluster


# In[57]:


# Use pandas filtering and get noisy datapoints -1
# df[df['cluster']==-1]


# In[56]:


pd.concat([df,cluster],axis=1)


# In[58]:


# as only 1 cluster is formed after changing eps value that means in this case dbscan is not that much effective to form clusters with this dataset
# So remove noisy data points and go for k-means clustering


# In[ ]:





# In[ ]:




