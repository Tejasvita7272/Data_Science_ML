#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import hierarchical clustering libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import scipy.cluster.hierarchy as sch # to build dendrogram and build the plotting
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings('ignore')


# In[2]:


univ = pd.read_csv("Universities.csv")
univ


# In[3]:


# Customized Normalization function 
# Here we can use standardized functions as well from sklearn but to show you how we can write customized function we used this code 
def norm_func(i): # function name is norm_func, we can give any name here.
    x = (i-i.min())/(i.max()-i.min()) # (Xi-min)/range (Feature Scaling), Range=max-min
    return x


# In[5]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(univ.iloc[:,1:]) # from 1st column because 0 index col is univ names
df_norm


# In[6]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_norm, method='single'))


# In[8]:


# create clusters, suppose got input from customer that go for 4 clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')
hc


# In[9]:


# save clusters for chart
y_hc = hc.fit_predict(df_norm) # apply fit_predict method on dataset df_norm. We will get cluster nos. in y_hc
Clusters=pd.DataFrame(y_hc,columns=['Clusters']) # append those no. of cluster numbers create dataframe
y_hc


# In[10]:


Clusters # Data point 0 belongs to 0th cluster, Data point 1 belongs to 3rd cluster


# In[11]:


# Now let us map this cluster membership to the data points
univ['h_clusterid'] = Clusters
univ  # Brown univ belongs to 0th cluster, CalTech belongs to 3rd cluster and so on. Total 4 clusters: 0,1,2,3


# In[12]:


univ1 = univ.sort_values("h_clusterid")
univ1.iloc[:,[0,7]]


# In[ ]:




