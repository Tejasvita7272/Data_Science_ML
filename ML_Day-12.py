#!/usr/bin/env python
# coding: utf-8

# # 25-01-2023  Thursday

# In[43]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

# We don't know the no. of clusters. So let's use K-Means and elbow method to choose this number of optimal clusters.
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


# In[44]:


univ = pd.read_csv("Universities.csv")
univ


# In[45]:


# Normalization / Standardisation function 
from sklearn.preprocessing import StandardScaler  # built in function for standardization
scaler = StandardScaler() # scaler is an object name of StandardScalar class, you can give any name
scaled_univ_df = scaler.fit_transform(univ.iloc[:,1:]) # fit_transform() is a method of StandardScalar class
scaled_univ_df


# In[46]:


# How to find optimum number of  cluster
#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:


# In[47]:


# Inertia measures how well a dataset was clustered by K-Means. 
# It is calculated by measuring the distance between each data point and its centroid, squaring this distance, 
# and summing these squares across one cluster.
# A good model is one with low inertia AND a low number of clusters (K). However, this is a tradeoff because as K increases, inertia decreases.

# To plot the elbow method graph, we need to compute the WCSS (Within Cluster Sum of Squares)
# Let us say max. no. of clusters could be 10. 
# As we are going to have 10 iterations we are going to write a for loop to create a list of 10 WCSS for the no. of clusters

wcss = [] # within cluster sum of square. Initialize WCSS and begin the loop
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,random_state=0)
#init='k-means++': init is random initialization method. We can choose random if choice of initial centroid is to be random.
# But as we don't want to fall into random initialization, we are going to use this initialization method.
# ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability distribution of the points’.
# It makes several trials at each sampling step and selects the best centroid among them.
# Refer: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#:~:text=init%7B'k%2Dmeans%2B%2B,contribution%20to%20the%20overall%20inertia.
# max_iter=300: Suppose dataset has 200 data points. We will make 10 clusters of 20 points each in 1st iteration. Such how many combinations you can form for data points?
# so default value is 300. so we will keep it as it is.
# random_state=0: Use an int to make the randomness deterministic i.e. same results every time. It determines random number generation for centroid initialization.

    kmeans.fit(scaled_univ_df) #  use fit method to fit the kmeans object to our scaled dataframe
    wcss.append(kmeans.inertia_) # another name for wcss is inertia. In WCSS list we will append all distances i.e if cluster is 10 what is the value, if 9 what is value and so on upto 1.
    
plt.plot(range(1, 11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[48]:


wcss


# In[49]:


kmeans = KMeans(n_clusters=4)
predict=kmeans.fit_predict(scaled_univ_df)
predict


# In[50]:


kmeans = KMeans(n_clusters=4,random_state=0)
predict1=kmeans.fit_predict(scaled_univ_df)
predict1


# In[51]:


# Build cluster algorithm
clusters_new = KMeans(3,random_state=42)
clusters_new.fit(scaled_univ_df)


# In[52]:


clusters_new.labels_


# In[53]:


# Assign cluster to the data set
univ['clusterid_new'] = clusters_new.labels_
univ


# In[54]:


#these are standardized values.
clusters_new.cluster_centers_


# In[55]:


univ.groupby('clusterid_new').agg(['mean']).reset_index()


# In[56]:


univ[univ['clusterid_new']==0]


# In[57]:


univ[univ['clusterid_new']==2]


# In[58]:


univ[univ['clusterid_new']==1]


# In[ ]:





# In[ ]:





# In[ ]:




