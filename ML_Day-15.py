#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install bioinfokit ')
# library for visualizing the things related to clinical research or pathological data


# In[2]:


# from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE 
from bioinfokit.visuz import cluster


# In[3]:


# load data
df = pd.read_csv('TSNE_data.csv')
df.head() # M Malignant - Cancer Tumor present, B Benign - not a cancer tumor. We want to find if there is a pattern in M tumor and B tumor


# In[4]:


# Split-out validation dataset
array = df.values
# separate array into input and output components
X = array[:,1:]
Y = array[:,0]


# In[5]:


df


# In[6]:


#TSNE visualization
data_tsne = TSNE(n_components=2).fit_transform(X) # no of components = 2, max you can go for 3
cluster.tsneplot(score=data_tsne)


# In[7]:


data_tsne


# In[8]:


newdf=pd.DataFrame(data_tsne).join(df['diagnosis'])
newdf


# In[9]:


# get a list of categories
color_class = df['diagnosis'].to_numpy() # legnedpos=position on upper right corner showing color difference, colorlist-show different color for M and B in diagnosis column
cluster.tsneplot(score=data_tsne, colorlist=color_class, legendpos='upper right',legendanchor=(1.15,1))
# plt.show()
#Plot will be stored in the default directory


# In[10]:


color_class


# In[11]:


data_tsne


# In[ ]:




