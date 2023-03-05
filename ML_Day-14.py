#!/usr/bin/env python
# coding: utf-8

# # 30 jan 2023
# # PCA 

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[2]:


uni  = pd.read_csv("Universities.csv")
uni.head()


# In[3]:


uni.describe() # data should be normalied


# In[4]:


# Considering only numerical data 
uni = uni.iloc[:,1:] # exculde )th columns i.e. univ. name
uni.head()


# In[5]:


# Normalizing the numerical data
std = StandardScaler()
uni_normal = std. fit_transform(uni)
uni_normal


# In[6]:


pca = PCA() # if you don't pass any parameter by default it will consider all 6 dimensions
#pca2=PCA(n_components=2) # or you can pass specific no. of variables
#pca_components=pd.DataFrame(pca2.fit_transform(uni_normal),columns=['a','b'])
pca_components=pca.fit_transform(uni_normal)
pca_components


# In[7]:


# The amount of variance that each PCA explain is 
pca.explained_variance_  # 1st column contains 4.80425527 variance.. upto 6th colcontains 0.02755274 variance 


# In[8]:


# in percentage - The amount of variance that each PCA explain is
var = pca.explained_variance_ratio_
var


# In[9]:


# cumulative variance 
var1 = np.cumsum(np.round(var,decimals=4)*100)
var1 # 76  76+13=89  89+4=93 and so on i.e. if you stop upto 3rd columns still you will get 95 %


# In[10]:


pca.components_ # 6 demensional components


# In[11]:


uni.columns


# In[12]:


# linear combination equation, constant values are taken from above array o/p
 # PCA1 = -0.45*SAT - 0.427*Top10 + 0.42*Accept + 0.39*SFRatio - 0.36*Expenses - 0.37*GradRate

 # Thus all 6 principal components can be calculated using above 6 values


# In[13]:


# variance plot for PCA components obtained
plt.plot(var1,color="red") # var1 is cumulative percentage on y axis and index on x axis 
# looking at graph you can decide how much percetage you want and accordingly go for that much column numbers


# In[14]:


pca_components[:,0:1] # 1st columns is 1st component value, then 2ne,3rd,4th


# In[15]:


# plot between PCA1 and PCA2
x = pca_components[:,0:1]
y = pca_components[:,1:2]
z = pca_components[:,2]
#plt.scatter(x,y)
#plt.scatter(x,z)
plt.scatter(y,z) # plot of 0 and 2
# plt.scatter(pca_components.a,pca_components.b)


# In[17]:


# why scatterplot?  - After applying PCA there should not be any correlation. thus multicollinearity is removed or not 

import seaborn as sns
sns.pairplot(pd.DataFrame(pca_components))


# In[18]:


sns.pairplot(uni) # can check scatterplot with original dataset and see the difference


# In[ ]:




