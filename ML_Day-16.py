#!/usr/bin/env python
# coding: utf-8

# # Association Rules          01-02-2023

# In[18]:


#Here we want to understand which features are strongly associated to survived = yes. eg. if 1st class, Male->Yes OR female->yes
#Install 'mlxtend' Library if not installed already
get_ipython().system('pip install mlxtend')


# In[19]:


import mlxtend


# In[20]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

import matplotlib.pyplot as plt


# In[21]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[22]:


titanic['Class'].value_counts()


#  # Pre-Processing 

# As the data is not in transaction formuation we are using transaction Encoder

# In[23]:


df = pd.get_dummies(titanic)
df.head()


# # Apriori Algorithm

# In[24]:


frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets  # output is frequent itemsets: 1-item itemsets, 2-item itemsets etc. with min support criteria


# # Rules Generation 

# In[25]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)  # min_threshold is Confidence
rules
rules.sort_values('lift',ascending = False)[0:20]


# In[26]:


rules[rules.lift>1]


# In[ ]:





# In[ ]:




