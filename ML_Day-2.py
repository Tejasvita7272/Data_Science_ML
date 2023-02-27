#!/usr/bin/env python
# coding: utf-8

# # Speed Up The EDA Process 
# # 11-01-2023

# In[7]:


get_ipython().system('pip install pandas-profiling==3.1.0')
get_ipython().system('pip install sweetviz ')


# In[8]:


import pandas_profiling as pp
import sweetviz as sv
import pandas as pd
import numpy as np
data = pd.read_csv("data_clean.csv")
data


# In[9]:


EDA_report= pp.ProfileReport(data,vars={"num":{"low_categorical_threshold":0}})


# In[10]:


EDA_report


# In[11]:


sweet_report = sv.analyze(data)
sweet_report.show_html('weather_report.html')


# In[ ]:




