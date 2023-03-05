#!/usr/bin/env python
# coding: utf-8

# # Recommender System     02-02-2023

# In[60]:


import pandas as pd
import numpy as np


# In[61]:


Movies = pd.read_csv("Movie.csv")
Movies


# In[62]:


Movies.head()


# In[63]:


Movies.sort_values('userId')
Movies.shape


# In[64]:


# number of unique users
len(Movies.userId.unique())


# In[65]:


Movies['rating'].value_counts()


# In[66]:


len(Movies.movie.unique())


# In[67]:


Movies.movie.value_counts()


# In[68]:


# change structure of dataset so that we can compute the similarity score
user_Movies = Movies.pivot(index='userId',
                          columns='movie',
                          values='rating')
user_Movies.head()


# In[69]:


user_Movies.fillna(0,inplace=True)
user_Movies


# In[70]:


# Calculating cosine similarity between users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[102]:


# User similarity - pairwise - taking 2 rows at a time it will compute distance
# That distance is scaled between 0 to 1
# Suppose distance between 2 rows is 0. We want similarity, not a distance.
# That is calculated as Similarity=1-distance. So 1-0=1. i.e. similarity in rows is 1.
# Suppose distance is 0.9. So 1-0.9=0.1 
user_sim = 1 - pairwise_distances(user_Movies.values,metric='cosine')
user_sim  # This is similarity matrix


# In[72]:


# Stores the result in dataframe
user_sim_df = pd.DataFrame(user_sim)
user_sim_df


# In[73]:


# set the index and columns names to user ids
user_sim_df.index = Movies.userId.unique()
user_sim_df.columns = Movies.userId.unique()


# In[101]:


user_sim_df.iloc[0:5,0:5]
# similarity 1 - between customer 3 to 3, 6 to 6.
# similarity between customer 11 and customer 3 is 1. i.e. they are very similar.


# In[100]:


# diagonal values are not givinig any information so make it zero. Now you can get nonzero values
np.fill_diagonal(user_sim, 0)
np.fill_diagonal(user_sim,0)
user_sim_df.iloc[0:5,0:5]


# In[99]:


#Most Similar Users
# idxmax() method returns a Series with the index of the maximum value for each column.
# By specifying the column axis (axis='columns' or 1), the idxmax() method returns a Series with the index of the maximum value for each row.
user_sim_df.idxmax(axis=1)[0:5]


# In[98]:


# eg. find movies watched by customer 6 and 168 as they are similar
Movies[(Movies['userId']==6) | (Movies['userId']==168)]
# Both watched Toy Story with good rating, 6 watched 2 more movies. 
# Now rating for Sabrina is more than other movie. So we can recommend that movie to 168.


# # Another way to find Movie name with User

# In[94]:


user_1 = Movies[Movies['userId']==6]  # Movie name watched by user 6


# In[95]:


user_2 = Movies[Movies['userId']==168]  # Movie name watched by user 168


# In[96]:


user_1.movie   # 60 in index no. here, Display Movie name watched by user 168.


# In[97]:


user_2.movie


# In[93]:


pd.merge(user_1,user_2,on='movie',how='outer')   # Merge 2 outputs of user_1 and user_2


# In[ ]:




