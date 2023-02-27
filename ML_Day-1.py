#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# # 10-01-2023
# 

# In[1]:


#load the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#data = pd.read_csv("data_clean.csv")
#data


# In[3]:


#pwd


# In[4]:


data = pd.read_csv("data_clean.csv")
data


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


# Data Structure
print(type(data))
print(data.shape)


# In[8]:


# data type 
data.dtypes   # check data types of all variables


# In[9]:


data.describe()


# In[10]:


data.info() # find missing values


# # Data type Conversion

# In[11]:


data['Month']=pd.to_numeric(data['Month'],errors='coerce') #replce with NA
data['Temp C']=pd.to_numeric(data['Temp C'],errors='coerce') # coerce will introduce NA values for non numeric data in the columns 
data['Weather']=data['Weather'].astype('category') # data['wind']=data.['wind'].astype('int64')
data.info()


# # Duplicated Values
# 

# In[12]:


data.duplicated() # if any 2 rows has same values


# In[13]:


# print duplicated rows
data[data.duplicated()]


# In[14]:


data[data.duplicated()].shape


# # Type of cleaing data 

# In[15]:


data_cleaned1=data.drop_duplicates()  # drop duplicates rows if any 
data_cleaned1.shape


# # Drop columns

# In[16]:


data_cleaned2=data_cleaned1.drop('Temp C',axis=1) # axis =1 refers to cols & axis =2 refer to rows
data_cleaned2.shape


# In[17]:


data_cleaned2


# # Rename the columns

# In[18]:


# rename the solas column, pass parameter in dictionary form 
data_cleaned3 = data_cleaned2.rename({'Solar.R': 'Solar'}, axis=1)
data_cleaned3


# # Outlier Detections

# In[19]:


# histogram
data_cleaned3['Ozone'].hist()


# In[20]:


#box plot
data_box=data_cleaned3.dropna() # data set is: data_box
data1_box=data_box.Ozone # in data1_box we are saving ozone cols
plt.boxplot(data1_box)


# In[21]:


box=plt.boxplot(data1_box)


# In[22]:


[item.get_ydata() for item in box['fliers']]


# In[23]:


[item.get_ydata() for item in box['whiskers']]


# In[24]:


data_cleaned3['Ozone'].describe()


# In[25]:


data_cleaned3


# In[26]:


#Bar plot - to identify outliers in categorical data - get count of unique values
data['Weather'].value_counts()


# In[27]:


#Bar plot - to identify outliers in categorical data
data['Weather'].value_counts().plot.bar()


# # Missing Value Imputation

# In[28]:


import seaborn as sns  # in seaborn documentation color codes are mentioned
cols = data_cleaned3.columns
colours = ['#000099', '#ffff00']# specify the colours - yellow is missing. blue is not missing.
#colours = ['pink' , 'blue']
sns.heatmap(data_cleaned3[cols].isnull(),
           cmap=sns.color_palette(colours))
#sns.heatmap(data_cleaned3[cols].isnull(),
 #           cmap=sns.color_palette(colours)) # map colors T = Yellow, F = Blue


# In[29]:


data_cleaned3[data_cleaned3.isnull().any(axis=1)].head()  # find cloumn wise null values


# In[30]:


data_cleaned3.isnull().sum()


# In[31]:


data_cleaned3.info()


# # Mean Imputation 

# In[32]:


mean = data_cleaned3['Ozone'].mean()
print(mean)


# In[33]:


data_cleaned3['Ozone'] = data_cleaned3['Ozone'].fillna(mean) # use mean of ozone col. to fill na values
data_cleaned3


# In[34]:


#Missing value imputation for categorical vlaue
#Get the object columns
obj_columns = data_cleaned3[['Weather']]


# In[35]:


obj_columns.isnull().sum()


# In[36]:


#missing values imputation for categorical values 
# df.mode
# df.median
obj_columns = obj_columns.fillna(obj_columns.mode().iloc[0]) #mode will return 2 values. So index 0 value is mode


# In[37]:


obj_columns.isnull().sum()


# In[38]:


obj_columns.shape


# In[39]:


# data_cleaned4 = data_cleaned3.drop(['Weather'],axis=1,inplace=True)


# In[40]:


#join the data set with imputed object dataset
data_cleaned4 = pd.concat([data_cleaned3,obj_columns],axis=1)


# In[41]:


data_cleaned4.isnull().sum()


# In[42]:


data_cleaned4


# # Scatter Plot and Correlation analysis
# 

# In[43]:


# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
#pd.plotting.scatter_matrix(data_cleaned3)
sns.pairplot(data_cleaned3)  # diagonal - histogram, other - scatter plot


# In[44]:


#Correlation
data_cleaned3.corr() # corr. between same variables is always 1.


# # Transformation  Dummy Variables

# In[45]:


data_cleaned4


# In[46]:


#Creating dummy variable for Weather column
data_cleaned4 = pd.get_dummies(data,columns=['Weather'])
data_cleaned4


# In[47]:


data_cleaned4 = data_cleaned4.dropna()
data_cleaned4


# # Normalization of The Data                                          
# # 11-01-2023
# 

# In[48]:


#Normalization of the data
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler


# In[49]:


data_cleaned4.values


# In[53]:


array = data_cleaned4.values

scaler = MinMaxScaler(feature_range=(0,1)) # range 0 to 1
rescaledX = scaler.fit_transform(array)# apply normalization on array

#transformed data
set_printoptions(precision=2)
print(rescaledX[0:5,:])


# In[54]:


# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler


# In[55]:


array = data_cleaned4.values
scaler = StandardScaler()
scaler.fit(array)
rescaledX = scaler.transform(array)

# summarize transformed data
set_printoptions(precision=2)
print(rescaledX[0:5,:])


# #  Speed Up The EDA Process

# In[56]:


get_ipython().system('pip install pandas-profiling==3.1.0')
get_ipython().system('pip install sweetviz ')


# In[57]:


import pandas_profiling as pp
import sweetviz as sv
import pandas as pd
import numpy as np


# In[58]:


EDA_report= pp.ProfileReport(data,vars={"num":{"low_categorical_threshold":0}})


# In[59]:


EDA_report


# In[60]:


sweet_report = sv.analyze(data)
sweet_report.show_html('weather_report.html')


# In[ ]:




