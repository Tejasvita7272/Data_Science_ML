#!/usr/bin/env python
# coding: utf-8

# # 13-01-2023  Friday

# In[2]:


import pandas as pd
data = pd.read_csv("NewspaperData.csv")
data.head()


# In[3]:


data


# In[4]:


data.info()


# # Correlations

# In[5]:


data.corr() # check daily vs sunday


# In[6]:


import seaborn as sns
sns.distplot(data['daily']) # density plot for daily - positive skewness in data


# In[7]:


import seaborn as sns
sns.distplot(data['sunday']) # density plot for sunday - positive skewness in data


# In[8]:


import statsmodels.formula.api as smf # model - Regression model
model = smf.ols("sunday~daily",data = data).fit() # Ordinary Least Squares, y Dep. variable - sunday, then give ~ symbol and then x Ind.Variable - daily, 
                                                  # dataset - data, fit() - fit the regression line


# In[9]:


#Coefficients are B0 and B1
model.params
# Reg. Equn: Sunday = B0+B1*daily
# In output: Intercept - B0, daily - coefficient of daily i.e.B1
# Sunday = 13.83 + 1.33 * daily
# we assumed daily circulation as 200
# So Sunday = 13.83 + 1.33 * 200


# In[10]:


sun=13.83+1.33*200
sun


# In[11]:


#R squared values - 0.92 - Good reg. equn.
(model.rsquared,model.rsquared_adj)


# In[12]:


# model.summary() # OLS - Ordinary Least Squares


# In[13]:


sns.regplot(x="daily", y="sunday", data=data);


# In[14]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)     
# we have estimates: B0=13.8, B1=1.33. We have t-value 0.38 and 18.93, p-value - 0.7, 6.01 is approx.= 0
#focus on B1. B1 is slope. Define H0 and H1.


# In[15]:


#1100 manual calculation
(1.3*1100)+13.835630


# In[16]:


# predict value for 150,240


# # Predict for new data point

# In[22]:


#Predict for 200 and 300 daily circulation
newdata=pd.Series([200,300]) # x value is (daily circulation) 200 at one branch and 300 at another branch
newdata


# In[25]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[26]:


model.predict(data_pred) # model - regression model


# In[ ]:




