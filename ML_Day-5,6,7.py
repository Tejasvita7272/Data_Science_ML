#!/usr/bin/env python
# coding: utf-8

# # 16-01-2023  
# # Multiple Linear Regression

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as snf


# read the data
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


# scatter matrix, corelation matrix
# pairplot, pd.plotting.scatter_matrix.


# In[4]:


cars.info()


# In[5]:


# check for missing values
cars.isna().sum()


# # correlation Matrix

# In[6]:


cars.corr()


# # Scatterplot Between variable along with histograms 

# In[7]:


#formate the plot background and scatter plots for all the variables 
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# # preparing a model

# In[8]:


# Build model
import statsmodels.formula.api as smf
model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit() # excess variable can be joined using + symbols


# In[9]:


#Coefficients
model.params


# In[10]:


# t and p-values 
print('*** t-values ***','\n',model.tvalues,'\n','*** p-values ***','\n',model.pvalues)
# p values of SP and HP only are significant 


# In[11]:


# R squared values
(model.rsquared,model.rsquared_adj)


# # simple linear regression model

# In[12]:


ml_v = smf.ols('MPG~VOL',data = cars).fit() #simple linear regression model for MPG Vs VOL alone 
# t - values & p - values
print(ml_v.tvalues,'\n',ml_v.pvalues)
# here p-values 3.822 is approx. = 0, and less than alpha so VOL variable 


# In[13]:


ml_w = smf.ols('MPG~WT',data = cars).fit()
# t - values & p - values
print(ml_w.tvalues,'\n',ml_w.pvalues)


# In[14]:


ml_wv = smf.ols('MPG~WT+VOL',data = cars).fit()
# t - values & p - values
print(ml_wv.tvalues,'\n',ml_wv.pvalues)


# # *************************** 17-01-2023  Tuesday *******************************

# # Calculating VIF formula

# In[15]:


dir(rsq_hp)


# In[16]:


rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared 
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared 
vif_wt = 1/(1-rsq_hp)

rsq_vol = smf.ols('VOL~WT+HP+SP',data=cars).fit().rsquared 
vif_vol = 1/(1-rsq_hp)

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared 
vif_sp = 1/(1-rsq_hp)

d1 = {'Variables' : ['HP','WT','VOL','SP'],'VIF' : [vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame = pd.DataFrame(d1)
vif_frame


# # Subset Selection
AIC
# # Residuals Analysis

# In[17]:


import statsmodels.api as sm
qqplot = sm.qqplot(model.resid,line = 'q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[18]:


list(np.where(model.resid>10))


# # Residual Plot for Homoscedasticity

# In[19]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[39]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))
plt.title('Residual plot')
plt.xlabel("standardized fitted values")
plt.ylabel("standardized residual values")
plt.show()


# In[21]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model,"VOL",fig=fig)
plt.show()


# In[22]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model,"SP",fig=fig)
plt.show()


# In[23]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model,"HP",fig=fig)
plt.show()


# In[24]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model,"WT",fig=fig)
plt.show()


# # *****************18-01-2023****************  Wednesday

# # MOdel Deletion Diagnostics

# # COOK's Distance

# In[25]:


from statsmodels.graphics.regressionplots import influence_plot

model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance


# In[26]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(cars)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[28]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# # High Influence Points

# In[29]:


cars.shape


# In[30]:


k = cars.shape[1]
n = cars.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[31]:


from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pyplot as plt

influence_plot(model,alhpa=0.5)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')

plt.show()

# From the above plot, it is evident that data point 70 and 76 are the influencers
# In[32]:


cars[cars.index.isin([70, 76])]


# In[33]:


#See the differences in HP and other variable values
cars.head()


# # Improving the model 

# In[34]:


#Load the data
cars_new = pd.read_csv("Cars.csv")


# In[35]:


#Discard the data points which are influencers and reasign the row number (reset_index())
car1=cars_new.drop(cars_new.index[[70,76]],axis=0).reset_index()
car1


# In[36]:


#Drop the original index
car1=car1.drop(['index'],axis=1)


# In[37]:


car1


# # Build Model

# In[40]:


#Exclude variable "WT" and generate R-Squared and AIC values
final_ml_V= smf.ols('MPG~VOL+SP+HP',data = cars).fit()


# In[41]:


(final_ml_V.rsquared,final_ml_V.aic,final_ml_V.bic)


# In[42]:


#Exclude variable "VOL" and generate R-Squared and AIC values
final_ml_W= smf.ols('MPG~WT+SP+HP',data = cars).fit()


# In[43]:


(final_ml_W.rsquared,final_ml_W.aic,final_ml_W.bic)


# ##### Comparing above R-Square and AIC values, model 'final_ml_V' has high R- square and low AIC value hence include variable 'VOL' so that multi collinearity problem would be resolved.

# In[44]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[45]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(car1)+2),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[46]:


#index of the data points where c is more than .5
(np.argmax(c_V),np.max(c_V))


# In[48]:


#Drop 76 and 77 observations
car2=car1.drop(car1.index[[76,77]],axis=0)
car2


# In[49]:


#Reset the index and re arrange the row values
car3=car2.reset_index()


# In[51]:


car4=car3.drop(['index'],axis=1)
car4


# In[52]:


#Build the model on the new data
final_ml_V= smf.ols('MPG~VOL+SP+HP',data = car4).fit()


# In[53]:


#Again check for influencers
model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[54]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(car4)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[55]:


#index of the data points where c is more than .5
(np.argmax(c_V),np.max(c_V))


# #### Since the value is <1 , we can stop the diagnostic process and finalize the model

# In[56]:


#Check the accuracy of the mode
final_ml_V= smf.ols('MPG~VOL+SP+HP',data = car4).fit()


# In[57]:


(final_ml_V.rsquared,final_ml_V.aic)


# # Predicting for new data 

# In[63]:


#New data for prediction
new_data=pd.DataFrame({'HP':40,"VOL":95,"SP":102,"WT":35},index=[1])
new_data


# In[64]:


final_ml_V.predict(new_data)


# In[60]:


final_ml_V.predict(cars_new.iloc[0:5,])


# In[62]:


pred_y = final_ml_V.predict(cars_new)
pred_y


# In[ ]:




