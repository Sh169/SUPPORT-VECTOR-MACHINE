#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libraries
import pandas as pd 
import numpy as np 
import seaborn as sns


# In[3]:


#Load the libraries
dataset = pd.read_csv("forestfires1.csv")
dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.shape


# In[6]:


dataset.describe()


# In[17]:


#Dropping the month and day columns
#dataset.drop(["month"],axis=1,inplace =True)
#dataset.drop(["day"],axis=1,inplace =True)


# In[16]:


dataset.head()


# In[18]:


#Normalising the data as there is scale difference
predictors = dataset.iloc[:,0:28]
target = dataset.iloc[:,28]


# In[19]:


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)


# In[20]:


fires = norm_func(predictors)


# In[21]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)


# ##### Applying different kernel Models

# In[24]:


# Kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)
np.mean(pred_test_linear==y_test)


# ###### Accuracy achieved for linear kernel model is 99.2%

# In[25]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)
np.mean(pred_test_rbf==y_test)


# ##### Accuracy achieved for rbf kernel model is 76.92%

# In[26]:


#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)
np.mean(pred_test_sig==y_test)


# ##### Accuracy achieved for sigmoid kernel model is 76.92
