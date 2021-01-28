#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[27]:


train=pd.read_csv("SalaryData_Train(1).csv")
test=pd.read_csv("SalaryData_Test(1).csv")


# In[16]:


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


# In[35]:


##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in string_columns:
    train[i]= number.fit_transform(train[i])
    test[i]=number.fit_transform(test[i])  


# In[38]:


##Capturing the column names which can help in futher process
colnames = train.columns
colnames
len(colnames)


# In[39]:


x_train = train[colnames[0:13]]
y_train = train[colnames[13]]
x_test = test[colnames[0:13]]
y_test = test[colnames[13]]


# In[44]:


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
x_train = norm_func(x_train)
x_test =  norm_func(x_test)


# In[45]:


model_linear = SVC(kernel = "linear")


# In[46]:


model_linear.fit(x_train,y_train)


# In[47]:


pred_test_linear = model_linear.predict(x_test)


# In[48]:


np.mean(pred_test_linear==y_test)


# ##### Accuracy achieved by applying linear kernel model is 81%

# ### Applying different Kernel Models

# In[49]:


# Kernel = poly
model_poly = SVC(kernel = "poly")


# In[50]:


model_poly.fit(x_train,y_train)


# In[52]:


pred_test_poly = model_poly.predict(x_test)


# In[53]:


np.mean(pred_test_poly==y_test) 


# ##### Accuracy achieved by applying poly kernel model is 84%

# In[54]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)
np.mean(pred_test_rbf==y_test)


# ##### Accuracy achieved by applying poly kernel model is 84%

# In[55]:


#kernel ='sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)
np.mean(pred_test_sig==y_test)


# ##### Accuracy achieved by applying poly kernel model is 84%
