#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn


# In[3]:


from sklearn import datasets


# In[9]:


dataset=datasets.load_diabetes()


# In[10]:


dataset


# In[11]:


X=dataset.data


# In[12]:


Y=dataset.target


# In[24]:


X


# In[17]:


dataset.feature_names


# In[21]:


import pandas as pd
df=pd.DataFrame(X)


# In[22]:


df.columns=dataset.feature_names


# In[23]:


df


# In[25]:


from sklearn import model_selection


# In[26]:


X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y)


# In[29]:


# X_train
# Y_train
# X_test
# Y_test


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


alg1=LinearRegression()


# In[32]:


alg1.fit(X_train,Y_train)


# In[34]:


Y_pred=alg1.predict(X_test)


# In[35]:


Y_pred


# In[36]:


Y_test


# In[38]:


import matplotlib.pyplot as plt
plt.scatter(Y_test,Y_pred)
# plt.axis([0,40,0,40])
plt.show()


# In[ ]:




