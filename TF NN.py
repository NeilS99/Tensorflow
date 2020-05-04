#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv('fake_reg.csv')


# In[3]:


df.head()


# In[4]:


sns.pairplot(df)


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X=df[['feature1','feature2']].values


# In[7]:


y=df['price'].values


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=42)


# In[9]:


from sklearn.preprocessing import MinMaxScaler


# In[10]:


scaler=MinMaxScaler()


# In[11]:


scaler.fit(X_train)


# In[12]:


X_train=scaler.transform(X_train)


# In[13]:


X_test=scaler.transform(X_test)


# In[14]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


# In[15]:





# In[17]:


model=Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')


# In[18]:


model.fit(x=X_train,y=y_train,epochs=250)


# In[19]:


model.history.history


# In[20]:


loss_df=pd.DataFrame(model.history.history)


# In[21]:


loss_df.plot()


# In[22]:


model.evaluate(X_test,y_test,verbose=0)


# In[23]:


model.evaluate(X_train,y_train,verbose=0)


# In[24]:


test_predictions=model.predict(X_test)


# In[44]:


test_predictions


# In[46]:


pred_df = pd.DataFrame(y_test,columns=['Test Y'])


# In[47]:


pred_df


# In[49]:


pred_df = pd.concat([pred_df,test_predictions],axis=1)


# In[50]:


pred_df.columns = ['Test Y','Model Predictions']


# In[51]:


pred_df


# In[52]:


sns.scatterplot(x='Test Y',y='Model Predictions',data=pred_df)


# In[53]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[54]:


mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions'])


# In[56]:


df.describe()


# In[57]:


mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions'])**0.5


# In[58]:


from tensorflow.keras.models import load_model


# In[59]:


model.save('Practice')


# In[ ]:




