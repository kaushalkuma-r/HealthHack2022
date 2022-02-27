#!/usr/bin/env python
# coding: utf-8

# In[29]:
import pandas as pd
import numpy as np
import pickle
import sklearn


# In[30]:


df = pd.read_csv(r'\Users\BROSHAL\jupyterlab\new1.csv')
df.head()


# In[18]:


null_data = pd.DataFrame(df.isna().sum().sort_values(ascending=False)).reset_index().rename(columns = {"index":"columns", 0:"missing_values"})
null_data["missing_percentage"] = null_data["missing_values"] / len(df) * 100
null_data["missing_percentage"] = null_data["missing_percentage"].round(2)
null_data


# In[31]:


df['gender'].unique()


# In[32]:


df['smoking_status'].unique()


# In[33]:


df['heart_disease'].unique()


# In[35]:


df['ever_married'].unique()


# In[36]:


df['work_type'].unique()


# In[37]:


df['Residence_type'].unique()


# In[38]:


df['smoking_status'].unique()


# In[39]:


df['hypertension'].unique()


# In[19]:


df.bmi = df.bmi.fillna(df.bmi.mode()[0])
df.bmi.isna().sum()


# In[20]:


cols = df.columns
numerical_cols = []
categorical_cols = []
for column in cols:
    if df[column].dtype == 'object':
        categorical_cols.append(column)
    else:
        numerical_cols.append(column)
# print(numerical_cols, len(numerical_cols), categorical_cols, len(categorical_cols), sep='\n')


# In[21]:


categorical_cols
from sklearn.preprocessing import StandardScaler, LabelEncoder
le= LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# In[22]:


X= df.drop(['stroke','id', 'bmi'], axis=1)
y= df.stroke


# In[23]:


scale = StandardScaler()

scale.fit(X)
x_scale = scale.transform(X)
X = pd.DataFrame(x_scale, columns=X.columns)
# X.head()


# In[24]:


from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0).fit(X,y)


# In[49]:


dtc_predict = dtc.predict(X_test)
df1=pd.DataFrame(dtc_predict)

pickle.dump(dtc, open('model1.pkl','wb'))

model = pickle.load(open('model1.pkl','rb'))

# In[ ]:





# In[ ]:




