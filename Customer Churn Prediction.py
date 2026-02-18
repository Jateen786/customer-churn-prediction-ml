#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# In[3]:


df = pd.read_csv("C:/Users/91701/Downloads/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# In[4]:


df.info()
df.describe()


# In[5]:


# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID
df.drop("customerID", axis=1, inplace=True)


# In[6]:


le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


# In[7]:


X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[8]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_pred_lr))


# In[10]:


rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_pred_rf))


# In[11]:


cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[12]:


plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=pd.read_csv("C:/Users/91701/Downloads/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
plt.title("Churn Distribution")
plt.show()


# In[13]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df['tenure'])
plt.title("Customer Tenure Distribution")
plt.show()


# In[15]:


importances = rf.feature_importances_
feat = pd.Series(importances, index=X.columns)
feat.nlargest(10).plot(kind='barh')
plt.title("Top Features Affecting Churn")
plt.show()

