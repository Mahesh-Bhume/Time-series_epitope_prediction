#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# for visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# for Pre-processing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# for Dimensionality reduction
from sklearn.decomposition import PCA

#for Models development 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier

#for validation
from sklearn.model_selection import cross_val_score,KFold


#for model Evaluation 
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report,matthews_corrcoef

import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


data=pd.read_csv('Ov_fullprots.csv')
data


# In[3]:


#Check for missing values 

data.isnull().sum().sort_values(ascending=False)


# In[4]:


# checking missing percentages 

missing_percentages = data.isnull().sum().sort_values(ascending=False) / len(data)
missing_percentages


# In[5]:


# visualization of null values 
missing_percentages[missing_percentages !=0].plot(kind='barh')


# # making a copy of original data frame

# In[4]:


df_master=data.copy()
df_master


# # Removing  the irrelevant features from the dataset.

# In[5]:


# data=data.drop("Info_organism_id", axis=1)
# data=data.drop("Info_epitope_id", axis=1)
# data=data.drop("Info_host_id", axis=1)
# data=data.drop("Info_nPos", axis=1)
# data=data.drop("Info_nNeg", axis=1)
# data=data.drop("Info_type", axis=1)
# data=data.drop("Info_window", axis=1)

# #Trying out Info_protein_id
# #data=data.drop("Info_protein_id", axis=1)


Remove_info_cols = []
for i in df_master.columns:
    if 'Info_' in i:
        Remove_info_cols.append(i)
for i in Remove_info_cols:
    df_master = df_master.drop(i,axis=1)


# In[6]:


df_master


# In[7]:


#Check for missing values after removing Info_ features

df_master.isnull().sum().sort_values(ascending=False)


# In[10]:


df_master['Class'].unique()


# # Selecting target feature

# In[8]:


x=df_master.drop(['Class'], axis=1) #axis=1 means we are working with columns and axis=0 means rows
y=df_master['Class']


# In[12]:


y.ndim


# In[13]:


y = np.array(y).reshape(1, -1)


# In[14]:


y.ndim


# # impute missing values (using Scikit-Learn KNN-Imputer)

# In[15]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(y)


# In[18]:


type(y)


# In[19]:


# df_master=df_master.merge(y, how='left')
# df_master


# In[20]:


#Check for missing values after imputing with KNN

df_master.isnull().sum().sort_values(ascending=False)


# # Class distribution
# 

# In[11]:


class_count = y.value_counts()
class_count


# In[12]:


#Class distribution visualization approch 2 using matplotlib library 
import matplotlib.pyplot as plt
fig1, ax1 =plt.subplots()
ax1.pie(y.value_counts(),autopct='%.2f', labels=class_count.index) # %.2f give us the values in 2 digits


# In[13]:


data['Info_protein_id'].describe()


# In[21]:


x.info()


# # Scalling
# 

# In[9]:


scaler= StandardScaler()
x_std =scaler.fit_transform(x)


# In[10]:


x_std


# # Principal Component Analysis (PCA)

# In[11]:


pca=PCA(n_components=0.99)
pca.fit_transform(x_std)


# In[12]:


pca.explained_variance_ratio_ 


# In[16]:


# Following wil tell, how many components we got, which is bascially given data columns(features)
PCA_df=pca.n_components_
PCA_df


# In[31]:


PCA_df=pd.DataFrame([PCA_df])
#df = pd.Dataframe([data])


# In[14]:


#Visualization of explained variance ratio

get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(range(1,20), pca.explained_variance_ratio_, color='teal')
plt.title("PCA Visualization")
plt.xlabel('principle components')
plt.ylabel('Explained variance ratio')
plt.show()


# Here PCA has selcted 19 features . All these  features are computed column. They are the new feature, so that now we can use this features for further pre-processing. Next use train-test split once again, but this time I wll supply new DataFrame(pca)

# # feature engineering in series-based data with Tsfresh

# To install tsfresh package first we need to install conda package on conda-forge using --> conda install -c conda-forge tsfresh

# In[11]:


get_ipython().system('pip install tsfresh')


# In[18]:


df_tfresh=df_master.copy()
df_tfresh


# In[19]:


x=df_tfresh.drop(['Class'], axis=1) #axis=1 means we are working with columns and axis=0 means rows
y=df_tfresh['Class']


# In[20]:


# y = np.array(y).reshape(1, -1)


# In[21]:


# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=2)
# imputer.fit_transform(y)


# In[11]:


#Check for missing values 

df_tfresh.isnull().sum().sort_values(ascending=False)


# In[12]:


# scaler= StandardScaler()
# x_TsFresh =scaler.fit_transform(x)


# In[22]:


from tsfresh.feature_extraction import ComprehensiveFCParameters


# In[25]:


settings = ComprehensiveFCParameters()


# In[23]:


from tsfresh.feature_extraction import extract_features


# In[32]:


extract_features(PCA_df, default_fc_parameters=settings)


# In[ ]:





# In[ ]:





# In[ ]:




