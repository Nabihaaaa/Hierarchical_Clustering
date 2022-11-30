#!/usr/bin/env python
# coding: utf-8

# In[1]:


#preprocessing library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
#Hierarchical Clustering Agglomerative
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


dataset = pd.read_csv('Groceries_dataset.csv')


# In[3]:


dataset.isnull().sum()


# In[4]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[5]:


#preprocessing 
dataset = MultiColumnLabelEncoder(columns = ['Date','itemDescription']).fit_transform(dataset)


# In[6]:


dataset


# In[7]:


#Tabel to array
X2 = dataset.iloc[:,[1,2]]
X2 = np.asarray(X2)
X2


# In[8]:


#dendrogram
dendrogram = sch.dendrogram(sch.linkage(X2, method = 'ward'))
plt.title('Dendrogram Groceries')
plt.xlabel('itemDescription')
plt.ylabel('Jarak Euclidean')
plt.show()


# In[9]:


#data before clustering
plt.scatter(X[:,1],X[:,2])


# In[10]:


#Agglomerative
hc = AgglomerativeClustering(n_clusters = 5, affinity= 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X2)
y_hc


# In[11]:


plt.scatter(X2[y_hc == 0,0], X2[y_hc == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X2[y_hc == 1,0], X2[y_hc == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X2[y_hc == 2,0], X2[y_hc == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X2[y_hc == 3,0], X2[y_hc == 3,1], s = 100, c = 'black', label = 'Cluster 4')
plt.scatter(X2[y_hc == 4,0], X2[y_hc == 4,1], s = 100, c = 'orange', label = 'Cluster 5')


# In[ ]:




