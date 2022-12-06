#!/usr/bin/env python
# coding: utf-8

# In[1]:
#preprocessing library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
#Hierarchical Clustering Agglomerative
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# In[2]:
dataset = pd.read_csv('mushroom_dataset_new.csv')
dataset.isnull().sum()

# In[3]:
dataset = dataset.iloc[:,].apply(LabelEncoder().fit_transform)
dataset

# In[4]:
data_scaled = normalize(dataset)
data_scaled = pd.DataFrame(data_scaled, columns=dataset.columns)
data_scaled

# In[5]:
plt.figure(figsize=(20, 10))  
plt.title("Complete-linkage Method Dendrograms")  
dend = sch.dendrogram(sch.linkage(data_scaled, method='complete'))

# In[6]:
plt.figure(figsize=(20, 10))  
plt.title("Single-linkage Method Dendrograms")  
dend = sch.dendrogram(sch.linkage(data_scaled, method='single'))

# In[7]:
plt.figure(figsize=(20, 10))  
plt.title("Dendrograms")  
dend = sch.dendrogram(sch.linkage(data_scaled, method='complete'))
plt.axhline(y=1.0, color='r', linestyle='--')

# In[8]:
cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='complete')  
cluster.fit_predict(data_scaled)[:100]

# In[9]:
#Sebelum clustering
plt.scatter(data_scaled['cap-shape'],data_scaled['cap-surface'])

# In[10]:
#Sesudah Custering
plt.scatter(data_scaled['cap-shape'], data_scaled['cap-surface'],s=100,c=cluster.labels_)
