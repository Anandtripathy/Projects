#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
Customer without any loan but are defaulters
'''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from kneed import KneeLocator
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
get_ipython().run_line_magic('matplotlib', 'inline')

#Code to read the .csv file and into the python environment(creating a dataframe)
dataset = pd.read_csv('/Users/swamy/Desktop/Visualization CA2/Clients.csv') 
dataset.isna().sum()

#cprint(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
converter = LabelEncoder()
dataset['job'] = converter.fit_transform(dataset['job'].astype(str))
dataset['default'] = converter.fit_transform(dataset['default'].astype(str))
dataset['education'] = converter.fit_transform(dataset['education'].astype(str))
dataset['default'] = converter.fit_transform(dataset['default'].astype(str))
dataset['marital'] = converter.fit_transform(dataset['marital'].astype(str))
dataset['housing'] = converter.fit_transform(dataset['housing'].astype(str))
dataset['personal'] = converter.fit_transform(dataset['personal'].astype(str))
dataset['term'] = converter.fit_transform(dataset['term'].astype(str))

no_loan_data = dataset[(dataset['housing'] == 0) & (dataset['personal'] == 0) & (dataset['term'] == 0)]

# Dividing dataset into label and feature sets
X = no_loan_data.drop('default' , axis = 1) # Features
Y = no_loan_data['default'] # Labels

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 2)
kmeans.fit(X_scaled)          

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 3, perplexity = 20 , n_iter = 3000)
x_tsne = tsne.fit_transform(X_scaled)      

default = list(no_loan_data['default'])
housing = list(no_loan_data['housing'])
personal = list(no_loan_data['personal'])
term = list(no_loan_data['term'])
data = [go.Scatter3d(x=x_tsne[:,0], y=x_tsne[:,1], z=x_tsne[:,2] , mode='markers',
                    marker = dict(color=Y, colorscale='Rainbow', opacity=0.5),
                                text=[f'default: {a}; housing:{b}, personal:{c}, term:{d}' 
                                      for a,b,c,d in list(zip(default,housing,personal,term))],
                                hoverinfo='text')]

layout = go.Layout(title = 'Defaulters without any Loan', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='no_loan_defaulters.html')


# In[ ]:




