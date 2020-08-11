#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
Here were are considering only the divorced entries, clustering them and coloring them as per their education level
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
dataset = pd.read_csv('C:/Users/Ritu/Desktop/MSc. DA Project/visualization code/Clients.csv') 
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

divorced_data = dataset[(dataset['marital'] == 0)]

# Dividing dataset into label and feature sets
X = divorced_data.drop('education' , axis = 1) # Features
Y = divorced_data['education'] # Labels

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

#Displaying the Elbow Plot    
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()    

#Computing the Distance between the Points in the Elbow plot
derivates = []
for i in range(len(inertia)):
    derivates.append(inertia[i] - inertia[i-1])    
print(derivates)    

x = range(1, len(derivates)+1)

#Locating the Knee Point
kn = KneeLocator(x, derivates, curve='convex', direction='decreasing')
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.plot(x, derivates, 'bx-')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.show()    

print("The Knee point is : ",
      kn.knee)    

# Implementing K-Means CLustering on dataset and visualizing clusters
kmeans = KMeans(n_clusters = kn.knee)
kmeans.fit(X_scaled)  

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 3, perplexity = 20 , n_iter = 3000)
x_tsne = tsne.fit_transform(X_scaled)      

education = list(divorced_data['education'])
age = list(divorced_data['age'])
marital = list(divorced_data['marital'])
data = [go.Scatter3d(x=x_tsne[:,0], y=x_tsne[:,1],z=x_tsne[:,2], mode='markers',
                    marker = dict(color=Y, colorscale='Rainbow', opacity=0.5),
                                text=[f'education: {a}; age:{b}, marital:{c}' 
                                      for a,b,c in list(zip(education,age,marital))],
                                hoverinfo='text')]

layout = go.Layout(title = 'Education Level for the Divorced Customers', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='divorced_education_level.html')


# In[ ]:




