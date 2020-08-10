#!/usr/bin/env python
# coding: utf-8

# In[10]:


'''
Here we have created a new column 'Elevation Level', where we are dividing the elevation into 3 levels
low (<2500), medium (2500-3000), high (>3000). the we are clustering them and coloring them according
to the elevation level
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', None)

dataset = pd.read_csv('Patches.csv') 
converter = LabelEncoder()
dataset['Tree'] = converter.fit_transform(dataset['Tree'].astype(str))

# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')

dataset.drop(['Vertical_Distance_To_Hydrology'], axis = 1) 

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(dataset)

# Creating Cutomer Category from their account balance
def converter(column):
    if column <= 2500:
        return 0 # Low Elevation
    elif column > 2500 and column <= 3000:
        return 1 # Medium Elevation
    else:
        return 2 # High Elevation

dataset['Elevation Level'] = dataset['Elevation'].apply(converter)

# Dividing dataset into label and feature sets
X = dataset.drop('Elevation Level', axis = 1) # Features
Y = dataset['Elevation Level'] # Labels

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

kmeans = KMeans(n_clusters = kn.knee)
kmeans.fit(X_scaled)

# Implementing PCA to visualize dataset
pca = PCA(n_components = 3)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))

'''
fig = px.scatter_3d(x_pca, x=x_pca[:,0], y=x_pca[:,1], z=x_pca[:,2],
              color=Y)
fig.show()
'''

print("Cluster Centers: \n",kmeans.cluster_centers_)

elevation = list(dataset['Elevation'])
elevation_level = list(dataset['Elevation Level'])
tree = list(dataset['Tree'])
fig = go.Figure(data=[go.Scatter3d(
    x=x_pca[:,0], y=x_pca[:,1], z=x_pca[:,2],
    mode='markers',
    marker=dict(color=Y, colorscale='Rainbow', opacity=0.5),text=[f'elevation: {a}; elevation level:{b}; tree:{c}' 
                                      for a,b,c in list(zip(elevation,elevation_level,tree))],
                                hoverinfo='text'
)],layout = go.Layout(title = 'Classification of the Spurce Trees according to there Elevation level', width = 700, height = 700,barmode='group'))

offline.plot(fig,filename='Patches_tree_by_elevation.html')


# In[ ]:





# In[ ]:




