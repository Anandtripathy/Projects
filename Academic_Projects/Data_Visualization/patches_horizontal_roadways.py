#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''
Here we have created a new column "Hor_Dist_Level" wherein we have divided the horizontal distance to roadways 
into 3 levels, low(<1000) , medium (1000-2000), high (>3000.), then we are clustering the daatset and 
coluring the data points according to the tree type 
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

dataset = pd.read_csv('/Users/swamy/Desktop/Visualization CA2/Patches.csv') 
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
    if column <= 1000:
        return 0 # Low distance
    elif column > 1000 and column <= 2000:
        return 1 # medium distance
    else:
        return 2 # High distance

dataset['Hor_Dist_Level'] = dataset['Horizontal_Distance_To_Roadways'].apply(converter)

# Dividing dataset into label and feature sets
X = dataset.drop('Hor_Dist_Level', axis = 1) # Features
Y = dataset['Hor_Dist_Level'] # Labels

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

print("Cluster Centers: \n",kmeans.cluster_centers_)

hor_dist = list(dataset['Horizontal_Distance_To_Roadways'])
hor_dist_level = list(dataset['Hor_Dist_Level'])
tree = list(dataset['Tree'])
fig = go.Figure(data=[go.Scatter3d(
    x=x_pca[:,0], y=x_pca[:,1], z=x_pca[:,2],
    mode='markers',
    marker=dict(color=Y, colorscale='Rainbow', opacity=0.5),text=[f'horizontal distance: {a}; horizontal distance:{b}; tree:{c}' 
                                      for a,b,c in list(zip(hor_dist,hor_dist_level,tree))],
                                hoverinfo='text'
)],layout=go.Layout(title = 'Classification of the Trees according to their Horizontal Distance Level to Roadways'
                    ,barmode='group'))

offline.plot(fig,filename='tree_type_by_hor_dist.html')


# In[ ]:




