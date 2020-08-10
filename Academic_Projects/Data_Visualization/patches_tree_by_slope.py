#!/usr/bin/env python
# coding: utf-8

# In[3]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
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

#Code to read the .csv file and into the python environment(creating a dataframe)
datas = pd.read_csv('Patches.csv') 
converter = LabelEncoder()

datas['Tree'] = converter.fit_transform(datas['Tree'].astype(str))

# Plotting Correlation Heatmap
corrs = datas.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')

datas.drop(['Vertical_Distance_To_Hydrology'], axis = 1).head() 

# Creating Cutomer Category from their account balance
def convertr(colun):
    if colun <= 17:
        return 0 # Low slope
    elif colun > 17 and colun <= 34:
        return 1 # Medium slope
    else:
        return 2 # High slope
    
datas['Slope Level']=datas['Slope'].apply(convertr)

# Dividing dataset into label and feature sets
X = datas.drop('Slope Level', axis = 1) # Features
Y = datas['Slope Level'] # Labels

#scatterplot
sns.set()
sns.pairplot(X, height = 2.5)
plt.show()

#correlation matrix
corrmat = datas.corr()
f, ax = plt.subplots(figsize=(10, 8))
print(corrmat)
sns.heatmap(corrmat, vmax=.7, square=True);

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(datas)

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

fig = px.scatter_3d(x_pca, x=x_pca[:,0], y=x_pca[:,1], z=x_pca[:,2],
              color=kmeans.labels_,title='Scatter Plot of PCA')
#fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='patches_pca.html')

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

pca = PCA(n_components = 3)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))

print("Cluster Centers: \n",kmeans.cluster_centers_)
Tree=list(datas['Tree'])
slope_level = list(datas['Slope Level'])
slope = list(datas['Slope'])

fig = go.Figure(data=[go.Scatter3d(
    x=x_pca[:,0], y=x_pca[:,1], z=x_pca[:,2],
    mode='markers',
    marker=dict(color=Y, colorscale='earth', opacity=0.5),text=[f'Tree: {a}; slope:{b}; slope level:{c} ' 
                                      for a,b,c in list(zip(Tree,slope,slope_level))],
                                hoverinfo='text')],layout = go.Layout(title = ' Trees as per their slope Level',barmode='group'))

offline.plot(fig,filename='Tree_by_Slope.html')


# In[ ]:




