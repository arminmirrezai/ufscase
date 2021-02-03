from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import seaborn as sbn
import ApiExtract
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# from trendypy.trendy import Trendy

# from scipy.spatial import distance
# from sklearn.neighbors import NearestNeighbors
# from fastdtw import fastdtw



years = range(2016,2021)
df = ApiExtract.extract(years,'NL')
 
all_series = []
# stl_series = []
for keyword in df.keyword.unique():
    product = df[df.keyword == keyword]['interest']
    time_series = pd.Series(product.tolist(), index = pd.date_range('1-1-2017', periods=len(product), freq='W'), name = keyword)
    stl = STL(time_series, robust=True).fit()
    # stl_series.append(stl)
    seasonal = stl.seasonal
    all_series.append(seasonal)
    # all_series.append(time_series.tolist())


# Hierarchical clustering_____________________________________________________________________________________________

for i in range(len(all_series)):
    length = len(all_series[i])
    all_series[i] = all_series[i].values.reshape((length, 1))
    # series[i] = series[i].reshape((length, 1))

# from sktime.distances.elastic_cython import dtw_distance
from dtw import *

n_series = len(all_series)
distance_matrix = np.zeros(shape=(n_series, n_series))

for i in range(n_series):
    for j in range(n_series):
        x = all_series[i]
        y = all_series[j]
        if i != j:
            # dist = dtw_distance(x, y)
            dist = dtw(x, y, keep_internals=True).distance
            distance_matrix[i, j] = dist

# for i in range(n_series):
#     for j in range(n_series):
#         x1 = stl_series[i].seasonal.values.reshape((len(stl_series[i].seasonal), 1))
#         y1 = stl_series[j].seasonal.values.reshape((len(stl_series[j].seasonal), 1))
#         x2 = stl_series[i].trend.values.reshape((len(stl_series[i].trend), 1))
#         y2 = stl_series[j].trend.values.reshape((len(stl_series[j].trend), 1))
#         if i != j:
#             dist = dtw_distance(x1, y1) + dtw_distance(x2, y2)
#             distance_matrix[i, j] = dist

from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram

def hierarchical_clustering(dist_mat, method='complete'):
    if method == 'complete':
        Z = complete(distance_matrix)
    if method == 'single':
        Z = single(distance_matrix)
    if method == 'average':
        Z = average(distance_matrix)
    if method == 'ward':
        Z = ward(distance_matrix)
    
    fig = plt.figure(figsize=(16, 8))
    dn = dendrogram(Z)
    plt.title(f"Dendrogram for {method}-linkage with dtw distance")
    plt.show()
    
    return Z

linkage_matrix = hierarchical_clustering(distance_matrix)

from scipy.cluster.hierarchy import fcluster

# select maximum number of clusters
cluster_labels = fcluster(linkage_matrix, 20, criterion='maxclust')
print(np.unique(cluster_labels))

# kNN_____________________________________________________________________________________________

# nbrs = NearestNeighbors(n_neighbors=2, radius=0.4)
# knn = NearestNeighbors(n_neighbors=3, metric=DTW)
# knn.fit(seasonal_series)

# hand-select an appropriate cut-off on the dendrogram
# cluster_labels = fcluster(linkage_matrix, 207000, criterion='distance')
# print(np.unique(cluster_labels))

# Agglomerative___________________________________________________________________________________

from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters=10, affinity='precomputed', linkage='average').fit(distance_matrix)
cluster_labels = clustering.labels_
print(clustering.labels_)


# Trendy package___________________________________________________________________________________

# trendy = Trendy(n_clusters=4)
# trendy.fit(seasonal_series[:10])
# print(trendy.labels_)


# plot_____________________________________________________________________________________________

# for i in range(3):
#     cluster = np.where(cluster_labels == i)
#     fig, axs = plt.subplots(cluster[0].size)
#     if cluster[0].size == 1:
#         axs.plot(all_series[cluster[0][0]])
#     else:
#         for j in range(cluster[0].size):
#             axs[j].plot(all_series[cluster[0][j]]) 
#     plt.show()

for i in range(20):
    cluster = np.where(cluster_labels == i)
    for j in range(cluster[0].size):
        plt.plot(all_series[cluster[0][j]])
    plt.show()

# F-measure_____________________________________________________________________________________________

from Decompositions import Decompose
dd = Decompose(df)
low_seasonal = []
mid_seasonal = []
high_seasonal =[]

for word in df.keyword.unique():
    dd.decompose_ma(word)
    # print(f"{word}: seasonal F-measure = {dd.seasonality_F()}")
    if dd.seasonality_F() < 0.3:
        low_seasonal.append(word)
    elif dd.seasonality_F() < 0.6:
        mid_seasonal.append(word)
    else:
        high_seasonal.append(word)

print(low_seasonal)
print(mid_seasonal)
print(high_seasonal)

series = []
for keyword in high_seasonal:
    product = df[df.keyword == keyword]['interest']
    time_series = pd.Series(product.tolist(), index = pd.date_range('1-1-2017', periods=len(product), freq='W'), name = keyword)
    series.append(time_series.to_numpy())




