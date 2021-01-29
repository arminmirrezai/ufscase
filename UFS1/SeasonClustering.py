from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import seaborn as sbn
import ApiExtract
from statsmodels.tsa.seasonal import STL
from croston import croston


# from trendypy.trendy import Trendy

# from sklearn.cluster import AgglomerativeClustering

# from tslearn.clustering import TimeSeriesKMeans


# from scipy.spatial import distance
# from sklearn.neighbors import NearestNeighbors
# from fastdtw import fastdtw

# Configuring Matplotlib
# from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
# register_matplotlib_converters()
# mpl.rcParams['figure.dpi'] = 300
# savefig_options = dict(format="png", dpi=300, bbox_inches="tight")


# def DTW(a, b):   
#     dtw_distance, warp_path = fastdtw(a, b, dist=euclidean)
#     return dtw_distance

# keywords = pd.read_csv(r'C:\Users\Naam\Desktop\seminar case study\ufscode\Data\KeyWords\NL_EN.txt')
# df = pd.read_csv(r'C:\Users\Naam\Desktop\seminar case study\ufscode\Data\data_UFS.csv')

years = range(2016,2021)
df = ApiExtract.extract(years,'NL')
keywords = df.keyword.unique()

i = 0
seasonal_series = np.ones((len(df[df.keyword==keywords[0]]['startDate']), len(keywords)))
for keyword in df.keyword.unique():
    product = df[df.keyword == keyword]['interest']
    time_series = pd.Series(product.tolist(), index = pd.date_range('1-1-2017', periods=len(product), freq='W'), name = keyword)
    stl = STL(time_series, seasonal=13).fit()
    seasonal = stl.trend
    seasonal_series[:,i] = np.array(seasonal)
    i += 1

print(seasonal_series)

# for i in range(len(seasonal_series)):
#     length = len(seasonal_series[i])
#     seasonal_series[i] = seasonal_series[i].values.reshape((length,1)) ################################## RESHAPE SERIES??

#from sktime.distances.elastic_cython import dtw_distance
from tslearn.metrics import dtw

n_series = len(seasonal_series)
distance_matrix = np.zeros(shape=(n_series, n_series))

for i in range(n_series):
    for j in range(n_series):
        x = seasonal_series[:,i]
        y = seasonal_series[:,j]
        if i != j:
            dist = dtw(x, y)
            distance_matrix[i, j] = dist ##################LOWER TRIANGULAR MATRIX?

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
    
    # fig = plt.figure(figsize=(16, 8))
    # dn = dendrogram(Z)
    # plt.title(f"Dendrogram for {method}-linkage with dtw distance")
    # plt.show()
    
    return Z

linkage_matrix = hierarchical_clustering(distance_matrix)

from scipy.cluster.hierarchy import fcluster

# select maximum number of clusters
cluster_labels = fcluster(linkage_matrix, 6, criterion='maxclust')
print(np.unique(cluster_labels))


# distances = []                  ###########################################ELBOW PLOT<<<<<<<<<<>>>>>>>>>>>
# bestClusters = []
# for c in range(1,10+1):
#     print(c)
#     cluster_labels = fcluster(linkage_matrix, t=c, criterion='maxclust')
#     numbOfClusters = max(cluster_labels)
#     cluster1 = []
#     cluster2 = []
#     cluster3 = []
#     cluster4 = []
#     cluster5 = []
#     cluster6 = []
#     cluster7 = []
#     cluster8 = []
#     cluster9 = []
#     cluster10 = []
#     for i in range(len(cluster_labels)):
#         if cluster_labels[i] == 1:
#             cluster1.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 2:
#             cluster2.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 3:
#             cluster3.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 4:
#             cluster4.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 5:
#             cluster5.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 6:
#             cluster6.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 7:
#             cluster7.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 8:
#             cluster8.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 9:
#             cluster9.append(seasonal_series[:,i])
#         elif cluster_labels[i] == 10:
#             cluster10.append(seasonal_series[:,i])
        
#     cluster1 = np.transpose(np.array(cluster1))
#     cluster2 = np.transpose(np.array(cluster2))
#     cluster3 = np.transpose(np.array(cluster3))
#     cluster4 = np.transpose(np.array(cluster4))
#     cluster5 = np.transpose(np.array(cluster5))
#     cluster6 = np.transpose(np.array(cluster6))
#     cluster7 = np.transpose(np.array(cluster7))
#     cluster8 = np.transpose(np.array(cluster8))
#     cluster9 = np.transpose(np.array(cluster9))
#     cluster10 = np.transpose(np.array(cluster10))

#     if i ==6:
#         bestClusters = [cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7]
    
#     averageDists = []

#     for cluster in [cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10]:
#         if len(cluster) == 0:
#             continue
#         n_series = len(cluster[0])
#         totDist = 0
#         for i in range(n_series):
#             for j in range(n_series):
#                 x = cluster[:,i]
#                 y = cluster[:,j]
#                 if i != j:
#                     dist = dtw(x, y)
#                     totDist += dist
#         if totDist > 0:
#             averageDists.append(totDist/len(cluster))

#     distances.append(np.sum(averageDists)/c)

# plt.figure()
# plt.plot(np.arange(1,10+1), distances)
# plt.title("Elbow plot of the average distance within differently sized clusters")
# plt.ylabel("Avg dist wthin clusters")
# plt.xlabel("Number of clusters")
# plt.show()                                 #END##########################################ELBOW PLOT<<<<<<<<<<>>>>>>>>>>>END


# >> 10 unique clusters

# hand-select an appropriate cut-off on the dendrogram
# cluster_labels = fcluster(linkage_matrix, 207000, criterion='distance')
# print(np.unique(cluster_labels))

# model = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=5, max_iter_barycenter=5).fit(seasonal_series[:25])

# km = TimeSeriesKMeans(n_clusters=4, metric="dtw")
# labels = km.fit_predict(seasonal_series)
# print(model.cluster_centers_.shape)

# nbrs = NearestNeighbors(n_neighbors=2, radius=0.4)

# clustering = AgglomerativeClustering(affinity=DTW, linkage='average').fit(seasonal_series[:25])
# print(clustering.labels_)


# trendy = Trendy(n_clusters=4)
# trendy.fit(seasonal_series[:10])
# print(trendy.labels_)

# fig, axs = plt.subplots(15)
# for i in range(15):
#     axs[i].plot(seasonal_series[i])

# no_clusters = np.unique(cluster_labels).size

# for i in range(no_clusters):
#     cluster = np.where(cluster_labels == i+1)
#     fig, axs = plt.subplots(cluster[0].size)
#     for j in range(cluster[0].size):
#         axs[j].plot(seasonal_series[cluster[0][j]])
#     plt.show()

# plt.plot(seasonal_series[i])

for c in range(1,7):
    cluster = np.where(cluster_labels == c)
    fig, axs = plt.subplots(cluster[0].size)
    for j in range(cluster[0].size):
        axs[j].plot(seasonal_series[:,cluster[0][j]])

    plt.show()

# i = 0
# for cluster in bestClusters:#####################################################                        PLOT All clusters
#     fig, axs = plt.subplots(len(cluster[0]))
#     for j in range(len(cluster[0])):
#         axs[j].plot(cluster[:,j])
#     plt.show()

# knn = NearestNeighbors(n_neighbors=3, metric=DTW)
# knn.fit(seasonal_series)

# sc = SpectralClustering(3, affinity='precomputed', n_init=100, assign_labels='discretize')
# sc.fit_predict(adjacency_matrix)

# for keyword in keywords["NL"]:
#     product = data.loc[(data["keyword"] == keyword)]["interest"]
#     time_series = pd.Series(product.tolist(), index = pd.date_range('1-1-2017', periods=len(product), freq='W'), name = keyword)

# trendy = Trendy(n_clusters=4)
# trendy.fit(seasonal_series)
# print(trendy.labels_)

# ackerbohne = pd.Series(ackerbohne, index = pd.date_range('1-1-2017', periods=len(ackerbohne), freq='W'), name = 'Ackerbohne')
# stl = STL(ackerbohne, seasonal=13).fit()
# seasonal = stl.seasonal
# fig = seasonal.plot()
# plt.show()
