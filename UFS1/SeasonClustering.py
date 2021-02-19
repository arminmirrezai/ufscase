from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import seaborn as sbn
import ApiExtract
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from c_index import calc_c_index
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.clarans import clarans


# Import data________________________________________________________________________________________________________
years = range(2016,2021)

# df = ApiExtract.extract(years,'NL')

df1 = ApiExtract.extract(years,'NL')
df2 = ApiExtract.extract(years,'DE')
df3 = ApiExtract.extract(years,'ES')
df = pd.concat([df1, df2, df3])


# Sparsity series____________________________________________________________________________________________________
from Description import Data
data = Data(df)
low_sparsity = []
mid_sparsity = []
high_sparsity = []

for word in df.keyword.unique():
    if data.statistics.sparsity(word) > 0.9:
        high_sparsity.append(word)
    elif data.statistics.sparsity(word) > 0.3:
        mid_sparsity.append(word)
    else:
        low_sparsity.append(word)

low_sparsity_series = []
for keyword in low_sparsity:
    product = df[df.keyword == keyword]
    for country in product.country.unique():
        time_series = product[product.country == country]["interest"]
        time_series.index = product.startDate.unique()
        low_sparsity_series.append(time_series.tolist())

mid_sparsity_series = []
for keyword in mid_sparsity:
    product = df[df.keyword == keyword]
    for country in product.country.unique():
        time_series = product[product.country == country]["interest"]
        time_series.index = product.startDate.unique()
        mid_sparsity_series.append(time_series.tolist())

no_sparsity = []
for word in df.keyword.unique():
    if data.statistics.sparsity(word) == 0:
        no_sparsity.append(word)

no_sparsity_series = []
for keyword in no_sparsity:
    product = df[df.keyword == keyword]
    for country in product.country.unique():
        time_series = product[product.country == country]["interest"]
        time_series.index = product.startDate.unique()
        no_sparsity_series.append(time_series.tolist())

series = low_sparsity_series

# -------------------------------------------------------------------------------------------------------------------
# Distance measures
# -------------------------------------------------------------------------------------------------------------------

# Manhattan Distance_________________________________________________________________________________________________
ts_matrix = np.array(series) 
distance_matrix = squareform(pdist(ts_matrix, 'cityblock'))

# Euclidean Distance_________________________________________________________________________________________________
ts_matrix = np.array(series) 
distance_matrix = squareform(pdist(ts_matrix, 'euclidean'))

# Chebyshev Distance_________________________________________________________________________________________________
ts_matrix = np.array(series) 
distance_matrix = squareform(pdist(ts_matrix, 'chebyshev'))

# Dynamic Time Warping_______________________________________________________________________________________________
n_series = len(series)
distance_matrix = np.zeros(shape=(n_series, n_series))

for i in range(n_series):
    for j in range(n_series):
        x = series[i]
        y = series[j]
        if i != j:
            dist = dtw(x, y, keep_internals=True).distance
            distance_matrix[i, j] = dist


# Longest Common Subsequence_________________________________________________________________________________________
from numba import jit as _jit

def check_arrays(X, Y):
    X = np.array(X, dtype=np.float)
    Y = np.array(Y, dtype=np.float)
    if X.ndim == 1:
        X = np.reshape(X, (1, X.size))
    if Y.ndim == 1:
        Y = np.reshape(Y, (1, Y.size))
    return X, Y

@_jit(nopython=True)
def _lcss_dist(X, Y, delta, epsilon):
    n_frame_X, n_frame_Y = X.shape[1], Y.shape[1]
    S = np.zeros((n_frame_X+1, n_frame_Y+1))
    for i in range(1, n_frame_X+1):
        for j in range(1, n_frame_Y+1):
            if np.all(np.abs(X[:, i-1]-Y[:, j-1]) < epsilon) and (
                np.abs(i-j) < delta):
                S[i, j] = S[i-1, j-1]+1
            else:
                S[i, j] = max(S[i, j-1], S[i-1, j])
    return 1-S[n_frame_X, n_frame_Y]/min(n_frame_X, n_frame_Y)

def lcss_dist(X, Y, delta, epsilon):
    X, Y = check_arrays(X, Y)
    dist = _lcss_dist(X, Y, delta, epsilon)
    return dist

n_series = len(series)
distance_matrix = np.zeros(shape=(n_series, n_series))

for i in range(n_series):
    for j in range(n_series):
        x = series[i]
        y = series[j]
        if i != j:
            dist = lcss_dist(x, y, delta=np.inf, epsilon=0.5)
            distance_matrix[i, j] = dist


# Discrete Fourier Transform_________________________________________________________________________________________
n_series = len(series)
distance_matrix = np.zeros(shape=(n_series, n_series))
for i in range(n_series):
    for j in range(n_series):
        x = series[i]
        y = series[j]
        if i != j:
            x.dft = np.fft.rfft(x)
            y.dft = np.fft.rfft(y)

            # sort x.dft and y.dft and take first 50/60/70 (variable)

            dist = np.linalg.norm(x.dft - y.dft) 
            distance_matrix[i, j] = dist



# -------------------------------------------------------------------------------------------------------------------
# Clustering methods
# -------------------------------------------------------------------------------------------------------------------

# Hierarchical clustering____________________________________________________________________________________________
def hierarchical_clustering(distance_matrix, method):
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

linkage_matrix = hierarchical_clustering(distance_matrix, 'complete')
linkage_matrix = hierarchical_clustering(distance_matrix, 'ward')

# K-Medoids clustering_______________________________________________________________________________________________

def kMedoids(distance_matrix, num_clusters):
    initial_medoids =  np.random.uniform(0,len(distance_matrix), num_clusters).astype(int).tolist()
    kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    cluster_labels = np.zeros(len(distance_matrix))
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            cluster_labels[clusters[i][j]] = i+1
    return cluster_labels


# CLARANS clustering_________________________________________________________________________________________________

def CLARANS(distance_matrix, num_clusters):
    clarans_instance = clarans(data=series, number_clusters=num_clusters, numlocal=2, maxneighbor=250)
    clarans_instance.process()
    clusters = clarans_instance.get_clusters()
    cluster_labels = np.zeros(len(distance_matrix))
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            cluster_labels[clusters[i][j]] = i+1
    return cluster_labels


# -------------------------------------------------------------------------------------------------------------------
# Determine number of clusters
# -------------------------------------------------------------------------------------------------------------------

# Calinski-Harabasz index____________________________________________________________________________________________
# only useful when using lock-step distance measure (not DTW/LCSS)
CH = []
for i in range(10):
    # cluster_labels = fcluster(linkage_matrix, i+2, criterion='maxclust')
    cluster_labels = kMedoids(distance_matrix, i+2)
    CH.append(metrics.calinski_harabasz_score(distance_matrix, cluster_labels))

# maximize
num_clusters = CH.index(max(CH)) + 2
# cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
cluster_labels = kMedoids(distance_matrix, num_clusters)
CH


# C index____________________________________________________________________________________________________________
X = np.array(series)

C = []
for i in range(15):
    # cluster_labels = fcluster(linkage_matrix, i+2, criterion='maxclust')
    cluster_labels = kMedoids(distance_matrix, i+2)
    cindex = calc_c_index(X, cluster_labels)
    C.append(cindex)

# minimize
num_clusters = C.index(min(C)) + 2
# cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
cluster_labels = kMedoids(distance_matrix, num_clusters)
C


# Silhouette score___________________________________________________________________________________________________
from sklearn.metrics import silhouette_score

silhouette = []
for i in range(15):
    # cluster_labels = fcluster(linkage_matrix, i+2, criterion='maxclust')
    cluster_labels = kMedoids(distance_matrix, i+2)
    silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
    silhouette.append(silhouette_avg)

# minimize
num_clusters = silhouette.index(max(silhouette)) + 2
# cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
cluster_labels = kMedoids(distance_matrix, num_clusters)
silhouette


# Gamma score________________________________________________________________________________________________________
def Baker_Hubert_Gamma(distance_matrix, cluster_labels):
    splus=0
    sminus=0
    pair_dis=squareform(distance_matrix)
    # numPair=len(pairDis)
    num_pair = len(pair_dis)
    # temp = np.zeros((len(self.classLabel),2))
    temp = np.zeros((len(cluster_labels),2))
    temp[:,0]= cluster_labels
    vecB= pdist(temp)
    #iterate through all the pairwise comparisons
    for i in range(num_pair-1):
        for j in range(i+1,num_pair):
            if vecB[i]>0 and vecB[j]==0:
                #heter points smaller than homo points
                if pair_dis[i]<pair_dis[j]:
                    splus=splus+1
                #heter points larger than homo points
                if pair_dis[i]>vecB[j]:
                    sminus=sminus+1
            if vecB[i]==0 and vecB[j]>0:
                #heter points smaller than homo points
                if pair_dis[j]<pair_dis[i]:
                    splus=splus+1
                #heter points larger than homo points
                if pair_dis[j]>vecB[i]:
                    sminus=sminus+1
    #compute the fitness
    gamma = (splus-sminus)/(splus+sminus)
    return gamma


# Gamma score________________________________________________________________________________________________________
def Baker_Hubert_Gamma(distance_matrix, cluster_labels):
    splus=0
    sminus=0
    pair_dis=squareform(distance_matrix)
    # numPair=len(pairDis)
    num_pair = len(pair_dis)
    # temp = np.zeros((len(self.classLabel),2))
    temp = np.zeros((len(cluster_labels),2))
    temp[:,0]= cluster_labels
    vecB= pdist(temp)
    #iterate through all the pairwise comparisons
    for i in range(num_pair-1):
        for j in range(i+1,num_pair):
            if vecB[i]>0 and vecB[j]==0:
                #heter points smaller than homo points
                if pair_dis[i]<pair_dis[j]:
                    splus=splus+1
                #heter points larger than homo points
                if pair_dis[i]>vecB[j]:
                    sminus=sminus+1
            if vecB[i]==0 and vecB[j]>0:
                #heter points smaller than homo points
                if pair_dis[j]<pair_dis[i]:
                    splus=splus+1
                #heter points larger than homo points
                if pair_dis[j]>vecB[i]:
                    sminus=sminus+1
    #compute the fitness
    gamma = (splus-sminus)/(splus+sminus)
    return gamma


# -------------------------------------------------------------------------------------------------------------------
# Miscellaneous
# -------------------------------------------------------------------------------------------------------------------

# Plot_______________________________________________________________________________________________________________
for i in range(num_clusters):
    cluster = np.where(cluster_labels == i+1)
    mean_series = np.zeros(np.array(series[cluster[0][0]]).size)
    for j in range(cluster[0].size):
        mean_series += np.array(series[cluster[0][j]])
        plt.plot(series[cluster[0][j]], 'k')
    mean_series = mean_series / cluster[0].size
    plt.plot(mean_series, 'r')
    plt.show()


# F-measure__________________________________________________________________________________________________________
from Decompositions import Decompose
dd = Decompose(df)
low_seasonal = []
mid_seasonal = []
high_seasonal =[]

for word in df.keyword.unique():
    dd.decompose_robustSTL(word)
    # print(f"{word}: seasonal F-measure = {dd.seasonality_F()}")
    if dd.seasonality_F() < 0.3:
        low_seasonal.append(word)
    elif dd.seasonality_F() < 0.6:
        mid_seasonal.append(word)
    else:
        high_seasonal.append(word)

seasonal_series = []
for keyword in high_seasonal:
    product = df[df.keyword == keyword]['interest']
    time_series = pd.Series(product.tolist(), index = pd.date_range('1-1-2017', periods=len(product), freq='W'), name = keyword)
    seasonal_series.append(time_series.tolist())

# all/stl series_____________________________________________________________________________________________________
all_series = []

for keyword in df.keyword.unique():
    product = df[df.keyword == keyword]
    for country in product.country.unique():
        time_series = product[product.country == country]["interest"]
        time_series.index = product.startDate.unique()
        all_series.append(time_series.tolist())