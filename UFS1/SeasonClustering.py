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


# Import data________________________________________________________________________________________________________
years = range(2016,2021)

df = ApiExtract.extract(years,'NL')

# df1 = ApiExtract.extract(years,'NL')
# df2 = ApiExtract.extract(years,'DE')
# df3 = ApiExtract.extract(years,'ES')
# df = pd.concat([df1, df2, df3])


# Sparsity series____________________________________________________________________________________________________
from Description import Data
data = Data(df)
low_sparsity = []
sparsity = []

for word in df.keyword.unique():
    if data.statistics.sparsity(word) > 0.1:
        sparsity.append(word)
    else:
        low_sparsity.append(word)

sparsity_series = []
for keyword in low_sparsity:
    product = df[df.keyword == keyword]
    for country in product.country.unique():
        time_series = product[product.country == country]["interest"]
        time_series.index = product.startDate.unique()
        sparsity_series.append(time_series.tolist())


# -------------------------------------------------------------------------------------------------------------------
# Distance measures
# -------------------------------------------------------------------------------------------------------------------

# Manhattan Distance_________________________________________________________________________________________________
ts_matrix = np.array(sparsity_series) 
distance_matrix = squareform(pdist(ts_matrix, 'cityblock'))

# Euclidean Distance_________________________________________________________________________________________________
ts_matrix = np.array(sparsity_series) 
distance_matrix = squareform(pdist(ts_matrix, 'euclidean'))

# Chebyshev Distance_________________________________________________________________________________________________
ts_matrix = np.array(sparsity_series) 
distance_matrix = squareform(pdist(ts_matrix, 'chebyshev'))

# Dynamic Time Warping_______________________________________________________________________________________________
n_series = len(sparsity_series)
distance_matrix = np.zeros(shape=(n_series, n_series))

for i in range(n_series):
    for j in range(n_series):
        x = sparsity_series[i]
        y = sparsity_series[j]
        if i != j:
            # dist = dtw_distance(x, y)
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

n_series = len(sparsity_series)
distance_matrix = np.zeros(shape=(n_series, n_series))

for i in range(n_series):
    for j in range(n_series):
        x = sparsity_series[i]
        y = sparsity_series[j]
        if i != j:
            dist = lcss_dist(x, y, delta=np.inf, epsilon=0.5)
            distance_matrix[i, j] = dist


# -------------------------------------------------------------------------------------------------------------------
# Clustering methods
# -------------------------------------------------------------------------------------------------------------------

# Hierarchical clustering____________________________________________________________________________________________
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


# -------------------------------------------------------------------------------------------------------------------
# Determine number of clusters
# -------------------------------------------------------------------------------------------------------------------

# Calinski-Harabasz index____________________________________________________________________________________________
# only useful when using lock-step distance measure (not DTW/LCSS)
CH = []
for i in range(10):
    cluster_labels = fcluster(linkage_matrix, i+2, criterion='maxclust')
    CH.append(metrics.calinski_harabasz_score(sparsity_series, cluster_labels))

# maximize
num_clusters = CH.index(max(CH)) + 2
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')


# C index____________________________________________________________________________________________________________
X = np.array(sparsity_series)

C = []
for i in range(15):
    cluster_labels = fcluster(linkage_matrix, i+2, criterion='maxclust')
    cindex = calc_c_index(X, cluster_labels)
    C.append(cindex)

# minimize
num_clusters = C.index(min(C)) + 2
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')


# -------------------------------------------------------------------------------------------------------------------
# Miscellaneous
# -------------------------------------------------------------------------------------------------------------------

# Plot_______________________________________________________________________________________________________________
for i in range(num_clusters):
    cluster = np.where(cluster_labels == i+1)
    mean_series = np.zeros(np.array(sparsity_series[cluster[0][0]]).size)
    for j in range(cluster[0].size):
        mean_series += np.array(sparsity_series[cluster[0][j]])
        plt.plot(sparsity_series[cluster[0][j]], 'k')
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
# stl_series = []

for keyword in df.keyword.unique():
    product = df[df.keyword == keyword]
    for country in product.country.unique():
        time_series = product[product.country == country]["interest"]
        time_series.index = product.startDate.unique()
        # stl = STL(time_series, robust=True).fit()
        # stl_series.append(stl)
        all_series.append(time_series.tolist())