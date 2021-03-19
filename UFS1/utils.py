from numba import jit as _jit
import numpy as np
# import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram


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


def hierarchical_clustering(distance_matrix, method):
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


